from typing import TYPE_CHECKING, Union

import torch
import torch_npu
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers import transformer_ernie_image
from diffusers.utils import is_torch_npu_available
from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers.models.attention_dispatch import _AttentionBackendRegistry, _check_device, _check_qkv_dtype_bf16_or_fp16, \
    _check_shape, _templated_context_parallel_attention, _npu_attention_forward_op, _npu_attention_backward_op, \
    _maybe_modify_attn_mask_npu

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig

from mindiesd import attention_forward, rotary_position_embedding


class ErnieImageSingleStreamAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ErnieImageSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE: same rotate_half logic as Megatron _apply_rotary_pos_emb_bshd (rotary_interleaved=False)
        # x_in: [B, S, heads, head_dim], freqs_cis: [B, S, 1, head_dim] with angles [θ0,θ0,θ1,θ1,...]
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            rot_dim = freqs_cis.shape[-1]
            x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
            cos_ = torch.cos(freqs_cis).to(x.dtype)
            sin_ = torch.sin(freqs_cis).to(x.dtype)
            # Non-interleaved rotate_half: [-x2, x1]
            x1, x2 = x.chunk(2, dim=-1)
            x_rotated = torch.cat((-x2, x1), dim=-1)
            return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)

        def apply_rotary_emb_npu(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            rot_dim = freqs_cis.shape[-1]
            x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
            cos_ = torch.cos(freqs_cis).to(x.dtype)
            sin_ = torch.sin(freqs_cis).to(x.dtype)

            out = rotary_position_embedding(x, cos_, sin_, rotated_mode="rotated_half")
            return torch.cat((out, x_pass), dim=-1)

        if freqs_cis is not None:
            if is_torch_npu_available():
                query = apply_rotary_emb_npu(query, freqs_cis)
                key = apply_rotary_emb_npu(key, freqs_cis)
            else:
                query = apply_rotary_emb(query, freqs_cis)
                key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]

            if attention_mask.ndim == 4 and attention_mask.shape[1:3] == (1, 1):
                attention_mask = attention_mask.expand(-1, attn.heads, query.shape[1], -1)

        # Compute joint attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        output = attn.to_out[0](hidden_states)

        return output


def is_supported_laser_attention(head_dim, q_seqlen, kv_seqlen, ):
    MAX_DIM = 128
    MIN_SEQLEN_SELF = 4000
    MIN_SEQLEN_CROSS = 118404
    MAX_SEQLEN_CROSS = 119056

    if head_dim > MAX_DIM:
        return False
    if q_seqlen == kv_seqlen:
        return q_seqlen >= MIN_SEQLEN_SELF
    else:
        return (MIN_SEQLEN_CROSS <= q_seqlen <= MAX_SEQLEN_CROSS) and \
            (MIN_SEQLEN_CROSS <= kv_seqlen <= MAX_SEQLEN_CROSS)


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_NPU,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _native_npu_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        return_lse: bool = False,
        _parallel_config: Union["ParallelConfig", None] = None
) -> torch.Tensor:
    if return_lse:
        raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
    if _parallel_config is None:
        query, key, value = (
            x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value)
        )

        B, N, Sq, D = query.shape
        B, N, Sk, D = key.shape

        if is_supported_laser_attention(D, Sq, Sk):
            out = attention_forward(query, key, value, opt_mode="manual",
                                    op_type="ascend_laser_attention", layout="BNSD", head_first=True)
        else:
            out = attention_forward(query, key, value, opt_mode="manual", attn_mask=attn_mask,
                                    op_type="fused_attn_score", layout="BNSD", head_first=True)

        out = out.permute(0, 2, 1, 3)

    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            None,
            scale,
            None,
            return_lse,
            forward_op=_npu_attention_forward_op,
            backward_op=_npu_attention_backward_op,
            _parallel_config=_parallel_config,
        )
    return out

transformer_ernie_image.ErnieImageSingleStreamAttnProcessor = ErnieImageSingleStreamAttnProcessor

