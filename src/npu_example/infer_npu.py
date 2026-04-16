import os
import argparse
import time
import torch

import npu_adapter

from diffusers import ErnieImagePipeline
from diffusers.models.transformers.transformer_ernie_image import ErnieImageTransformer2DModel
from diffusers.utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="/models/ERNIE-Image",
                        help="Path to model checkpoint.",
                        )
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="num_inference_steps.",
                        )
    parser.add_argument("--output-path", type=str, default="ernie-image-output.png",
                        help="Path to output.",
                        )
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    device = torch.device("npu")
    width = 1024
    height = 1024
    random_seed = 42
    model_path = args.model_path
    output_path = args.output_path
    num_inference_steps = args.num_inference_steps
    torch_dtype = torch.bfloat16

    generator = torch.Generator(device).manual_seed(random_seed)

    pipe = ErnieImagePipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    # npu attention
    pipe.transformer.set_attention_backend("_native_npu")

    pipe.to(device)

    prompt = "一只黑白相间的中华田园犬"

    print("=====================  Warm up 2 steps ...")
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=4.0,
        generator=generator,
        use_pe=True,
    ).images

    print("========================= Generating image ...")
    torch.npu.synchronize()
    begin = time.time()

    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.0,
        generator=generator,
        use_pe=True,
    ).images
    images[0].save(output_path)

    torch.npu.synchronize()
    end = time.time()
    print(f"============== Generating image used time {end - begin: .4f}s")


if __name__ == '__main__':
    main()

