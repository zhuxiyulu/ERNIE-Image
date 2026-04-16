export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
export HCCL_OP_EXPANSION_MODE="AIV"

export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH="/models/ERNIE-Image"
OUTPUT_PATH="ernie-image-output-npu.png"

python infer.py \
    --model-path $MODEL_PATH \
    --num-inference-steps 50 \
    --output-path $OUTPUT_PATH
