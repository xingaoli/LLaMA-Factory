#!/bin/bash

#export DISABLE_VERSION_CHECK=1
export API_PORT=8001
llamafactory-cli api \
    --model_name_or_path=ckpt/Qwen2.5-VL-7B-Instruct \
    --template=qwen2_vl \
    --infer_backend=huggingface \
    --trust_remote_code=true \
    --max_length=65536 \
    --quantization_bit=4 \
    --quantization_method=bnb