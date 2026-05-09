#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CXX=/usr/bin/g++-11
export CC=/usr/bin/gcc-11

# --enforce-eager
# --language-model-only
vllm serve ckpts/Qwen3.6-27B-int4-AutoRound \
    --host 0.0.0.0 \
    --port 8080 \
    --reasoning-parser qwen3 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 3 \
    --max-num-batched-tokens 4128 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --no-scheduler-reserve-full-isl \
    --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --dtype float16 \
    --quantization auto_round
