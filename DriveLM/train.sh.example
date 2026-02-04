#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
#export FORCE_TORCHRUN=1
export TOKENIZERS_PARALLELISM=false
llamafactory-cli train \
    --model_name_or_path=ckpt/Qwen2.5-VL-7B-Instruct-add-token \
    --quantization_bit=4 \
    --quantization_method=bnb  \
    --trust_remote_code=true \
    --image_max_pixels: 200704 \
    --image_min_pixels: 3136 \
    --stage=sft \
    --do_train=true \
    --finetuning_type=lora \
    --lora_rank=8 \
    --lora_target=all \
    --dataset_dir=data \
    --dataset=nuscenes_drivelm_train \
    --eval_dataset=nuscenes_drivelm_val \
    --template=qwen2_vl \
    --cutoff_len=4096 \
    --max_samples=100 \
    --overwrite_cache=false \
    --tokenized_path=data/cache/nuscenes_drivelm \
    --preprocessing_num_workers=8 \
    --dataloader_num_workers=8 \
    --output_dir=output/Qwen2.5-VL-7B-Instruct-add-token/debug \
    --logging_steps=10 \
    --save_steps=500 \
    --plot_loss=true \
    --overwrite_output_dir=true \
    --save_only_model=false \
    --report_to=none \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1.0e-4 \
    --num_train_epochs=3.0 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.1 \
    --bf16=true \
    --ddp_timeout=180000000 \
    --per_device_eval_batch_size=1 \
    --eval_strategy=steps \
    --eval_steps=500