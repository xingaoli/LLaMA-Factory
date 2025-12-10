## 这个仓库主要是LLaMA-Factory的一些用法，有的api很难找

1. 训练相关参数
```yaml
### model
model_name_or_path: ckpt/Qwen2.5-VL-7B-Instruct-add-token
quantization_bit: 4  # choices: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]
quantization_method: bnb  # choices: [bnb, hqq, eetq]
trust_remote_code: true
image_max_pixels: 200704
image_min_pixels: 3136

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
# 给embed_token也加一层lora还是不fix直接训？反正是想训lm_head用下面这个参数
additional_target: embed_tokens,lm_head
deepspeed: /home/sa1ad/xingao/code/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### dataset
dataset_dir: data # 指定数据检索的路径，默认data
dataset: nuscenes_drivelm_train_offset
eval_dataset: nuscenes_drivelm_val_offset
template: qwen2_vl
cutoff_len: 16384

#max_samples: 50
# 依稀记得数据集缓存cache这玩意儿机制很复杂，反正如果你想显示的控制生成的数据集缓存，那就用下边的两个参数，cache重写设为false，然后指定数据集cache路径，第一次会生成arrow data，后续都会直接用这个数据集，改了配置想重新生成就得换个路径，不然不会生成新的数据（不确定？是否是改变了一些重大的config，即使路径里有cache也会被overwrite？如果不想折腾就别用这俩参数，默认好像是在~/.cache/huggingface下边无感存储的）
overwrite_cache: false
tokenized_path: data/cache/nuscenes_drivelm_offset
preprocessing_num_workers: 64
dataloader_num_workers: 16

#上述数据处理是预处理的，如果数据集比较大就要等很久，还有一种预处理的streaming方案。
#streaming: True
#max_steps: 100
#buffer_size: 128
#preprocessing_batch_size: 128
#preprocessing_num_workers: 8
#dataloader_num_workers: 8
#accelerator_config:
#  dispatch_batches: false

### output
output_dir: output/model/Qwen2.5-VL-7B-Instruct-add-token/lora-sft-adapter-offset-0819-2147
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 10.0
lr_scheduler_type: constant
#warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

## eval
val_size: 0
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```
训练命令：
```shell
#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export FORCE_TORCHRUN=1 
# 内部调用torchrun的，如果环境是直接离线移植的确认torchrun路径
llamafactory-cli train train.yaml
```

2. 推理相关参数
```yaml
model_name_or_path: ckpt/Qwen2.5-VL-7B-Instruct-add-token
#adapter_name_or_path: output/InternVL2_5-8B-MPO-hf/qlora-sft-self-cognition-adapter
template: qwen2_vl
infer_backend: huggingface  # choices: [huggingface, vllm, sglang]
trust_remote_code: true
quantization_bit: 4  # choices: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]
quantization_method: bnb  # choices: [bnb, hqq, eetq]
```
如果是网页推理：
```shell
llamafactory-cli webchat interence.yaml
```

如果是cli推理：
```shell
llamafactory-cli chat interence.yaml
```

如果是部署一个api服务来进行批量推理：
```shell
export API_PORT=8001
llamafactory-cli api interence.yaml
# 部署完需要在开一份代码，跟使用gpt api一样调用这个模型
```
当然还有不使用llamafactory-cli来部署批量推理的更高级用法，可以参考Impromptu-VLA中的vllm_infer.py,sglang_infer.py，他们没有直接用llamafactory-cli，但是依然是用了很多llamafactory的api和代码。我个人觉得使用llamafactory-cli api部署再调用的推理代码更简洁。

3. lora微调的adapter合并到原权重：
```yaml
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters
# 合并的似乎不能是量化模型

### model
model_name_or_path: ckpt/Qwen2.5-VL-7B-Instruct-add-token
adapter_name_or_path: output/Qwen2.5-VL-7B-Instruct-add-token/xxx
template: qwen2_vl
trust_remote_code: true

### export
export_dir: ckpt_my/Qwen2.5-VL-7B-Instruct-add-token-sft
export_size: 5 # 合并后导出的safetensor个数？
export_device: cpu  # choices: [cpu, auto]
export_legacy_format: false
```
指令：
```shell
llamafactory-cli export merge.yaml
```
