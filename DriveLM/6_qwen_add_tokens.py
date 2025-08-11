import os
import torch
import numpy as np
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def save_model_with_updated_config(
        model,
        tokenizer,
        output_dir,
        max_shard_size="4GB",
        safe_serialization=True
):
    """
    智能保存模型（自动更新所有配置）
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 更新词汇表相关配置
    model.config.vocab_size = len(tokenizer)

    # # 更新特殊token ID
    # updated_config.update({
    #     "bos_token_id": tokenizer.bos_token_id,
    #     "eos_token_id": tokenizer.eos_token_id,
    #     "unk_token_id": tokenizer.unk_token_id,
    #     "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    # })

    # 3. 保存分词器
    tokenizer.save_pretrained(output_dir)

    # 4. 保存模型权重和更新后的配置
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size
    )

    # 5. 验证文件完整性
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",  # 多模态模型需要
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
    if missing_files:
        print(f"警告：缺少以下文件: {missing_files}")
    else:
        print(f"模型和配置已完整保存到: {output_dir}")


def save_multimodal_model(model, tokenizer, original_dir, output_dir):
    """处理多模态模型的特殊配置"""
    # 1. 获取原始processor配置（如果存在）
    processor_config = {}
    processor_path = os.path.join(original_dir, "preprocessor_config.json")
    if os.path.exists(processor_path):
        with open(processor_path, "r") as f:
            processor_config = json.load(f)

    # # 2. 更新视觉部分配置
    # if hasattr(model.config, "vision_config"):
    #     processor_config.update({
    #         "image_size": model.config.vision_config.image_size,
    #         "patch_size": model.config.vision_config.patch_size,
    #         # 其他视觉参数...
    #     })

    # 3. 保存processor配置
    with open(os.path.join(output_dir, "preprocessor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)

    # 4. 调用通用保存函数
    save_model_with_updated_config(model, tokenizer, output_dir)

# 配置参数
base_model_path = "ckpt/Qwen2.5-VL-7B-Instruct"
new_model_path = "ckpt/Qwen2.5-VL-7B-Instruct-add-token"
new_x_tokens = [f"x_{i}" for i in range(256)]
new_y_tokens = [f"y_{i}" for i in range(256)]
all_new_tokens = new_x_tokens + new_y_tokens

# ===== 加载原始模型和分词器 =====
print("加载原始模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True
)

# =====  添加新token =====
print(f"原始词表大小: {len(tokenizer)}")
num_added = tokenizer.add_tokens(all_new_tokens)
print(f"新增 {num_added} token，当前词表: {len(tokenizer)}")

# 扩展嵌入维度
model.resize_token_embeddings(len(tokenizer))
new_vocab_size = len(tokenizer)

embed_layer = model.get_input_embeddings()
lm_head = model.get_output_embeddings()
# 获取数字token的嵌入向量
digit_tokens = [str(i) for i in range(10)]  # "0"到"9"
digit_ids = tokenizer.convert_tokens_to_ids(digit_tokens)
digit_embeddings = embed_layer.weight.data[digit_ids]

# 计算数字token的平均向量
base_vector = digit_embeddings.mean(dim=0)

# 检查是否共享权重
tie_weights  = id(embed_layer.weight) == id(lm_head.weight)
print(f"Embedding & lm_head share the same weight? {tie_weights}")

# ===== 初始化新token向量 =====
print("初始化新token的嵌入向量...")
new_token_ids = tokenizer.convert_tokens_to_ids(all_new_tokens)
new_embeddings = []

# 为每个新token创建嵌入向量
for token in all_new_tokens:
    if token.startswith("x_") or token.startswith("y_"):
        # 数字相关token初始化
        token_number = int(token.split("_")[1])
        # 基于十进制位的组合
        digits = [int(d) for d in str(token_number)]
        digit_weights = digit_embeddings[[d for d in digits]]

        # 根据数字位加权平均
        weights = torch.tensor([10 ** (len(digits) - i - 1) for i in range(len(digits))])
        weights = weights / weights.sum()
        weighted_avg = torch.sum(digit_weights * weights.view(-1, 1), dim=0)

        # 添加随机扰动保持多样性
        new_vec = weighted_avg + torch.randn_like(weighted_avg) * 0.005
    else:
        # 普通token初始化策略
        new_vec = base_vector + torch.randn_like(base_vector) * 0.02

    new_embeddings.append(new_vec.clone())

new_embeddings = torch.as_tensor(torch.stack(new_embeddings), dtype=torch.bfloat16)
# 批量更新权重
with torch.no_grad():
    embed_layer.weight.data[new_token_ids] = new_embeddings
    if not tie_weights:
        lm_head.weight.data[new_token_ids] = new_embeddings

# ===== 5. 完整保存新模型 =====
print("保存扩展后的模型...")
os.makedirs(new_model_path, exist_ok=True)

save_multimodal_model(model, tokenizer, base_model_path, new_model_path)

print(f"扩展模型已完整保存至: {new_model_path}")