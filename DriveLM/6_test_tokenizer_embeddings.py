import os
import torch
import numpy as np
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


new_model_path = "ckpt/Qwen2.5-VL-7B-Instruct-add-token"
# ===== 加载原始模型和分词器 =====
print("加载原始模型...")
tokenizer = AutoTokenizer.from_pretrained(new_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    new_model_path,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True
)

ids = tokenizer.encode("x_0", add_special_tokens=False)
print(ids, tokenizer.convert_ids_to_tokens(ids))   # 期望得到 [..., x_0_id, ...]

cos = torch.nn.CosineSimilarity(dim=-1)
sim = cos(model.base_model.language_model.embed_tokens.weight[tokenizer.convert_tokens_to_ids("x_12")],
          model.base_model.language_model.embed_tokens.weight[tokenizer.convert_tokens_to_ids("9")])
print(sim.item())   # 应接近 1（因为 x_0 用 0 的向量初始化）