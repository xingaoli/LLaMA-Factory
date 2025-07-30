# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/home/xingao/code/LLaVA/ckpt/llava-v1.5-7b")
# vocab = tokenizer.get_vocab()  # 获取 {token: id} 字典
# print(list(vocab.items())[:10])  # 打印前10个 token-ID 对


import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="InternVL2_5-tokenizer.model")
vocab_size = sp.get_piece_size()  # 词汇表大小
for i in range(vocab_size):
    token = sp.id_to_piece(i)
    print(f"{token}: {i}")  # 输出 token 和 ID
