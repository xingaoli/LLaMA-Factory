from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "ckpts/Qwen3.5-0.8B", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("ckpts/Qwen3.5-0.8B")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "<0.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000000.jpg"},
            {"type": "text",
             "text": "<1.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000005.jpg"},
            {"type": "text",
             "text": "<2.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000010.jpg"},
            {"type": "text",
             "text": "<3.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000015.jpg"},
            {"type": "text",
             "text": "<4.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000020.jpg"},
            {"type": "text",
             "text": "<5.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000025.jpg"},
            {"type": "text",
             "text": "<6.0 seconds>"},
            {"type": "image",
             "image": "Qwen/frames/frame_000029.jpg"},
            {"type": "text", "text": "这是一个车辆在道路前进的前视摄像头视频，请你识别并定位每一帧路面上的常见障碍物，如汽车、非机动车、行人等，结果以json格式返回：{\"time\": 0.0s, \"position\": [x1,y1,x2,y2], \"time\": 1.0s, \"position\": [x1,y1,x2,y2], ...}"},

        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)