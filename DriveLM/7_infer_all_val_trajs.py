import json
import os
import sys

import numpy as np
from openai import OpenAI
import time
import base64
import random

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

eval_data = json.load(open('data/nuscenes_drivelm/val_trajs_offset.json', 'r'))
eval_data = random.sample(eval_data, 20)

traj_offset_map = json.load(open('data/nuscenes_drivelm/token_to_traj_offset_map_offset.json', 'r'))
x_bin_map = traj_offset_map['x_bins']
y_bin_map = traj_offset_map['y_bins']
output = {}
for idx, data in enumerate(eval_data):
    id = data['id']
    print("Process Idx:", idx)
    print("Process Sample:", id)
    scene_token, sample_token = id.split('_')
    image_path = data['images']
    image_path = [os.path.join('data', p) for p in image_path]
    image_path_fl = image_path[0]
    image_path_f = image_path[1]
    image_path_fr = image_path[2]
    image_path_bl = image_path[3]
    image_path_b = image_path[4]
    image_path_br = image_path[5]
    base64_qwen_fl = image_to_base64(image_path_fl)
    base64_qwen_f = image_to_base64(image_path_f)
    base64_qwen_fr = image_to_base64(image_path_fr)
    base64_qwen_bl = image_to_base64(image_path_bl)
    base64_qwen_b = image_to_base64(image_path_b)
    base64_qwen_br = image_to_base64(image_path_br)
    system = data['system']
    behavior = data['conversations'][0]['value'].split('\n')[1]

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_fl}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_f}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_fr}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_bl}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_b}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_qwen_br}"}},
                {"type": "text", "text": behavior},
            ],
        }
    ]
    start_time = time.time()
    completion = client.chat.completions.create(
        model="qwen2.5-vl",
        messages=messages,
        # max_tokens=8192,
        # temperature=0.7
    )
    end_time = time.time()
    print("Time:", end_time - start_time)
    print("Completion result:", completion.choices[0].message.content)
    if scene_token not in output:
        output[scene_token] = {}
    # traj_token_list = completion.choices[0].message.content.split(' ')
    traj_token_list = ['x_101', 'y_53', 'x_105', 'y_77', 'x_108', 'y_88']
    try:
        traj_offset_list = [x_bin_map[int(t[2:])] if t.startswith('x_') else y_bin_map[int(t[2:])] for t in traj_token_list]
        traj_offset = np.array(traj_offset_list).reshape(-1, 2)
        output[scene_token][sample_token] = traj_offset.tolist()
    except Exception as e:
        print(f"Error processing tokens: {e}\nCurrent token list: {traj_token_list}")

json.dump(output, open('data/nuscenes_drivelm/val_trajs_pred_offset.json', 'w'), indent=2)
