import numpy as np
import json


SYSTEM = """You are an autonomous driving trajectory planner. Process the following inputs in EXACT ORDER:
1. Left-front camera image
2. Front camera image
3. Right-front camera image
4. Left-rear camera image
5. Rear camera image
6. Right-rear camera image
7. Textual behavior description (e.g., 'turning left slowly')

Output requirements:
- Generate ONLY space-separated trajectory tokens like 'x_127 y_21'
- Never describe or explain your output
- Example valid output: x_120 y_15 x_122 y_18 x_125 y_25"""

def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    with open(trajs_root, 'r') as f:
        trajs_file = json.load(f)

    output = []
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("../", "") for key in image_paths.keys()]
            image_paths = [key.replace("nuscenes", "nuscenes_drivelm") for key in image_paths]
            image_paths = [image_paths[1], image_paths[0], image_paths[2], image_paths[4], image_paths[3], image_paths[5]]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["behavior"]

            traj = ' '.join(trajs_file[scene_id][frame_id])

            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id,
                        "system": SYSTEM,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image><image><image><image><image><image>\n" + answer
                            },
                            {
                                "from": "gpt",
                                "value": traj
                            },
                        ],
                        "images": image_paths
                    }
                )

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    root = "data/nuscenes_drivelm/trainval.json"
    trajs_root = "data/nuscenes_drivelm/scene_sample_token_to_traj_token_offset.json"
    dst = "data/nuscenes_drivelm/trainval_trajs_offset.json"
    convert2llama(root, dst)
