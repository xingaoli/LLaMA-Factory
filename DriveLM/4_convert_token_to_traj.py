import json
import random
from tqdm import tqdm
import numpy as np

token_data = json.load(open('data/nuscenes_drivelm/scene_sample_token_to_traj_token_offset.json', 'r'))

traj_offset_map = json.load(open('data/nuscenes_drivelm/token_to_traj_offset_map_offset.json', 'r'))
x_bin_map = traj_offset_map['x_bins']
y_bin_map = traj_offset_map['y_bins']

detokenized_data = {}
for scene_id, frames in tqdm(token_data.items(), desc="Processing scenes"):
    detokenized_data[scene_id] = {}
    for frame_id, traj_token_list in frames.items():

        try:
            traj_token_list = [x_bin_map[int(t[2:])] if t.startswith('x_') else y_bin_map[int(t[2:])] for t in
                                traj_token_list]
            traj_offset = np.array(traj_token_list).reshape(-1, 2)
            detokenized_data[scene_id][frame_id] = traj_offset.tolist()
        except Exception as e:
            print(f"Error processing tokens: {e}\nCurrent token list: {traj_token_list}")

json.dump(detokenized_data, open('data/nuscenes_drivelm/scene_sample_token_to_traj_token_detokenize_offset.json', 'w'), indent=2)
