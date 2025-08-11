import json
import numpy as np

data = json.load(open('data/nuscenes_drivelm/trainval_trajs.json', 'r'))

data_val = np.random.choice(data, 400).tolist()
data_train = [d for d in data if d not in data_val]

json.dump(data_val, open('data/nuscenes_drivelm/val_trajs.json', 'w'), indent=2)
json.dump(data_train, open('data/nuscenes_drivelm/train_trajs.json', 'w'), indent=2)