import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def draw_traj_on_map_patch(scene_token, frame_token, traj_offset, nusc, map_folder, seconds=3.0, freq=2.0):
    sample = nusc.get('sample', frame_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    calib_lidar = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

    ego2global_t = np.array(ego_pose['translation'])
    ego2global_r = Quaternion(ego_pose['rotation'])
    lidar2ego_t = np.array(calib_lidar['translation'])
    lidar2ego_r = Quaternion(calib_lidar['rotation'])

    lidar2global_t = ego2global_t + ego2global_r.rotate(lidar2ego_t)
    lidar2global_r = ego2global_r * lidar2ego_r

    # traj_offset -> global traj
    traj_offset = np.array(traj_offset).cumsum(axis=0)
    abs_pos = [lidar2global_t[:2]]
    for step in traj_offset:
        cur = np.array([step[0], step[1], 0.0])
        pt_global = lidar2global_t + lidar2global_r.rotate(cur)
        abs_pos.append(pt_global[:2])
    abs_pos = np.array(abs_pos)

    # 获取 ground truth ego pose 轨迹（每 0.5 秒 = 2Hz）
    timestamps = []
    sample_cur = sample
    gt_traj = np.zeros((int(seconds * freq) + 1, 2))
    for i in range(int(seconds * freq) + 1):
        lidar_sd_gt = nusc.get('sample_data', sample_cur['data']['LIDAR_TOP'])
        ego_pose_gt = nusc.get('ego_pose', lidar_sd_gt['ego_pose_token'])
        calib_sensor_gt = nusc.get('calibrated_sensor', lidar_sd_gt['calibrated_sensor_token'])
        pos_gt = np.array(ego_pose_gt['translation']) + Quaternion(ego_pose_gt['rotation']).rotate(
            np.array(calib_sensor_gt['translation'])
        )
        gt_traj[i] = pos_gt[:2]
        timestamps.append(ego_pose_gt['timestamp'])  # 微秒数
        if sample_cur['next'] == '':
            gt_traj[i+1:] = gt_traj[i]
            break
        sample_cur = nusc.get('sample', sample_cur['next'])

    # 打印时间间隔检查
    print("timestamp(ms):", [(ts - timestamps[0]) / 1e3 for ts in timestamps])
    print("time length(s):", (timestamps[-1] - timestamps[0]) / 1e6)
    # 地图渲染
    map_name = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
    nusc_map = NuScenesMap(dataroot=map_folder, map_name=map_name)
    center = abs_pos[0]
    patch_radius = 10
    xmin, ymin = center[0] - patch_radius, center[1] - patch_radius
    xmax, ymax = center[0] + patch_radius, center[1] + patch_radius
    patch_polygon = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    fig, ax = plt.subplots(figsize=(10, 10))
    for layer_name in ['road_segment', 'lane', 'road_block']:
        records = getattr(nusc_map, layer_name)
        for record in records:
            if record['polygon_token'] == '':
                continue
            polygon = nusc_map.extract_polygon(record['polygon_token'])
            if polygon.intersects(patch_polygon):
                x, y = polygon.exterior.xy
                ax.plot(x, y, linewidth=1, alpha=0.6, color='black')


    # 红线：预测轨迹
    ax.plot(abs_pos[:, 0], abs_pos[:, 1], color='red', linewidth=2, label='Pred')

    # 蓝点：真实轨迹
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], color='blue', linestyle='--', label='GT')

    ax.set_title(f"scene token: {scene_token[:6]} sample token: {frame_token[:6]}")
    ax.legend()
    ax.axis('off')
    plt.axis('equal')
    plt.grid(False)
    fig.canvas.draw()
    plt.tight_layout()
    # plt.show()
    save_name = frame_token
    plt.savefig(
        f'output/vis/{save_name}.png',
        dpi=300,
        bbox_inches='tight',
        transparent=True,
        facecolor='white'
    )

# 设置路径
drivelm_result_path = "data/nuscenes_drivelm/val_trajs_pred.json"
nusc_root = "data/nuscenes_drivelm"
map_folder = "data/nuscenes_drivelm"
nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root)

with open(drivelm_result_path, "r") as f:
    traj_data = json.load(f)

for scene_token, frame_dict in traj_data.items():
    for frame_token, traj_offset in frame_dict.items():
        draw_traj_on_map_patch(scene_token, frame_token, traj_offset, nusc, map_folder)
    #     break  # 只看一帧
    # break
