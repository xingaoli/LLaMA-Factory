import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def extract_drivelm_scene_keyframe_tokens(json_path):
    """
    提取 DriveLM 数据集中每个 scene 的 scene_token 及其对应的 key_frame token 列表。
    
    参数:
        json_path: str, DriveLM JSON 文件路径（如 drivelm_nuscenes.json）
    
    返回:
        dict: 结构为 {scene_token: [key_frame_token1, key_frame_token2, ...]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    scene_to_keyframes = {}

    for scene_token, scene_info in data.items():
        key_frame_tokens = list(scene_info.get("key_frames", {}).keys())
        scene_to_keyframes[scene_token] = key_frame_tokens

    return scene_to_keyframes

def get_global_sensor_pose(sample, nusc, inverse=False):
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    calib_sensor = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

    if not inverse:
        global_from_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=False)
        ego_from_sensor = transform_matrix(calib_sensor['translation'], Quaternion(calib_sensor['rotation']), inverse=False)
        return global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(calib_sensor['translation'], Quaternion(calib_sensor['rotation']), inverse=True)
        ego_from_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)
        return sensor_from_ego.dot(ego_from_global)

def get_ego_future_trajectory_offset(nusc, sample_token, seconds=3.0, freq=2.0):
    """
    提取未来轨迹点（相对 LIDAR 当前帧坐标系），返回 step-wise offset（以 local lidar 坐标系表示）
    """
    fut_ts = int(seconds * freq)
    sample = nusc.get('sample', sample_token)

    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

    ego_fut_trajs = np.zeros((fut_ts+1, 3))
    sample_cur = sample

    for i in range(fut_ts+1):
        pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
        ego_fut_trajs[i] = pose_mat[:3, 3]
        if sample_cur['next'] == '':
            ego_fut_trajs[i+1:] = ego_fut_trajs[i]
            break
        sample_cur = nusc.get('sample', sample_cur['next'])

    # global to ego at lidar
    ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
    rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
    rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

    # per-step offset
    traj_offsets = ego_fut_trajs[1:] - ego_fut_trajs[:-1]  # shape (fut_ts, 3)

    return ego_fut_trajs[1:, :2], traj_offsets[:, :2]  # 只返回 x, y

def extract_lidar_offset_trajectory(nusc, sample_token, seconds=3.0, freq=2.0):
    """
    正确地以当前帧 LIDAR 坐标系为参考系提取未来 3s 的自车轨迹 offset
    步骤：
    1. 提取当前帧 pose（ego_pose + calibrated_sensor）形成参考坐标系
    2. 提取 t+0 ~ t+N 每帧的 global LIDAR 位置（每0.5秒一帧）
    3. 全部点转换到当前 LIDAR 坐标系下
    4. 做差分构造 offset
    返回：N×2 的轨迹点序列（单位：米）
    """
    fut_ts = int(seconds * freq)
    sample = nusc.get('sample', sample_token)

    # 获取当前帧 pose（参考系）
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    calib_sensor = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ego2global_t = np.array(ego_pose['translation'])
    ego2global_r = Quaternion(ego_pose['rotation'])
    lidar2ego_t = np.array(calib_sensor['translation'])
    lidar2ego_r = Quaternion(calib_sensor['rotation'])

    lidar_global_T = ego2global_t + ego2global_r.rotate(lidar2ego_t)
    lidar_global_R = ego2global_r * lidar2ego_r

    # 获取所有未来帧的 global LIDAR 坐标
    sample_cur = sample
    global_points = np.zeros((fut_ts + 1, 3))
    for i in range(fut_ts + 1):
        sd_cur = nusc.get('sample_data', sample_cur['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', sd_cur['ego_pose_token'])
        calib_lidar = nusc.get('calibrated_sensor', sd_cur['calibrated_sensor_token'])

        lidar2global_t = np.array(ego_pose['translation']) + Quaternion(ego_pose['rotation']).rotate(np.array(calib_lidar['translation']))
        global_points[i] = lidar2global_t

        if sample_cur['next'] == '':
            global_points[i+1:] = global_points[i]
            break
        sample_cur = nusc.get('sample', sample_cur['next'])


    # 将所有 global 点变换到当前 LIDAR 坐标系（t=0）
    local_points = []
    for p in global_points:
        next2cur_t_in_global_coord = p - lidar_global_T
        next2cur_t = lidar_global_R.inverse.rotate(next2cur_t_in_global_coord)
        local_points.append(next2cur_t)

    local_points = np.array(local_points)

    # 计算 offset（差分）
    offsets = local_points[1:]- local_points[:-1]

    return local_points[1:, :2], offsets[:, :2]  # 只要 x, y

def extract_trajectory_for_drivelm(drivelm_json_path, nusc_root, nusc_version='v1.0-trainval'):
    """
    集成版本：对每个 DriveLM 的 scene/frame，提取对应的 sample_token 和未来轨迹（3 秒，2Hz）
    返回结构：{scene_token: {frame_token: trajectory_points}}
    """
    nusc = NuScenes(version=nusc_version, dataroot=nusc_root)
    scene_to_keyframes = extract_drivelm_scene_keyframe_tokens(drivelm_json_path)
    result = {}

    for scene_token, key_frames in scene_to_keyframes.items():
        result[scene_token] = {}
        for frame_token in key_frames:
            try:
                traj, traj_offset = extract_lidar_offset_trajectory(nusc, frame_token, seconds=3.0, freq=2.0)
                result[scene_token][frame_token] = traj_offset.tolist()
            except Exception as e:
                print(f"跳过无效 frame_token: {frame_token}, 错误: {e}")
                continue

    return result

# 示例用法：
if __name__ == "__main__":
    drivelm_json_path = "data/nuscenes_drivelm/v1_1_train_nus.json"  # TODO: 替换为你的实际路径
    nusc_root = "data/nuscenes_drivelm"

    result = extract_trajectory_for_drivelm(
        drivelm_json_path,
        nusc_root,
        nusc_version='v1.0-trainval'
    )
    # 保存结果
    with open("data/nuscenes_drivelm/scene_sample_token_to_traj_offset.json", "w") as f:
        json.dump(result, f, indent=2)

    print("轨迹提取完成，已保存为 scene_sample_token_to_traj_offset.json")
