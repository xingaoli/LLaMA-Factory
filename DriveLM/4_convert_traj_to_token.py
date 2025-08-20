import json
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

# 加载轨迹数据
with open('data/nuscenes_drivelm/scene_sample_token_to_traj_offset.json', 'r') as f:
    traj_data = json.load(f)

# 收集所有x/y坐标以计算真实范围
all_x, all_y, all_offset = [], [], []
for scene in traj_data.values():
    for traj_offset in scene.values():
        traj_offset = np.array(traj_offset)
        all_x.extend(traj_offset[:, 0])
        all_y.extend(traj_offset[:, 1])
        all_offset.append(traj_offset)
all_offset = np.vstack(all_offset)
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)
print(f"X_MIN: {x_min:.3f}, X_MAX: {x_max:.3f}")
print(f"Y_MIN: {y_min:.3f}, Y_MAX: {y_max:.3f}")

all_angles_rad = []  # 存储所有角度（弧度）

# 计算每个差分向量的角度
for dx, dy in zip(all_x, all_y):
    # 使用arctan2计算方向角（-π到π弧度）
    angle_rad = math.atan2(dy, dx)
    all_angles_rad.append(angle_rad)

# 转换为numpy数组便于计算
angles_rad = np.array(all_angles_rad)

# 计算角度范围（弧度）
angle_min_rad = np.min(angles_rad)
angle_max_rad = np.max(angles_rad)

# 转换为角度制（便于理解）
angles_deg = np.degrees(angles_rad)
angle_min_deg = np.min(angles_deg)
angle_max_deg = np.max(angles_deg)

# 输出结果
print(f"ANGLE_MIN (rad): {angle_min_rad:.3f}, ANGLE_MAX (rad): {angle_max_rad:.3f}")
print(f"ANGLE_MIN (deg): {angle_min_deg:.3f}, ANGLE_MAX (deg): {angle_max_deg:.3f}")

# 对速度分布进行统计分析
'''
# 时间间隔（秒）
dt = 0.5
# 速度（m/s）和角度（弧度）
speeds = np.linalg.norm(all_offset, axis=1)
angles = np.arctan2(all_offset[:, 1], all_offset[:, 0])
# 打印统计信息
print(f"🔢 样本数: {len(speeds)}")
print(f"✅ 平均速度: {np.mean(speeds):.3f} delta_m")
print(f"✅ 速度分位数 [5,25,50,75,95]%: {np.percentile(speeds, [5,25,50,75,95])}")
print(f"✅ 平均角度: {np.mean(angles):.3f} rad")
print(f"✅ 角度分位数 [5,25,50,75,95]%: {np.percentile(angles, [5,25,50,75,95])}")
# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(speeds, bins=30, color='orange', edgecolor='black')
plt.title("Speed Distribution (m/s)")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(angles, bins=30, color='skyblue', edgecolor='black')
plt.title("Angle Distribution (radian)")
plt.xlabel("Angle")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
'''

# 对角度分布进行统计分析
'''
# 1. 基本统计量
angle_range = angle_max_rad - angle_min_rad
angle_mean = circmean(angles_rad)  # 环形数据均值[8](@ref)
angle_std = circstd(angles_rad)    # 环形数据标准差[8](@ref)

# 2. 角度分布直方图（36个区间，每10°一个区间）
hist, bin_edges = np.histogram(angles_rad, bins=36, range=(-np.pi, np.pi))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3. 百分位数计算（25%、50%、75%）[6](@ref)
percentiles = np.percentile(angles_rad, [25, 50, 75])

# 4. 输出统计结果
print("===== angle distributed statistic =====")
print(f"角度范围: {np.degrees(angle_min_rad):.1f}° ~ {np.degrees(angle_max_rad):.1f}°")
print(f"平均方向: {np.degrees(angle_mean):.1f}°")
print(f"方向标准差: {np.degrees(angle_std):.1f}°")
print(f"25%分位数: {np.degrees(percentiles[0]):.1f}°")
print(f"中位数: {np.degrees(percentiles[1]):.1f}°")
print(f"75%分位数: {np.degrees(percentiles[2]):.1f}°")

# 5. 可视化角度分布
plt.figure(figsize=(12, 6))
plt.subplot(121, polar=True)
plt.bar(bin_centers, hist, width=np.pi/18, alpha=0.7)
plt.title("angle distributed pole", pad=20)

plt.subplot(122)
plt.bar(np.degrees(bin_centers), hist, width=10)
plt.xlabel("degree(°)")
plt.ylabel("frequency")
plt.title("angle distributed hist")
plt.tight_layout()
plt.show()
'''

# 定义离散化区间（每个坐标轴分256个桶）
GRID_SIZE = 256
# 参数
X_MIN, X_MAX = -14, 12  # 扩一点边界防止截断
Y_MIN, Y_MAX = -2, 55
x_bins = np.linspace(X_MIN, X_MAX, GRID_SIZE)
y_bins = np.linspace(Y_MIN, Y_MAX, GRID_SIZE)

# 定义 token 前缀和特殊标记
SOT_TOKEN = '<sot>'
EOT_TOKEN = '<eot>'
TOKEN_PREFIX_X = 'x_'
TOKEN_PREFIX_Y = 'y_'

def discretize(value, bins):
    return np.digitize(value, bins) - 1  # 映射到0~255

# 构建token化数据
tokenized_data = {}
for scene_id, frames in tqdm(traj_data.items(), desc="Processing scenes"):
    tokenized_data[scene_id] = {}
    for frame_id, traj_offset in frames.items():
        traj_offset = np.array(traj_offset)

        x_seq = [TOKEN_PREFIX_X + str(discretize(x, x_bins)) for x in traj_offset[:, 0]]
        y_seq = [TOKEN_PREFIX_Y + str(discretize(y, y_bins)) for y in traj_offset[:, 1]]

        # 拼接最终token序列
        # tokens = [SOT_TOKEN]
        tokens = []
        for x_token, y_token in zip(x_seq, y_seq):
            tokens.extend([x_token, y_token])
        # tokens.append(EOT_TOKEN)

        tokenized_data[scene_id][frame_id] = tokens

token_to_traj_offset_map = {'x_bins': x_bins.tolist(), 'y_bins': y_bins.tolist()}
with open('data/nuscenes_drivelm/token_to_traj_offset_map_offset.json', 'w') as f:
    json.dump(token_to_traj_offset_map, f, indent=2)

with open('data/nuscenes_drivelm/scene_sample_token_to_traj_token_offset.json', 'w') as f:
    json.dump(tokenized_data, f, indent=2)

print("✅ Tokenized trajectory data saved to scene_sample_token_to_traj_token_offset.json")
