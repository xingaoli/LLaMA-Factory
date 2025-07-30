import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Blip2Processor
import json
import os


class TrajectoryBinner:
    def __init__(self):
        # 基于nuScenes的坐标分布特征 (ego坐标系)
        self.x_range = (-15, 15)  # 横向范围 (左到右)
        self.y_range = (-40, 60)  # 纵向范围 (后到前)

        # BLIP-2处理器加载
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.tokenizer = self.processor.tokenizer

        # 数字token准备
        self._prepare_number_tokens()

    def _prepare_number_tokens(self):
        """预加载数字token并建立映射关系"""
        self.number_tokens = []

        # 尝试加载三位数格式的token（000-255）
        for num in range(256):
            num_str = f"{num:03d}"  # 固定三位数格式
            if num_str in self.tokenizer.vocab:
                self.number_tokens.append(num_str)
            else:
                # 尝试两位或一位数格式
                alt_str = str(num)
                if alt_str in self.tokenizer.vocab:
                    self.number_tokens.append(alt_str)
                else:
                    # 最后手段：使用第一个未使用token（确保安全）
                    unused_tokens = [tok for tok, idx in self.tokenizer.vocab.items() if tok.startswith("<unused")]
                    if unused_tokens:
                        self.number_tokens.append(unused_tokens[0])
                    else:
                        raise ValueError(f"Cannot find token for bin {num}")

        print(f"Successfully mapped {len(self.number_tokens)} bins to tokens.")

    def generate_simulated_data(self, num_points=10000):
        """生成符合nuScenes分布规律的模拟轨迹点"""
        np.random.seed(42)  # 固定随机种子以便复现

        # 横向坐标：集中在0附近的正态分布
        x_points = np.random.normal(0, 4, num_points)
        x_points = np.clip(x_points, *self.x_range)

        # 纵向坐标：大部分点在车辆前方（0-60m），少量在后方（-40m-0）
        # 使用混合分布：75%的点在前方，25%在后方
        front_mask = np.random.rand(num_points) > 0.25
        # 前方：均值为20m，标准差为15m的正态分布（取绝对值确保非负）
        front_points = np.abs(np.random.normal(20, 15, num_points))
        # 后方：均匀分布
        rear_points = np.random.uniform(-40, 0, num_points)

        y_points = np.where(front_mask, front_points, rear_points)
        y_points = np.clip(y_points, *self.y_range)

        return np.column_stack((x_points, y_points))

    def compute_bins(self, points=None, strategy='quantile'):
        """
        计算分bin边界
        strategy:
          'uniform' - 均匀分bin
          'quantile' - 按密度分bin
          'mixed' - 横向均匀，纵向按密度分（推荐）
        """
        if points is None:
            points = self.generate_simulated_data(10000)

        x_points, y_points = points[:, 0], points[:, 1]

        if strategy == 'uniform':
            # 两个轴都均匀分bin
            x_bins = np.linspace(*self.x_range, 257)[1:-1]
            y_bins = np.linspace(*self.y_range, 257)[1:-1]

        elif strategy == 'quantile':
            # 两个轴都按密度分bin
            x_bins = np.unique(np.percentile(x_points, np.linspace(0, 100, 257)))
            y_bins = np.unique(np.percentile(y_points, np.linspace(0, 100, 257)))
            # 确保边界数量为255（256个区间）
            if len(x_bins) < 255:
                x_bins = np.linspace(min(x_points), max(x_points), 257)[1:-1]
            if len(y_bins) < 255:
                y_bins = np.linspace(min(y_points), max(y_points), 257)[1:-1]
            # 截取255个边界点
            x_bins = x_bins[:255]
            y_bins = y_bins[:255]

        elif strategy == 'mixed':
            # 横向（X轴）均匀分bin（因为横向范围固定）
            x_bins = np.linspace(*self.x_range, 257)[1:-1]

            # 纵向（Y轴）按密度分bin（考虑前方需要更精细）
            # 我们将纵向分成两个区域：后方（-40~0）和前方（0~60）
            rear_points = y_points[y_points <= 0]
            front_points = y_points[y_points > 0]

            # 后方分配64个bins，前方分配192个bins
            rear_bins = np.percentile(rear_points, np.linspace(0, 100, 65)) if len(rear_points) > 0 else []
            front_bins = np.percentile(front_points, np.linspace(0, 100, 193)) if len(front_points) > 0 else []

            # 合并并去重
            y_bins = np.unique(np.concatenate([rear_bins, front_bins]))
            # 确保边界数量不超过255
            if len(y_bins) > 255:
                y_bins = np.percentile(y_points, np.linspace(0, 100, 257))[1:-1]
            elif len(y_bins) < 255:
                y_bins = np.linspace(*self.y_range, 257)[1:-1]

        return {
            'x': {'bounds': x_bins, 'min': self.x_range[0], 'max': self.x_range[1]},
            'y': {'bounds': y_bins, 'min': self.y_range[0], 'max': self.y_range[1]}
        }

    def plot_distribution(self, points, bins, strategy='mixed'):
        """可视化坐标分布和分bin结果"""
        plt.figure(figsize=(14, 6))

        # X轴分布
        plt.subplot(1, 2, 1)
        sns.histplot(points[:, 0], kde=True)
        for b in bins['x']['bounds']:
            plt.axvline(b, color='r', alpha=0.3, linewidth=0.5)
        plt.title("X-coordinate Distribution")
        plt.xlabel("Meters (Lateral)")

        # Y轴分布
        plt.subplot(1, 2, 2)
        sns.histplot(points[:, 1], kde=True)
        for b in bins['y']['bounds']:
            plt.axvline(b, color='r', alpha=0.3, linewidth=0.5)
        plt.title(f"Y-coordinate Distribution (Strategy: {strategy})")
        plt.xlabel("Meters (Longitudinal)")

        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/trajectory_bin_{strategy}.png", dpi=200)
        plt.close()
        print(f"Saved distribution plot for strategy '{strategy}'")

    def point_to_bin_id(self, point, bins):
        """将轨迹点映射到bin ID (0-255)"""
        x, y = point
        # 处理x坐标
        if x <= bins['x']['min']:
            x_idx = 0
        elif x >= bins['x']['max']:
            x_idx = 255
        else:
            x_idx = np.searchsorted(bins['x']['bounds'], x, side='right')

        # 处理y坐标
        if y <= bins['y']['min']:
            y_idx = 0
        elif y >= bins['y']['max']:
            y_idx = 255
        else:
            y_idx = np.searchsorted(bins['y']['bounds'], y, side='right')

        # 确保索引在0-255范围内
        x_idx = min(max(x_idx, 0), 255)
        y_idx = min(max(y_idx, 0), 255)

        return x_idx, y_idx

    def bin_id_to_token(self, bin_id):
        """将bin ID映射到数字token"""
        if bin_id < 0 or bin_id > 255:
            raise ValueError(f"Bin ID {bin_id} out of range [0, 255]")
        return self.number_tokens[bin_id]

    def create_bin_mapping_table(self, bins):
        """创建完整的bin映射表（用于记录）"""
        mapping = []
        for i in range(256):
            # X轴范围计算
            if i == 0:
                x_low = bins['x']['min']
                x_high = bins['x']['bounds'][0]
            elif i == 255:
                x_low = bins['x']['bounds'][-1]
                x_high = bins['x']['max']
            else:
                x_low = bins['x']['bounds'][i - 1]
                x_high = bins['x']['bounds'][i]

            # Y轴范围同理（实际应用需分别处理XY，此处简化）
            token_str = self.bin_id_to_token(i)
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)

            mapping.append({
                "bin_id": i,
                "token": token_str,
                "token_id": token_id,
                "x_range": (round(float(x_low), 2), round(float(x_high), 2))
            })
        return mapping

    def save_mapping(self, bins, strategy='mixed', file_path=None):
        """保存完整的映射关系表"""
        if file_path is None:
            file_path = f"outputs/bin_mapping_{strategy}.json"

        mapping = {
            "strategy": strategy,
            "coordinate_ranges": {
                "x": [float(self.x_range[0]), float(self.x_range[1])],
                "y": [float(self.y_range[0]), float(self.y_range[1])]
            },
            "bin_specification": {
                "x_num_bins": len(bins['x']['bounds']) + 1,
                "y_num_bins": len(bins['y']['bounds']) + 1,
                "x_bins": [round(float(b), 4) for b in bins['x']['bounds']],
                "y_bins": [round(float(b), 4) for b in bins['y']['bounds']]
            },
            "token_mapping": [
                {
                    "bin_id": i,
                    "token": self.number_tokens[i],
                    "token_id": self.tokenizer.convert_tokens_to_ids(self.number_tokens[i])
                }
                for i in range(256)
            ]
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved bin mapping to {file_path}")
        return mapping

    def process_full_trajectory(self, trajectory, bins):
        """处理完整轨迹样本示例"""
        # 输入轨迹: N x 2 的数组
        bin_ids = []
        for point in trajectory:
            x_idx, y_idx = self.point_to_bin_id(point, bins)
            bin_ids.append((x_idx, y_idx))

        # 展平：每个点产生两个token（x_bin, y_bin）
        tokens = []
        for (x_id, y_id) in bin_ids:
            tokens.append(self.bin_id_to_token(x_id))
            tokens.append(self.bin_id_to_token(y_id))

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return {
            "original_trajectory": trajectory,
            "bin_ids": bin_ids,
            "tokens": tokens,
            "token_ids": token_ids
        }


# ===================== 完整执行流程 =====================
if __name__ == "__main__":
    # 1. 初始化处理器
    binner = TrajectoryBinner()

    # 2. 生成模拟数据
    points = binner.generate_simulated_data(num_points=10000)

    # 3. 计算分bin边界（采用混合策略）
    bins = binner.compute_bins(points, strategy='mixed')

    # 4. 可视化分布
    binner.plot_distribution(points, bins, strategy='mixed')

    # 5. 测试单个点转换
    test_point = (2.3, 15.8)
    x_bin, y_bin = binner.point_to_bin_id(test_point, bins)
    print(f"Point {test_point} → X-bin: {x_bin} ({binner.bin_id_to_token(x_bin)}), "
          f"Y-bin: {y_bin} ({binner.bin_id_to_token(y_bin)})")

    # 6. 创建并保存完整映射表
    mapping = binner.save_mapping(bins, strategy='mixed')

    # 7. 模拟轨迹转换
    sample_trajectory = [
        [-3.2, 10.1],
        [-1.5, 20.3],
        [0.8, 32.7],
        [2.5, 45.2]
    ]
    tokenized = binner.process_full_trajectory(sample_trajectory, bins)
    print("\nSample Trajectory Tokenization:")
    print(json.dumps(tokenized, indent=2))

    # 8. 输出混合策略的精度分析
    total_x_error = 0
    total_y_error = 0
    for point in sample_trajectory:
        x, y = point
        x_bin, y_bin = binner.point_to_bin_id(point, bins)

        # 获取bin范围
        if x_bin == 0:
            x_min, x_max = bins['x']['min'], bins['x']['bounds'][0]
        elif x_bin == 255:
            x_min, x_max = bins['x']['bounds'][-1], bins['x']['max']
        else:
            x_min, x_max = bins['x']['bounds'][x_bin - 1], bins['x']['bounds'][x_bin]

        x_center = (x_min + x_max) / 2
        x_error = abs(x - x_center)
        total_x_error += x_error

        # Y轴同理
        if y_bin == 0:
            y_min, y_max = bins['y']['min'], bins['y']['bounds'][0]
        elif y_bin == 255:
            y_min, y_max = bins['y']['bounds'][-1], bins['y']['max']
        else:
            y_min, y_max = bins['y']['bounds'][y_bin - 1], bins['y']['bounds'][y_bin]

        y_center = (y_min + y_max) / 2
        y_error = abs(y - y_center)
        total_y_error += y_error

        print(f"Point ({x}, {y}): X-bin error={x_error:.4f}m, Y-bin error={y_error:.4f}m")

    avg_x_error = total_x_error / len(sample_trajectory)
    avg_y_error = total_y_error / len(sample_trajectory)
    print(f"\nAverage error: X={avg_x_error:.4f}m, Y={avg_y_error:.4f}m")
    print(f"X-axis bin size: min={np.min(np.diff(bins['x']['bounds'])):.4f}m, "
          f"max={np.max(np.diff(bins['x']['bounds'])):.4f}m")
    print(f"Y-axis bin size: min={np.min(np.diff(bins['y']['bounds'])):.4f}m, "
          f"max={np.max(np.diff(bins['y']['bounds'])):.4f}m")
