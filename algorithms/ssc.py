import numpy as np
from typing import Tuple

def ssc_depth_estimation(
    photon_events: np.ndarray,
    sigma: int = 5,
    num_candidates: int = 5,
    window_size: int = 5,
    aging_lambda: float = 0.9375,
) -> np.ndarray:
    """
    实现 SSC（Spatial and Statistical Correlation）深度估计方法。
    
    参数:
        photon_events: 形状为 (h, w, t) 的光子事件数组，每个元素为 0/1。
        sigma: 脉冲持续时间对应的 bin 数（统计相关性参数）。
        num_candidates: 每个像素存储的候选深度数量。
        window_size: 滑动窗口大小（空间相关性参数）。
        aging_lambda: 去噪阶段的老化系数（默认为 0.9375）。
    
    返回:
        depth_map: 形状为 (h, w) 的深度图。
    """
    h, w, t = photon_events.shape
    half_win = window_size // 2

    # 初始化候选深度和频率数组
    candidates_depth = np.ones((h, w, num_candidates), dtype=int) * -1  # 候选深度值
    candidates_freq = np.ones((h, w, num_candidates), dtype=int) * -1  # 候选频率值

    # 主循环：处理每个光子周期
    for cycle in range(t):
        # 当前周期的光子事件（二值图）
        current_events = photon_events[:, :, cycle]  # (h, w)

        # 步骤1：预处理阶段 - 计算频率图
        frequency_map = np.zeros((h, w), dtype=int)
        for i in range(h):
            for j in range(w):
                if current_events[i, j] == -1:
                    continue
                # 获取滑动窗口区域
                i_min = max(0, i - half_win)
                i_max = min(h, i + half_win + 1)
                j_min = max(0, j - half_win)
                j_max = min(w, j + half_win + 1)
                # 统计与中心像素深度差小于 sigma 的像素数
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if current_events[ii, jj] != -1 and abs(current_events[ii, jj] - current_events[i, j]) <= sigma:
                            frequency_map[i, j] += 1

        # 步骤2：深度测量阶段 - 更新候选
        for i in range(h):
            for j in range(w):
                if current_events[i, j] == -1:
                    continue
                # 2.1 从相邻像素中选取频率最高的 num_candidates 个深度作为临时候选
                temp_depths = []
                temp_freqs = []
                i_min = max(0, i - half_win)
                i_max = min(h, i + half_win + 1)
                j_min = max(0, j - half_win)
                j_max = min(w, j + half_win + 1)
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        if current_events[ii, jj] != -1:
                            temp_depths.append(current_events[ii, jj])  # 当前周期作为深度值（简化）
                            temp_freqs.append(frequency_map[ii, jj])
                # 按频率排序，选择前 num_candidates 个
                if len(temp_depths) > 0:
                    sorted_idx = np.argsort(temp_freqs)[-num_candidates:]
                    temp_depths = np.array(temp_depths)[sorted_idx]
                    temp_freqs = np.array(temp_freqs)[sorted_idx]
                else:
                    temp_depths = np.array([])
                    temp_freqs = np.array([])

                # 2.2 用上一周期的候选校准临时候选频率（统计相关性）
                for idx, d in enumerate(temp_depths):
                    for k in range(num_candidates):
                        if candidates_depth[i, j, k] != -1 and abs(candidates_depth[i, j, k] - d) < sigma:
                            temp_freqs[idx] += candidates_freq[i, j, k]

                # 2.3 合并上一周期候选和临时候选，选择频率最高的 num_candidates 个
                all_depths = np.concatenate([candidates_depth[i, j], temp_depths])
                all_freqs = np.concatenate([candidates_freq[i, j], temp_freqs])
                # 选择频率最高的 num_candidates 个
                if len(all_depths) > 0:
                    sorted_idx = np.argsort(all_freqs)[-num_candidates:]
                    candidates_depth[i, j] = all_depths[sorted_idx]
                    candidates_freq[i, j] = all_freqs[sorted_idx]

        # 步骤3：去噪阶段 - 应用老化系数
        candidates_freq = (candidates_freq * aging_lambda).astype(int)

    # 最终深度计算：选择频率高于最高频率一半的候选，计算平均深度
    depth_map = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            freq = candidates_freq[i, j]
            depths = candidates_depth[i, j]
            if np.max(freq) == -1:
                depth_map[i, j] = -1
                continue
            # 选择频率高于最高频率一半的候选
            mask = freq > (0.5 * np.max(freq))
            if np.any(mask):
                depth_map[i, j] = np.mean(depths[mask])
            else:
                depth_map[i, j] = depths[np.argmax(freq)]

    return depth_map

# 示例用法
if __name__ == "__main__":
    # 模拟一个 32x32x1000 的光子事件数组（随机二值）
    h, w, t = 8, 8, 100
    photon_events = np.random.randint(0, 2, (h, w, t))

    # 运行 SSC 深度估计
    depth_map = ssc_depth_estimation(
        photon_events,
        sigma=5,
        num_candidates=5,
        window_size=3,
        aging_lambda=0.9375
    )

    print("Depth map shape:", depth_map.shape)
    print("Depth map sample (top-left 5x5):")
    print(depth_map[:5, :5])