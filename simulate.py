import sys

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image

sys.path.append(".")

from spad_lidar import SPAD_LiDAR


def load_depth(path):

    if path.endswith('.png'):
        # 读取16位PNG图像
        depth_pil = Image.open(path)
        depth_vis = np.array(depth_pil, dtype=np.uint16)

        # 模仿 MATLAB 的位移和位或操作
        depth_shifted_right = np.right_shift(depth_vis, 3)
        depth_shifted_left = np.left_shift(depth_vis, 13)  # 16 - 3 = 13
        depth_inpaint = np.bitwise_or(depth_shifted_right, depth_shifted_left)

        # 转换为 float32，单位从 mm 转为 m
        depth_inpaint = depth_inpaint.astype(np.float32) / 1000.0

        # 大于 8 米的设为 8
        depth_inpaint[depth_inpaint > 8.0] = 8.0
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return depth_inpaint


def parallel_bincount_3d(clicks_3d, minlength=1025):
    """
    并行统计三维数组每个位置 (i, j) 的频率。
    参数:
        clicks_3d: 形状为 (I, J, L) 的cupy数组。
        minlength: bincount的最小长度。
    返回:
        counts: 形状为 (I, J, minlength) 的数组，counts[i, j] 是位置(i,j)的频率分布。
    """
    I, J, L = clicks_3d.shape
    
    # 1. 重塑：将 (I, J, L) 变为 (I*J, L)，视作一个大批次
    clicks_2d = clicks_3d.reshape(-1, L)
    
    # 2. 核心：为每个批次计算bincount。这里需要一个技巧：
    # 由于cupy没有直接的"沿轴bincount"，我们通过偏移量将不同批次的数据“隔离”到不同的数值区间。
    # 为每个批次的数据加上一个大的偏移量 (batch_idx * offset)
    offset = minlength  # 偏移量需要 >= minlength，确保不同批次计数不重叠
    batch_indices = cp.arange(clicks_2d.shape[0])[:, cp.newaxis]  # (I*J, 1)
    # 制造偏移：让批次0的数据在 [0, minlength)，批次1在 [minlength, 2*minlength) ...
    shifted_data = clicks_2d + batch_indices * offset
    
    # 3. 一次性计算所有批次的联合bincount
    flat_shifted_counts = cp.bincount(shifted_data.ravel().astype(cp.int32), minlength=clicks_2d.shape[0] * offset)
    
    # 4. 将结果重新分割成各个批次
    # 重塑为 (I*J, offset)，然后取前minlength列
    batch_counts = flat_shifted_counts.reshape(clicks_2d.shape[0], offset)[:, :minlength]
    
    # 5. 恢复成 (I, J, minlength) 的形状并返回
    return batch_counts.reshape(I, J, minlength)


if __name__ == "__main__":
    
    depth_img = load_depth("rgbd.png")
    config = OmegaConf.load("config1.yaml")
    sensor_cfg = config.RX.sensor
    camera_cfg = config.RX.camera
    laser_cfg = config.TX.laser
    jitter = config.jitter
    bkg_lux = config.bkg_lux
    lidar = SPAD_LiDAR(
        sensor_cfg, camera_cfg, 
        laser_cfg, jitter, bkg_lux
    )
    clicks = lidar.record_a_frame_clicks(0.08956, depth_img)
    freq = lidar.calc_freqency(clicks)
    depth_ =  lidar.calc_depth(freq)
    depth = lidar.crop_from_center(depth_img)
    print(np.abs(depth_.get() - depth).mean())
    plt.bar(np.arange(lidar.bins), freq[0, 0].get())
    plt.show()
    