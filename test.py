import sys

import numpy as np
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


if __name__ == "__main__":
    
    depth = load_depth("1.png")
    config = OmegaConf.load("config.yaml")
    sensor_cfg = config.RX.sensor
    camera_cfg = config.RX.camera
    laser_cfg = config.TX.laser
    jitter = config.jitter
    bkg_lux = config.bkg_lux
    lidar = SPAD_LiDAR(
        sensor_cfg, camera_cfg, 
        laser_cfg, jitter, bkg_lux
    )
    clicks = lidar.take_a_frame(0.08956, depth)
    clicks += 1
    freq = np.bincount(clicks[0, 0].get().astype(np.int32), minlength=1025)[1:]
    plt.bar(np.arange(1024), freq)
    