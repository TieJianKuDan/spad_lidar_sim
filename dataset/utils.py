import numpy as np
from PIL import Image


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