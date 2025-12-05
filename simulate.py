import sys

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(".")

from algorithms.freq_estimate import SpaceSaving
from dataset.utils import load_depth
from simulator.spad_lidar import SPAD_LiDAR

if __name__ == "__main__":
    
    # depth_img = load_depth("rgbd.png")
    depth_img = np.ones((8, 8))*3
    config = OmegaConf.load("configs/config.yaml")
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
    depth_ =  lidar.calc_depth(freq, "argmax")
    depth = lidar.crop_from_center(depth_img)
    print(np.abs(depth_.get() - depth).mean())
    plt.bar(np.arange(lidar.bins), freq[0, 0].get())
    plt.show()
    
    # h, w, p = clicks.shape
    # ss = SpaceSaving(10)
    # freq_ = freq.argmax(axis=-1)
    # for i in tqdm(range(h)):
    #     for j in tqdm(range(w), leave=False):
    #         for e in range(p):
    #             cur_click = int(clicks[i, j, e])
    #             if cur_click == -1:
    #                 continue
    #             ss.update(cur_click)
    #         freq_[i, j] = ss.query()[0][1]
    #         ss.counters = {}
    #         ss.min_heap = []