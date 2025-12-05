import sys

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

sys.path.append(".")

from dataset.utils import load_depth
from simulator.spad_lidar import SPAD_LiDAR

if __name__ == "__main__":
    
    depth_img = load_depth("rgbd.png")
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
    clicks = lidar.record_a_frame_clicks(0.08956, depth_img)
    freq = lidar.calc_freqency(clicks)
    depth_ =  lidar.calc_depth(freq, "centroid")
    depth = lidar.crop_from_center(depth_img)
    print(np.abs(depth_.get() - depth).mean())
    plt.bar(np.arange(lidar.bins), freq[0, 0].get())
    plt.show()
    