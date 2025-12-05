from tqdm import tqdm

try:
    import cupy as cp
    from cupyx import scipy
    print(f"Find {cp.cuda.runtime.getDeviceCount()} GPUs")
except RuntimeError:
    import numpy as cp
    import scipy


class SPAD_LiDAR:

    def __init__(self, sensor_cfg, camera_cfg, laser_cfg, jitter, bkg_lux):
        self.c_speed = 2.99e8
        self.plank = 6.626e-34
        self.C_atm = (10**(-1*(7e-4/10.0)))
        self.jitter = jitter
        self.bkg_lux = bkg_lux
        self.w_bkg = self.estimate_irradiance(bkg_lux, laser_cfg.wavelength)
        self.photos_bkg = (laser_cfg.wavelength * self.w_bkg) / (self.plank * self.c_speed)

        self.rep_rate = laser_cfg.rep_rate
        self.energy_p_pulse = laser_cfg.energy_p_pulse
        self.pulse_fwhm = laser_cfg.pulse_fwhm
        self.sigma = self.pulse_fwhm / (2 * cp.sqrt(2 * cp.log(2)))
        self.fibre_core = laser_cfg.fibre_core
        self.illum_lens = laser_cfg.illum_lens
        self.wavelength = laser_cfg.wavelength
        self.photos_p_pulse = (self.wavelength * self.energy_p_pulse) / (self.plank * self.c_speed)

        self.f_num = camera_cfg.f_num
        self.bins = sensor_cfg.bins
        self.t_res = sensor_cfg.t_res
        self.fps = sensor_cfg.fps
        self.depth_range = sensor_cfg.depth_range
        self.q_efficiency = sensor_cfg.q_efficiency
        self.effictive_pix_size = sensor_cfg.effictive_pix_size
        self.dcr = sensor_cfg.dcr
        self.img_res = sensor_cfg.img_res

        self.t_res = ((self.depth_range[1] - self.depth_range[0]) * 2 / self.c_speed) / self.bins
        self.bin1_t = self.depth_range[0] * 2 / self.c_speed
        self.pulse_per_frame = int((1 / self.fps) * self.rep_rate)


    def estimate_irradiance(self, illuminance, wavelength, bandwidth=10):
        AM15G_REF_ILLUMINANCE = 120e3
        AM15G_SPECTRAL_IRRADIANCE = {
            671e-9: 1.409,
            870e-9: 0.9610,
            905e-9: 0.8131,   
            940e-9: 0.4486,
        }
        if wavelength not in AM15G_SPECTRAL_IRRADIANCE:
            raise ValueError(f"不支持的波长 {wavelength}nm")
        
        base_spec_irrad = AM15G_SPECTRAL_IRRADIANCE[wavelength]
        scaling_factor = illuminance / AM15G_REF_ILLUMINANCE
        estimated_spec_irrad = base_spec_irrad * scaling_factor
        total_irradiance = estimated_spec_irrad * bandwidth
        return total_irradiance

    def calc_attenuation(self, ref, depth):
        illum_radius = ((depth / self.illum_lens) * self.fibre_core) / 2.0
        illum_area = cp.pi*(illum_radius) ** 2
        attenuation1 = self.q_efficiency * ref * (self.C_atm ** (2 * depth)) / 8
        attenuation2 = (self.effictive_pix_size[0] * self.effictive_pix_size[1]) / (illum_area * (self.f_num ** 2))
        return attenuation1 * attenuation2, illum_area

    def calc_liklyhood(self, ref, depth):
        attenuation, illum_area = self.calc_attenuation(ref, depth)
        Ppp = self.photos_p_pulse * attenuation
        Cbkg = self.photos_bkg * attenuation * illum_area
        return Ppp, Cbkg

    def first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr!=0
        return cp.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def calc_all_clicks(self, tof, Ppp, Cbkg):
        bins_lb = cp.linspace(0, self.bins - 1, self.bins) * self.t_res + self.bin1_t
        bins_ub = cp.linspace(1, self.bins, self.bins) * self.t_res + self.bin1_t
        bins_lb = cp.tile(bins_lb, [self.pulse_per_frame, 1])
        bins_ub = cp.tile(bins_ub, [self.pulse_per_frame, 1])
        jit = (cp.random.rand(self.pulse_per_frame)) * self.jitter
        jit = cp.tile(jit.reshape((-1, 1)), [1, self.bins])

        noise_click = (Cbkg + self.dcr) * self.t_res
        laser_click = (Ppp/2)*(
            scipy.special.erf((tof + jit - bins_lb) / (self.sigma * cp.sqrt(2))) \
                - scipy.special.erf((tof + jit - bins_ub)/(self.sigma * cp.sqrt(2)))
        )
        bino_prob = cp.fmin((noise_click + laser_click), 1.0)

        trials = cp.ones((self.pulse_per_frame, self.bins), dtype=int)
        clicks = cp.random.binomial(trials, bino_prob, dtype=cp.float32)
        clicks = self.first_nonzero(clicks, axis=1, invalid_val=-1)
        return clicks

    def crop_from_center(self, depth):
        if depth.ndim != 2:
            raise ValueError("输入一个二维矩阵")
    
        rows, cols = depth.shape
        start_row = (rows - self.img_res[0]) // 2
        start_col = (cols - self.img_res[1]) // 2
        cropped = depth[
            start_row:start_row + self.img_res[0], 
            start_col:start_col + self.img_res[1]
        ]
        return cropped

    def calc_tof(self, depth):
        return (depth.astype(cp.float64) * 2) / self.c_speed

    def record_a_frame_clicks(self, ref, depth):
        depth = self.crop_from_center(cp.asarray(depth))
        h, w = depth.shape
        Ppp, Cbkg = self.calc_liklyhood(ref, depth)
        tof = self.calc_tof(depth)
        all_clicks = cp.zeros((h, w, self.pulse_per_frame))
        for i in tqdm(range(h)):
            for j in tqdm(range(w), leave=False):
                all_clicks[i, j] = self.calc_all_clicks(tof[i, j], Ppp[i, j], Cbkg[i, j])
        return all_clicks
    
    def calc_freqency(self, clicks):
        h, w, _ = clicks.shape
        freq = cp.zeros((h, w, self.bins))
        for i in tqdm(range(h)):
            for j in tqdm(range(w), leave=False):
                freq[i, j] = cp.bincount(clicks[i, j].astype(cp.int32) + 1, minlength=self.bins + 1)[1:]
        return freq

    def calc_depth(self, freq, algo="argmax"):
        if algo == "argmax":
            tof = (freq.argmax(axis=-1) + 0.5) * self.t_res + self.bin1_t
            return (tof / 2) * self.c_speed
        elif algo == "centroid":
            t_bin = (cp.linspace(0, self.bins - 1, self.bins) + 0.5) * self.t_res + self.bin1_t
            tof = cp.sum((t_bin[None, None, :] * freq), axis=-1) / cp.sum(freq, axis=-1)
            return (tof / 2) * self.c_speed
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
