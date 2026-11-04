import os, json
from . import config as C

_P = os.path.join(os.path.dirname(__file__), "..", "results", "noise_params.json")


def load_noise():
    d = dict(wheel_scale=1.0, track_eff=C.TRACK, k_odo=C.K_ODO,
             sig_r0=C.SIG_R0, sig_r1=C.SIG_R1, sig_p0=C.SIG_P0, sig_p1=C.SIG_P1,
             bias_r=0.0, bias_phi=0.0)
    if os.path.exists(_P):
        d.update(json.load(open(_P)))
    
    d["sig_r0"] = max(d["sig_r0"], 0.006)
    d["sig_p0"] = max(d["sig_p0"], 0.005)
    d["lid_sig_r"] = max(2.0 * d.get("lidar_sig_r", 0.01), 0.008)
    d["lid_sig_pr"] = max(2.0 * d.get("lidar_sig_pr", 0.01), 0.008)
    return d
