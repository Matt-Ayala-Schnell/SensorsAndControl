import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tb3.sim import Sim
from tb3 import config as C
from tb3.calib import load_noise

nz = load_noise()
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
rows = []
for w in (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5):
    ratios = []
    for rep in range(4):
        s = Sim(seed=100 + rep, render=False)
        s.reset([0.0, -2.0, rep * 0.7]); s.odometry_increment()
        y_prev, tr, od = s.truth_pose()[2], 0.0, 0.0
        T = 1.2 / w * 3.0                     
        for k in range(int(T / C.CTRL_DT)):
            s.set_cmd(0.0, w); s.step()
            dl, dr = s.odometry_increment()
            od += (dr - dl) * nz["wheel_scale"] / nz["track_eff"]
            y = s.truth_pose()[2]; tr += wrap(y - y_prev); y_prev = y
        ratios.append(od / tr)
    rows.append((w, float(np.mean(ratios)), float(np.std(ratios))))
    print(f"w = {w:4.2f} rad/s   odometry / true rotation = {np.mean(ratios):.3f} +/- {np.std(ratios):.3f}")
json.dump(rows, open("results/odom_yaw_rate.json", "w"))
w, m, sd = np.array(rows).T
plt.figure(figsize=(5.5, 3.6)); plt.errorbar(w, (m - 1) * 100, sd * 100, fmt="o-")
plt.axhline(0, color="k", lw=0.5); plt.xlabel("commanded yaw rate (rad/s), in-place spin from rest"); plt.ylabel("odometry yaw over-estimate (%)")
plt.title("Odometry yaw error grows with yaw rate (wheel slip)"); plt.tight_layout(); plt.savefig("results/fig_odom_yaw_rate.png", dpi=130)
