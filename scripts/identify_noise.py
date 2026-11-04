"""Parts B/C: Run: python scripts/identify_noise.py"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tb3.sim import Sim
from tb3 import config as C, vision as V
from tb3.world import all_markers
from tb3.mapmodel import clearance

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
K, dist = V.load_calib()
rng = np.random.default_rng(11)
s = Sim(seed=5)

dst, dso, dtt, dto, dsr_l, dsl_l = [], [], [], [], [], []
for run in range(14):
    while True:   
        p0 = [rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5), rng.uniform(-3, 3)]
        if clearance(p0[0], p0[1]) > 0.6:
            break
    s.reset(p0)
    p_prev = s.truth_pose(); s.odometry_increment()
    v, w = 0.0, 0.0
    for k in range(160):
        if k % 20 == 0:
            v, w = rng.uniform(0.03, 0.15), rng.uniform(-1.2, 1.2)
        near = clearance(*s.truth_pose()[:2]) < 0.45      
        s.set_cmd(0.0, 1.0) if near else s.set_cmd(v, w); s.step()
        p = s.truth_pose(); dl, dr = s.odometry_increment()
        dth = wrap(p[2] - p_prev[2]); mid = p_prev[2] + dth / 2
        ds = (p[0] - p_prev[0]) * np.cos(mid) + (p[1] - p_prev[1]) * np.sin(mid)
        dst.append(ds); dtt.append(dth); dso.append((dl + dr) / 2); dto.append((dr - dl) / C.TRACK)
        dsr_l.append(dr); dsl_l.append(dl); p_prev = p
dst, dso, dtt, dto, dsr_l, dsl_l = map(np.array, (dst, dso, dtt, dto, dsr_l, dsl_l))
beta = float(dst @ dso / (dso @ dso))                  
alpha = float(dtt @ dto / (dto @ dto))                 
track_eff = C.TRACK / alpha
res_s = dst - beta * dso
dto_c = (dsr_l - dsl_l) / track_eff * 1.0
res_t = dtt - (dsr_l * beta - dsl_l * beta) / track_eff
act = (np.abs(dsr_l) + np.abs(dsl_l))

k_s = float((res_s ** 2).sum() / (act / 4).sum())
k_t = float((res_t ** 2).sum() * 1 / ((act / track_eff ** 2).sum()))
k_odo = float(0.5 * (k_s + k_t))
print(f"odometry: wheel scale beta={beta:.4f}, rot gain alpha={alpha:.4f} -> effective track {track_eff*1000:.1f} mm (nominal {C.TRACK*1000:.0f})")
print(f"  k_s={k_s:.2e}  k_theta={k_t:.2e}  -> k_odo={k_odo:.2e} m (var per metre per wheel)")


mk = all_markers()
rows = []
s2 = Sim(seed=6)
for n in range(700):
    mid = int(rng.choice(list(mk)))
    mx, my, myaw = mk[mid]

    r = rng.uniform(0.5, 3.4)
    ang = myaw + rng.uniform(-0.8, 0.8)                    
    px, py = mx + r * np.cos(ang), my + r * np.sin(ang)
    if abs(px) > 2.7 or abs(py) > 2.7:
        continue
    yaw = np.arctan2(my - py, mx - px) + rng.uniform(-0.3, 0.3)
    s2.reset([px, py, yaw])
    tp = s2.truth_pose()
    for d in V.detect(s2.camera(), K, dist):
        if d["id"] != mid:
            continue
        rm, bm = V.range_bearing(d["tvec"])
        dx, dy = mx - tp[0], my - tp[1]
        rows.append([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - tp[2]), rm, bm])

rows = np.array(rows)
rt, bt, rm, bm = rows.T
er, eb = rm - rt, wrap(bm - bt)
er_raw, eb_raw = er.copy(), eb.copy()
c0 = float((er * rt).sum() / (rt ** 2).sum()); er = er - c0 * rt; eb = eb - eb.mean()
c_scale = c0
print(f"marker samples: {len(rows)}; raw mean err r={er_raw.mean()*100:.2f} cm, bearing={np.rad2deg(eb_raw.mean()):.3f} deg")
bins = np.array([0.4, 1.0, 1.6, 2.2, 2.8, 3.6])
cen, sr, sb = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (rt >= lo) & (rt < hi)
    if m.sum() > 8:
        cen.append(rt[m].mean()); sr.append(er[m].std()); sb.append(eb[m].std())
cen, sr, sb = map(np.array, (cen, sr, sb))
A = np.stack([np.ones_like(cen), cen ** 2], 1); r0, r1 = np.linalg.lstsq(A, sr, rcond=None)[0]
B = np.stack([np.ones_like(cen), cen], 1); p0, p1 = np.linalg.lstsq(B, sb, rcond=None)[0]
r0, r1, p0, p1 = max(r0, 0.002), max(r1, 0.0), max(p0, 0.002), max(p1, 0.0)
print(f"range scale bias c={c_scale:.4f} (meas = true*(1+c))")
print("range std by bin (m):", dict(zip(np.round(cen, 2), np.round(sr, 4))))
print("bearing std by bin (rad):", dict(zip(np.round(cen, 2), np.round(sb, 4))))
print(f"fit: sigma_r = {r0:.4f} + {r1:.4f} r^2 ; sigma_phi = {p0:.4f} + {p1:.4f} r")
out = dict(wheel_scale=beta, track_eff=track_eff, k_odo=k_odo, sig_r0=float(r0), sig_r1=float(r1), sig_p0=float(p0), sig_p1=float(p1),
           bias_r=float(er_raw.mean()), bias_phi=float(eb_raw.mean()), range_scale_bias=c_scale, n_marker_samples=int(len(rows)),
           bins=cen.tolist(), sigma_r_bins=sr.tolist(), sigma_phi_bins=sb.tolist())

from tb3.ekf import pillar_measurements
ANG = np.linspace(0, 2 * np.pi, C.LIDAR_N, endpoint=False)
s3 = Sim(seed=8, render=False)
lr, lp = [], []
for n in range(500):
    while True:
        pp = [rng.uniform(-2.7, 2.7), rng.uniform(-2.7, 2.7), rng.uniform(-3.1, 3.1)]
        if clearance(pp[0], pp[1]) > 0.25:
            break
    s3.reset(pp); tp = s3.truth_pose()
    for (z, sg, m) in pillar_measurements(s3.lidar(), ANG, tp):     
        dx, dy = m[0] - tp[0], m[1] - tp[1]
        lr.append((np.hypot(dx, dy), z[0] - np.hypot(dx, dy), wrap(z[1] - (np.arctan2(dy, dx) - tp[2]))))
lr = np.array(lr)
print(f"lidar pillar samples {len(lr)}: sigma_r={lr[:,1].std()*1000:.1f} mm (bias {lr[:,1].mean()*1000:.1f} mm), sigma_phi*r = {np.std(lr[:,2]*lr[:,0])*1000:.1f} mm")
out["lidar_sig_r"] = float(lr[:, 1].std()); out["lidar_sig_pr"] = float(np.std(lr[:, 2] * lr[:, 0])); out["lidar_bias_r"] = float(lr[:, 1].mean())
json.dump(out, open("results/noise_params.json", "w"), indent=1)
np.save("results/marker_residuals.npy", np.stack([rt, er, eb], 1))
