import numpy as np
import cv2
from . import config as C
from . import model as M
from . import vision as V
from . import mapmodel as MM
from . import planner as PL
from .sim import Sim
from .ekf import EKF, marker_measurement, pillar_measurements
from .controller import LQRTracker
from .calib import load_noise
from .world import all_markers

DEFAULTS = dict(seed=0, variant="cam_lidar", q_scale=1.0, r_scale=1.0, lqr_q_scale=1.0, lqr_r_scale=1.0,
                friction=1.0, jitter=True, unmapped=None, dropout=None, pixel_noise_scale=1.0, outlier_prob=0.0,
                push=False, t_max=170.0, record=False, init_sd=(0.10, 0.10, 0.10), gate=C.GATE_CHI2_2,
                lidar_replan=True, use_markers=None, use_pillars=None, marker_rmax=3.6)
ANGLES = np.linspace(0, 2 * np.pi, C.LIDAR_N, endpoint=False)
wrap = M.wrap


def run_trial(**kw):
    cfg = {**DEFAULTS, **kw}
    rng = np.random.default_rng(cfg["seed"])
    nz = load_noise()
    K_cam, dist = V.load_calib()
    use_m = cfg["use_markers"] if cfg["use_markers"] is not None else cfg["variant"] in ("cam", "cam_lidar")
    use_p = cfg["use_pillars"] if cfg["use_pillars"] is not None else cfg["variant"] in ("lidar", "cam_lidar")
    sim = Sim(seed=cfg["seed"] + 1000, unmapped=cfg["unmapped"], wheel_friction=cfg["friction"])
    sim.pixel_noise = C.PIXEL_NOISE * cfg["pixel_noise_scale"]
    start = C.START_NOMINAL.copy()
    if cfg["jitter"]:
        start = start + rng.normal(0, [0.15, 0.15, 0.20])
    sim.reset(start)

    sd = np.array(cfg["init_sd"])
    x0 = start + rng.normal(0, sd)
    ekf = EKF(x0, np.diag(sd ** 2), nz, cfg["q_scale"], cfg["r_scale"], cfg["gate"])
    markers = all_markers()
    # planning
    dyn_count = np.zeros(MM.grid_shape(), int)
    ref = PL.plan_to_dock(ekf.x[:2])
    Q, R = C.LQR_Q * cfg["lqr_q_scale"], C.LQR_R * cfg["lqr_r_scale"]
    ctl = LQRTracker(ref, Q=Q, R=R)
    log = dict(t=[], truth=[], est=[], P=[], cmd=[], phase=[], nees=[], pe=[])
    events = dict(nis=[], replans=[], dets=[], lidar_hits=[])
    frames, scans = [], []
    n_steps = int(cfg["t_max"] / C.CTRL_DT)
    collided, docked_t, step_after_done = False, None, 0
    v_cmd = w_cmd = 0.0
    next_push, push_end = (rng.uniform(8, 15) if cfg["push"] else np.inf), -1
    last_replan_t = -9.0
    n_replans = 0
    for k in range(n_steps):
        t = sim.t

        dsl, dsr = sim.odometry_increment()
        ekf.predict(dsl, dsr)
        if k % C.CAM_EVERY == 0 and (use_m or cfg["record"]):
            img = sim.camera()
            blocked = cfg["dropout"] is not None and cfg["dropout"][0] <= t < cfg["dropout"][1]
            dets = [] if blocked else V.detect(img, K_cam, dist)
            for d in dets:
                if d["id"] not in markers:
                    continue
                (r, phi), sig = marker_measurement(d["tvec"], nz)
                if rng.random() < cfg["outlier_prob"]:
                    r += rng.uniform(0.5, 1.5)
                if use_m and (cfg["marker_rmax"] is None or r <= cfg["marker_rmax"]):   
                    acc, nis = ekf.update_rb((r, phi), markers[d["id"]][:2], sig)
                    events["nis"].append((t, nis, d["id"], acc))
                events["dets"].append((t, d["id"], r))
            if cfg["record"]:
                frames.append((t, cv2.resize(img, (320, 240)), [(d["id"], d["corners"] / 2) for d in dets]))
        if k % C.LIDAR_EVERY == 0:
            scan = sim.lidar()
            if cfg["record"]:
                scans.append((t, scan))
            if use_p:
                for (z, sig, m) in pillar_measurements(scan, ANGLES, ekf.x, nz=nz):
                    acc, nis = ekf.update_rb(z, m, sig)
                    events["nis"].append((t, nis, -1, acc))

            est = ekf.x
            exp = MM.expected_ranges(est, C.LIDAR_POS_B, ANGLES)
            sxy, sth = np.sqrt(ekf.P[0, 0] + ekf.P[1, 1]), np.sqrt(ekf.P[2, 2])
            tol = 0.20 + 3 * sxy + 3 * sth * np.where(np.isfinite(scan), scan, 0.0)     
            bad = np.isfinite(scan) & (scan < 2.0) & (scan < exp - tol)
            if bad.any():
                a = est[2] + ANGLES[bad]
                lx = est[0] + np.cos(est[2]) * C.LIDAR_POS_B[0]; ly = est[1] + np.sin(est[2]) * C.LIDAR_POS_B[0]
                px, py = lx + scan[bad] * np.cos(a), ly + scan[bad] * np.sin(a)
                keep = MM.clearance(px, py) > 0.25
                if keep.any():
                    i, j = MM.world_to_grid(px[keep], py[keep])
                    ok = (i >= 0) & (i < 120) & (j >= 0) & (j < 120)
                    np.add.at(dyn_count, (i[ok], j[ok]), 1)

        if cfg["lidar_replan"] and not ctl.done and t - last_replan_t > 2.0 and (dyn_count >= 2).any():
            dyn = MM.inflated(dyn_count >= 2, 0.04)
            infl = MM.inflated(dyn, PL.INFL - 0.05)
            i0 = ctl.i_proj
            ahead = np.arange(i0, min(len(ref["s"]), i0 + 70), 3)
            gi, gj = MM.world_to_grid(ref["x"][ahead], ref["y"][ahead])
            if infl[gi, gj].any():
                new = PL.plan_to_dock(ekf.x[:2], dyn)
                last_replan_t = t
                events["replans"].append((t, new is not None))
                n_replans += 1
                if new is not None:
                    ref, ctl = new, LQRTracker(new, Q=Q, R=R)

        v_cmd, w_cmd, info = ctl.step(ekf.x)
        if k % C.LIDAR_EVERY == 0:
            front = np.abs(wrap(ANGLES)) < np.deg2rad(20)
            if np.nanmin(np.where(np.isfinite(scan[front]), scan[front], 9.0)) < 0.17:
                v_cmd = 0.0                       # safety layer
        v_cmd, w_cmd = sim.set_cmd(v_cmd, w_cmd)

        if t >= next_push:
            ang = rng.uniform(0, 2 * np.pi)
            sim.push = np.array([2.5 * np.cos(ang), 2.5 * np.sin(ang), 0, 0, 0, rng.uniform(-0.04, 0.04)])
            push_end, next_push = t + 0.1, t + rng.uniform(10, 18)
        if t >= push_end:
            sim.push = np.zeros(6)
        sim.step()
        
        #EVALUATION ONLY (ground truth is never read by anything above)
        tr = sim.truth_pose()
        e = tr - ekf.x; e[2] = wrap(e[2])
        log["t"].append(sim.t); log["truth"].append(tr); log["est"].append(ekf.x.copy()); log["P"].append(ekf.P.copy())
        log["cmd"].append((v_cmd, w_cmd)); log["phase"].append(info["phase"])
        log["nees"].append(float(e @ np.linalg.solve(ekf.P, e)))
        d2 = np.hypot(ref["x"] - tr[0], ref["y"] - tr[1]); log["pe"].append(float(d2.min()))
        if sim.truth_collision():
            collided = True
            break
        if ctl.done:
            step_after_done += 1
            if docked_t is None:
                docked_t = sim.t
            if step_after_done > 20:
                break
    sim.step(0)
    tr = sim.truth_pose()
    A = {k: np.array(v) for k, v in log.items() if k not in ("phase",)}
    A["phase"] = np.array(log["phase"])
    pos_err_dock = float(np.hypot(*(tr[:2] - C.DOCK_POSE[:2])))
    yaw_err_dock = float(abs(wrap(tr[2] - C.DOCK_POSE[2])))
    success = bool(ctl.done and not collided and pos_err_dock <= C.DOCK_TOL_POS and yaw_err_dock <= C.DOCK_TOL_YAW)
    ee = A["truth"] - A["est"]; ee[:, 2] = wrap(ee[:, 2])
    dt = C.CTRL_DT
    reason = "ok" if success else ("collision" if collided else ("timeout" if not ctl.done else "dock_miss"))
    metrics = dict(success=success, reason=reason, t_dock=docked_t, pos_err_dock=pos_err_dock, yaw_err_dock_deg=float(np.rad2deg(yaw_err_dock)),
                   est_err_final=float(np.hypot(*(ee[-1, :2]))), rmse_pos=float(np.sqrt((ee[:, :2] ** 2).sum(1).mean())),
                   rmse_yaw_deg=float(np.rad2deg(np.sqrt((ee[:, 2] ** 2).mean()))), max_pos_err=float(np.hypot(ee[:, 0], ee[:, 1]).max()),
                   path_rmse=float(np.sqrt((A["pe"] ** 2).mean())), path_max=float(A["pe"].max()),
                   effort=float(((A["cmd"] ** 2).sum(1)).sum() * dt), anees=float(A["nees"].mean()),
                   mean_nis=float(np.mean([n[1] for n in events["nis"]])) if events["nis"] else float("nan"),
                   n_updates=ekf.n_updates, n_rejected=ekf.n_rejected, n_replans=n_replans, collided=collided,
                   duration=float(sim.t))
    out = dict(metrics=metrics, log=A, events=events, ref=ref, cfg=cfg, dyn=(dyn_count >= 2), frames=frames, scans=scans)
    sim.close()
    return out


if __name__ == "__main__":
    import json, sys, time
    t0 = time.time()
    r = run_trial(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(json.dumps(r["metrics"], indent=1, default=str)); print("wall", time.time() - t0)
