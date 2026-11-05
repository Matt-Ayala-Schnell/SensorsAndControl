"""Part B: Run: python scripts/calibrate_camera.py"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from tb3.sim import Sim
import mujoco  
from tb3 import config as C, vision as V

N_VIEWS = 60

FLAGS = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
rng = np.random.default_rng(7)
s = Sim(seed=3, checker=True)
s.reset([-2.0, 2.0, 0])
nx, ny, sq = 9, 6, 0.66 / 12
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * sq
R0 = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float).T  
R0 = np.stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], 1)
objpts, imgpts, kept = [], [], 0
for k in range(N_VIEWS * 3):
    if kept >= N_VIEWS:
        break
    yaw, pitch, roll = rng.uniform(-50, 50), rng.uniform(-45, 45), rng.uniform(-40, 40)
    R = Rot.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix() @ R0
    pos = np.array([-2.0 + rng.uniform(0.7, 1.6), 2.0 + rng.uniform(-0.35, 0.35), rng.uniform(0.0, 0.30)])
    s.d.mocap_pos[0] = pos
    q = Rot.from_matrix(R).as_quat()  
    s.d.mocap_quat[0] = [q[3], q[0], q[1], q[2]]
    mujoco.mj_forward(s.m, s.d)
    img = s.camera()
    ok, c = cv2.findChessboardCorners(img, (nx, ny), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)    
    if not ok:
        continue
    c = cv2.cornerSubPix(img, c, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
    objpts.append(objp); imgpts.append(c); kept += 1
print("views used:", kept)
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (C.CAM_W, C.CAM_H), np.eye(3), np.zeros(5), flags=FLAGS)
Kt = V.true_intrinsics()

rms_tr, K30, d30, _, _ = cv2.calibrateCamera(objpts[:40], imgpts[:40], (C.CAM_W, C.CAM_H), np.eye(3), np.zeros(5), flags=FLAGS)
errs = []
for o, i in zip(objpts[40:], imgpts[40:]):
    ok, rv, tv = cv2.solvePnP(o, i, K30, d30)
    p, _ = cv2.projectPoints(o, rv, tv, K30, d30)
    errs.append(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1))))
out = dict(K=K.tolist(), dist=dist.flatten().tolist(), rms_reproj_px=float(rms), n_views=kept,
           
           true_K=Kt.tolist(), fx_err_pct=float(100 * (K[0, 0] - Kt[0, 0]) / Kt[0, 0]), fy_err_pct=float(100 * (K[1, 1] - Kt[1, 1]) / Kt[1, 1]),
           cx_err_px=float(K[0, 2] - Kt[0, 2]), cy_err_px=float(K[1, 2] - Kt[1, 2]),
           heldout_reproj_px_mean=float(np.mean(errs)))

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/camera_calib.json", "w"), indent=1)
print({k: (round(v,3) if isinstance(v,float) else v) for k, v in out.items() if k not in ("K", "true_K")}); print("K=\n", K)
