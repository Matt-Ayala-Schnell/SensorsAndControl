import numpy as np
import cv2
from . import config as C

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DET = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
S = C.MARKER_SIZE
OBJ = np.array([[-S / 2, S / 2, 0], [S / 2, S / 2, 0], [S / 2, -S / 2, 0], [-S / 2, -S / 2, 0]], np.float64)

R_B_CV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   
T_B_CV = C.CAM_POS_B


def true_intrinsics():
    f = 0.5 * C.CAM_H / np.tan(np.deg2rad(C.CAM_FOVY_DEG) / 2)
    return np.array([[f, 0, (C.CAM_W - 1) / 2], [0, f, (C.CAM_H - 1) / 2], [0, 0, 1]], float)


def detect(gray, K, dist=None):
    corners, ids, _ = DET.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        c = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(OBJ, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(4, 2) - c, axis=1).mean()
        out.append(dict(id=int(i), corners=c, rvec=rvec, tvec=tvec.flatten(), reproj=err))
    return out


def range_bearing(tvec_cv):
    p_b = R_B_CV @ tvec_cv + T_B_CV
    return float(np.hypot(p_b[0], p_b[1])), float(np.arctan2(p_b[1], p_b[0]))


def sigma_rb(r):
    return C.SIG_R0 + C.SIG_R1 * r ** 2, C.SIG_P0 + C.SIG_P1 * r


def load_calib(path=None):
    import json, os
    path = path or os.path.join(os.path.dirname(__file__), "..", "results", "camera_calib.json")
    if os.path.exists(path):
        j = json.load(open(path))
        return np.array(j["K"]), np.array(j["dist"])
    return true_intrinsics(), np.zeros(5)
