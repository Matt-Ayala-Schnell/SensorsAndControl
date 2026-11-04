import numpy as np
from . import config as C
from . import model as M
from .calib import load_noise


class EKF:
    def __init__(self, x0, P0, noise=None, q_scale=1.0, r_scale=1.0, gate=C.GATE_CHI2_2, q_floor=(1e-7, 1e-7, 1e-6)):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.nz = noise or load_noise()
        self.q_scale, self.r_scale, self.gate = q_scale, r_scale, gate
        self.q_floor = np.diag(q_floor)
        self.log_nis, self.n_rejected, self.n_updates = [], 0, 0

    def predict(self, dsl, dsr):
        b, k = self.nz["track_eff"], self.nz["k_odo"] * self.q_scale
        dsl, dsr = dsl * self.nz["wheel_scale"], dsr * self.nz["wheel_scale"]
        ds, dth = 0.5 * (dsr + dsl), (dsr - dsl) / b
        G = M.jac_G(self.x, ds, dth, b)
        Sw = np.diag([k * abs(dsr) + 1e-9, k * abs(dsl) + 1e-9])     
        F = M.jac_F(self.x, ds, dth)
        self.P = F @ self.P @ F.T + G @ Sw @ G.T + self.q_floor * self.q_scale
        self.x = M.f_odom(self.x, ds, dth)

    def update_rb(self, z, m, sig, tag=None):

        R = np.diag([(sig[0]) ** 2, (sig[1]) ** 2]) * self.r_scale
        H = M.jac_H(self.x, m)
        y = np.array([z[0], z[1]]) - M.h_range_bearing(self.x, m)
        y[1] = M.wrap(y[1])
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        self.log_nis.append(nis)
        if nis > self.gate:
            self.n_rejected += 1
            return False, nis
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.x[2] = M.wrap(self.x[2])
        I = np.eye(3)
        self.P = (I - Kg @ H) @ self.P @ (I - Kg @ H).T + Kg @ R @ Kg.T      
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        return True, nis

    def pose(self):
        return self.x.copy()


def marker_measurement(det_tvec, nz):
    from .vision import range_bearing
    r, phi = range_bearing(det_tvec)
    r = r / (1.0 + nz.get("range_scale_bias", 0.0))
    phi = M.wrap(phi - nz.get("bias_phi", 0.0))
    sr = nz["sig_r0"] + nz["sig_r1"] * r ** 2
    sp = nz["sig_p0"] + nz["sig_p1"] * r
    return (r, phi), (sr, sp)


def pillar_measurements(ranges, angles, pose_est, pillars=C.PILLARS, nz=None):
    out = []
    x, y, th = pose_est
    ox, oy = C.LIDAR_POS_B[0], C.LIDAR_POS_B[1]
    for (px, py, rad) in pillars:

        dx, dy = px - x, py - y
        cb = np.array([np.cos(th) * dx + np.sin(th) * dy - ox, -np.sin(th) * dx + np.cos(th) * dy - oy])
        rp, bp = np.hypot(*cb), np.arctan2(cb[1], cb[0])
        if rp > C.LIDAR_MAX - 0.3 or rp < 0.3:
            continue
        half = np.arcsin(min(1.0, (rad + 0.15) / rp))
        da = M.wrap(angles - bp)
        sel = (np.abs(da) < half) & np.isfinite(ranges) & (np.abs(ranges - (rp - rad)) < 0.30)
        if sel.sum() < 4:
            continue
        pts = np.stack([ranges[sel] * np.cos(angles[sel]), ranges[sel] * np.sin(angles[sel])], 1)
        c = cb.copy()
        for _ in range(6):               
            d = pts - c
            n = np.linalg.norm(d, axis=1)
            res = n - rad
            J = -d / n[:, None]
            c = c - np.linalg.lstsq(J, res, rcond=None)[0]
        if np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - rad) ** 2)) > 0.03:
            continue                      

        cbase = c + np.array([ox, oy])
        r, phi = np.hypot(*cbase), np.arctan2(cbase[1], cbase[0])
        sr = nz["lid_sig_r"] if nz else C.LIDAR_PILLAR_SIG[0]
        sp = (nz["lid_sig_pr"] if nz else 0.01) / max(r, 0.3) + 0.002
        sig = (sr, sp)
        out.append(((r, phi), sig, (px, py)))
    return out
