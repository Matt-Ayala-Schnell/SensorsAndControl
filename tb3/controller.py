"""Part D:"""
import numpy as np
import scipy.linalg as sla
from . import config as C
from . import model as M


class LQRTracker:
    def __init__(self, ref, dt=C.CTRL_DT, Q=None, R=None, hold_dist=0.20):
        self.ref, self.dt = ref, dt
        self.Q = C.LQR_Q if Q is None else Q
        self.R = C.LQR_R if R is None else R
        self.s_ref, self.i_proj, self.hold = 0.0, 0, hold_dist
        self.aligned = False
        self.done = False

    def gain(self, v_r, w_r):
        A, B = M.error_model(max(v_r, 0.02), w_r)         
        Ad, Bd = np.eye(3) + A * self.dt, B * self.dt
        P = sla.solve_discrete_are(Ad, Bd, self.Q, self.R)
        return np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    def _project(self, p):
        r = self.ref
        lo = max(0, self.i_proj - 10); hi = min(len(r["s"]), self.i_proj + 60)
        d = np.hypot(r["x"][lo:hi] - p[0], r["y"][lo:hi] - p[1])
        self.i_proj = lo + int(d.argmin())
        return self.i_proj

    def step(self, est):
        r = self.ref
        ip = self._project(est)
        s_proj = r["s"][ip]
        if not self.aligned:                                   
            e_th = M.wrap(r["th"][0] - est[2])
            if abs(e_th) < 0.15:
                self.aligned = True
            else:
                return 0.0, float(np.clip(1.5 * e_th, -C.W_MAX, C.W_MAX)), dict(phase="align", e=np.array([0, 0, e_th]))
        j = int(np.searchsorted(r["s"], self.s_ref))
        j = min(j, len(r["s"]) - 1)
        self.s_ref = min(self.s_ref + r["v"][j] * self.dt, s_proj + self.hold, r["L"])
        j = min(int(np.searchsorted(r["s"], self.s_ref)), len(r["s"]) - 1)
        v_r, w_r = r["v"][j], r["w"][j]
        c, s_ = np.cos(est[2]), np.sin(est[2])
        dx, dy = r["x"][j] - est[0], r["y"][j] - est[1]
        e = np.array([c * dx + s_ * dy, -s_ * dx + c * dy, M.wrap(r["th"][j] - est[2])])
        K = self.gain(v_r, w_r)
        du = -K @ e
        v, w = v_r + du[0], w_r + du[1]

        e_end_x = np.cos(r["th"][-1]) * (r["x"][-1] - est[0]) + np.sin(r["th"][-1]) * (r["y"][-1] - est[1])
        if s_proj > r["L"] - 0.05 and e_end_x < 0.012:
            self.done = True
            return 0.0, 0.0, dict(phase="done", e=e)
        return float(np.clip(v, 0.0, C.V_MAX)), float(np.clip(w, -C.W_MAX, C.W_MAX)), dict(phase="track", e=e, K=K, v_r=v_r, w_r=w_r, s=s_proj)
