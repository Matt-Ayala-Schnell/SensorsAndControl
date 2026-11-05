import numpy as np
from . import config as C

wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi 

"""KINEMATICS ESTIMATOR MODELS"""
def f_odom(x, ds, dth):
  th = x[2] + dth / 2
  return np.array([x[0] + ds * np.cos(th), x[1] + ds * np.sin(th), wrap(x[2] + dth)])

def jac_F(x, ds, dth):
  th = x[2] + dth / 2
  return np.array([[1, 0, -ds * np.sin(th)], [0, 1, ds * np.cos(th)], [0, 0, 1]])

def jac_G(x, ds, dth, b):
    """d f / d (ds_right, ds_left): ds=(dr+dl)/2, dth=(dr-dl)/b."""
    th = x[2] + dth / 2
    c, s = np.cos(th), np.sin(th)
    dfd = np.array([[c, -0.5 * ds * s], [s, 0.5 * ds * c], [0, 1]])         
    dud = np.array([[0.5, 0.5], [1 / b, -1 / b]])                           
    return dfd @ dud


def h_range_bearing(x, m):
    """m=(mx,my): z = [r, phi]."""
    dx, dy = m[0] - x[0], m[1] - x[1]
    r = np.hypot(dx, dy)
    return np.array([r, wrap(np.arctan2(dy, dx) - x[2])])


def jac_H(x, m):
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]])

def continuous_5state(tau_v, tau_w):
    def AB(theta0, v0):
        A = np.zeros((5, 5))
        A[0, 2] = -v0 * np.sin(theta0); A[0, 3] = np.cos(theta0)
        A[1, 2] = v0 * np.cos(theta0);  A[1, 3] = np.sin(theta0)
        A[2, 4] = 1.0
        A[3, 3] = -1 / tau_v; A[4, 4] = -1 / tau_w
        B = np.zeros((5, 2)); B[3, 0] = 1 / tau_v; B[4, 1] = 1 / tau_w
        return A, B
    return AB

def error_model(v_r, w_r):
    A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]], float)
    B = np.array([[-1, 0], [0, 0], [0, -1]], float)
    return A, B

def ctrb_rank(A, B, tol=1e-9):
    n = A.shape[0]
    M = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    return int(np.linalg.matrix_rank(M, tol=tol)), M

def obsv_rank_ltv(x_traj, markers_per_step, ds_dth, tol=1e-7):
    n = 3
    Phi = np.eye(n)
    rows = []
    x = np.array(x_traj[0], float)
    for k, ms in enumerate(markers_per_step):
        for m in ms:
            rows.append(jac_H(x, m) @ Phi)
        ds, dth = ds_dth[k]
        Phi = jac_F(x, ds, dth) @ Phi
        x = f_odom(x, ds, dth)
    O = np.vstack(rows) if rows else np.zeros((1, n))
    sv = np.linalg.svd(O, compute_uv=False)
    return int((sv > tol * max(1.0, sv.max())).sum()), sv

def analysis_report():
    out = {}
    for name, (v, w) in {"v=0.10,w=0": (0.10, 0.0), "v=0.10,w=0.5": (0.10, 0.5), "v=0,w=0.5 (spin)": (0.0, 0.5), "v=0,w=0": (0.0, 0.0)}.items():
        A, B = error_model(v, w)
        r, M = ctrb_rank(A, B)
        out[f"ctrb[{name}]"] = (r, float(np.linalg.svd(M, compute_uv=False).min()))

    mA, mB = (3.0, 0.0), (0.0, 3.0)
    x0 = np.array([0.5, -1.0, 0.4])
    still = [(0.0, 0.0)] * 20
    move = [(0.02, 0.01)] * 20
    out["obs: odometry only, moving"] = obsv_rank_ltv([x0] * 1, [[]] * 20, move)[0]
    out["obs: 1 marker, stationary"] = obsv_rank_ltv([x0], [[mA]] * 20, still)[0]
    out["obs: 1 marker, moving (odometry known)"] = obsv_rank_ltv([x0], [[mA]] * 20, move)[0]
    out["obs: 2 markers, stationary"] = obsv_rank_ltv([x0], [[mA, mB]] * 20, still)[0]

    H = jac_H(x0, mA)
    _, _, Vt = np.linalg.svd(H)
    out["null direction, 1 marker stationary"] = Vt[-1].tolist()
    return out

#NEED TO SET UP REPORT DOC FOR RESULTS
if __name__ == "__main__":
    for k, v in analysis_report().items():
        print(k, v)