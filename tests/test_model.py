import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tb3 import model as M


def test_ctrb_rank_and_obsv():
    r = M.analysis_report()
    assert r["ctrb[v=0.10,w=0]"][0] == 3 and r["ctrb[v=0,w=0]"][0] == 2
    assert r["obs: 1 marker, stationary"] == 2 and r["obs: 2 markers, stationary"] == 3


def test_jacobians_match_finite_difference():
    x = np.array([0.3, -0.2, 0.7]); m = (2.0, 1.0); eps = 1e-6
    H = M.jac_H(x, m)
    Hn = np.stack([(M.h_range_bearing(x + eps * np.eye(3)[i], m) - M.h_range_bearing(x - eps * np.eye(3)[i], m)) / (2 * eps) for i in range(3)], 1)
    assert np.allclose(H, Hn, atol=1e-5)
