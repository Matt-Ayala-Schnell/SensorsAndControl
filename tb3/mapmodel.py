import numpy as np
from . import config as C

RES = 0.05

def clearance(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = C.ROOM - np.maximum(np.abs(x), np.abs(y))
    for (px, py, r) in C.PILLARS:
        d = np.minimum(d, np.hypot(x - px, y - py) - r)
    for (px, py, hx, hy) in C.BOXES:
        qx, qy = np.abs(x - px) - hx, np.abs(y - py) - hy
        d = np.minimum(d, np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0))
    return d


def grid_shape():
    n = int(round(2 * C.ROOM / RES))
    return n, n


def world_to_grid(x, y):
    return np.floor((np.asarray(x) + C.ROOM) / RES).astype(int), np.floor((np.asarray(y) + C.ROOM) / RES).astype(int)


def grid_to_world(i, j):
    return -C.ROOM + (np.asarray(i) + 0.5) * RES, -C.ROOM + (np.asarray(j) + 0.5) * RES


def static_occupied():
    n, _ = grid_shape()
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    X, Y = grid_to_world(I, J)
    return clearance(X, Y) <= 0.0


def inflated(occ, radius):
    """Dilate a boolean grid by `radius` (m) using a distance threshold."""
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(~occ) * RES
    return dt <= radius


def expected_ranges(pose_xy_yaw, offset_b, angles, rmax=C.LIDAR_MAX):
    x, y, th = pose_xy_yaw
    c, s = np.cos(th), np.sin(th)
    ox, oy = x + c * offset_b[0] - s * offset_b[1], y + s * offset_b[0] + c * offset_b[1]
    steps = np.arange(0.05, rmax + 0.02, 0.02)
    a = th + angles
    X = ox + np.outer(np.cos(a), steps)
    Y = oy + np.outer(np.sin(a), steps)
    hit = clearance(X, Y) <= 0.0
    first = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)
