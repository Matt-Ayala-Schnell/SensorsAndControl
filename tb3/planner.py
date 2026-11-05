"""This file produces path for obstacles"""
import heapq
import numpy as np
from scipy.interpolate import splprep, splev
from . import config as C
from . import mapmodel as MM

INFL = C.ROBOT_RADIUS + 0.10          
SAFE = C.ROBOT_RADIUS + 0.03          
V_REF, A_LAT, V_END = 0.12, 0.08, 0.03


def astar(occ, start, goal):
    n = occ.shape[0]
    s = tuple(int(v) for v in MM.world_to_grid(*start))
    g = tuple(int(v) for v in MM.world_to_grid(*goal))
    if occ[g]:
        return None
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq = [(0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    while openq:
        _, u = heapq.heappop(openq)
        if u == g:
            break
        for dx, dy, w in nb:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < n and 0 <= v[1] < n) or (occ[v] and v != s):
                continue
            nc = cost[u] + w
            if v not in cost or nc < cost[v]:
                cost[v] = nc; came[v] = u
                heapq.heappush(openq, (nc + np.hypot(v[0] - g[0], v[1] - g[1]), v))
    if g not in came:
        return None
    path, u = [], g
    while u is not None:
        path.append(u); u = came[u]
    path = path[::-1]
    xs, ys = MM.grid_to_world([p[0] for p in path], [p[1] for p in path])
    return np.stack([xs, ys], 1)


def _los(occ, a, b):
    d = np.linalg.norm(b - a)
    t = np.linspace(0, 1, max(2, int(d / 0.02)))
    p = a[None] + t[:, None] * (b - a)[None]
    i, j = MM.world_to_grid(p[:, 0], p[:, 1])
    return not occ[i, j].any()


def shortcut(path, occ):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _los(occ, path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return np.array(out)


def _fillet(P, Rc):
    P = np.asarray(P, float)
    d = np.diff(P, axis=0); l = np.linalg.norm(d, axis=1); u = d / l[:, None]
    X, Y, TH, KA = [], [], [], []
    cur = P[0].copy()

    def line(a, b):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.02)))
        t = np.linspace(0, 1, n + 1)[:-1]
        pts = a[None] + t[:, None] * (b - a)[None]
        X.extend(pts[:, 0]); Y.extend(pts[:, 1])
        TH.extend([np.arctan2(*(b - a)[::-1])] * len(t)); KA.extend([0.0] * len(t))
    for i in range(1, len(P) - 1):
        d1, d2 = u[i - 1], u[i]
        phi = np.arccos(np.clip(d1 @ d2, -1, 1))
        if phi < 1e-3:
            continue
        sgn = np.sign(d1[0] * d2[1] - d1[1] * d2[0])

        r = min(Rc, 0.5 * min(l[i - 1], l[i]) / np.tan(phi / 2))
        T = r * np.tan(phi / 2)
        A = P[i] - d1 * T
        line(cur, A)
        nL = np.array([-d1[1], d1[0]]); th0 = np.arctan2(d1[1], d1[0])
        n = max(2, int(np.ceil(r * phi / 0.02)))
        ang = np.linspace(0, phi, n + 1)[:-1]
        pts = A[None] + r * (np.sin(ang)[:, None] * d1[None] + sgn * (1 - np.cos(ang))[:, None] * nL[None])
        X.extend(pts[:, 0]); Y.extend(pts[:, 1]); TH.extend(th0 + sgn * ang); KA.extend([sgn / r] * len(ang))
        cur = P[i] + d2 * T
    line(cur, P[-1])
    X.append(P[-1][0]); Y.append(P[-1][1]); TH.append(TH[-1]); KA.append(0.0)
    return np.array(X), np.array(Y), np.unwrap(np.array(TH)), np.array(KA)


def build_reference(waypoints, extra_clear=None):
   
    for Rc in (0.60, 0.45, 0.30, 0.20, 0.12):
        x, y, th, kappa = _fillet(waypoints, Rc)
        ok = (MM.clearance(x, y) >= SAFE).all() and (extra_clear is None or extra_clear(x, y))
        if ok:
            break
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    Ltot = s[-1]
    v = np.minimum(V_REF, np.sqrt(A_LAT / np.maximum(np.abs(kappa), 1e-6)))
    v = np.minimum(v, V_END + 0.35 * (Ltot - s))          
    v = np.minimum(v, 0.04 + 0.25 * s)                   
    ds = np.maximum(seg, 1e-3)
    for i in range(len(v) - 2, -1, -1):                   
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * 0.15 * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * 0.15 * ds[i - 1]))
    return dict(s=s, x=x, y=y, th=th, kappa=kappa, v=v, w=v * kappa, L=Ltot, Rc=Rc,
                min_clear=float(MM.clearance(x, y).min()), clear_ok=bool(ok))


def plan_to_dock(start_xy, dyn_occ=None):
    occ = MM.static_occupied()
    if dyn_occ is not None:
        occ = occ | dyn_occ
    occ = MM.inflated(occ, INFL)
    approach = np.array([C.DOCK_POSE[0] - 1.2, 0.0])
    path = astar(occ, np.asarray(start_xy), approach)
    if path is None:
        return None
    wp = shortcut(path, occ)
    wp[0] = np.asarray(start_xy)
    wp[-1, 1] = 0.0                                         
    wp = np.vstack([wp, [C.DOCK_POSE[0], 0.0]])           
    dyn_inf = MM.inflated(dyn_occ, INFL - 0.05) if dyn_occ is not None else None

    def extra(x, y):
        if dyn_inf is None:
            return True
        i, j = MM.world_to_grid(x, y)
        return not dyn_inf[np.clip(i, 0, 119), np.clip(j, 0, 119)].any()
    ref = build_reference(wp, extra)
    ref["waypoints"] = wp
    return ref
