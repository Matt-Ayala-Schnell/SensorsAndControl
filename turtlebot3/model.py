import numpy as np
#from . import config as C

wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi 

"""KINEMATICS ESTIMATOR MODELS"""
def f_odom(x, ds, dth):
  th = x[2] + dth / 2
  return np.array([x[0] + ds * np.cos(th), x[1] + ds * np.sin(th), wrap(x[2] + dth)])

def jac_F(x, ds, dth):
  th = x[2] + dth / 2
  return np.array([1, 0, -ds * np.sin(th)], [0, 1, ds * np.cos(th), [0,0, 1]])

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

