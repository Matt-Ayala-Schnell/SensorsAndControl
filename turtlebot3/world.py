import os
import numpy as np
import cv2
from . import config as C

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

def make_marker_png(mid: int, path: str, px: int = 400):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cell = px // 7
    m = cv2.aruco.generateImageMarker(d, mid, cell * 5)
    img = np.full((cell * 7, cell * 7), 255, np.uint8)
    img[cell:cell * 6, cell:cell * 6] = m
    cv2.imwrite(path, img)

def make_checker_png(path: str, nx=9, ny=6, sq=60):
    inner = np.zeros(((ny + 1) * sq, (nx + 1) * sq), np.uint8)
    for j in range(ny + 1):
        for i in range(nx + 1):
            inner[j * sq:(j + 1) * sq, i * sq:(i + 1) * sq] = 0 if (i + j) % 2 == 0 else 255
    img = np.full(((ny + 3) * sq, (nx + 3) * sq), 255, np.uint8)
    img[sq:-sq, sq:-sq] = inner
    cv2.imwrite(path, img)

def marker_frame(x, y, yaw):
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, n)          
    R = np.stack([right, up, n], axis=1)
    return R, np.array([x, y, C.MARKER_Z])

def all_markers():
    d = dict(C.MARKERS)
    d[C.DOCK_ID] = C.DOCK_MARKER
    return d

def build_xml(extra_obstacle=None, wheel_friction=1.0, checker=False, unmapped=None):
    os.makedirs(ASSET_DIR, exist_ok=True)
    mk = all_markers()
    tex, mat, geoms = [], [], []
    for mid in mk:
        p = os.path.join(ASSET_DIR, f"m{mid}.png")
        if not os.path.exists(p):
            make_marker_png(mid, p)
        tex.append(f'<texture name="t{mid}" type="2d" file="{p}"/>')
        mat.append(f'<material name="mm{mid}" texture="t{mid}" specular="0" shininess="0" emission="0.6"/>')
    ck = os.path.join(ASSET_DIR, "checker.png")
    if not os.path.exists(ck):
        make_checker_png(ck)
    tex.append(f'<texture name="tck" type="2d" file="{ck}"/>')
    mat.append('<material name="mck" texture="tck" specular="0" shininess="0" emission="0.6"/>')
    s = C.MARKER_SIZE * 7 / 5 / 2   

    for mid, (x, y, yaw) in mk.items():
        R, t = marker_frame(x, y, yaw)
        t = t + R[:, 2] * 0.003
        xy = f"{R[0,0]:.6f} {R[1,0]:.6f} {R[2,0]:.6f} {R[0,1]:.6f} {R[1,1]:.6f} {R[2,1]:.6f}"
        geoms.append(f'<geom name="mk{mid}" type="box" size="{s:.5f} {s:.5f} 0.002" pos="{t[0]:.5f} {t[1]:.5f} {t[2]:.5f}" '
                     f'xyaxes="{xy}" material="mm{mid}" contype="0" conaffinity="0"/>')
        
    for i, (x, y, r) in enumerate(C.PILLARS):
        geoms.append(f'<geom name="pillar{i}" type="cylinder" size="{r}  0.2" pos="{x} {y} 0.2" rgba="0.8 0.5 0.2 1"/>')
    
    for i, (x, y, hx, hy) in enumerate(C.BOXES):
        geoms.append(f'<geom name="box{i}" type="box" size="{hx} {hy} 0.2" pos="{x} {y} 0.2" rgba="0.3 0.5 0.8 1"/>')
    
    if unmapped is not None:
        x, y, hx, hy = unmapped
        geoms.append(f'<geom name="unmapped" type="box" size="{hx} {hy} 0.2" pos="{x} {y} 0.2" rgba="0.9 0.2 0.2 1"/>')
    
    W, H = C.ROOM, C.WALL_H / 2
    T = 0.02   
    
    for nm, (px, py, sx, sy) in dict(wn=(0, W + T, W + 2 * T, T), ws=(0, -W - T, W + 2 * T, T),
                                     we=(W + T, 0, T, W + 2 * T), ww=(-W - T, 0, T, W + 2 * T)).items():
        geoms.append(f'<geom name="{nm}" type="box" size="{sx} {sy} {H}" pos="{px:.3f} {py:.3f} {H}" rgba="0.85 0.85 0.85 1"/>')
    
    if checker:   
        geoms.append('<body name="checker" mocap="true" pos="1.2 0 0.15"><geom name="ck" type="box" size="0.33 0.2475 0.002" material="mck" '
                     'contype="0" conaffinity="0"/></body>')
    
    r, b = C.WHEEL_R, C.TRACK
    cx, cy, cz = C.CAM_POS_B
    lx, ly, lz = C.LIDAR_POS_B
    fov = C.CAM_FOVY_DEG
    ##NEXT SEGMENT SOURCED ONLINE:
    return f""" 
<mujoco model="tb3_room">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{C.SIM_DT}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="{C.CAM_W}" offheight="{C.CAM_H}"/><quality shadowsize="1024"/></visual>
  <asset>
    {''.join(tex)}{''.join(mat)}
    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1="0.35 0.38 0.40" rgb2="0.30 0.33 0.35"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0.0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9" ambient="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="{W+0.2} {W+0.2} 0.1" material="floor" friction="{wheel_friction} 0.005 0.0001"/>
    {''.join(geoms)}
    <body name="base" pos="0 0 0.033">
      <freejoint name="root"/>
      <geom name="chassis" type="cylinder" size="0.07 0.045" pos="0 0 0.045" mass="0.9" rgba="0.15 0.15 0.15 1" group="1" contype="2" conaffinity="1"/>
      <geom name="deck" type="cylinder" size="0.05 0.02" pos="-0.02 0 0.11" mass="0.1" rgba="0.9 0.9 0.9 1" group="1" contype="0" conaffinity="0"/>
      <geom name="lidar_hat" type="cylinder" size="0.035 0.02" pos="{lx} {ly} {lz-0.033-0.02}" mass="0.001" rgba="0.1 0.1 0.6 1" group="1" contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.008" pos="0.065 0 -0.025" friction="0.0 0.0 0.0" priority="1" mass="0.01" group="1" contype="2" conaffinity="1"/>
      <geom name="caster_b" type="sphere" size="0.008" pos="-0.065 0 -0.025" friction="0.0 0.0 0.0" priority="1" mass="0.01" group="1" contype="2" conaffinity="1"/>
      <body name="wl" pos="0 {b/2} 0"><joint name="jl" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="cylinder" size="{r} 0.009" euler="1.5708 0 0" mass="0.03" friction="{wheel_friction} 0.005 0.0001" rgba="0.1 0.1 0.1 1" group="1" contype="2" conaffinity="1"/></body>
      <body name="wr" pos="0 {-b/2} 0"><joint name="jr" type="hinge" axis="0 1 0" damping="0.002"/>
        <geom type="cylinder" size="{r} 0.009" euler="1.5708 0 0" mass="0.03" friction="{wheel_friction} 0.005 0.0001" rgba="0.1 0.1 0.1 1" group="1" contype="2" conaffinity="1"/></body>
      <site name="lidar" pos="{lx} {ly} {lz-0.033}"/>
      <camera name="cam" pos="{cx} {cy} {cz-0.033}" xyaxes="0 -1 0 0 0 1" fovy="{fov}"/>
      <site name="camsite" pos="{cx} {cy} {cz-0.033}" size="0.003"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="al" joint="jl" kv="0.03" ctrlrange="-30 30" forcerange="-0.06 0.06"/>
    <velocity name="ar" joint="jr" kv="0.03" ctrlrange="-30 30" forcerange="-0.06 0.06"/>
  </actuator>
</mujoco>"""

