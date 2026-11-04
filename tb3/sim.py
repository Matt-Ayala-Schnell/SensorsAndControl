import os
##os.environ.setdefault("MUJOCO_GL", "egl") ##Potential to be linux only...
import numpy as np
import mujoco
from . import config as C
from .world import build_xml


class Sim:
    def __init__(self, seed=0, unmapped=None, wheel_friction=1.0, checker=False, render=True, enc_noise=True):
        self.rng = np.random.default_rng(seed)
        self.m = mujoco.MjModel.from_xml_string(build_xml(unmapped=unmapped, wheel_friction=wheel_friction, checker=checker))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.jl = self.m.joint("jl").qposadr[0]
        self.jr = self.m.joint("jr").qposadr[0]
        self.lid_site = self.m.site("lidar").id
        self.cam_id = self.m.camera("cam").id
        self.renderer = mujoco.Renderer(self.m, C.CAM_H, C.CAM_W) if render else None
        self.enc_noise = enc_noise
        self.pixel_noise = C.PIXEL_NOISE
        self.gm = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # LiDAR sees group 0 only (world), not the robot (group 1)
        ang = np.linspace(0, 2 * np.pi, C.LIDAR_N, endpoint=False)
        self.lid_ang = ang
        self.lid_dirs_b = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
        self._enc_last = None
        self.push = np.zeros(6)
        self.t = 0.0

    # ----- setup -----
    def reset(self, pose):
        mujoco.mj_resetData(self.m, self.d)
        self.d.ctrl[:] = 0.0
        x, y, th = pose
        self.d.qpos[0:3] = [x, y, 0.033]
        self.d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
        mujoco.mj_forward(self.m, self.d)
        for _ in range(40):                      # settle
            mujoco.mj_step(self.m, self.d)
        self.d.qvel[:] = 0
        self.t = 0.0
        self._enc_phase = self.rng.uniform(-1, 1, 2)  # encoder zero offset (sub-tick)
        self._enc_last = self.encoders()

    def set_wheel_speeds(self, wl, wr):
        self.d.ctrl[0], self.d.ctrl[1] = wl, wr

    def set_cmd(self, v, w):
        v = np.clip(v, -C.V_MAX, C.V_MAX); w = np.clip(w, -C.W_MAX, C.W_MAX)
        self.set_wheel_speeds((v - w * C.TRACK / 2) / C.WHEEL_R, (v + w * C.TRACK / 2) / C.WHEEL_R)
        return v, w

    def step(self, n=C.SUB):
        for _ in range(n):
            self.d.xfrc_applied[self.bid, :] = self.push
            mujoco.mj_step(self.m, self.d)
        self.t = self.d.time

    # ----- sensors -----
    def encoders(self):
        """Cumulative wheel angles (rad), quantised to ENC_TICKS per rev. (left, right)."""
        q = np.array([self.d.qpos[self.jl], self.d.qpos[self.jr]])
        tick = 2 * np.pi / C.ENC_TICKS
        return np.round(q / tick) * tick

    def odometry_increment(self):
        e = self.encoders()
        de = e - self._enc_last
        self._enc_last = e
        return C.WHEEL_R * de[0], C.WHEEL_R * de[1]    # ds_left, ds_right (m)

    def lidar(self):
        """360 ranges (m) in the lidar frame; inf where nothing within LIDAR_MAX."""
        pos = self.d.site_xpos[self.lid_site].copy()
        R = self.d.site_xmat[self.lid_site].reshape(3, 3)
        dirs = (R @ self.lid_dirs_b.T).T
        dist = np.full(C.LIDAR_N, -1.0)
        geomid = np.zeros(C.LIDAR_N, dtype=np.int32)
        mujoco.mj_multiRay(self.m, self.d, pos, dirs.flatten(), self.gm, 1, -1, geomid, dist, None, C.LIDAR_N, C.LIDAR_MAX + 1)
        r = dist.copy()
        r[(r < 0)] = np.inf
        r = r + self.rng.normal(0, C.LIDAR_SIGMA, r.shape)
        r[(r > C.LIDAR_MAX) | (r < C.LIDAR_MIN)] = np.inf
        return r

    def camera(self):
        """Grayscale uint8 image with sensor noise."""
        self.renderer.update_scene(self.d, "cam")
        rgb = self.renderer.render()
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
        g = g + self.rng.normal(0, self.pixel_noise, g.shape)
        return np.clip(g, 0, 255).astype(np.uint8)

    # ----- evaluation-only ground truth -----
    def truth_pose(self):
        p = self.d.xpos[self.bid]
        q = self.d.xquat[self.bid]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.array([p[0], p[1], yaw])

    def truth_collision(self):
        """True if any robot geom (group 1) touches a world geom (group 0) other than the floor. Evaluation only."""
        fl = self.m.geom("floor").id
        gg = self.m.geom_group
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if {gg[g1], gg[g2]} == {0, 1} and fl not in (g1, g2):
                return True
        return False

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
