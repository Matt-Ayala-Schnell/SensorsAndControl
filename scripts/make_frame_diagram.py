import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from tb3 import config as C

fig = plt.figure(figsize=(13, 5.6))
ax = fig.add_axes([0.02, 0.05, 0.50, 0.88]); ax.set_aspect("equal"); ax.axis("off")


def arrow(p, q, c="k", lw=1.6, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=c, lw=lw, ls=ls))


arrow((0, 0), (1.1, 0), "C0"); arrow((0, 0), (0, 1.1), "C0"); ax.text(1.15, -0.1, "x_W", color="C0"); ax.text(-0.25, 1.1, "y_W", color="C0"); ax.text(-0.45, -0.25, "W (world)", color="C0", weight="bold")


# robot pose
x, y, th = 2.6, 1.3, 0.55
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
body = Circle((x, y), 0.30, fc="#dddddd", ec="k"); ax.add_patch(body)
bx = R @ [0.9, 0]; by = R @ [0, 0.9]
arrow((x, y), (x + bx[0], y + bx[1]), "C3"); arrow((x, y), (x + by[0], y + by[1]), "C3")
ax.text(x + bx[0] + 0.05, y + bx[1] - 0.12, "x_B", color="C3"); ax.text(x + by[0] - 0.2, y + by[1] + 0.05, "y_B", color="C3"); ax.text(1.35, 0.35, "B (base, on wheel axle)", color="C3", weight="bold")
cam = np.array([x, y]) + R @ [0.17, 0]
ax.plot(*cam, "gs", ms=7); ax.text(3.35, 0.72, "camera Cv\n(0.05, 0, 0.15) m", color="g", fontsize=8)
lid = np.array([x, y]) + R @ [-0.11, 0]
ax.plot(*lid, "mo", ms=6); ax.text(0.35, 1.0, "LiDAR\n(-0.032, 0, 0.17) m", color="m", fontsize=8)

ang = np.linspace(0, th, 20); ax.plot(x + 0.6 * np.cos(ang), y + 0.6 * np.sin(ang), "k-", lw=1); ax.text(x + 0.62, y + 0.12, "θ", fontsize=11)

mx, my = 5.2, 3.2
ax.add_patch(Rectangle((mx - 0.03, my - 0.5), 0.06, 1.0, fc="#999999"))
arrow((mx, my), (mx - 0.9, my), "C2"); arrow((mx, my), (mx, my + 0.9), "C2"); ax.text(mx - 1.15, my + 0.1, "z_M (out of face)", color="C2", fontsize=8); ax.text(mx + 0.08, my + 0.8, "y_M (up)", color="C2", fontsize=8)
ax.text(2.6, 3.75, "M (ArUco marker,\nknown pose in W)", color="C2", weight="bold", fontsize=9)

arrow((x, y), (mx, my), "k", 1.2, "--", "-|>"); ax.text((x + mx) / 2 + 0.15, (y + my) / 2 - 0.35, "r", fontsize=12)
phi = np.arctan2(my - y, mx - x) - th
ang = np.linspace(th, th + phi, 20); ax.plot(x + 1.25 * np.cos(ang), y + 1.25 * np.sin(ang), "k-", lw=1); ax.text(x + 1.3, y + 0.75, "φ", fontsize=11)
ax.set_xlim(-0.8, 6.0); ax.set_ylim(-0.6, 4.2)
ax.set_title("Geometry: measurement = (range r, bearing φ) to a known landmark, about the base origin", fontsize=10)

ax2 = fig.add_axes([0.54, 0.05, 0.45, 0.88]); ax2.axis("off"); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
steps = [("Image corners (pixels)", "ArUco detector, 4 corners per marker"),
         ("t_Cv,M  (marker in camera frame)", "solvePnP IPPE_SQUARE with CALIBRATED K, side 0.20 m"),
         ("p_B = R_B,Cv · t + t_B,Cv", "R_B,Cv = [[0,0,1],[-1,0,0],[0,-1,0]],  t_B,Cv = (0.05, 0, 0.15) m"),
         ("(r, φ) = (|p_B,xy|, atan2(p_B,y , p_B,x))", "bias-corrected (scale 1.5 %, bearing -5.7 mrad); σ from fitted noise model"),
         ("EKF update vs known m_W:  h(x) = [ |m−p| , atan2(m−p) − θ ]", "gated on NIS (χ²₂ 99 %), Joseph form; marker range ≤ 3.6 m"),
         ("x̂_W = (x, y, θ) with covariance P", "this and only this goes to planner and LQR"),
         ("e = R(θ̂)ᵀ (p_ref − p̂),  e_θ = θ_ref − θ̂   in B", "LQR: u = u_ref − K e  →  (v, ω) → wheel speeds")]
h = 0.105
for i, (t1, t2) in enumerate(steps):
    yy = 0.9 - i * 0.135
    ax2.add_patch(FancyBboxPatch((0.02, yy - 0.04), 0.96, h, boxstyle="round,pad=0.01", fc="#f2f6fb", ec="#4a6fa5"))
    ax2.text(0.04, yy + 0.035, t1, fontsize=9.5, weight="bold", va="center"); ax2.text(0.04, yy - 0.012, t2, fontsize=8, va="center", color="#333")
    if i < len(steps) - 1: ax2.annotate("", xy=(0.5, yy - 0.06), xytext=(0.5, yy - 0.032), arrowprops=dict(arrowstyle="-|>", color="#4a6fa5"))
ax2.set_title("Frame chain: sensor measurement → quantity the controller acts on", fontsize=10)
plt.savefig("results/fig_frames.png", dpi=140)
print("wrote results/fig_frames.png")
