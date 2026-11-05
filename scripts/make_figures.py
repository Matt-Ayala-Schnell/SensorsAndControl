import sys, os, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from tb3 import config as C

rows = [json.loads(l) for l in open("results/trials.jsonl")]
groups = {}
for r in rows:
    groups.setdefault(r["group"], []).append(r)


def agg(rs, k):
    v = np.array([r[k] for r in rs if r.get(k) is not None and r["reason"] not in ("crash",) and not str(r["reason"]).startswith("crash")], float)
    return (np.nanmedian(v), np.nanpercentile(v, 90)) if len(v) else (np.nan, np.nan)


lines = ["| group | n | success | dock pos err med/p90 (cm) | RMSE pos (cm) | ANEES (ideal 3) | path RMSE (cm) | effort | replans | failures |", "|---|---|---|---|---|---|---|---|---|---|"]
for g, rs in groups.items():
    n = len(rs); ok = sum(bool(r["success"]) for r in rs)
    reasons = {}
    for r in rs:
        if not r["success"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    pe = agg(rs, "pos_err_dock"); rm = agg(rs, "rmse_pos"); an = agg(rs, "anees"); pa = agg(rs, "path_rmse"); ef = agg(rs, "effort"); rp = agg(rs, "n_replans")
    lines.append(f"| {g} | {n} | {ok}/{n} | {pe[0]*100:.1f} / {pe[1]*100:.1f} | {rm[0]*100:.1f} | {an[0]:.1f} | {pa[0]*100:.1f} | {ef[0]:.2f} | {rp[0]:.0f} | {reasons or '-'} |")
open("results/summary_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

if os.path.exists("results/demo.pkl"):
    d = pickle.load(open("results/demo.pkl", "rb"))
    L, ref = d["log"], d["ref"]
    t, tr, es, P = L["t"], L["truth"], L["est"], L["P"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    a = ax[0]
    a.set_aspect("equal"); a.set_xlim(-3.1, 3.1); a.set_ylim(-3.1, 3.1)
    for (x, y, r) in C.PILLARS: a.add_patch(Circle((x, y), r, color="orange"))
    for (x, y, hx, hy) in C.BOXES: a.add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, color="steelblue"))
    from tb3.world import all_markers
    for mid, (x, y, yaw) in all_markers().items():
        a.plot([x], [y], "ks", ms=5); a.text(x * 0.93, y * 0.93, str(mid), fontsize=8)
    a.plot(ref["x"], ref["y"], "g--", lw=1, label="reference")
    a.plot(tr[:, 0], tr[:, 1], "k-", lw=1.5, label="truth (eval only)")
    a.plot(es[:, 0], es[:, 1], "r:", lw=1.5, label="EKF estimate")
    a.plot(*C.DOCK_POSE[:2], "m*", ms=12, label="dock pose"); a.legend(loc="lower left", fontsize=8); a.set_title("Nominal run: trajectory")
    e = tr - es; e[:, 2] = (e[:, 2] + np.pi) % (2 * np.pi) - np.pi
    sd = np.sqrt(np.stack([P[:, i, i] for i in range(3)], 1))
    b = ax[1]
    b.plot(t, e[:, 0] * 100, "C0", label="x error"); b.fill_between(t, -3 * sd[:, 0] * 100, 3 * sd[:, 0] * 100, color="C0", alpha=0.2, label="±3σ (x)")
    b.plot(t, e[:, 1] * 100, "C1", label="y error"); b.fill_between(t, -3 * sd[:, 1] * 100, 3 * sd[:, 1] * 100, color="C1", alpha=0.2, label="±3σ (y)")
    b.set_ylim(-25, 25); b.set_xlabel("t (s)"); b.set_ylabel("cm"); b.legend(fontsize=8, ncol=2); b.set_title("Estimation error vs EKF 3σ bounds")
    plt.tight_layout(); plt.savefig("results/fig_demo.png", dpi=130); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].semilogy(t, np.maximum(L["nees"], 1e-3)); ax[0].axhline(7.81, color="r", ls="--", label="χ²₃ 95%"); ax[0].set_title("NEES (truth vs EKF)"); ax[0].set_xlabel("t (s)"); ax[0].legend()
    ax[1].plot(t, L["cmd"][:, 0], label="v (m/s)"); ax[1].plot(t, L["cmd"][:, 1], label="ω (rad/s)"); ax[1].legend(); ax[1].set_title("Commands"); ax[1].set_xlabel("t (s)")
    plt.tight_layout(); plt.savefig("results/fig_nees_cmd.png", dpi=130); plt.close()
