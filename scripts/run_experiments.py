"""Part E:"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiprocessing import Pool
from tb3.mission import run_trial

OUT = "results/trials.jsonl"
HARD = dict(unmapped=(-0.05, -0.9, 0.15, 0.15), push=True, friction=0.7, outlier_prob=0.03, dropout=(20.0, 32.0), pixel_noise_scale=2.0)
GROUPS = {
    "nominal": (dict(), range(0, 10)),
    "odom_only": (dict(variant="odom"), range(0, 4)),
    "cam_only": (dict(variant="cam"), range(0, 4)),
    "lidar_only": (dict(variant="lidar"), range(0, 4)),
    "R_x0.1": (dict(r_scale=0.1), range(0, 4)),
    "R_x10": (dict(r_scale=10.0), range(0, 4)),
    "Q_x10": (dict(q_scale=10.0), range(0, 4)),
    "LQR_Q_x0.1": (dict(lqr_q_scale=0.1), range(0, 4)),
    "LQR_Q_x10": (dict(lqr_q_scale=10.0), range(0, 4)),
    "no_range_limit": (dict(marker_rmax=None), range(0, 6)),
    "disturbed": (HARD, range(0, 8)),
    "disturbed_no_gate": (dict(HARD, gate=1e9), range(0, 4)),
}


def job(a):
    name, kw, seed = a
    try:
        r = run_trial(seed=seed, **kw)
        m = r["metrics"]
        rec = dict(group=name, seed=seed, **{k: (v if not hasattr(v, "item") else v.item()) for k, v in m.items()})
    except Exception as e:  
        rec = dict(group=name, seed=seed, success=False, reason="crash:" + repr(e)[:80])
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            j = json.loads(l); done.add((j["group"], j["seed"]))
    jobs = [(n, kw, s) for n, (kw, ss) in GROUPS.items() for s in ss if (n, s) not in done]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            print(r["group"], r["seed"], r["success"], r.get("reason"), flush=True)
