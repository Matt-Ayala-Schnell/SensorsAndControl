import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for mod in ["numpy", "scipy", "matplotlib", "cv2", "mujoco", "imageio"]:
    try:
        m = importlib.import_module(mod); print("OK  ", mod, getattr(m, "__version__", ""))
    except Exception as e:
        print("FAIL", mod, repr(e), "-> pip install -r requirements.txt")
from tb3.sim import Sim
s = Sim(render=False); s.reset([0, 0, 0]); s.set_cmd(0.1, 0); s.step()
print("OK   simulation step, pose =", s.truth_pose().round(3))
s2 = Sim(render=True); s2.reset([2, 0, 0]); print("OK   camera render", s2.camera().shape)