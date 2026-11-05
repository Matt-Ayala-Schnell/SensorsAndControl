import os , sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")