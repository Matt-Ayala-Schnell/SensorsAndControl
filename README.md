# SensorsAndControl
41014 Integrated Project Track A:

Group Number: --
Members: Matt Ayala-Schnell, Massimo Zanni, --
Github URL: 
Video Link:
Contribution Statement:
AI Declaration:

##Robot Task:
The robot starts in a randomised pose in a 6 x 6m room with pillars, boxes and 9 ArUco markers. It plans a path to a charging station and tracks it using LQR on the EKF estimate and replans if that obstacle is not on the map. If it is, it docks at 0.4m from the marker. 

##TO RUN:
```bash
pip install -r requirements.txt            
export MUJOCO_GL=egl                       
python -m tb3.model                        # Part A
python scripts/calibrate_camera.py         # Part B
python scripts/identify_noise.py           # Parts B/C
python -m tb3.mission 0                    # Metrics
python scripts/run_experiments.py          
python scripts/make_figures.py                           
python -m pytest tests
```

