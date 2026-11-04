import numpy as np

WHEEL_R = 0.033          
TRACK = 0.160            
ROBOT_RADIUS = 0.105     
V_MAX = 0.15             
W_MAX = 1.5              
ENC_TICKS = 4096   

SIM_DT = 0.005           
CTRL_DT = 0.05           
CAM_EVERY = 4            
LIDAR_EVERY = 4          
SUB = int(round(CTRL_DT / SIM_DT))

CAM_W, CAM_H = 640, 480
CAM_FOVY_DEG = 48.8                 
CAM_POS_B = np.array([0.05, 0.0, 0.15])  
MARKER_SIZE = 0.20       
PIXEL_NOISE = 2.0

LIDAR_N = 360
LIDAR_MAX = 3.5
LIDAR_MIN = 0.12
LIDAR_SIGMA = 0.01       
LIDAR_POS_B = np.array([-0.032, 0.0, 0.17])

"""MAP SETUP"""
ROOM = 3.0              
WALL_H = 0.5
PILLARS = [(-0.6, 0.9, 0.15), (0.7, -0.8, 0.15), (1.2, 1.0, 0.15)]
BOXES = [(-1.2, -1.2, 0.25, 0.25), (0.2, 0.2, 0.2, 0.15)]
MARKERS = {
    1: (-3.0, 1.5, 0.0), 2: (-3.0, -1.5, 0.0),
    3: (-1.5, 3.0, -np.pi / 2), 4: (1.5, 3.0, -np.pi / 2),
    5: (-1.5, -3.0, np.pi / 2), 6: (1.5, -3.0, np.pi / 2),
    7: (3.0, 2.0, np.pi), 8: (3.0, -2.0, np.pi),
}
DOCK_ID = 0
DOCK_MARKER = (3.0, 0.0, np.pi)        
DOCK_STANDOFF = 0.40                   
DOCK_POSE = np.array([3.0 - DOCK_STANDOFF, 0.0, 0.0])
DOCK_TOL_POS, DOCK_TOL_YAW = 0.05, np.deg2rad(5.0)
MARKER_Z = 0.17
START_NOMINAL = np.array([-2.4, -2.2, 0.6])

##ESTIMATOR
K_ODO = 2.0e-4

SIG_R0, SIG_R1 = 0.010, 0.0060
SIG_P0, SIG_P1 = 0.004, 0.0030
GATE_CHI2_2 = 9.21       
LIDAR_PILLAR_SIG = (0.025, 0.020)  

##CONTROLLER##
LQR_Q = np.diag([30.0, 60.0, 8.0])
LQR_R = np.diag([2.0, 0.5])


