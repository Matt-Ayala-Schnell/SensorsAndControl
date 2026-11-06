# 41014 Integrated Project — Track A: visual waypoint navigation and ArUco docking (TurtleBot3, MuJoCo)

41014 Integrated Project Track A:

Group Number: 44
Members: Matt Ayala-Schnell, Massimo Zanni
Github URL: https://github.com/Matt-Ayala-Schnell/SensorsAndControl
Video Link: https://youtu.be/FSR8wzv_H_A
Contribution Statement: Each member contributed a fair amount to the project.
AI Declaration: AI (Claude by Anthropic) was utilised to assist the creation of the demo video. 

## 1. Project overview
This project implements an integrated perception, estimation and control
pipeline for a TurtleBot3-style differential-drive robot operating in a
simulated 6 m × 6 m environment. The environment contains pillars, boxes
and nine wall-mounted ArUco markers. The robot begins from a randomised
pose, navigates around obstacles and visually docks at a charging
station identified by ArUco marker ID 0 on the east wall.

The navigation pipeline combines wheel-encoder odometry, camera-based
ArUco measurements and LiDAR-based landmark detection in an Extended
Kalman Filter (EKF). The estimated pose is used by an A\* path planner
and an LQR trajectory tracker. When the LiDAR detects an obstacle that
is not represented in the known map, the robot can replan its route.
During docking, the robot approaches to 0.40 m from the marker plane,
with a target tolerance of 5 cm in position and 5° in heading.

The system is designed so that the estimator, planner and controller
operate on the estimated robot state rather than simulator ground truth.
Ground-truth values (`sim.truth_*`) are accessed only in the section
explicitly labelled `### EVALUATION ONLY ###` in `tb3/mission.py` and by
offline identification scripts.

### System pipeline
``` text
Wheel encoders ──> EKF prediction ───────────────┐
                                                 │
Camera ──> ArUco detection ──> PnP range/bearing ├──> EKF update
                                                 │       │
LiDAR ──> pillar detection and circle fitting ───┘       │
                                                         v
                                               Estimated pose and
                                                  covariance
                                                         │
                                                         v
                                             A* planner + path fillet
                                                         │
                                                         v
                                                  LQR tracker
                                                         │
                                                         v
                                               Linear/angular velocity
                                                         │
                                                         v
                                                   Robot wheels

LiDAR scan + known map + pose-uncertainty gate
             └──> unexpected-obstacle detection ──> path replanning
```

## 2. Installation and execution
The project uses MuJoCo, OpenCV-contrib and the Python dependencies
listed in `requirements.txt`.

### Setup
``` bash
pip install -r requirements.txt
export MUJOCO_GL=egl
```

### Run the project
Run the following commands from the repository root:

``` bash
# Part A: model, controllability and observability analysis
python -m tb3.model

# Part B: camera intrinsic calibration
python scripts/calibrate_camera.py

# Parts B and C: identify odometry, marker and LiDAR noise parameters
python scripts/identify_noise.py

# Run one complete navigation and docking trial
python -m tb3.mission 0

# Run Monte Carlo trials and ablation experiments
python scripts/run_experiments.py

# Generate result tables and figures
python scripts/make_figures.py

# Generate the demonstration video
python scripts/make_video.py

# Run the test suite
python -m pytest tests
```

The experiment script runs a set of Monte Carlo trials and ablation
studies. On the tested setup, it takes approximately 15 minutes using
two CPU cores. It writes trial data to `results/trials.jsonl`; the
figure-generation script produces the corresponding tables and figures.

## 3. Coordinate frames and sensor measurements
The implementation uses the following coordinate-frame conventions:

-   **World frame (`W`):** x points east, y points north and z points
    upwards.
-   **Robot base frame (`B`):** x points forwards, y points left, with
    the origin at the wheel axle.
-   **Camera optical frame (`Cv`):** x points right, y points down and z
    points forwards.
-   **Marker frame (`M`):** x points right, y points upwards and z
    points outwards from the marker face.

The camera is mounted at `t_B,Cv = (0.05, 0, 0.15) m` and faces along
the robot's positive x-axis. Its rotation relative to the base frame is:

``` text
R_B,Cv = [[ 0,  0,  1],
          [-1,  0,  0],
          [ 0, -1,  0]]
```

For an ArUco observation, image corners are converted into a
camera-frame pose using `solvePnP(IPPE_SQUARE)` and the calibrated
camera matrix. The resulting translation, `t_Cv,M`, is transformed into
the robot base frame:

``` text
p_B = R_B,Cv t_Cv,M + t_B,Cv
r   = |p_B,xy|
phi = atan2(p_B,y, p_B,x)
```

The EKF then compares the measured range and bearing with the expected
observation of the marker at its known world position. LiDAR pillar
detections are converted into the same range-and-bearing measurement
form. The LiDAR is mounted at `(-0.032, 0) m` relative to the robot
base, and this offset is included in the measurement conversion.

## 4. Part A --- Platform model, controllability and observability

### 4.1 Odometry-based state model
The estimator state is:

``` text
x = [x, y, theta]
```

Wheel-encoder increments are used as the input. The robot pose is
propagated using midpoint integration:

``` text
ds  = (ds_R + ds_L) / 2
dth = (ds_R - ds_L) / b

x+     = x + ds cos(theta + dth/2)
y+     = y + ds sin(theta + dth/2)
theta+ = theta + dth
```

This model assumes planar motion and rolling without slip. In practice,
some wheel slip is present, so its effect is represented in the
process-noise covariance, `Q`.

The effective wheel track was identified as **163.5 mm**, compared with
the nominal value of 160 mm. The identified wheel scale factor is
**1.0013**.

### 4.2 Measurement model
For a landmark at known world position `(mx, my)`, the predicted
range-and-bearing measurement is:

``` text
h(x) = [
    sqrt((mx - x)^2 + (my - y)^2),
    atan2(my - y, mx - x) - theta
]
```

### 4.3 Actuator and tracking models
The five-state actuated model, `[x, y, theta, v, w]`, includes
first-order linear and angular velocity actuator lags. It is implemented
in `tb3/model.py::continuous_5state`. The simulator's velocity actuators
exhibit an estimated lag of approximately 30 ms (`kv = 0.03` over
reflected inertia). The LQR tracking model does not explicitly include
this lag; this simplification is discussed in Section 7.

The LQR uses a tracking-error state defined in the robot frame:

``` text
e   = R(theta)^T (p_ref - p)
e_theta = theta_ref - theta
```

Linearising around reference velocities `(v_r, w_r)` gives:

``` text
A = [[ 0,   w_r,  0],
     [-w_r, 0,    v_r],
     [ 0,   0,    0]]

B = [[-1, 0],
     [ 0, 0],
     [ 0, -1]]
```

### 4.4 Controllability and observability
The following results are calculated by `python -m tb3.model` and are
also checked by the test suite.

  -----------------------------------------------------------------------
  Analysis                Result                  Interpretation
  ----------------------- ----------------------- -----------------------
  Controllability at      Rank 3; smallest        All tracking-error
  `v_r = 0.10 m/s`,       singular value 0.10     states are controllable
  `w_r = 0`                                       while moving. Lateral
                                                  correction is achieved
                                                  through heading
                                                  changes, and its
                                                  authority depends on
                                                  forward speed.

  Controllability at      Rank 2                  From rest, the
  `v_r = 0`, `w_r = 0`                            nonholonomic robot
                                                  cannot directly correct
                                                  lateral error.

  Observability using     Rank 0                  Absolute pose is
  odometry only                                   unobservable from
                                                  odometry alone, so
                                                  accumulated drift
                                                  cannot be corrected.

  One visible marker,     Rank 2; null direction  A single landmark does
  static or moving        `(0.35, -0.87, 0.35)`   not constrain every
                                                  pose direction;
                                                  uncertainty remains
                                                  along the unobservable
                                                  direction.

  Two visible markers     Rank 3                  The full robot pose is
                                                  observable.
  -----------------------------------------------------------------------

These results explain why the number of visible markers affects
localisation. When visual landmarks are unavailable, odometry error
accumulates and the pose covariance increases. With only one marker
visible, one direction remains unobservable, so the corresponding
uncertainty continues to grow.

## 5. Part B --- Sensing, calibration and noise characterisation

Sensor calibration and noise identification were carried out offline.
The resulting parameters are used by the estimator, with measurement
validity limits applied where the fitted models are reliable.

### 5.1 Camera calibration
Camera intrinsics were estimated from 60 rendered checkerboard images
captured at randomised poses using `cv2.calibrateCamera` with one
radial-distortion parameter.

-   Estimated radial coefficient: `k1 = -0.0017`.
-   Reprojection RMS error: **0.86 px**.
-   Held-out reprojection error: **0.38 px**.
-   Relative error against simulator intrinsics: **0.47% for `fx`** and
    **0.23% for `fy`**.
-   Principal-point difference from simulator values: **−3.2 px in x**
    and **+4.5 px in y**.

The principal point is comparatively weakly observable from the
calibration views. The resulting offset contributes to an estimated
bearing bias of approximately 5.7 mrad.

### 5.2 ArUco measurement bias and noise
Marker measurements were compared with simulator truth using 583 static
samples. The measured range scale bias was **1.53%**, and the bearing
bias was **−5.7 mrad**. Both biases are corrected before measurements
are supplied to the EKF.

Over the fitted operating range of 0.4--3.6 m, the marker noise model
is:

``` text
sigma_r   = 0.002 + 0.0041 r^2  m
sigma_phi = 3.2 mrad
```

The estimator applies minimum standard deviations of **6 mm** in range
and **5 mrad** in bearing to account for systematic error. These floors
were selected with reference to the NIS and NEES results discussed in
Section 6.

Marker detections beyond **3.6 m** are rejected. At greater distances,
the marker is approximately 16 pixels wide or smaller, and the PnP pose
becomes biased and non-Gaussian. This behaviour was identified during a
failed trial and is discussed in Section 9.

### 5.3 Wheel odometry
Odometry calibration identified a wheel scale factor of **1.0013** and
an effective wheel track of **163.5 mm**. The wheel-increment variance
is modelled as:

``` text
var(ds_i) = k |ds_i|
k = 9.7e-5 m
```

The parameter `k` was fitted using 14 random-drive experiments.

### 5.4 LiDAR landmark measurements
The simulated LiDAR provides 360 rays over a maximum range of 3.5 m,
with 1 cm measurement noise. Pillars are identified using pose-based
gating followed by a circle fit with the known pillar radius.

Across 1,005 samples, the fitted standard deviations were **3.7 mm in
range** and **3.9 mm in cross-range**. The measurement standard
deviation is inflated by a factor of two in the estimator, giving an 8
mm value to provide additional tolerance.

### 5.5 Frame consistency
Camera and LiDAR measurements are transformed into the robot base frame
before being represented as range and bearing. The camera and LiDAR
mounting offsets are included in these transformations. This allows
marker and pillar observations to be processed using a consistent
measurement representation.

## 6. Part C --- State estimation using an EKF
The EKF combines wheel-encoder odometry with exteroceptive measurements
from the camera and LiDAR.

### 6.1 Prediction and update
-   **Prediction:** performed at 20 Hz using wheel increments.
-   **Camera updates:** performed at 5 Hz for each detected marker.
-   **LiDAR updates:** performed at 5 Hz for each visible pillar.
-   **Update order:** measurements are processed sequentially.
-   **Numerical update:** Joseph-form covariance update.
-   **Outlier rejection:** NIS gating using the 99% chi-squared
    threshold for two measurement dimensions, `chi2(2) = 9.21`.

The initial pose estimate is generated by adding Gaussian error to the
true initial pose, with standard deviations of **0.1 m in x**, **0.1 m
in y** and **0.1 rad in heading**. The initial covariance, `P0`, is set
consistently with these uncertainties.

### 6.2 Process and measurement noise selection
The process-noise covariance is calculated from the fitted
wheel-odometry noise parameter:

``` text
Q = G diag(k|ds_R|, k|ds_L|) G^T
    + diag(1e-7, 1e-7, 1e-6)
```

The odometry parameter `k` and the marker and pillar measurement
standard deviations are obtained from the offline identification script,
`scripts/identify_noise.py`. Minimum noise values are then applied to
account for residual systematic errors. The final noise settings were
assessed using NEES and NIS consistency statistics.

### 6.3 Estimator consistency
On nominal successful runs, the average normalised estimation error
squared (ANEES) is approximately **4--6**, compared with an ideal value
of 3 for this three-state estimator. This indicates that the filter is
mildly overconfident.

The mean NIS is approximately **0.5--1 for marker measurements** and
**0.4 for pillar measurements**, compared with an ideal value of 2.
These values indicate that the observed measurement residuals are
smaller than expected under the assumed measurement-noise model.

The two observations are not necessarily contradictory. Time-correlated
errors, including residual calibration bias, can be under-represented by
NIS while still contributing to the state-estimation error measured by
NEES. The consistency gap was reduced through noise floors but was not
fully resolved. This remains a limitation of the estimator.

## 7. Part D --- Path planning and feedback control

### 7.1 Path planning
The planner uses A\* on a **5 cm grid**. Obstacles are inflated by
**0.205 m**, comprising the robot footprint radius of 0.105 m and an
additional 0.10 m clearance. The resulting path is shortened where
possible and converted into a line-and-arc fillet path.

The fillet path uses a nominal corner radius of **0.6 m**, reduced where
necessary to maintain at least **0.135 m clearance**. The final approach
to the charging station is constrained to an exactly straight **1.2 m
segment**, providing a stable alignment for docking.

The speed profile applies the following limits:

-   Maximum planned speed: **0.12 m/s**.
-   Maximum lateral acceleration: **0.08 m/s²**.
-   Maximum along-track acceleration: **0.15 m/s²**.
-   Docking approach speed: **0.03 m/s**.

### 7.2 LQR trajectory tracking
An LQR controller tracks the planned path using the EKF pose estimate.
The controller is recalculated at each step for the current reference
velocities `(v_r, w_r)`, using a discrete Riccati solution and an Euler
discretisation step of **0.05 s**.

The selected weighting matrices are:

``` text
Q = diag(30, 60, 8)
R = diag(2, 0.5)
```

The lateral tracking error is weighted twice as strongly as the
along-track error. The heading weight was selected to give a comparable
penalty to a 4 cm lateral error for a heading error of 0.1 rad. The
control law is:

``` text
u = u_r - K e
```

The applied command limits are **0.15 m/s** for linear velocity and
**±1.5 rad/s** for angular velocity. The controller uses the EKF pose
estimate only. To prevent the reference trajectory from advancing too
far ahead of the robot, its reference clock pauses if the robot falls
more than **0.2 m** behind.

A separate heading-alignment phase turns the robot in place before
trajectory tracking begins. This is necessary because the LQR model is
linearised around the reference trajectory and is not intended to handle
large initial heading errors.

### 7.3 LQR tuning trade-off
The position weighting was varied by factors of 0.1, 1 and 10. The
resulting control effort was **1.67, 2.08 and 3.46**, respectively,
while path RMSE was **3.8, 3.5 and 4.4 cm**.

Increasing the position weight therefore increased control effort
without improving tracking accuracy. In these trials, performance
appears to be limited more by actuator saturation and odometry slip than
by insufficient controller gain.

### 7.4 Model limitation
The actuator model indicates a velocity lag of approximately 30 ms, but
this lag is not included in the LQR tracking model. This is a known
simplification and a potential area for further controller improvement.

## 8. Part E --- Experimental evaluation and robustness

### 8.1 Evaluation method
The evaluation comprises **60 trials** recorded in
`results/trials.jsonl`, with summary results in
`results/summary_table.md`. Initial robot poses and initial estimate
errors are randomised between trials.

A trial is counted as successful when the robot reaches the docking
condition with its **true simulated pose** within **5 cm position
error** and **5° heading error**, without collision and within **170
seconds**. Ground truth is used to assess performance only; it is not
supplied to the estimator, planner or controller.

### 8.2 Quantitative results

  ---------------------------------------------------------------------------------------------------------
  Experiment      Trials    Success       Dock   Position   ANEES    Path   Control   Replans Failures
                   (`n`)              position  RMSE (cm)            RMSE    effort           
                                        error:                       (cm)                     
                                      median /                                                
                                      p90 (cm)                                                
  ------------- -------- ---------- ---------- ---------- ------- ------- --------- --------- -------------
  Nominal             10   **7/10** 1.6 / 56.0        7.3     6.2     6.6      2.29         0 2 dock
                                                                                              misses, 1
                                                                                              timeout

  Odometry only        4        0/4 35.6 / 289        9.1     0.8     7.7      1.77         0 2 collisions,
                                                                                              2 dock misses

  Camera only          4        3/4  2.5 / 272        9.0     7.4     7.4      3.21         1 1 collision

  LiDAR pillars        4        3/4   1.4 / 38        3.7     4.8     3.4      2.08         0 1 dock miss
  only                                                                                        

  Measurement          4        4/4  1.1 / 1.3        3.9    19.2     3.5      2.08         0 ---
  `R × 0.1`                                                                                   

  Measurement          4        4/4  2.1 / 2.1        3.7     1.5     3.5      2.08         0 ---
  `R × 10`                                                                                    

  Process noise        4        4/4  0.8 / 1.0        3.8     2.0     3.5      2.06         0 ---
  `Q × 10`                                                                                    

  LQR position         4        4/4  1.5 / 1.5        3.9     4.1     3.8      1.67         0 ---
  weight                                                                                      
  `Q × 0.1`                                                                                   

  LQR position         4        4/4  1.5 / 1.8        4.1    10.2     4.4      3.46         0 ---
  weight                                                                                      
  `Q × 10`                                                                                    

  No marker            6        6/6  1.4 / 1.6        3.1     6.1     3.5      1.85         0 ---
  range limit                                                                                 

  Disturbed            8    **6/8**  1.4 / 199        4.8     7.1     4.5      3.75         3 1 dock miss,
                                                                                              1 collision

  Disturbed,           4        4/4  1.2 / 1.4        3.8    14.7     3.4      3.00         3 ---
  gating                                                                                      
  disabled                                                                                    
  ---------------------------------------------------------------------------------------------------------

The disturbed condition includes an unmapped box, external pushes,
friction reduced to 0.7, a 12-second marker blackout, 3% measurement
outliers and twice the nominal pixel noise.

### 8.3 Interpreting the results
The groups generally contain only four trials, and their random
seeds overlap. In particular, seeds 0--3 are mostly the easier cases,
while all three nominal failures occur on seeds 4, 6 and 7.
Consequently, the small ablation groups do not support strong
conclusions about the effect of every individual setting. For example,
the results do not establish that removing the marker range limit
improves performance or that disabling gating is beneficial. The larger
and more defensible observations are:

-   The odometry-only configuration achieved 0 successes in 4 trials. This
    supports the need for exteroceptive measurements to correct
    accumulated odometry drift.
-   **Measurement-noise settings affect estimator consistency and
    responsiveness.** With `R × 0.1`, ANEES increased to 19.2,
    indicating substantial overconfidence. With `R × 10`, ANEES
    decreased to 1.5, indicating a more conservative estimate. The
    median docking errors were 1.1 cm and 2.1 cm, respectively. The more
    conservative filter therefore finished with approximately 1 cm
    greater median docking error in these trials.
-   **Increasing LQR position weights increased effort without improving
    tracking.** At `Q × 10`, control effort rose to 3.46 compared with
    1.67 at `Q × 0.1`, while path RMSE increased from 3.8 cm to 4.4 cm.
-   **The pipeline retained useful performance under disturbances.** Six
    of the eight disturbed trials succeeded. The two failures were
    associated with estimator divergence rather than an identified
    failure of the trajectory-tracking controller.

These findings should be interpreted in the context of the trial counts
and seed selection. The results provide quantitative evidence of the
pipeline's behaviour, but the smaller ablation groups limit the strength
of comparisons between individual parameter settings.

## 9. Development issues, failure analysis and limitations
Several issues were identified during development and addressed through
changes to the path representation, sensor validity checks and
obstacle-detection logic. Some estimator failures remain unresolved due to 
time consteitions/ other subject responsibilities. 

### 9.1 Excessive curvature near the docking approach --- addressed
An earlier smoothing-spline path, constrained to finish with a straight
docking segment, produced peak curvature of approximately **17 rad/m**.
The controller reached its angular-velocity limit of **1.5 rad/s**,
causing wheel skid and an odometry heading error of approximately
0.3--0.4 rad. The robot consequently missed the docking heading by
7--9°.

The spline was replaced with a line-and-arc fillet path with bounded
curvature of no more than **1.67 rad/m**, followed by an exactly
straight final approach. After this change, the docking yaw error
decreased to **0.0--0.4°**.

### 9.2 Biased long-range marker poses --- addressed
At ranges of approximately 5--7 m, ArUco markers are around 16 pixels
wide. PnP estimates in this range were biased and non-Gaussian,
producing yaw-estimate shifts of approximately 0.2--0.3 rad.

The estimator now rejects marker detections beyond **3.6 m**,
restricting updates to the range over which the measurement-noise model
was fitted.

### 9.3 False unexpected-obstacle detections --- addressed
When the estimated pose error reached 0.1--0.2 m, mapped walls could
appear displaced by approximately 0.3 m when the LiDAR scan was
interpreted using the estimated pose. This caused false obstacle
detections and repeated replanning.

The obstacle-detection tolerance was made dependent on pose uncertainty,
and the check was restricted to the near field (less than 2 m). These
changes were intended to reduce false detections caused by localisation
uncertainty.

### 9.4 Estimator divergence on selected trials --- unresolved
The main remaining limitation is estimator divergence in nominal seeds
4, 6 and 7, and disturbed seeds 4 and 7. In these trials, the estimate
error reaches approximately **0.5--2.3 m**, meaning that the estimated
docking pose can differ substantially from the true simulated pose.

In seed 4, the heading error grows at approximately **0.02 rad/s**
during the first 18 seconds, when no marker is within the 3.6 m
measurement limit. The planner also replans repeatedly during this
period. Separate odometry tests from the same initial pose produced
heading errors within 0--9% of the true value, so the cause has not been
established.

The current suspected causes, in order of concern, are:
1.  Wheel slip during repeated stop-and-go alignment and replanning at
    low speed.
2.  Initial pose error (including a heading standard deviation of 0.1
    rad) remaining uncorrected during the period without visible
    markers.
3.  Unexpected-obstacle replanning still being triggered incorrectly in
    these cases.

These explanations have not been verified. Further investigation should
log the difference between estimated and true pose at each mission phase
and repeat the affected seeds with replanning disabled. This would help
separate estimator drift from errors associated with the replanning
behaviour.

### 9.5 Residual estimator overconfidence
Nominal ANEES values of approximately 5--7 exceed the ideal value of 3,
while the `R × 0.1` experiment reaches 19.2. The applied noise floors
reduce the effect but do not fully resolve the inconsistency. Further
work should investigate time-correlated measurement errors and refine
the process- and measurement-noise models.

## 10. Hardware implementation
A physical hardware implementation was **not attempted**. All results
presented in this README are from simulation.

## 11. Independent contribution beyond the weekly tutorials

In addition to adapting the core modelling, estimation and control
concepts introduced in the weekly tutorials, the project includes the
following extensions:
-   A LiDAR-based pillar landmark estimator, evaluated against
    camera-only and odometry-only configurations.
-   Pose-uncertainty-gated detection of unmapped obstacles, with
    automatic path replanning.
-   Offline identification of sensor noise and measurement biases.
-   Parameter sweeps across EKF `Q` and `R` settings and LQR weighting
    matrices, evaluated using tracking error, control effort and
    NEES/NIS statistics.
-   Quantitative failure analysis, including investigation of long-range
    marker bias, path curvature and estimator divergence.


## 12. Individual contribution statements
-   Team member 1 --- Matt Ayala-Schnell 
    Contributed to code production and video creation

-   Team member 2 --- Massimo Zanni 
    Contributed to project management, ensuring figures are correct within code creation and background research

Each member should be prepared to explain their own contribution and how
it connects to the complete sensing, estimation, planning and control
pipeline during the individual viva voce.

## 13. Generative AI declaration
Claude (by Antropic) was utilised in aiding the creation of the final demo video.

## 14. Acknowledgements and external resources
The kinematic model, EKF structure and LQR tracking approach were
developed with reference to the weekly 41014 tutorial material. The
implementation uses MuJoCo, OpenCV ArUco and SciPy.

The TurtleBot3 Burger dimensions were referenced from the ROBOTIS
datasheet. The robot model used in this project was hand-built and does
not use the ROBOTIS mesh.


