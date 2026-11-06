| group vs nominal (same seeds) | seeds | success: group / nominal | median est-RMSE difference (cm) | Wilcoxon p |
|---|---|---|---|---|
| odom_only | 10 | 0/10 vs 10/10 | +11.4 | 0.002 |
| cam_only | 10 | 10/10 vs 10/10 | +0.3 | 0.002 |
| lidar_only | 10 | 8/10 vs 10/10 | +5.7 | 0.002 |
| no_localise_sweep | 10 | 10/10 vs 10/10 | -0.1 | 0.695 |
| range_limit_3.6m | 10 | 9/10 vs 10/10 | +0.9 | 0.131 |
| no_accel_limit | 10 | 10/10 vs 10/10 | -0.4 | 0.275 |
| R_x0.1 | 6 | 6/6 vs 6/6 | +2.4 | 0.031 |
| R_x0.3 | 6 | 6/6 vs 6/6 | +2.2 | 0.062 |
| R_x3 | 6 | 6/6 vs 6/6 | -0.2 | 0.562 |
| R_x10 | 6 | 6/6 vs 6/6 | -0.2 | 0.688 |
| Q_x0.1 | 6 | 6/6 vs 6/6 | +0.5 | 0.312 |
| Q_x10 | 6 | 6/6 vs 6/6 | +0.0 | 0.844 |
| LQR_Q_x0.1 | 6 | 6/6 vs 6/6 | -0.0 | 1.000 |
| LQR_Q_x0.3 | 6 | 6/6 vs 6/6 | +0.0 | 0.562 |
| LQR_Q_x3 | 6 | 6/6 vs 6/6 | -0.0 | 0.312 |
| LQR_Q_x10 | 6 | 6/6 vs 6/6 | -0.0 | 0.688 |
| disturbed | 10 | 10/10 vs 10/10 | -0.4 | 0.049 |
| disturbed_no_gate | 10 | 10/10 vs 10/10 | +0.5 | 0.492 |
