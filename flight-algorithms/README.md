# AERIS flight algorithms

This folder publishes the current simulation algorithm source, including the Kalman tracker, temporal detection window, planning, mission logic, LiDAR avoidance, bounded trajectory commands and the PX4/Gazebo bridge. Code comments and docstrings have been removed; explanations are provided here in English.

## Algorithm map

| Module | Role |
| --- | --- |
| [tracking.py](tracking.py) | Constant-velocity Kalman state estimator and sliding detection window |
| [avoidance.py](avoidance.py) | LiDAR clustering/association, motion prediction and local velocity selection |
| [trajectory.py](trajectory.py) | Acceleration/jerk limits and attitude-aware command stabilization |
| [planning.py](planning.py) | Static-grid A*, visibility-graph routing and obstacle-clearance geometry |
| [coverage.py](coverage.py) | Camera footprint projection, persistent observed-cell memory and viewpoint skipping |
| [coverage_route.py](coverage_route.py) | Greedy footprint set cover, route-distance caching and 2-opt refinement |
| [survey_mission.py](survey_mission.py) | Search, station acquisition, refill, suppression verification and return decisions |
| [perception.py](perception.py) | Controlled color/shape landmarks and image-to-ground projection |
| [controller.py](controller.py) | Local SITL MAVLink telemetry, commands, watchdogs and mission orchestration |
| [vision.py](vision.py) | Gazebo camera/LiDAR subscriptions and perception pipeline |
| [world_effects.py](world_effects.py) | Separate Gazebo-only scenario placement and abstract water effects |
| [traffic_world.py](traffic_world.py) | Kinematic simulation traffic actors |

The source is a simulation snapshot. Runtime installers, local environment configuration, dashboard application assets, raw flight/event logs, scenario-audit recordings, binaries and personal paths are excluded. There is no credential-bearing configuration in this folder. Omitted files remain local; this publication does not create a private GitHub repository.

## Kalman filtering

The tracked state is `[east, north, east_velocity, north_velocity]`. A constant-velocity prediction is corrected by the current LiDAR-cluster center. The covariance uses a Joseph-form update. Initial position standard deviation is 0.3 m, velocity standard deviation 1 m/s; measurement standard deviation is 0.3 m and assumed acceleration noise is 0.8 m/s². These are initial model parameters, not measured calibration values.

Tracks are associated using predicted centers with a 1.8 m gate and one association per track per scan. A gap of 0.8 s or an inconsistent cluster starts a new track. Velocity enters avoidance after three observations. Raw LiDAR surface points remain in the clearance check, so smoothing does not replace close measured returns with smoothed centers. This is a linear Kalman filter, not an EKF or a complete multi-object association solution.

## Temporal windowing

A station or fire is confirmed by at least three positive observations in the latest five distinct camera frames, all within 1.2 s. Multiple blobs matching one fire in one frame count as one vote. Old evidence expires instead of accumulating indefinitely. Confirmed-target memory and separate stable-camera extinguishment verification remain distinct from this detection window.

This is a sliding **time window**. It is not a Hann/Hamming spectral window. Gaussian, median and sharpening options exist in the vision pipeline, but the default image preprocessing mode is `none`; the temporal confirmation remains active.

## Communication and control

The bridge uses **pymavlink / MAVLink directly** with a loopback SITL endpoint. **MAVROS is not used.** PX4 supplies the inner flight/attitude control; this code sends local position/velocity/yaw setpoints and does not implement a replacement motor controller. Some legacy task branches remain in the controller; their presence does not imply that every branch is exposed in the current dashboard.

`world_effects.py` owns simulation target placement and effect acknowledgements. Target positions/counts are not supplied to the autonomous search logic. World-pose information used to apply abstract water/contact effects is not target-discovery evidence. No real pump, thermal or fluid model is included.

## Run the tests

With Python, NumPy and OpenCV installed, these tests require no running simulator:

```bash
cd flight-algorithms
python -m unittest -v test_tracking test_avoidance test_survey_mission test_trajectory
```

The additional `test_mission_logic` suite imports the PX4/Gazebo bridge. It requires pymavlink plus matching Gazebo Transport 13 / Messages 10 Python bindings, but does not start a flight:

```bash
python -m unittest discover -s . -p 'test_*.py'
```

If the Gazebo bindings are installed outside the active interpreter's import path, point `AERIS_GZ_PYTHONPATH` at that installation's compatible site-packages directory. The Python ABI and native Gazebo libraries must match; pip requirements alone do not install Gazebo or PX4.

**Verification of this published copy: 47 tests passed in the configured project environment.** These cover trajectory bounds, obstacle routing, stale-data behavior, coverage retention, camera-based mission transitions, Kalman noise response and temporal-window edge cases. They are not physical-flight certification. The earlier v0.7 mission benchmark in the main README predates the Kalman/window update; its performance improvements are not attributed to these filters.

## Synthetic filtering experiment

```bash
python validate_tracking.py
```

This writes `validation/tracking-filter-comparison.json`. The bundled [result](tracking-filter-comparison.json) uses seed 42, an 8 Hz constant-velocity target and independent Gaussian position noise of 0.3 m. Position RMSE decreased from approximately 0.404 m for raw measurements to 0.162 m for the Kalman estimate. It is a synthetic estimator comparison, not a full Gazebo mission or a guarantee for turning targets and changing clusters.

## Integration contract

Running the mission bridge additionally requires a matching Gazebo world and vehicle model with `/aeris/camera`, `/aeris/front`, `/aeris/lidar`, `/world/aeris/clock` and pose/effect services, plus a running PX4 SITL endpoint. The complete platform setup and dashboard assets are not included in this folder. The source supports algorithm inspection and integration into a compatible simulation; it is not presented as a one-command reproduction of the complete local application.

Copyright and publication scope are described in [the repository notice](../NOTICE.md).
