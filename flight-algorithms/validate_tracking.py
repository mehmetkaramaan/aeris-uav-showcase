import json
from pathlib import Path
import numpy as np
from tracking import KalmanTrack

def run():
    rng = np.random.default_rng(42)
    k = KalmanTrack([0, 0], 0)
    last = np.zeros(2)
    ema = np.zeros(2)
    rows = []
    for i in range(1, 161):
        t = i * 0.125
        truth = np.array([0.7 * t, -0.3 * t])
        z = truth + rng.normal(0, 0.3, 2)
        delta = (z - last) / 0.125
        if np.linalg.norm(delta) < 3.5:
            ema = 0.8 * ema + 0.2 * delta
        last = z
        k.update(z, t)
        if i > 20:
            rows.append([np.linalg.norm(z - truth) ** 2, np.linalg.norm(k.x[:2] - truth) ** 2, np.linalg.norm(ema - [0.7, -0.3]) ** 2, np.linalg.norm(k.x[2:] - [0.7, -0.3]) ** 2])
    rmse = np.sqrt(np.mean(rows, axis=0))
    return dict(experiment='Synthetic constant-velocity target; seed 42; 8 Hz; 0.3 m independent Gaussian position noise; first 20 samples excluded', raw_position_rmse_m=float(rmse[0]), kalman_position_rmse_m=float(rmse[1]), previous_ema_velocity_rmse_m_s=float(rmse[2]), kalman_velocity_rmse_m_s=float(rmse[3]), limitation='Not a Gazebo mission or a guarantee for turning targets, changing LiDAR clusters or association errors.')
if __name__ == '__main__':
    result = run()
    path = Path(__file__).parent / 'validation/tracking-filter-comparison.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
