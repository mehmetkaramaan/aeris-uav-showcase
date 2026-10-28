from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyulog import ULog
out = Path(__file__).parent / 'validation'
log = ULog(str(out / 'gazebo-flight.ulg'))
pos = log.get_dataset('vehicle_local_position').data
land = log.get_dataset('vehicle_land_detected').data
status = log.get_dataset('vehicle_status').data
start = min(pos['timestamp'][0], land['timestamp'][0]) / 1000000.0
pt = pos['timestamp'] / 1000000.0 - start
lt = land['timestamp'] / 1000000.0 - start
st = status['timestamp'] / 1000000.0 - start
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(pt, -pos['z'], color='#167c9c', label='PX4 local altitude estimate')
axes[0].axhline(2.5, color='#b46518', ls='--', label='2.5 m takeoff target')
axes[0].set_ylabel('Up position (m)')
axes[0].legend(loc='upper left')
axes[1].plot(pt, pos['x'], label='North')
axes[1].plot(pt, pos['y'], label='East')
axes[1].set_ylabel('Horizontal position (m)')
axes[1].legend(loc='upper left')
axes[2].step(lt, land['landed'], where='post', label='Landed (1) / airborne (0)')
axes[2].step(st, (status['arming_state'] == 2).astype(int), where='post', label='Arm (1) / disarm (0)')
axes[2].set_ylabel('State')
axes[2].set_xlabel('Time from ULog start (s)')
axes[2].legend(loc='upper left')
axes[2].set_yticks([0, 1])
for ax in axes:
    ax.grid(alpha=0.22)
fig.suptitle('Gazebo + PX4 SITL: takeoff, position hold and landing\nRecorded simulation ULog', fontsize=14)
fig.savefig(out / 'flight-validation.png', dpi=160)
summary = {'source': 'Complete PX4 ULog', 'log': 'gazebo-flight.ulg', 'max_local_up_m': float(max(-pos['z'])), 'final_landed': bool(land['landed'][-1]), 'final_armed': bool(status['arming_state'][-1] == 2), 'airborne_samples': int(sum(land['landed'] == 0)), 'duration_s': float(pt[-1] - pt[0])}
(out / 'complete_flight_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
