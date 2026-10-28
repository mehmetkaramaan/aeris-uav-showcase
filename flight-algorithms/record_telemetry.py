import argparse, csv, json, math, time
from pathlib import Path
from pymavlink import mavutil
p = argparse.ArgumentParser()
p.add_argument('--seconds', type=float, default=45)
a = p.parse_args()
m = mavutil.mavlink_connection('udpin:127.0.0.1:14540')
if m.wait_heartbeat(timeout=10) is None:
    raise SystemExit('No local PX4 heartbeat; start Baslat.command first.')
rows = []
state = {}
start = time.monotonic()
while time.monotonic() - start < a.seconds:
    msg = m.recv_match(blocking=True, timeout=0.5)
    if msg is None:
        continue
    typ = msg.get_type()
    if typ == 'ATTITUDE':
        state.update(roll_deg=math.degrees(msg.roll), pitch_deg=math.degrees(msg.pitch), yaw_deg=math.degrees(msg.yaw))
    elif typ == 'EXTENDED_SYS_STATE':
        state['landed_state'] = msg.landed_state
    elif typ == 'HEARTBEAT':
        state['armed'] = int(bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
    elif typ == 'LOCAL_POSITION_NED':
        state.update(north_m=msg.x, east_m=msg.y, up_m=-msg.z, vn_ms=msg.vx, ve_ms=msg.vy, vu_ms=-msg.vz)
    elif typ == 'GLOBAL_POSITION_INT':
        state.update(relative_altitude_m=msg.relative_alt / 1000)
        rows.append(dict(t_s=time.monotonic() - start, **state))
out = Path(__file__).parent / 'validation'
out.mkdir(exist_ok=True)
fields = ['t_s', 'relative_altitude_m', 'north_m', 'east_m', 'up_m', 'vn_ms', 've_ms', 'vu_ms', 'roll_deg', 'pitch_deg', 'yaw_deg', 'landed_state', 'armed']
with (out / 'flight_telemetry.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
summary = {'source': 'PX4 SITL / Gazebo Harmonic, local MAVLink UDP 14540', 'duration_s': a.seconds, 'samples': len(rows), 'max_relative_altitude_m': max((r['relative_altitude_m'] for r in rows)), 'first': rows[0], 'last': rows[-1]}
(out / 'flight_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
