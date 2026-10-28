import json, urllib.request, sys, signal, time
from pathlib import Path
import psutil
root = Path(__file__).resolve().parent
runtime = (root / '../../work/aeris-gazebo').resolve()
px4 = (runtime / 'PX4-Autopilot/build/px4_sitl_default/bin/px4').resolve()
try:
    state = json.load(urllib.request.urlopen('http://127.0.0.1:8772/api/state', timeout=3))
except Exception:
    raise SystemExit('Cannot read panel state. Shutdown refused without confirming flight state.')
if state['telemetry'].get('armed'):
    raise SystemExit('Vehicle is armed. Land through the panel and wait for DISARM first.')
for p in psutil.process_iter(['exe']):
    try:
        if p.info['exe'] and Path(p.info['exe']).resolve() == px4:
            p.send_signal(signal.SIGINT)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
print('Stopping the AERIS session. Use the local session launcher to restart.')
