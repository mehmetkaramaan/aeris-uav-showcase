import sys, os, time, json, random, math, secrets
from pathlib import Path
if os.environ.get('AERIS_GZ_PYTHONPATH'):
    sys.path.append(os.environ['AERIS_GZ_PYTHONPATH'])
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from coverage import Coverage
runtime = Path(os.environ.get('AERIS_RUNTIME', '.aeris-runtime'))
runtime.mkdir(parents=True, exist_ok=True)
node = Node()
fires = []
station = None
cells = Coverage().cells.tolist()
audit = runtime / 'scenario-audit.json'

def persist():
    audit.write_text(json.dumps({'fires': fires, 'station': station}, indent=2))

def move(name, x, y, z):
    p = Pose()
    p.name = name
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.w = 1
    ok, result = node.request('/world/aeris/set_pose', p, Pose, Boolean, 500)
    return bool(ok and result.data)

def randomize():
    global fires, station
    rng = random.Random(secrets.randbits(32))
    available = [p for p in cells if math.hypot(*p) > 7]
    rng.shuffle(available)
    station = available.pop()
    fires = []
    for p in available:
        if math.dist(p, station) > 7 and all((math.dist(p, f['position']) > 8 for f in fires)):
            fires.append({'name': f'fire_visual_marker_{len(fires)}', 'position': p, 'strength': 1.0})
        if len(fires) >= 3:
            break
    fires = fires[:max(2, min(3, int(os.environ.get('AERIS_VALIDATION_COUNT', rng.randint(2, 3)))))]
    if os.environ.get('AERIS_VALIDATION_LAYOUT'):
        fixture = json.loads(Path(os.environ['AERIS_VALIDATION_LAYOUT']).read_text())
        station = fixture['station']
        fires = [{'name': f'fire_visual_marker_{i}', 'position': f['position'], 'strength': 1.0} for i, f in enumerate(fixture['fires'])]
    ready = False
    for attempt in range(15):
        results = [move('water_station', *station, 0)]
        for i in range(4):
            f = next((f for f in fires if f['name'] == f'fire_visual_marker_{i}'), None)
            results.append(move(f'fire_visual_marker_{i}', *(f['position'] if f else (0, 0)), 0.65 if f else -20))
        if all(results):
            ready = True
            break
        time.sleep(0.3)
    persist()
    print(json.dumps({'scene_ready': ready}), flush=True)
for line in sys.stdin:
    try:
        d = json.loads(line)
        if d['action'] == 'randomize':
            randomize()
        elif d['action'] == 'move':
            move(d['name'], d['x'], d['y'], d['z'])
        elif d['action'] == 'drop':
            x, y, z = d['position']
            for k in range(7):
                move('water_payload', x, y, max(0.08, z - 0.5 * 9.81 * (k * 0.14) ** 2))
                time.sleep(0.07)
            move('water_payload', x, y, -4)
            for f in fires:
                r = math.dist((x, y), f['position'])
                if r < 1.7 and 5.5 < z < 8.8:
                    f['strength'] = max(0, f['strength'] - 0.4 * math.exp(-r * r / 5))
                    move(f['name'], *f['position'], -20 if f['strength'] < 0.15 else 1.3 * f['strength'] - 0.65)
            persist()
            print(json.dumps({'drop_applied': True}), flush=True)
        elif d['action'] == 'refill':
            p = d.get('position', [])
            ok = station is not None and len(p) == 3 and (math.dist(p[:2], station) < 1.4) and (2 < p[2] < 4.2)
            print(json.dumps({'refill_result': ok}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
