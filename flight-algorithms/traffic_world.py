import os, sys, time, json, math, random
from pathlib import Path
if os.environ.get('AERIS_GZ_PYTHONPATH'):
    sys.path.append(os.environ['AERIS_GZ_PYTHONPATH'])
from gz.transport13 import Node
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from planning import astar, blocked
node = Node()
config = {'enabled': True, 'seed': 42, 'count': 2, 'speed': 0.5, 'pattern': 'random', 'center_e': 20.0, 'center_n': 20.0}
actors = []
own = None
clock = 0.0
generation = 0
pub = node.advertise('/aeris/traffic/state', StringMsg)

def setup():
    global actors, generation, epoch, minimum
    rng = random.Random(int(config['seed']))
    actors = []
    generation += 1
    epoch = clock
    minimum = None
    for i in range(5):
        x = 8 + i * 6
        y = rng.uniform(-3, 3)
        z = 7 if i != 2 else 8.5
        actors.append({'name': f'traffic_{i}', 'east': x, 'north': y, 'z': z, 'route': [], 'speed': config['speed'] * rng.uniform(0.7, 1.2), 'rng': rng, 'yaw': 0.0})

def control(msg):
    try:
        d = json.loads(msg.data)
        config.update({k: d[k] for k in config if k in d})
        config['count'] = max(1, min(5, int(config['count'])))
        config['speed'] = max(0.3, min(1.5, float(config['speed'])))
        setup()
    except Exception as e:
        print(e, flush=True)

def pose_cb(msg):
    global own
    for p in msg.pose:
        if p.name == 'aeris_x500_0':
            own = (p.position.x, p.position.y, p.position.z)

def clock_cb(msg):
    global clock
    clock = msg.sim.sec + msg.sim.nsec / 1000000000.0
node.subscribe(StringMsg, '/aeris/traffic/control', control)
node.subscribe(Pose_V, '/world/aeris/dynamic_pose/info', pose_cb)
node.subscribe(Clock, '/world/aeris/clock', clock_cb)
setup()
previous = 0.0
while True:
    dt = min(0.25, max(0, clock - previous))
    previous = clock
    poses = Pose_V()
    visible = []
    for i, a in enumerate(actors):
        enabled = config['enabled'] and i < config['count']
        x, y, z = (a['east'], a['north'], a['z'])
        if enabled:
            if config['pattern'] == 'crossing':
                phase = (clock - epoch) * (config['speed'] / 12) + i * 0.33
                if i == 0:
                    x = 26 - 18 * (phase % 2 / 2)
                    y = 0.0
                    z = 4
                    a['yaw'] = math.pi
                elif i == 1:
                    x = 15
                    y = 5 * math.sin(phase * math.pi)
                    z = 4
                else:
                    x = 33 + 7 * math.cos(phase)
                    y = 19 + 4 * math.sin(phase)
                    z = 5.5
            elif config['pattern'] == 'fire':
                phase = (clock - epoch) * config['speed'] / 7 + i * 2.1
                cx = float(config['center_e'])
                cy = float(config['center_n'])
                if i == 0:
                    x = cx + 7 * math.sin(phase)
                    y = cy + 3 * math.sin(2 * phase)
                    z = 7.0
                else:
                    x = cx + 6 * math.cos(phase)
                    y = cy + 5 * math.sin(phase)
                    z = 7.0 if i != 2 else 8.5
                a['yaw'] = phase + math.pi / 2
            else:
                if not a['route']:
                    for _ in range(12):
                        target = (a['rng'].uniform(6, 44), a['rng'].uniform(-18, 22))
                        if not blocked(*target):
                            try:
                                a['route'] = astar((x, y), target)
                                break
                            except ValueError:
                                pass
                if a['route']:
                    tx, ty = a['route'][0]
                    dx, dy = (tx - x, ty - y)
                    d = math.hypot(dx, dy)
                    step = a['speed'] * dt
                    if d < max(0.25, step):
                        a['route'].pop(0)
                    else:
                        x += dx / d * step
                        y += dy / d * step
                        a['yaw'] = math.atan2(dy, dx)
            a.update(east=x, north=y, z=z)
            visible.append({'id': i, 'name': a['name'], 'east': x, 'north': y, 'z': z, 'kind': 'Sabit kanat' if i == 2 else 'Kargo quadcopter'})
        p = poses.pose.add()
        p.name = a['name']
        p.position.x = x
        p.position.y = y
        p.position.z = z if enabled else -20
        p.orientation.w = math.cos(a['yaw'] / 2)
        p.orientation.z = math.sin(a['yaw'] / 2)
    if clock:
        try:
            node.request('/world/aeris/set_pose_vector', poses, Pose_V, Boolean, 150)
        except Exception:
            pass
    separation = min((math.dist(own, (a['east'], a['north'], a['z'])) for a in visible), default=None) if own else None
    if separation is not None and own[2] > 1:
        minimum = min(minimum or separation, separation)
    state = {'config': config, 'actors': visible, 'separation_truth': separation, 'minimum_separation_truth': minimum, 'generation': generation}
    msg = StringMsg()
    msg.data = json.dumps(state)
    pub.publish(msg)
    time.sleep(0.1)
