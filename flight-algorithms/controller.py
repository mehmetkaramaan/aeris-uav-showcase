import math, time, queue, threading, json, subprocess, sys, atexit
from collections import deque
from pathlib import Path
from pymavlink import mavutil
from planning import astar, shortest_path, visible, blocked, OBSTACLES
from vision import Vision
from trajectory import Trajectory
from survey_mission import SurveyMission, TASKS
CLAMP = lambda v, a, b: max(a, min(b, v))

class Controller:

    def __init__(self):
        self.mission = SurveyMission()
        self.scene_ready = False
        self.pending_mission = None
        self.data_hold = False
        self.data_stale_since = 0
        self.sim_paused_at = 0
        self.resume_until = 0
        self.event_id = 0
        self.log_queue = queue.SimpleQueue()
        threading.Thread(target=self.write_events, daemon=True).start()
        self.trajectory = Trajectory()
        self.home = {'east': 0.0, 'north': 0.0}
        self.flown_distance = 0.0
        self.last_flown = None
        self.return_distance = None
        self.return_recorded_distance = None
        self.mission_path = []
        self.auto_return = False
        self.return_route = []
        self.last_record = None
        self.zones = []
        self.checkpoints = []
        self.challenge = None
        self.avoid_enabled = True
        self.last_avoid_log = 0
        self.last_replan = 0
        self.last_dynamic_plan = 0
        self.route_zones = []
        self.detour_expiry = 0
        self.ground_up = None
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.t = {}
        self.hb = 0
        self.pos_time = 0
        self.events = deque(maxlen=300)
        self.trail = deque(maxlen=2400)
        self.task = 'IDLE'
        self.target = None
        self.route = []
        self.scan = []
        self.warm = 0
        self.engaged = False
        self.started = 0
        self.last_mode = 0
        self.lost_line = 0
        self.water = 0
        self.last_drop = 0
        self.takeoff_at = 0
        self.height = 4.0
        self.speed = 0.65
        self.fire_target = None
        self.fire_hits = 0
        self.last_frame = 0
        self.sector = 1
        self.task_deadline = 0
        self.effects = subprocess.Popen([sys.executable, str(Path(__file__).with_name('world_effects.py'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.effects.terminate)
        threading.Thread(target=self.effect_messages, daemon=True).start()
        self.m = mavutil.mavlink_connection('udpin:127.0.0.1:14540', source_system=245, source_component=191)
        self.vision = Vision(self.telemetry)
        self.effect({'action': 'randomize'})
        threading.Thread(target=self.run, daemon=True).start()

    def effect_messages(self):
        for line in self.effects.stdout:
            try:
                d = json.loads(line)
                if d.get('error'):
                    self.log('World service: ' + d['error'])
                if 'scene_ready' in d:
                    self.scene_ready = d['scene_ready']
                    self.log('Randomized scene ready.' if self.scene_ready else 'Scene preparation failed.')
                if 'refill_result' in d:
                    self.mission.refill_result = d['refill_result']
            except ValueError:
                pass

    def effect(self, data):
        self.effects.stdin.write(json.dumps(data) + '\n')
        self.effects.stdin.flush()

    def write_events(self):
        folder = Path(__file__).parent / 'logs'
        folder.mkdir(exist_ok=True)
        while True:
            event = self.log_queue.get()
            with (folder / ('events-' + time.strftime('%Y%m%d') + '.jsonl')).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def log(self, text):
        from datetime import datetime
        self.event_id += 1
        level = 'WARNING' if any((k in text.lower() for k in ['fail', 'lost', 'delayed', 'low', 'incomplete', 'error', 'rejected', 'not applied', 'insufficient'])) else 'INFO'
        event = {'id': self.event_id, 'timestamp': datetime.now().astimezone().isoformat(), 'time': time.strftime('%H:%M:%S'), 'level': level, 'text': text}
        self.events.appendleft(event)
        self.log_queue.put(event)
        print(text, flush=True)

    def telemetry(self):
        with self.lock:
            return dict(self.t)

    def state(self):
        with self.lock:
            return {'telemetry': dict(self.t), 'connected': time.monotonic() - self.hb < 3, 'position_fresh': time.monotonic() - self.pos_time < 1, 'task': self.task, 'target': self.target, 'route': list(self.route), 'water': self.water, 'mission': self.mission.state(), 'scene_ready': self.scene_ready, 'data_hold': self.data_hold, 'vision': self.vision.state(), 'events': list(self.events), 'trail': list(self.trail), 'obstacles': OBSTACLES, 'speed': self.speed, 'height_target': self.height, 'safety': self.vision.safety.state(), 'traffic': self.vision.traffic, 'zones': self.zones, 'challenge': self.challenge, 'home': self.home, 'trajectory': self.trajectory.state(), 'flown_distance': self.flown_distance, 'return_distance': self.return_distance, 'return_recorded_distance': self.return_recorded_distance, 'mission_path': self.mission_path, 'detour_zones': self.route_zones[len(self.zones):]}

    def command(self, action, data):
        if action not in {'arm', 'disarm', 'takeoff', 'hold', 'land', 'rtl', 'goto', 'search', 'fire', 'drop', 'reset_fire', 'config', 'move', 'traffic', 'scenario', 'guard', 'mission', 'randomize'}:
            raise ValueError('Bilinmeyen komut.')
        self.commands.put((action, data))

    def send(self, cmd, *params):
        self.m.mav.command_long_send(1, 1, cmd, 0, *(list(params) + [0] * 7)[:7])

    def mode(self, name):
        self.m.set_mode(name)

    def cancel(self, reason, px4_mode='LOITER', outcome='INCOMPLETE'):
        self.pending_mission = None
        if self.mission.active:
            self.mission.active = False
            self.mission.outcome = outcome
        self.route_zones = list(self.zones)
        self.detour_expiry = 0
        self.task = px4_mode if px4_mode in {'LAND', 'RTL'} else 'HOLD'
        self.route = []
        self.scan = []
        self.engaged = False
        self.warm = 0
        self.target = None
        if px4_mode:
            self.mode(px4_mode)
        self.trajectory.reset((self.t.get('ve', 0), self.t.get('vn', 0)))
        self.log(reason)

    def begin_return(self, reason, outcome='INCOMPLETE'):
        if self.mission.active:
            self.mission.active = False
            self.mission.outcome = outcome
        self.height = max(7.0, self.height)
        self.scan = []
        self.auto_return = False
        home = getattr(self, 'home', {'east': 0.0, 'north': 0.0})
        self.plan(home['east'], home['north'], 'RETURN_HOME')
        points = [(self.t['east'], self.t['north'])] + self.route
        self.return_route = list(self.route)
        self.return_distance = sum((math.dist(a, b) for a, b in zip(points, points[1:])))
        self.return_recorded_distance = self.flown_distance
        self.log(f'{reason} Return to the recorded launch point: planned {self.return_distance:.1f} m; outbound trace {self.return_recorded_distance:.1f} m.')

    def save_flight(self):
        record = {'home': dict(self.home), 'path': list(self.mission_path), 'horizontal_distance_m': self.flown_distance, 'outbound_distance_m': self.return_recorded_distance, 'planned_return_distance_m': self.return_distance, 'planned_return_route': list(self.return_route), 'observed_mission': self.mission.state(), 'water_remaining': self.water, 'last_task': self.task, 'final_position': {'east': self.t.get('east'), 'north': self.t.get('north')}}
        self.last_record = record

        def write():
            folder = Path(__file__).parent / 'flight_logs'
            folder.mkdir(exist_ok=True)
            (folder / (time.strftime('%Y%m%d-%H%M%S') + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        threading.Thread(target=write, daemon=True).start()

    def require_air(self):
        if not self.t.get('armed') or self.t.get('height', 0) < 1:
            raise ValueError('Take off and climb above 1 m first.')
        if time.monotonic() - self.pos_time > 1:
            raise ValueError('Position telemetry is stale.')
        if self.t.get('battery', 100) < 20:
            raise ValueError('Mission requires battery above 20%; restart the simulation.')

    def plan(self, e, n, task):
        self.route_zones = list(getattr(self, 'zones', []))
        self.detour_expiry = 0
        continuing = self.engaged and self.t.get('mode') == 'OFFBOARD'
        self.route = shortest_path((self.t.get('east', 0), self.t.get('north', 0)), (e, n), getattr(self, 'zones', []))
        self.target = {'east': float(e), 'north': float(n)}
        self.task = task
        self.warm = time.monotonic()
        self.engaged = continuing
        self.task_deadline = self.warm + (1200 if task in TASKS else 180)

    def handle(self, a, d):
        if a == 'traffic':
            self.vision.configure_traffic(d)
            self.log('Moving-traffic settings sent.')
            return
        if a == 'config':
            self.height = 7.0 if self.task in TASKS - {'REFILL'} else CLAMP(float(d.get('height', self.height)), 3, 10)
            self.speed = CLAMP(float(d.get('speed', self.speed)), 0.2, 1.5)
            self.vision.filter = d.get('filter', self.vision.filter) if d.get('filter', self.vision.filter) in {'none', 'gaussian', 'median', 'sharpen'} else 'none'
            self.vision.noise = int(CLAMP(float(d.get('noise', self.vision.noise)), 0, 35))
            return
        if time.monotonic() - self.hb > 3:
            raise ValueError('No PX4 connection; start the simulation first.')
        if a == 'arm':
            self.mode('LOITER')
            self.send(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.log('Arm command sent; monitor telemetry for the result.')
        elif a == 'disarm':
            if self.t.get('landed') != 1:
                raise ValueError('Airborne disarm is disabled. Use Land or Emergency Land.')
            self.send(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
            self.log('Disarm command sent.')
        elif a == 'takeoff':
            if self.t.get('landed', 1) != 1:
                raise ValueError('Vehicle is already airborne.')
            if 'alt_amsl' not in self.t:
                raise ValueError('GPS/altitude reference is not ready.')
            self.home = {'east': self.t.get('east', 0.0), 'north': self.t.get('north', 0.0)}
            self.flown_distance = 0.0
            self.last_flown = None
            self.mission_path = []
            self.return_distance = None
            self.return_recorded_distance = None
            self.return_route = []
            self.trajectory.reset()
            self.height = CLAMP(float(d.get('height', 4)), 3, 10)
            self.mode('LOITER')
            self.send(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.task = 'TAKEOFF'
            self.takeoff_at = time.monotonic() + 1
            self.task_deadline = time.monotonic() + 25
            self.log(f'{self.height:.1f} m takeoff preparation.')
        elif a == 'hold':
            self.cancel('Mission stopped; PX4 holds position.', outcome='ABORTED')
        elif a == 'land':
            self.cancel('Landing started.', 'LAND', outcome='ABORTED')
        elif a == 'rtl':
            self.require_air()
            self.begin_return('Return home requested.', outcome='ABORTED')
        elif a in {'goto', 'move'}:
            self.require_air()
            e = float(d.get('east', self.t['east']))
            n = float(d.get('north', self.t['north']))
            if a == 'move':
                e += self.t['east']
                n += self.t['north']
            self.plan(e, n, 'GOTO')
            self.log(f'A* route: east {e:.1f}, Kuzey {n:.1f} m.')
        elif a == 'guard':
            self.require_air()
            self.plan(self.t['east'], self.t['north'], 'GUARD')
            self.log('LiDAR-protected hold enabled.')
        elif a == 'scenario':
            self.require_air()
            kind = d.get('scenario', 'crossing')
            self.challenge = kind
            self.height = 4.0
            self.zones = []
            self.checkpoints = []
            self.vision.configure_traffic({'enabled': True, 'seed': int(d.get('seed', 42)), 'count': 3, 'speed': 0.8, 'pattern': 'crossing' if kind == 'crossing' else 'random'})
            if kind == 'nfz':
                self.zones = [(15, 0, 4)]
            if kind == 'checkpoints':
                self.checkpoints = [(20, 2), (32, 0), (42, -3)]
            if kind == 'visibility':
                self.vision.noise = 25
                self.vision.filter = 'gaussian'
            goal = (8, 0) if kind == 'checkpoints' else (40, 0)
            self.plan(*goal, 'CHALLENGE')
            self.task_deadline = time.monotonic() + 300
            self.log('Training scenario started: ' + kind)
        elif a in {'mission', 'fire', 'search'}:
            if not self.scene_ready:
                raise ValueError('Scene is not ready.')
            if self.vision.state()['age'] is None or self.vision.state()['age'] > 2:
                raise ValueError('Camera stream is not ready for the mission.')
            mode = d.get('mode', 'fire')
            if self.t.get('landed') == 1:
                self.handle('takeoff', {'height': 7})
                self.pending_mission = mode
            else:
                self.require_air()
                self.mission.start(self, mode)
        elif a in {'reset_fire', 'randomize'}:
            if self.t.get('armed'):
                raise ValueError('Land and disarm before creating a new scene.')
            self.mission = SurveyMission()
            self.water = 0
            self.scene_ready = False
            self.effect({'action': 'randomize'})
            self.log('Preparing a new scene.')
        elif a == 'drop':
            self.drop()

    def drop(self):
        now = time.monotonic()
        if now - self.last_drop < 2.5:
            return
        if self.water < 25:
            raise ValueError('Su deposu yetersiz.')
        self.require_air()
        if not 5.5 <= self.t['height'] <= 8:
            raise ValueError('Water-drop altitude must be 5.5-8 m.')
        if math.hypot(self.t.get('ve', 0), self.t.get('vn', 0)) > 0.25 or max(abs(self.t.get('roll', 0)), abs(self.t.get('pitch', 0))) > math.radians(6):
            return
        if self.vision.safety.nearest is not None and self.vision.safety.nearest < 3:
            return
        if self.vision.truth is None:
            raise ValueError('Simulation effect position is not ready.')
        self.water -= 25
        self.last_drop = now
        self.effect({'action': 'drop', 'position': list(self.vision.truth)})
        self.log(f'Water applied. Tank %{self.water}; the result is monitored by the camera.')

    def parse(self, msg):
        typ = msg.get_type()
        now = time.monotonic()
        if typ == 'HEARTBEAT' and msg.get_srcSystem() == 1:
            was_armed = self.t.get('armed', False)
            self.hb = now
            self.t.update(armed=bool(msg.base_mode & 128), mode=mavutil.mode_string_v10(msg), system_status=msg.system_status)
            if was_armed and (not self.t['armed']):
                self.save_flight()
        elif typ == 'LOCAL_POSITION_NED':
            self.pos_time = now
            if self.ground_up is None or (self.t.get('landed') == 1 and (not self.t.get('armed'))):
                self.ground_up = -msg.z
            self.t.update(north=msg.x, east=msg.y, local_up=-msg.z, height=max(0, -msg.z - self.ground_up), vn=msg.vx, ve=msg.vy, vup=-msg.vz)
            if self.t.get('armed') and self.t['height'] > 1:
                point = (msg.y, msg.x)
                if self.last_flown is not None and math.dist(point, self.last_flown) < 2:
                    self.flown_distance += math.dist(point, self.last_flown)
                self.last_flown = point
                if not self.mission_path or math.dist(point, self.mission_path[-1]) > 0.3:
                    self.mission_path.append(point)
            if not self.trail or now - self.trail[-1][0] > 0.25:
                self.trail.append([now, msg.y, msg.x])
        elif typ == 'GLOBAL_POSITION_INT':
            self.t.update(gps_relative_height=msg.relative_alt / 1000, alt_amsl=msg.alt / 1000, lat=msg.lat / 10000000.0, lon=msg.lon / 10000000.0)
        elif typ == 'ATTITUDE':
            self.t.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw, rollspeed=msg.rollspeed, pitchspeed=msg.pitchspeed, yawspeed=msg.yawspeed)
        elif typ == 'EXTENDED_SYS_STATE':
            self.t['landed'] = msg.landed_state
        elif typ == 'SYS_STATUS':
            self.t['battery'] = msg.battery_remaining
        elif typ == 'GPS_RAW_INT':
            self.t.update(gps_fix=msg.fix_type, satellites=msg.satellites_visible)
        elif typ == 'COMMAND_ACK' and msg.result not in (0, 5):
            self.log(f'PX4 command response: {msg.command} / result {msg.result}.')
        elif typ == 'STATUSTEXT' and msg.severity <= 4:
            self.log('PX4: ' + msg.text)

    def setpoint(self, e, n, height):
        de = e - self.t.get('east', 0)
        dn = n - self.t.get('north', 0)
        norm = max(1, math.hypot(de, dn) / self.speed)
        ve = de / norm
        vn = dn / norm
        vz = CLAMP((self.t.get('height', 0) - height) * 0.8, -0.7, 0.7)
        if self.task == 'CHALLENGE' and self.challenge == 'checkpoints' and (18 < self.t.get('east', 0) < 30):
            length = math.hypot(ve, vn)
            if length > 0.5:
                ve *= 0.5 / length
                vn *= 0.5 / length
        if self.avoid_enabled and self.t.get('height', 0) > 1:
            safe = self.vision.safety.choose([self.t.get('east', 0), self.t.get('north', 0)], [ve, vn], min(self.speed, 1.0), self.zones, velocity=[self.t.get('ve', 0), self.t.get('vn', 0)])
            ve, vn = (float(safe[0]), float(safe[1]))
            if self.vision.safety.active and time.monotonic() - self.last_avoid_log > 3:
                self.log('LiDAR: ' + self.vision.safety.reason)
                self.last_avoid_log = time.monotonic()
        self.publish_velocity([ve, vn], height)

    def publish_velocity(self, velocity, height):
        ve, vn = map(float, self.trajectory.step(velocity, self.t))
        if self.task in TASKS - {'REFILL'} or self.task == 'RETURN_HOME':
            height = max(7.0, height)
        z = -(float(self.ground_up or 0) + height)
        self.m.mav.set_position_target_local_ned_send(0, 1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED, 2531, 0, 0, z, vn, ve, 0, 0, 0, 0, math.pi / 2, 0)

    def tick(self, now):
        if self.task == 'TAKEOFF':
            if self.takeoff_at and now >= self.takeoff_at:
                if self.t.get('armed'):
                    self.send(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, math.nan, math.nan, math.nan, self.t['alt_amsl'] + self.height)
                    self.takeoff_at = 0
                elif now > self.task_deadline:
                    self.cancel('Arming failed; takeoff cancelled.', None)
            if self.t.get('height', 0) > self.height - 0.35 and (not self.takeoff_at):
                self.task = 'HOLD'
                self.log('Takeoff complete.')
                if self.pending_mission:
                    mode = self.pending_mission
                    self.pending_mission = None
                    self.mission.start(self, mode)
            elif now > self.task_deadline:
                self.cancel('Takeoff timeout.', 'LAND')
            return
        active = self.task in TASKS | {'GOTO', 'RETURN_HOME', 'GUARD', 'CHALLENGE'}
        if not active:
            return
        v = self.vision.state()
        fresh = v['age'] is not None and v['age'] < 1.5
        sim_age = v.get('sim_age')
        if sim_age is not None and sim_age > 1:
            if not self.sim_paused_at:
                self.sim_paused_at = now
                self.log('Simulation clock paused; mission suspended.')
            self.data_hold = True
            self.mission.absent_since = 0
            self.mission.stable_since = 0
            self.publish_velocity([0, 0], self.height)
            return
        if self.sim_paused_at:
            self.task_deadline += now - self.sim_paused_at
            self.sim_paused_at = 0
            self.resume_until = now + 3
            self.trajectory.reset((self.t.get('ve', 0), self.t.get('vn', 0)))
        telemetry_bad = now - self.pos_time > 1.5 or now - self.hb > 3
        camera_bad = self.task in TASKS and (not fresh)
        if telemetry_bad or camera_bad:
            if not self.data_stale_since:
                self.data_stale_since = now
                self.log('Camera/telemetry delayed; controlled hold.')
            self.data_hold = True
            self.mission.absent_since = 0
            self.mission.stable_since = 0
            self.publish_velocity([0, 0], self.height)
            if now - self.data_stale_since < 2 or now < self.resume_until:
                return
            if telemetry_bad:
                self.cancel('Telemetry lost; mission abandoned.', 'LAND')
            else:
                self.begin_return('Camera stream is stale; target mission abandoned.')
            return
        if self.data_hold:
            self.log('Data stream fresh; mission continues.')
        self.data_hold = False
        self.data_stale_since = 0
        if self.t.get('landed') == 1 or not self.t.get('armed'):
            self.cancel('Vehicle landed/disarmed; mission stopped.', None)
            return
        if self.t.get('battery', 100) >= 0 and self.t['battery'] < 15:
            self.cancel('Low battery: landing at the current position.', 'LAND')
            return
        if now > self.task_deadline:
            if self.task in TASKS:
                self.begin_return('Mission duration expired.')
            else:
                self.cancel('Mission timed out.')
            return
        if self.engaged and now - self.warm > 5 and (self.t.get('mode') != 'OFFBOARD'):
            self.cancel('PX4/ground station took control; autonomous mission abandoned.', None)
            return
        if self.mission.active:
            self.mission.tick(self, v, now)
        e, n = (self.t['east'], self.t['north'])
        te, tn = (self.target['east'], self.target['north']) if self.target else (e, n)
        if self.detour_expiry and now > self.detour_expiry:
            self.route_zones = list(self.zones)
            self.detour_expiry = 0
        if self.route and self.target and (now - self.last_dynamic_plan > 8):
            extra = self.vision.safety.detour((e, n), (self.target['east'], self.target['north']), self.speed)
            if extra:
                self.last_dynamic_plan = now
                try:
                    self.route = shortest_path((e, n), (self.target['east'], self.target['north']), self.zones + extra)
                    self.route_zones = self.zones + extra
                    self.detour_expiry = now + 16
                    self.vision.safety.side = 0
                    self.log('Temporary detour planned from LiDAR motion estimates.')
                except ValueError:
                    pass
        if self.route and (not visible((e, n), self.route[0], self.route_zones)) and (now - getattr(self, 'last_replan', 0) > 2):
            self.last_replan = now
            try:
                self.route = shortest_path((e, n), (self.target['east'], self.target['north']), self.route_zones)
                self.vision.safety.side = 0
                self.log('Route recalculated after avoidance with obstacle clearance.')
            except ValueError:
                pass
        if self.route:
            while len(self.route) > 1 and visible((e, n), self.route[1], self.route_zones):
                self.route.pop(0)
            te, tn = self.route[0]
            if math.hypot(te - e, tn - n) < (0.2 if len(self.route) > 1 else 0.55):
                self.route.pop(0)
                if self.route:
                    te, tn = self.route[0]
        if self.task == 'GUARD':
            te, tn = (self.target['east'], self.target['north'])
        elif not self.route:
            if self.task == 'CHALLENGE':
                if self.checkpoints:
                    goal = self.checkpoints.pop(0)
                    self.plan(*goal, 'CHALLENGE')
                    self.task_deadline = now + 300
                else:
                    self.cancel('Training scenario complete.')
                    return
            elif self.task == 'RETURN_HOME':
                if math.dist((e, n), (self.home['east'], self.home['north'])) < 0.6 and math.hypot(self.t.get('ve', 0), self.t.get('vn', 0)) < 0.15 and (self.vision.safety.nearest is None or self.vision.safety.nearest > 3):
                    self.cancel('Launch area reached; automatic landing.', 'LAND')
                    return
                te, tn = (self.home['east'], self.home['north'])
            elif self.task == 'GOTO':
                self.cancel('Goal reached.')
                return
        if not self.engaged:
            self.setpoint(e, n, self.t['height'])
            if now - self.warm > 1.3 and now - self.last_mode > 1:
                self.mode('OFFBOARD')
                self.last_mode = now
                self.engaged = True
        else:
            self.setpoint(te, tn, self.height)

    def run(self):
        self.log('Controller ready; waiting for PX4 and cameras.')
        last_beat = 0
        last_tick = 0
        while True:
            now = time.monotonic()
            try:
                with self.lock:
                    for _ in range(150):
                        msg = self.m.recv_match(blocking=False)
                        if msg is None:
                            break
                        self.parse(msg)
                    if now - last_beat > 1:
                        self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
                        last_beat = now
                    for _ in range(4):
                        try:
                            a, d = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            self.handle(a, d)
                        except Exception as e:
                            self.log('Command not applied: ' + str(e))
                    if now - last_tick > 0.05:
                        try:
                            self.tick(now)
                        except Exception as e:
                            self.cancel('Mission error: ' + str(e))
                        last_tick = now
            except Exception as e:
                self.log('Connection error: ' + str(e))
                time.sleep(0.2)
            time.sleep(0.005)
