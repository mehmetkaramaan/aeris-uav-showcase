import math, time
from coverage import Coverage
from tracking import DetectionWindow
TASKS = {'STATION_SEARCH', 'STATION_APPROACH', 'REFILL', 'SURVEY', 'FIRE_APPROACH', 'EXTINGUISH', 'VERIFY_FIRE'}

class SurveyMission:

    def __init__(self):
        self.station_window = DetectionWindow()
        self.fire_windows = {}
        self.coverage = Coverage()
        self.active = False
        self.station = None
        self.fires = []
        self.current = None
        self.refills = 0
        self.phase = 'IDLE'
        self.last_frame = -1
        self.station_hits = 0
        self.fire_hits = {}
        self.absent_since = 0
        self.stable_since = 0
        self.refill_pending = False
        self.refill_result = None
        self.outcome = 'NOT_STARTED'
        self.pass_number = 0
        self.cue = None

    def state(self):
        return {'active': self.active, 'phase': self.phase, 'station': self.station, 'fires': self.fires, 'refills': self.refills, 'coverage': self.coverage.state(), 'pass': self.pass_number, 'outcome': self.outcome, 'detection_filter': '3_of_5_frames_within_1.2s'}

    def start(self, c, mode='fire'):
        self.__init__()
        self.active = True
        self.outcome = 'IN_PROGRESS'
        self.mode = mode
        c.water = 0
        c.height = 7.0
        self.coverage.home = (c.home['east'], c.home['north'])
        self.coverage.reset()
        self.pass_number = 1
        self.scan(c, station=mode == 'fire')
        c.log('Mission started with an empty tank; station and targets are searched using cameras only.')

    def scan(self, c, station=False):
        self.phase = 'STATION_SEARCH' if station else 'SURVEY'
        p = self.coverage.next((c.t['east'], c.t['north']))
        if p is None:
            if self.coverage.planning:
                c.route = []
                c.target = {'east': c.t['east'], 'north': c.t['north']}
                c.task = self.phase
                return
            if station:
                self.finish(c, 'Station not found; mission incomplete.', 'INCOMPLETE')
            elif self.coverage.complete:
                self.finish(c, 'Camera coverage of the searchable area is complete.', 'COMPLETE')
            else:
                self.finish(c, 'Coverage incomplete: no accessible viewpoint route.', 'INCOMPLETE')
        else:
            c.height = 7.0
            c.speed = 1.0
            c.plan(*p, self.phase)

    def finish(self, c, text, outcome):
        self.active = False
        self.outcome = outcome
        self.phase = 'RETURN_HOME'
        c.begin_return(text)

    def station_route(self, c):
        if self.station is None:
            self.coverage.replan()
            self.scan(c, True)
            return
        c.speed = 0.75
        self.phase = 'STATION_APPROACH'
        self.stable_since = 0
        c.height = 7.0
        c.plan(self.station['east'], self.station['north'], self.phase)
        c.log('Approaching the visually recorded water station.')

    def choose_fire(self, c):
        if self.mode != 'fire':
            return False
        active = [f for f in self.fires if f['status'] == 'ACTIVE']
        if not active:
            return False
        self.current = min(active, key=lambda f: math.dist((c.t['east'], c.t['north']), (f['east'], f['north'])))
        c.speed = 0.75
        self.phase = 'FIRE_APPROACH'
        self.absent_since = 0
        c.height = 7.0
        c.plan(self.current['east'], self.current['north'], self.phase)
        c.log(f"Kamera hedefi {self.current['id']}: approaching.")
        return True

    def perceive(self, c, v, now):
        if v['frames'] == self.last_frame:
            return False
        self.last_frame = v['frames']
        self.coverage.observe(c.t, v['frames'])
        ds = [d for d in v['detections'] if d['bbox'][0] > 8 and d['bbox'][1] > 8 and (d['bbox'][0] + d['bbox'][2] < 632) and (d['bbox'][1] + d['bbox'][3] < 472)]
        stations = [d for d in ds if d['label'] == 'Station']
        self.station_hits = self.station_window.update(v['frames'], now, bool(stations))
        if self.station_hits >= 3 and stations:
            d = max(stations, key=lambda d: d['area'])
            self.station = {'east': d['east'], 'north': d['north']}
        observed = set()
        for d in [a for a in ds if a['label'] == 'Fire']:
            f = min(self.fires, key=lambda f: math.dist((f['east'], f['north']), (d['east'], d['north'])), default=None)
            if f is None or math.dist((f['east'], f['north']), (d['east'], d['north'])) > 3:
                f = {'id': len(self.fires) + 1, 'east': d['east'], 'north': d['north'], 'status': 'CANDIDATE', 'hits': 0}
                self.fires.append(f)
            if f['id'] in observed:
                continue
            observed.add(f['id'])
            f['last_seen'] = now
            f['east'] = 0.8 * f['east'] + 0.2 * d['east']
            f['north'] = 0.8 * f['north'] + 0.2 * d['north']
        for f in self.fires:
            window = self.fire_windows.setdefault(f['id'], DetectionWindow())
            f['hits'] = window.update(v['frames'], now, f['id'] in observed)
            if f['hits'] >= 3 and f['id'] in observed and (f['status'] != 'ACTIVE'):
                f['status'] = 'ACTIVE'
                c.log(f"Fire {f['id']} confirmed in camera observations.")
        return True

    def tick(self, c, v, now):
        fresh = self.perceive(c, v, now)
        position = (c.t['east'], c.t['north'])
        speed = math.hypot(c.t.get('ve', 0), c.t.get('vn', 0))
        if self.phase in {'STATION_SEARCH', 'SURVEY'} and c.route and self.coverage.target_satisfied():
            self.coverage.skipped_views += 1
            self.scan(c, station=self.phase == 'STATION_SEARCH')
        if self.phase == 'STATION_SEARCH':
            if self.station is not None:
                self.station_route(c)
            elif not c.route:
                cues = [d for d in v.get('front_landmarks', []) if d['label'] == 'Station']
                if cues and (self.cue is None or now - self.cue > 15):
                    self.cue = now
                    self.coverage.replan()
                    d = cues[0]
                    try:
                        c.plan(d['east'], d['north'], 'STATION_SEARCH')
                        c.log('Station cue in the forward camera; moving for closer observation.')
                    except ValueError:
                        self.scan(c, True)
                else:
                    self.scan(c, True)
        elif self.phase == 'STATION_APPROACH':
            if not c.route and self.station_hits >= 3:
                self.phase = 'REFILL'
                c.task = 'REFILL'
                c.speed = 0.45
                c.height = 3.0
                self.stable_since = 0
                self.refill_pending = False
                self.refill_result = None
                c.log('Station reacquired; controlled refill approach.')
        elif self.phase == 'REFILL':
            near = self.station and math.dist(position, (self.station['east'], self.station['north'])) < 0.65 and (abs(c.t['height'] - 3) < 0.3) and (speed < 0.18) and (c.vision.safety.nearest is None or c.vision.safety.nearest > 3)
            if near:
                if not self.stable_since:
                    self.stable_since = now
                if now - self.stable_since > 3 and (not self.refill_pending):
                    self.refill_pending = True
                    c.effect({'action': 'refill', 'position': list(c.vision.truth or ())})
            else:
                self.stable_since = 0
            if self.refill_result is not None:
                if self.refill_result:
                    c.water = 100
                    self.refills += 1
                    c.height = 7.0
                    self.coverage.replan()
                    self.phase = 'SURVEY'
                    c.log('Refill confirmed; tank 100%. Fire search continues.')
                    if not self.choose_fire(c):
                        self.scan(c)
                else:
                    self.refill_pending = False
                    self.refill_result = None
                    self.stable_since = now
                    c.log('Station refill unconfirmed; waiting for alignment.')
        elif self.phase == 'SURVEY':
            if self.mode == 'fire' and c.water < 25:
                self.station_route(c)
            elif self.choose_fire(c):
                pass
            elif self.coverage.complete:
                self.finish(c, 'All searchable cells have been observed by the camera.' + (' No active fire.' if self.mode == 'fire' else ' Survey complete.'), 'COMPLETE')
            elif not c.route:
                cues = [d for d in v.get('front_landmarks', []) if d['label'] == 'Fire' and (not any((f['status'] in {'ACTIVE', 'CLEARED'} and math.dist((d['east'], d['north']), (f['east'], f['north'])) < 4 for f in self.fires)))]
                if cues and (self.cue is None or now - self.cue > 15):
                    d = min(cues, key=lambda d: math.dist(position, (d['east'], d['north'])))
                    self.cue = now
                    self.coverage.replan()
                    try:
                        c.plan(d['east'], d['north'], 'SURVEY')
                        c.log('Close observation route for a forward-camera cue.')
                    except ValueError:
                        self.scan(c)
                else:
                    self.scan(c)
        elif self.phase == 'FIRE_APPROACH':
            if c.water < 25:
                self.station_route(c)
            elif not c.route:
                self.phase = 'EXTINGUISH'
                c.task = 'EXTINGUISH'
                c.speed = 0.45
                self.absent_since = 0
        elif self.phase in {'EXTINGUISH', 'VERIFY_FIRE'}:
            target = (self.current['east'], self.current['north'])
            c.target = {'east': target[0], 'north': target[1]}
            c.height = 7.0
            visible = self.coverage.sees(target, c.t) and math.dist(position, target) < 0.9 and (speed < 0.22) and (max(abs(c.t.get('roll', 0)), abs(c.t.get('pitch', 0))) < math.radians(5)) and (c.vision.safety.nearest is None or c.vision.safety.nearest > 3)
            burning = any((d['label'] == 'Fire' and math.dist((d['east'], d['north']), target) < 3 for d in v['detections']))
            if fresh and visible and (now - c.last_drop > 1.6):
                if burning:
                    self.absent_since = 0
                else:
                    if not self.absent_since:
                        self.absent_since = now
                    if now - self.absent_since >= 2.5:
                        self.current['status'] = 'CLEARED'
                        self.current['cleared_at'] = now
                        c.log(f"Fire {self.current['id']}: flame disappearance verified with a stable camera.")
                        self.coverage.replan()
                        self.current = None
                        if c.water < 25:
                            self.station_route(c)
                        elif not self.choose_fire(c):
                            self.scan(c)
                        return
            elif not visible:
                self.absent_since = 0
            if burning and visible:
                if c.water < 25:
                    self.station_route(c)
                elif now - c.last_drop > 2.5:
                    c.drop()
