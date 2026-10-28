import math, time, threading
import numpy as np
from tracking import KalmanTrack
from planning import blocked, clearance_ok, OBSTACLES

class Avoidance:

    def __init__(self):
        self.lock = threading.RLock()
        self.last = 0
        self.points = np.empty((0, 2))
        self.vel = np.empty((0, 2))
        self.tracks = []
        self.next_id = 1
        self.frames = 0
        self.interventions = 0
        self.active = False
        self.nearest = None
        self.chosen = [0, 0]
        self.reason = 'Waiting for LiDAR'
        self._was_active = False
        self.side = 0
        self.clear_since = 0
        self.decision_time = 0
        self.blocked_since = 0

    def update(self, msg, st):
        if not all((k in st for k in ['east', 'north', 'height', 'yaw'])):
            return
        now = time.monotonic()
        h = msg.count
        v = msg.vertical_count or 1
        ranges = np.asarray(msg.ranges).reshape(v, h)
        az = np.linspace(msg.angle_min, msg.angle_max, h)
        el = np.linspace(msg.vertical_angle_min, msg.vertical_angle_max, v)
        xx = ranges * np.cos(el[:, None]) * np.cos(az)
        yy = ranges * np.cos(el[:, None]) * np.sin(az)
        zz = ranges * np.sin(el[:, None])
        cr, sr = (math.cos(st.get('roll', 0)), math.sin(st.get('roll', 0)))
        cp, sp = (math.cos(st.get('pitch', 0)), math.sin(st.get('pitch', 0)))
        cy, sy = (math.cos(st['yaw']), math.sin(st['yaw']))
        R = np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy], [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy], [-sp, sr * cp, cr * cp]])
        xyz = np.stack([xx, -yy, -zz], axis=-1)
        ned = xyz @ R.T
        valid = np.isfinite(ranges) & (ranges > 0.85) & (ranges < 39) & (np.abs(ned[:, :, 2]) < 1.4)
        d = np.where(valid, ranges, np.inf)
        j = np.argmin(d, axis=0)
        k = np.arange(h)
        mask = np.isfinite(d[j, k])
        local = ned[j[mask], k[mask], :]
        points = np.stack([local[:, 1] + st['east'], local[:, 0] + st['north']], axis=-1)
        groups = []
        for p in points:
            if not groups or np.linalg.norm(p - groups[-1][-1]) > 1.4:
                groups.append([p])
            else:
                groups[-1].append(p)
        if len(groups) > 1 and np.linalg.norm(np.asarray(groups[0][0]) - groups[-1][-1]) < 1.4:
            groups[0] = groups.pop() + groups[0]
        tracks = []
        velocities = []
        used = set()
        with self.lock:
            for group in groups:
                arr = np.asarray(group)
                center = arr.mean(axis=0)
                span = np.max(np.linalg.norm(arr - center, axis=1))
                candidates = [t for t in self.tracks if t['id'] not in used and 0 < now - t['time'] < 0.8]
                old = min(candidates, key=lambda t: np.linalg.norm(center - t['filter'].predict(now)[0][:2]), default=None)
                velocity = np.zeros(2)
                age = 1
                if old is not None and np.linalg.norm(center - old['filter'].predict(now)[0][:2]) < 1.8 and (span < 2) and (old['span'] < 2) and (np.linalg.norm(center - old['measurement']) / max(0.08, now - old['time']) < 3.5):
                    filt = old['filter']
                    filt.update(center, now)
                    velocity = filt.x[2:].copy()
                    age = old['age'] + 1
                    tid = old['id']
                    used.add(tid)
                else:
                    filt = KalmanTrack(center, now)
                    tid = self.next_id
                    self.next_id += 1
                tracks.append(dict(id=tid, center=filt.x[:2].copy(), measurement=center, span=span + float(np.linalg.norm(center - filt.x[:2])), velocity=velocity, time=now, age=age, filter=filt))
                velocities.extend([velocity if age >= 3 else np.zeros(2)] * len(group))
            self.points = points
            self.vel = np.asarray(velocities).reshape(-1, 2)
            self.tracks = tracks
            self.last = now
            self.frames += 1
            self.nearest = float(np.min(np.linalg.norm(points - [st['east'], st['north']], axis=1))) if len(points) else None

    def state(self):
        with self.lock:
            return {'tracking_filter': 'constant_velocity_kalman', 'age': round(time.monotonic() - self.last, 2) if self.last else None, 'frames': self.frames, 'nearest': self.nearest, 'active': self.active, 'reason': self.reason, 'interventions': self.interventions, 'chosen_velocity': self.chosen, 'tracks': [{'id': t['id'], 'east': float(t['center'][0]), 'north': float(t['center'][1]), 'speed': float(np.linalg.norm(t['velocity'])), 'radius': float(t['span'])} for t in self.tracks], 'points': self.points[::2].tolist()}

    def detour(self, position, goal, speed):
        pos = np.asarray(position)
        direction = np.asarray(goal) - pos
        length = np.linalg.norm(direction)
        if length < 6:
            return []
        unit = direction / length
        own = unit * speed
        hazards = []
        with self.lock:
            if time.monotonic() - self.last > 1:
                return []
            for t in self.tracks:
                center = t['center']
                velocity = t['velocity']
                span = t['span']
                if t['age'] < 4 or span > 1.8 or np.linalg.norm(velocity) < 0.12:
                    continue
                if any((x - 0.7 <= center[0] <= xx + 0.7 and y - 0.7 <= center[1] <= yy + 0.7 for x, y, xx, yy in OBSTACLES)):
                    continue
                relative = center - pos
                rv = velocity - own
                ttc = -np.dot(relative, rv) / max(0.01, np.dot(rv, rv))
                if not 1 < ttc < 16 or np.linalg.norm(relative) > 19:
                    continue
                if np.linalg.norm(relative + rv * ttc) > 4:
                    continue
                future = center + velocity * min(ttc, 12)
                radius = max(0.8, min(1.8, span + 0.6))
                if np.linalg.norm(future - np.asarray(goal)) < radius + 3.3:
                    continue
                hazards.append((ttc, (float(future[0]), float(future[1]), radius)))
        return [min(hazards, key=lambda a: a[0])[1]] if hazards else []

    def choose(self, position, desired, max_speed=1.0, zones=(), velocity=None):
        with self.lock:
            points = self.points.copy()
            vel = self.vel.copy()
            fresh = time.monotonic() - self.last < 1.2
        desired = np.asarray(desired, dtype=float)
        pos = np.asarray(position, dtype=float)
        previous = np.asarray(self.chosen)
        current = previous if velocity is None else np.asarray(velocity, dtype=float)
        if not fresh:
            self.active = True
            self.reason = 'Stale LiDAR: controlled braking'
            self.chosen = [0, 0]
            return np.zeros(2)
        now = time.monotonic()
        candidates = [desired, np.zeros(2), previous]
        for speed in [min(0.45, max_speed), max_speed]:
            for angle in np.linspace(0, 2 * math.pi, 32, endpoint=False):
                candidates.append(speed * np.array([math.cos(angle), math.sin(angle)]))
        candidates = np.asarray(candidates)
        times = np.linspace(0.2, 8.0, 32)
        delta = candidates - current
        duration = np.maximum(np.linalg.norm(delta, axis=1) / 0.35, 0.1)
        accel = delta / duration[:, None]
        ramp = np.minimum(times[None, :], duration[:, None])
        travel = current[None, None, :] * ramp[:, :, None] + 0.5 * accel[:, None, :] * ramp[:, :, None] ** 2 + candidates[:, None, :] * np.maximum(0, times[None, :] - duration[:, None])[:, :, None]
        if len(points):
            relative = points[None, None, :, :] + vel[None, None, :, :] * times[None, :, None, None] - pos[None, None, None, :] - travel[:, :, None, :]
            clearance = np.linalg.norm(relative, axis=-1).min(axis=(1, 2))
            threatened = clearance[0] < 3.2
        else:
            clearance = np.full(len(candidates), 40.0)
            threatened = False
        map_ok = np.array([clearance_ok(pos, pos + path[::3], zones, 3.0) for path in travel])
        safe = (clearance >= 2.8) & map_ok
        turn = desired[0] * candidates[:, 1] - desired[1] * candidates[:, 0]
        switching = turn * self.side < -0.02 if self.side else np.zeros(len(candidates), dtype=bool)
        progress = candidates @ desired / max(0.1, np.linalg.norm(desired))
        viable = safe & (progress > 0.2)
        if self.side and np.any(viable) and (not np.any(viable & ~switching)):
            if not self.blocked_since:
                self.blocked_since = now
            if now - self.blocked_since > 1.5:
                self.side = 0
                switching[:] = False
        else:
            self.blocked_since = 0
        score = np.linalg.norm(candidates - desired, axis=1) + 0.35 * np.linalg.norm(candidates - previous, axis=1) + 0.4 / np.maximum(clearance, 0.2) + switching * 3.0
        if np.any(safe):
            index = int(np.argmin(np.where(safe, score, np.inf)))
            reason = 'Predictive passing: committed side'
        else:
            index = int(np.argmax(np.where(map_ok, clearance, -100)))
            reason = 'Nearby traffic: controlled separation'
            if not np.any(map_ok):
                index = 1
                reason = 'Route blocked: controlled braking'
        result = candidates[index]
        active = bool(threatened or np.linalg.norm(result - desired) > 0.15)
        if threatened:
            self.clear_since = 0
            cross = desired[0] * result[1] - desired[1] * result[0]
            if not self.side and abs(cross) > 0.03:
                self.side = 1 if cross > 0 else -1
        else:
            if not self.clear_since:
                self.clear_since = now
            if now - self.clear_since > 3:
                self.side = 0
        if active and (not self._was_active):
            self.interventions += 1
        self._was_active = active
        self.active = active
        self.reason = reason if active else 'Route clear'
        self.chosen = result.tolist()
        return result
