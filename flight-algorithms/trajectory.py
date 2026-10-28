import math, time
import numpy as np

class Trajectory:

    def __init__(self):
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.last = 0
        self.stabilizing = False

    def reset(self, velocity=(0, 0)):
        self.velocity = np.asarray(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        self.last = 0

    @staticmethod
    def limit(v, maximum):
        return v * min(1.0, maximum / max(1e-09, np.linalg.norm(v)))

    def step(self, target, st, now=None):
        now = time.monotonic() if now is None else now
        dt = min(0.1, max(0.01, now - self.last)) if self.last else 0.05
        self.last = now
        tilt = max(abs(st.get('roll', 0)), abs(st.get('pitch', 0)))
        rate = max(abs(st.get('rollspeed', 0)), abs(st.get('pitchspeed', 0)))
        if tilt > math.radians(10) or rate > math.radians(35):
            self.stabilizing = True
        elif tilt < math.radians(5) and rate < math.radians(12):
            self.stabilizing = False
        target = np.zeros(2) if self.stabilizing else np.asarray(target, dtype=float)
        desired = self.limit((target - self.velocity) * 1.2, 0.35)
        self.acceleration += self.limit(desired - self.acceleration, 0.7 * dt)
        self.acceleration = self.limit(self.acceleration, 0.35)
        next_velocity = self.velocity + self.acceleration * dt
        self.velocity = next_velocity
        return self.velocity.copy()

    def state(self):
        return {'velocity': self.velocity.tolist(), 'acceleration': self.acceleration.tolist(), 'stabilizing': self.stabilizing, 'acceleration_limit': 0.35, 'jerk_limit': 0.7}
