from collections import deque
import numpy as np

class KalmanTrack:

    def __init__(self, position, now):
        self.x = np.r_[np.asarray(position, dtype=float), [0.0, 0.0]]
        self.P = np.diag([0.09, 0.09, 1.0, 1.0])
        self.time = now

    def predict(self, now):
        dt = max(0.0, now - self.time)
        F = np.eye(4)
        F[:2, 2:] = np.eye(2) * dt
        G = np.vstack([np.eye(2) * dt * dt / 2, np.eye(2) * dt])
        return (F @ self.x, F @ self.P @ F.T + G @ G.T * 0.8 ** 2)

    def update(self, position, now):
        if not 0 < now - self.time < 0.8:
            self.__init__(position, now)
            return
        x, P = self.predict(now)
        H = np.eye(2, 4)
        R = np.eye(2) * 0.3 ** 2
        K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
        self.x = x + K @ (np.asarray(position) - H @ x)
        A = np.eye(4) - K @ H
        self.P = A @ P @ A.T + K @ R @ K.T
        self.time = now

class DetectionWindow:

    def __init__(self):
        self.samples = deque(maxlen=5)

    def update(self, frame, now, present):
        if self.samples and now < self.samples[-1][1]:
            self.samples.clear()
        if self.samples and frame == self.samples[-1][0]:
            return self.hits(now)
        self.samples.append((frame, now, bool(present)))
        return self.hits(now)

    def hits(self, now):
        while self.samples and now - self.samples[0][1] > 1.2:
            self.samples.popleft()
        return sum((s[2] for s in self.samples))
