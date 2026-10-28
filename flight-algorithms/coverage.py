import math
import numpy as np
from planning import blocked, visible, shortest_path
from coverage_route import POOL, optimize
BOUNDS = (-4, 44, -24, 24)
_FOOTPRINTS = None

def rotation(st):
    cr, sr = (math.cos(st.get('roll', 0)), math.sin(st.get('roll', 0)))
    cp, sp = (math.cos(st.get('pitch', 0)), math.sin(st.get('pitch', 0)))
    cy, sy = (math.cos(st.get('yaw', math.pi / 2)), math.sin(st.get('yaw', math.pi / 2)))
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy], [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy], [-sp, sr * cp, cr * cp]])

class Coverage:

    def __init__(self):
        self.cells = np.array([(x, y) for x in range(-3, 44, 2) for y in range(-23, 24, 2) if all((not blocked(x + dx, y + dy, 2.9) for dx in [-1, 1] for dy in [-1, 1]))], dtype=float)
        self.seen = np.zeros(len(self.cells), dtype=int)
        self.last_frame = -1
        self.covers = []
        self.home = (0.0, 0.0)
        self.future = None
        self.queue = []
        self.active_view = None
        self.skipped_views = 0
        self.plan_stats = {}
        self.generation = 0
        self.future_generation = 0
        global _FOOTPRINTS
        if _FOOTPRINTS is None:
            _FOOTPRINTS = [self.visible_indices({'east': p[0], 'north': p[1], 'height': 7.0, 'yaw': math.pi / 2}) for p in self.cells]
        self.covers = _FOOTPRINTS

    def reset(self):
        self.seen[:] = 0
        self.last_frame = -1
        self.replan()

    def replan(self):
        self.generation += 1
        self.queue = []
        self.active_view = None
        if self.future is not None:
            self.future.cancel()
        self.future = None

    @property
    def planning(self):
        return self.future is not None

    def target_satisfied(self):
        return self.active_view is not None and bool(np.all(self.seen[self.covers[self.active_view]] >= 3))

    @property
    def complete(self):
        return bool(np.all(self.seen >= 3))

    def observe(self, st, frame):
        if frame == self.last_frame:
            return
        self.last_frame = frame
        if not 6.6 < st.get('height', 0) < 7.5 or max(abs(st.get('roll', 0)), abs(st.get('pitch', 0))) > math.radians(6):
            return
        ids = self.visible_indices(st)
        self.seen[ids] = np.minimum(3, self.seen[ids] + 1)

    def visible_indices(self, st):
        corners = self.cells[:, None, :] + np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])[None, :, :]
        ned = np.stack([corners[:, :, 1] - st['north'], corners[:, :, 0] - st['east'], np.full(corners.shape[:2], st['height'] - 0.1)], axis=-1)
        body = ned @ rotation(st)
        f = 640 / (2 * math.tan(0.7))
        u = 320 + f * body[:, :, 1] / body[:, :, 2]
        v = 240 - f * body[:, :, 0] / body[:, :, 2]
        mask = np.all((body[:, :, 2] > 1) & (u > 32) & (u < 608) & (v > 24) & (v < 456), axis=1)
        return np.array([i for i in np.flatnonzero(mask) if visible((st['east'], st['north']), self.cells[i], margin=0)], dtype=int)

    def sees(self, point, st):
        ned = np.array([point[1] - st['north'], point[0] - st['east'], st['height'] - 0.1])
        b = rotation(st).T @ ned
        f = 640 / (2 * math.tan(0.7))
        return b[2] > 1 and abs(f * b[1] / b[2]) < 220 and (abs(f * b[0] / b[2]) < 150)

    def next(self, position):
        if self.complete:
            if self.future is not None:
                self.future.cancel()
            self.future = None
            self.queue = []
            self.active_view = None
            return None
        if self.future is not None:
            if not self.future.done():
                return None
            result = self.future.result()
            self.future = None
            self.queue = result.pop('views')
            self.plan_stats = result
        while self.queue:
            i = self.queue.pop(0)
            if np.all(self.seen[self.covers[i]] >= 3):
                self.skipped_views += 1
                continue
            self.active_view = i
            return tuple(self.cells[i])
        self.future_generation = self.generation
        self.future = POOL.submit(optimize, self.cells.copy(), self.covers, (self.seen < 3).copy(), tuple(position), tuple(self.home))
        self.active_view = None
        return None

    def state(self):
        return {'percent': round(100 * float(np.mean(self.seen >= 3)), 1), 'observed': int(np.sum(self.seen >= 3)), 'total': len(self.cells), 'cells': [[float(p[0]), float(p[1]), int(n >= 3)] for p, n in zip(self.cells, self.seen)], 'bounds': BOUNDS, 'planning': self.planning, 'remaining_views': len(self.queue), 'skipped_views': self.skipped_views, 'plan': dict(self.plan_stats)}
