import unittest, numpy as np, math, time
from types import SimpleNamespace
from trajectory import Trajectory
from controller import Controller

class FlightSafetyTests(unittest.TestCase):

    def test_reversal_has_bounded_acceleration_and_jerk(self):
        p = Trajectory()
        rows = []
        for i in range(200):
            v = p.step([1, 0] if i < 100 else [-1, 0], {}, now=1 + i * 0.05)
            rows.append((v, p.acceleration.copy()))
        self.assertLessEqual(max((np.linalg.norm(a) for v, a in rows)), 0.350001)
        self.assertLessEqual(max((np.linalg.norm(rows[i][1] - rows[i - 1][1]) / 0.05 for i in range(1, len(rows)))), 0.700001)

    def test_gyro_tilt_guard_has_hysteresis(self):
        p = Trajectory()
        p.step([1, 0], {'rollspeed': 1.0})
        self.assertTrue(p.stabilizing)
        p.step([1, 0], {'roll': math.radians(7)})
        self.assertTrue(p.stabilizing)
        p.step([1, 0], {})
        self.assertFalse(p.stabilizing)

    def test_fire_setpoint_holds_height_and_ignores_vertical_velocity(self):
        c = Controller.__new__(Controller)
        c.t = {'east': 20, 'north': 20, 'height': 4}
        c.speed = 0.6
        c.task = 'EXTINGUISH'
        c.avoid_enabled = False
        c.ground_up = 0.2
        c.trajectory = Trajectory()
        calls = []
        c.m = SimpleNamespace(mav=SimpleNamespace(set_position_target_local_ned_send=lambda *a: calls.append(a)))
        c.setpoint(20, 20, 2)
        a = calls[0]
        self.assertEqual(a[4], 2531)
        self.assertEqual(a[7], -7.2)
        self.assertEqual(a[10], 0)

    def test_drop_rejects_low_altitude_and_nearby_traffic(self):
        c = Controller.__new__(Controller)
        c.last_drop = 0
        c.water = 100
        c.require_air = lambda: None
        c.t = {'height': 3}
        with self.assertRaises(ValueError):
            c.drop()
        c.t = {'height': 7}
        c.vision = SimpleNamespace(safety=SimpleNamespace(nearest=2))
        c.drop()
        self.assertEqual(c.water, 100)

    def test_global_route_clearance_matches_local_surface_margin(self):
        from planning import shortest_path, OBSTACLES
        route = [(20, 20)] + shortest_path((20, 20), (0, 0))
        for a, b in zip(route, route[1:]):
            for t in np.linspace(0, 1, 60):
                x, y = np.asarray(a) * (1 - t) + np.asarray(b) * t
                for x0, y0, x1, y1 in OBSTACLES:
                    distance = math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
                    self.assertGreater(distance, 2.8)

    def test_detour_position_gets_a_new_static_clear_route(self):
        from planning import visible, shortest_path
        position = (7.95, 4.6)
        self.assertFalse(visible(position, (40, 0)))
        route = shortest_path(position, (40, 0))
        self.assertGreater(len(route), 1)
        self.assertLess(route[0][1], 4.6)
        self.assertTrue(visible(position, route[0]))

    def test_buffer_penetration_has_monotonic_escape_route(self):
        from planning import shortest_path, clearance_ok
        start = (8.2, 4.2)
        route = shortest_path(start, (40, 0))
        self.assertGreater(len(route), 1)
        self.assertTrue(route[0][0] < 8.2 or route[0][1] < 4.2)
        samples = [np.array(start) * (1 - t) + np.array(route[0]) * t for t in np.linspace(0, 1, 30)]
        self.assertTrue(clearance_ok(start, samples, margin=2.9))

    def test_oncoming_traffic_plans_open_southern_side(self):
        from avoidance import Avoidance
        from planning import shortest_path
        a = Avoidance()
        a.last = time.monotonic()
        a.tracks = [{'center': np.array([21.0, 0.0]), 'velocity': np.array([-0.6, 0.0]), 'span': 0.5, 'age': 10}]
        zones = a.detour((5, 0), (40, 0), 0.65)
        self.assertTrue(zones)
        route = shortest_path((5, 0), (40, 0), zones)
        self.assertLess(min((p[1] for p in route)), -3)
if __name__ == '__main__':
    unittest.main()
