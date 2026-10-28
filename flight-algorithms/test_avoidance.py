import unittest, time
import numpy as np
from avoidance import Avoidance
from planning import astar

class AvoidanceTests(unittest.TestCase):

    def sensor(self, points, vel=None):
        a = Avoidance()
        a.last = time.monotonic()
        a.points = np.asarray(points, dtype=float).reshape(-1, 2)
        a.vel = np.zeros_like(a.points) if vel is None else np.asarray(vel, dtype=float)
        return a

    def test_clear_path_preserves_velocity(self):
        a = self.sensor([])
        a.chosen = [0.7, 0]
        np.testing.assert_allclose(a.choose([0, 0], [0.7, 0]), [0.7, 0])

    def test_head_on_rejected(self):
        a = self.sensor([[4, 0]], [[-0.8, 0]])
        v = a.choose([0, 0], [1, 0])
        self.assertGreater(np.linalg.norm(v - [1, 0]), 0.15)
        ts = np.linspace(0.25, 3, 8)
        self.assertGreaterEqual(min((np.linalg.norm([4, 0] + np.array([-0.8, 0]) * t - v * t) for t in ts)), 1.8)

    def test_crossing_rejected(self):
        a = self.sensor([[2, -2]], [[0, 1]])
        v = a.choose([0, 0], [1, 0])
        self.assertTrue(a.active)
        self.assertTrue(np.isfinite(v).all())

    def test_stale_scan_brakes(self):
        a = self.sensor([])
        a.last = 0
        np.testing.assert_equal(a.choose([0, 0], [1, 0]), [0, 0])

    def test_vertical_contacts_filtered_and_frame_rotated(self):
        from types import SimpleNamespace
        scan = SimpleNamespace(count=3, vertical_count=1, ranges=[5.0, 5.0, 5.0], angle_min=-0.1, angle_max=0.1, vertical_angle_min=0.0, vertical_angle_max=0.0)
        a = Avoidance()
        a.update(scan, dict(east=0, north=0, height=4, yaw=np.pi / 2))
        self.assertTrue(np.all(a.points[:, 0] > 4.9))
        self.assertTrue(np.all(np.abs(a.points[:, 1]) < 0.6))
        scan.vertical_angle_min = scan.vertical_angle_max = 1.0
        a.update(scan, dict(east=0, north=0, height=4, yaw=np.pi / 2))
        self.assertEqual(len(a.points), 0)

    def test_nofly_global_route(self):
        zones = [(15, 0, 4)]
        route = astar((0, 0), (40, 0), zones)
        self.assertTrue(all((np.linalg.norm(np.asarray(p) - [15, 0]) >= 5.1 for p in route)))

    def test_small_measurement_noise_preserves_passing_side(self):
        a = self.sensor([[5.0, 0]], [[-0.8, 0]])
        sides = []
        for i in range(60):
            a.points = np.array([[5.0, 0.015 * (-1) ** i]])
            a.last = time.monotonic()
            a.decision_time = time.monotonic() - 0.05
            v = a.choose([0, 0], [0.7, 0])
            if abs(v[1]) > 0.05:
                sides.append(np.sign(v[1]))
        self.assertTrue(sides)
        self.assertEqual(len(set(sides)), 1)

    def test_clear_path_accelerates_without_step(self):
        from trajectory import Trajectory
        a = self.sensor([])
        target = a.choose([0, 0], [0.7, 0])
        v = Trajectory().step(target, {})
        self.assertGreater(v[0], 0)
        self.assertLessEqual(np.linalg.norm(v), 0.033)

    def test_visible_direct_and_obstacle_detour(self):
        from planning import shortest_path, visible
        self.assertEqual(shortest_path((0, 0), (40, 0)), [(40, 0)])
        route = [(0, 0)] + shortest_path((0, 0), (20, 20))
        self.assertTrue(all((visible(a, b) for a, b in zip(route, route[1:]))))
        self.assertLess(len(route), 8)
if __name__ == '__main__':
    unittest.main()
