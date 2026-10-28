import unittest, time, math
from types import SimpleNamespace
from planning import astar, blocked
from vision import Vision
from controller import Controller

class MissionTests(unittest.TestCase):

    def test_astar_clearance_and_segments(self):
        route = [(0, 0)] + astar((0, 0), (20, 20))
        self.assertEqual(route[-1], (20, 20))
        for a, b in zip(route, route[1:]):
            for k in range(21):
                self.assertFalse(blocked(a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20))

    def test_reject_obstacle_goal(self):
        with self.assertRaises(ValueError):
            astar((0, 0), (15, 9))

    def test_camera_projection_ned_axes(self):
        v = Vision.__new__(Vision)
        st = {'east': 10, 'north': 20, 'yaw': math.pi / 2}
        self.assertAlmostEqual(v.project(320, 240, 5, st)['east'], 10)
        self.assertLess(v.project(420, 240, 5, st)['north'], 20)
        self.assertGreater(v.project(320, 140, 5, st)['east'], 10)

    def test_airborne_disarm_rejected(self):
        c = Controller.__new__(Controller)
        c.hb = time.monotonic()
        c.t = {'landed': 2}
        with self.assertRaisesRegex(ValueError, 'Airborne'):
            c.handle('disarm', {})

    def watchdog_fixture(self, now):
        c = Controller.__new__(Controller)
        c.task = 'SURVEY'
        c.pos_time = now
        c.hb = now
        c.t = {'landed': 2, 'armed': True, 'battery': 90}
        c.task_deadline = now + 20
        c.sim_paused_at = 0
        c.resume_until = 0
        c.data_hold = False
        c.data_stale_since = 0
        c.height = 7
        c.mission = SimpleNamespace(absent_since=3, stable_since=3)
        c.publish_velocity = lambda *a: None
        c.log = lambda *a: None
        return c

    def test_old_telemetry_lands_after_controlled_wait(self):
        now = time.monotonic()
        c = self.watchdog_fixture(now)
        c.pos_time = 0
        c.vision = SimpleNamespace(state=lambda: {'age': 0, 'sim_age': 0})
        calls = []
        c.cancel = lambda *x: calls.append(x)
        c.tick(now)
        self.assertEqual(calls, [])
        self.assertTrue(c.data_hold)
        c.tick(now + 2.1)
        self.assertEqual(calls[0][1], 'LAND')

    def test_camera_loss_returns_after_controlled_wait(self):
        now = time.monotonic()
        c = self.watchdog_fixture(now)
        c.vision = SimpleNamespace(state=lambda: {'age': 4, 'sim_age': 0})
        calls = []
        c.begin_return = lambda *x: calls.append(x)
        c.tick(now)
        self.assertEqual(calls, [])
        c.pos_time = c.hb = now + 2.1
        c.tick(now + 2.1)
        self.assertIn('Camera', calls[0][0])

    def test_paused_simulator_does_not_become_a_sensor_failure(self):
        now = time.monotonic()
        c = self.watchdog_fixture(now)
        c.pos_time = 0
        c.vision = SimpleNamespace(state=lambda: {'age': 5, 'sim_age': 5})
        calls = []
        c.cancel = lambda *x: calls.append(x)
        c.tick(now)
        c.tick(now + 10)
        self.assertEqual(calls, [])
        self.assertTrue(c.data_hold)
        self.assertEqual(c.mission.absent_since, 0)

    def test_drop_does_not_block_on_gazebo_service(self):
        c = Controller.__new__(Controller)
        c.last_drop = 0
        c.water = 100
        c.t = {'height': 7}
        c.require_air = lambda: None
        c.vision = SimpleNamespace(truth=(20, 20, 7), safety=SimpleNamespace(nearest=None))
        effects = []
        c.effect = effects.append
        c.log = lambda x: None
        c.drop()
        self.assertEqual(c.water, 75)
        self.assertEqual(effects[0]['action'], 'drop')
        self.assertNotIn('markers', effects[0])

    def test_return_uses_recorded_takeoff_location(self):
        c = Controller.__new__(Controller)
        c.hb = time.monotonic()
        c.height = 4
        c.require_air = lambda: None
        c.log = lambda x: None
        calls = []
        c.plan = lambda *x: calls.append(x)
        c.mission = SimpleNamespace(active=False)
        c.t = {'east': 20, 'north': 10}
        c.route = [(2, 3)]
        c.home = {'east': 2, 'north': 3}
        c.flown_distance = 30
        c.handle('rtl', {})
        self.assertEqual(calls, [(2, 3, 'RETURN_HOME')])
        self.assertAlmostEqual(c.return_distance, math.hypot(18, 7))

    def test_visibility_route_respects_nofly_segments(self):
        from planning import shortest_path, visible
        zones = [(15, 0, 4)]
        route = [(0, 0)] + shortest_path((0, 0), (40, 0), zones)
        self.assertGreater(len(route), 2)
        self.assertTrue(all((visible(a, b, zones) for a, b in zip(route, route[1:]))))
if __name__ == '__main__':
    unittest.main()
