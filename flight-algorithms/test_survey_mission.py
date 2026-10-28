import unittest, math
from types import SimpleNamespace
import numpy as np, cv2
from coverage import Coverage
from survey_mission import SurveyMission
from perception import landmarks

class SurveyTests(unittest.TestCase):

    def controller(self):
        c = SimpleNamespace(t={'east': 3.0, 'north': -11.0, 'height': 7.0, 've': 0.0, 'vn': 0.0, 'yaw': math.pi / 2}, route=[], target=None, water=100, last_drop=-100, height=7.0, task='SURVEY', speed=1.0, events=[], effects=[])
        c.vision = SimpleNamespace(safety=SimpleNamespace(nearest=None), truth=(3, -11, 3))
        c.log = c.events.append
        c.effect = c.effects.append
        c.begin_return = lambda text: c.events.append('RETURN ' + text)

        def plan(e, n, task):
            c.route = [(e, n)]
            c.target = {'east': e, 'north': n}
            c.task = task
        c.plan = plan
        return c

    def vision(self, n, ds=[]):
        return {'frames': n, 'detections': ds, 'front_landmarks': []}

    def mission(self):
        m = SurveyMission()
        m.active = True
        m.mode = 'fire'
        m.pass_number = 1
        return m

    def test_coverage_requires_distinct_observed_frames(self):
        c = self.controller()
        g = Coverage()
        g.observe(c.t, 1)
        g.observe(c.t, 1)
        self.assertEqual(int(g.seen.max()), 1)
        g.observe(c.t, 2)
        g.observe(c.t, 3)
        self.assertGreater(g.state()['observed'], 0)

    def test_camera_loss_does_not_create_coverage(self):
        c = self.controller()
        c.t['height'] = 3
        g = Coverage()
        g.observe(c.t, 1)
        self.assertEqual(int(g.seen.sum()), 0)

    def test_refill_requires_world_confirmation(self):
        c = self.controller()
        c.t['height'] = 3
        c.water = 0
        m = self.mission()
        m.phase = 'REFILL'
        m.station = {'east': 3, 'north': -11}
        m.tick(c, self.vision(1), 1)
        m.tick(c, self.vision(2), 5)
        self.assertEqual(c.water, 0)
        self.assertEqual(c.effects[0]['action'], 'refill')
        m.refill_result = True
        m.tick(c, self.vision(3), 6)
        self.assertEqual(c.water, 100)
        self.assertEqual(m.refills, 1)

    def test_extinguished_target_keeps_observed_area_and_continues(self):
        c = self.controller()
        m = self.mission()
        m.phase = 'EXTINGUISH'
        m.current = {'id': 1, 'east': 3, 'north': -11, 'status': 'ACTIVE', 'hits': 3}
        m.fires = [m.current]
        m.coverage.seen[:10] = 3
        m.tick(c, self.vision(1), 10)
        m.tick(c, self.vision(2), 13)
        self.assertEqual(m.fires[0]['status'], 'CLEARED')
        self.assertEqual(m.phase, 'SURVEY')
        self.assertTrue(np.all(m.coverage.seen[:10] >= 3))
        self.assertTrue(m.active)
        self.assertFalse(any((e.startswith('RETURN') for e in c.events)))

    def test_nearby_traffic_prevents_false_extinguish_confirmation(self):
        c = self.controller()
        c.vision.safety.nearest = 2
        m = self.mission()
        m.phase = 'EXTINGUISH'
        m.current = {'id': 1, 'east': 3, 'north': -11, 'status': 'ACTIVE', 'hits': 3}
        m.fires = [m.current]
        m.tick(c, self.vision(1), 10)
        m.tick(c, self.vision(2), 20)
        self.assertEqual(m.current['status'], 'ACTIVE')

    def test_full_coverage_returns_only_without_known_active_fire(self):
        c = self.controller()
        m = self.mission()
        m.phase = 'SURVEY'
        m.coverage.seen[:] = 3
        m.tick(c, self.vision(1), 10)
        self.assertEqual(m.outcome, 'COMPLETE')
        self.assertFalse(m.active)
        self.assertTrue(any((e.startswith('RETURN') for e in c.events)))
        c = self.controller()
        m = self.mission()
        m.phase = 'SURVEY'
        m.coverage.seen[:] = 3
        m.fires = [{'id': 1, 'east': 3, 'north': -11, 'status': 'ACTIVE', 'hits': 3}]
        m.tick(c, self.vision(1), 10)
        self.assertEqual(m.phase, 'FIRE_APPROACH')
        self.assertTrue(m.active)

    def test_empty_tank_returns_to_observed_station_not_home(self):
        c = self.controller()
        c.water = 0
        m = self.mission()
        m.phase = 'SURVEY'
        m.station = {'east': -1, 'north': -15}
        m.tick(c, self.vision(1), 10)
        self.assertEqual(m.phase, 'STATION_APPROACH')
        self.assertEqual(c.target, {'east': -1, 'north': -15})

    def test_station_pattern_and_fire_color_are_distinct_from_traffic(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 150), (300, 250), (255, 255, 0), -1)
        cv2.line(image, (220, 170), (280, 230), (255, 0, 255), 10)
        ds = landmarks(image, {'height': 7}, lambda *a: {'east': 0, 'north': 0})
        self.assertTrue(any((d['label'] == 'Station' for d in ds)))
        image[:] = 0
        cv2.rectangle(image, (200, 150), (300, 250), (30, 220, 245), -1)
        ds = landmarks(image, {'height': 7}, lambda *a: {'east': 0, 'north': 0})
        self.assertFalse(any((d['label'] == 'Fire' for d in ds)))

    def test_no_observation_route_is_not_reported_as_full_coverage(self):
        c = self.controller()
        m = self.mission()
        m.coverage.next = lambda p: None
        m.scan(c)
        self.assertEqual(m.outcome, 'INCOMPLETE')

    def test_reconnaissance_does_not_extinguish_known_targets(self):
        c = self.controller()
        c.water = 0
        m = self.mission()
        m.mode = 'survey'
        m.phase = 'SURVEY'
        m.fires = [{'id': 1, 'east': 3, 'north': -11, 'status': 'ACTIVE', 'hits': 3}]
        m.coverage.seen[:] = 3
        m.tick(c, self.vision(1), 10)
        self.assertEqual(m.fires[0]['status'], 'ACTIVE')
        self.assertEqual(m.outcome, 'COMPLETE')
        self.assertEqual(c.effects, [])

    def test_reconnaissance_does_not_revisit_the_same_front_camera_cue(self):
        c = self.controller()
        m = self.mission()
        m.mode = 'survey'
        m.phase = 'SURVEY'
        m.fires = [{'id': 1, 'east': 3, 'north': -11, 'status': 'ACTIVE', 'hits': 3}]
        v = self.vision(1)
        v['front_landmarks'] = [{'label': 'Fire', 'east': 3, 'north': -11}]
        m.tick(c, v, 10)
        self.assertIsNone(m.cue)

    def test_fully_observed_viewpoint_is_skipped_before_physical_arrival(self):
        c = self.controller()
        m = self.mission()
        m.phase = 'SURVEY'
        i = 0
        m.coverage.active_view = i
        m.coverage.seen[m.coverage.covers[i]] = 3
        c.route = [tuple(m.coverage.cells[i])]
        m.tick(c, self.vision(1), 10)
        self.assertGreater(m.coverage.skipped_views, 0)
        self.assertTrue(m.active)
        self.assertNotEqual(c.route, [tuple(m.coverage.cells[i])])

    def test_planning_never_marks_predicted_cells_observed(self):
        g = Coverage()
        self.assertIsNone(g.next((0.0, 0.0)))
        g.future.result(timeout=30)
        self.assertIsNotNone(g.next((0.0, 0.0)))
        self.assertEqual(int(g.seen.sum()), 0)
        self.assertFalse(g.complete)

    def test_refill_does_not_forget_camera_coverage(self):
        c = self.controller()
        c.t['height'] = 3
        c.water = 0
        m = self.mission()
        m.phase = 'REFILL'
        m.station = {'east': 3, 'north': -11}
        m.coverage.seen[:20] = 3
        m.refill_result = True
        m.tick(c, self.vision(1), 10)
        self.assertTrue(np.all(m.coverage.seen[:20] >= 3))
        self.assertEqual(c.water, 100)
if __name__ == '__main__':
    unittest.main()
