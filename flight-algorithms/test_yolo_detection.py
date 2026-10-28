import threading
import unittest
from yolo_detection import YoloBridge, project_detections

class YoloTests(unittest.TestCase):

    def box(self, **kw):
        return dict({'class': 'fire_marker', 'confidence': 0.9, 'xyxy': [300, 220, 340, 260]}, **kw)

    def test_invalid_and_unknown_classes_are_rejected(self):
        project = lambda *a: {'east': 1, 'north': 2}
        boxes = [self.box(**{'class': 'person'}), self.box(confidence=0.1), self.box(xyxy=[10, 10, 5, 5]), self.box(xyxy=[float('nan'), 0, 4, 4])]
        self.assertEqual(project_detections(boxes, (480, 640, 3), {}, project), [])

    def test_projection_uses_capture_telemetry_and_marker_plane(self):
        seen = []

        def project(u, v, h, st):
            seen.append((u, v, h, st['east']))
            return {'east': st['east'], 'north': st['north']}
        out = project_detections([self.box()], (480, 640, 3), {'height': 7, 'east': 10, 'north': 20}, project)
        self.assertEqual(seen, [(320.0, 240.0, 5.6000000000000005, 10)])
        self.assertEqual(out[0]['label'], 'Fire')
        self.assertEqual(out[0]['detector'], 'yolo')

    def test_station_and_clipped_box(self):
        out = project_detections([self.box(**{'class': 'water_station', 'xyxy': [-10, -10, 40, 40]})], (480, 640, 3), {}, lambda *a: {'east': 0, 'north': 0})
        self.assertEqual(out[0]['bbox'], [0.0, 0.0, 40.0, 40.0])
        self.assertEqual(out[0]['label'], 'Station')

    def test_front_horizon_is_not_projected(self):
        self.assertEqual(project_detections([self.box(xyxy=[300, 0, 340, 20])], (360, 640, 3), {'height': 7}, lambda *a: None, front=True), [])

    def test_missing_weights_do_not_start_worker(self):
        bridge = YoloBridge(python='/missing/python', weights='/missing/model.pt')
        self.assertEqual(bridge.state()['status'], 'unavailable')
        self.assertFalse(bridge.state()['mission_authority'])
        self.assertIsNone(bridge.snapshot('down'))
        bridge.close()

    def test_stale_results_are_never_reused(self):
        bridge = YoloBridge(python='/missing', weights='/missing')
        bridge.results['down'] = {'captured': 10.0, 'detections': []}
        self.assertIsNotNone(bridge.snapshot('down', now=10.5))
        self.assertIsNone(bridge.snapshot('down', now=10.81))
        self.assertIsNone(bridge.snapshot('down', now=9.9))
        bridge.close()
if __name__ == '__main__':
    unittest.main()
