import unittest
import numpy as np
from tracking import KalmanTrack, DetectionWindow
from survey_mission import SurveyMission
from types import SimpleNamespace

class TrackingTests(unittest.TestCase):

    def test_kalman_reduces_noisy_constant_velocity_error(self):
        rng = np.random.default_rng(42)
        k = KalmanTrack([0, 0], 0)
        raw = []
        filtered = []
        for i in range(1, 161):
            t = i * 0.125
            truth = np.array([0.7 * t, -0.3 * t])
            z = truth + rng.normal(0, 0.3, 2)
            k.update(z, t)
            if i > 20:
                raw.append(np.linalg.norm(z - truth))
                filtered.append(np.linalg.norm(k.x[:2] - truth))
        self.assertLess(np.mean(np.square(filtered)), np.mean(np.square(raw)) * 0.6)
        np.testing.assert_allclose(k.x[2:], [0.7, -0.3], atol=0.18)
        np.testing.assert_allclose(k.P, k.P.T, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(k.P).min(), 0)

    def test_kalman_gap_resets_old_velocity(self):
        k = KalmanTrack([0, 0], 0)
        k.update([0.2, 0], 0.1)
        k.update([12, 4], 2)
        np.testing.assert_equal(k.x, [12, 4, 0, 0])

    def test_window_ignores_duplicate_frames_and_expires(self):
        w = DetectionWindow()
        self.assertEqual(w.update(1, 0, True), 1)
        self.assertEqual(w.update(1, 0.01, True), 1)
        self.assertEqual(w.update(2, 0.1, True), 2)
        self.assertEqual(w.update(3, 2, True), 1)

    def test_window_three_of_five_and_negative_evidence(self):
        w = DetectionWindow()
        for i, p in enumerate([True, False, True, False, True]):
            hits = w.update(i, i * 0.1, p)
        self.assertEqual(hits, 3)
        self.assertEqual(w.update(5, 0.5, False), 2)

    def test_fire_multiple_blobs_not_multiple_votes(self):
        m = SurveyMission()
        c = SimpleNamespace(t={}, log=lambda s: None)
        m.coverage.observe = lambda *a: None
        d = {'label': 'Fire', 'bbox': [100, 100, 20, 20], 'east': 3.0, 'north': 4.0, 'area': 100}
        for frame, now, ds in [(1, 0, [d, d, d]), (2, 0.1, []), (3, 0.2, [d]), (4, 0.3, []), (5, 0.4, [d])]:
            m.perceive(c, dict(frames=frame, detections=ds), now)
            if frame == 1:
                self.assertEqual(m.fires[0]['status'], 'CANDIDATE')
        self.assertEqual(m.fires[0]['status'], 'ACTIVE')

    def test_sparse_fire_sightings_never_accumulate_forever(self):
        m = SurveyMission()
        c = SimpleNamespace(t={}, log=lambda s: None)
        m.coverage.observe = lambda *a: None
        d = {'label': 'Fire', 'bbox': [100, 100, 20, 20], 'east': 3.0, 'north': 4.0, 'area': 100}
        for i in range(5):
            m.perceive(c, dict(frames=i, detections=[d]), i * 2.0)
        self.assertEqual(m.fires[0]['status'], 'CANDIDATE')
if __name__ == '__main__':
    unittest.main()
