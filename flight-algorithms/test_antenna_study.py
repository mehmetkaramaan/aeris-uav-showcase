import unittest
import numpy as np
from antenna_study import fspl, pattern, evaluate, CONFIGS, ENVS, recommend

class AntennaStudyTests(unittest.TestCase):

    def route(self):
        n = 100
        return dict(t=np.arange(n) * 0.5, east=np.linspace(2, 50, n), north=np.linspace(-10, 10, n), height=np.full(n, 7.0), roll=np.zeros(n), pitch=np.zeros(n))

    def test_fspl_reference_and_distance_scaling(self):
        self.assertAlmostEqual(float(fspl(1000, 2400)), 100.054, places=2)
        self.assertAlmostEqual(float(fspl(200, 868) - fspl(100, 868)), 6.0206, places=4)

    def test_directional_backside_penalty_and_omni_symmetry(self):
        az = np.array([0.0, 180.0])
        el = np.zeros(2)
        p = pattern(CONFIGS[3], az, el)
        self.assertGreater(p[0] - p[1], 20)
        np.testing.assert_equal(*pattern(CONFIGS[1], az, el))

    def test_repeatable_shared_seed_and_power_shift(self):
        r = self.route()
        a = evaluate(r, CONFIGS[0], ENVS[0])
        b = evaluate(r, CONFIGS[0], ENVS[0])
        np.testing.assert_equal(a['margin'], b['margin'])
        np.testing.assert_allclose(evaluate(r, CONFIGS[0], ENVS[0], eirp=23)['margin'] - a['margin'], 3)

    def test_shadow_loss_reduces_margin_with_identical_fading(self):
        r = self.route()
        env = dict(ENVS[0], obstruction=20.0)
        a = evaluate(r, CONFIGS[0], ENVS[0])
        b = evaluate(r, CONFIGS[0], env)
        self.assertTrue(np.all(b['margin'] <= a['margin']))
        self.assertLess(b['margin'].min(), a['margin'].min())

    def test_recommendation_can_refuse_all(self):
        self.assertIsNone(recommend([dict(coverage_min=98, outage_max=0)]))

    def test_recommendation_prefers_simple_eligible(self):
        rows = [dict(id='a', coverage_min=100, outage_max=0, cost=1, p05_mean=20), dict(id='b', coverage_min=100, outage_max=0, cost=2, p05_mean=30)]
        self.assertEqual(recommend(rows), 'a')
if __name__ == '__main__':
    unittest.main()
