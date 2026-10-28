"""Tests for the independent public samples, not for the private flight system."""
import math
import tempfile
import unittest
from pathlib import Path
from astar_demo import astar
from telemetry_report import summarize


class RouteTests(unittest.TestCase):
    def test_unobstructed_diagonal_is_shortest(self):
        path = astar((0, 0), (4, 4), set(), 5, 5)
        self.assertAlmostEqual(sum(math.dist(a, b) for a, b in zip(path, path[1:])), 4*math.sqrt(2))

    def test_diagonal_does_not_cut_blocked_corner(self):
        with self.assertRaises(ValueError):
            astar((0, 0), (1, 1), {(0, 1), (1, 0)}, 2, 2)

    def test_route_passes_through_only_open_gap(self):
        wall = {(2, y) for y in range(5) if y != 3}
        path = astar((0, 0), (4, 0), wall, 5, 5)
        self.assertIn((2, 3), path)
        self.assertTrue(all(p not in wall for p in path))

    def test_blocked_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            astar((0, 0), (2, 2), {(2, 2)}, 3, 3)


class TelemetryTests(unittest.TestCase):
    def test_known_distance_and_decreasing_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.csv'
            path.write_text('elapsed_s,east_m,north_m,altitude_m,ground_speed_ms,coverage_percent,water_percent\n'
                            '0,0,0,7,1,20,100\n1,3,4,7,1,10,75\n')
            report = summarize(path)
            self.assertEqual(report['sampled_horizontal_distance_m'], 5)
            self.assertFalse(report['coverage_monotonic'])

    def test_bundled_flight_reaches_full_coverage(self):
        report = summarize(Path(__file__).with_name('sample-telemetry.csv'))
        self.assertEqual(report['final_coverage_percent'], 100)
        self.assertTrue(report['coverage_monotonic'])


if __name__ == '__main__':
    unittest.main()
