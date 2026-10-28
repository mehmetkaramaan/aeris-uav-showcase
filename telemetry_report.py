"""Analyse the bundled, sanitised 1 Hz simulation telemetry sample.

No networking or vehicle commands. Standard library only.
Run: python telemetry_report.py [sample-telemetry.csv]
"""
import csv
import json
import math
import sys
from pathlib import Path


def summarize(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Telemetry file contains no observations.')
    numeric = ['elapsed_s', 'east_m', 'north_m', 'altitude_m',
               'ground_speed_ms', 'coverage_percent', 'water_percent']
    values = [{k: float(row[k]) for k in numeric} for row in rows]
    if any(not math.isfinite(v) for row in values for v in row.values()):
        raise ValueError('Telemetry must contain finite numbers.')
    if any(b['elapsed_s'] <= a['elapsed_s'] for a, b in zip(values, values[1:])):
        raise ValueError('Observation timestamps must increase.')
    distance = sum(math.hypot(b['east_m'] - a['east_m'],
                              b['north_m'] - a['north_m'])
                   for a, b in zip(values, values[1:]))
    return {
        'samples': len(rows),
        'duration_s': values[-1]['elapsed_s'] - values[0]['elapsed_s'],
        'sampled_horizontal_distance_m': round(distance, 3),
        'max_altitude_m': max(r['altitude_m'] for r in values),
        'max_ground_speed_ms': max(r['ground_speed_ms'] for r in values),
        'final_coverage_percent': values[-1]['coverage_percent'],
        'coverage_monotonic': all(b['coverage_percent'] >= a['coverage_percent']
                                  for a, b in zip(values, values[1:])),
        'final_water_percent': values[-1]['water_percent'],
        'note': 'Distance is approximated from 1 Hz samples; the full-rate controller log is authoritative.'
    }


if __name__ == '__main__':
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('sample-telemetry.csv')
    print(json.dumps(summarize(source), indent=2))
