import argparse
import csv
import json
import math
import time
import urllib.request
from pathlib import Path

FIELDS = ['t', 'east', 'north', 'height', 'roll', 'pitch']
STATE_URL = 'http://127.0.0.1:8772/api/state'


def read_telemetry():
    with urllib.request.urlopen(STATE_URL, timeout=3) as response:
        state = json.load(response)

    if not state.get('connected'):
        raise RuntimeError('The local simulator is not connected to PX4')

    telemetry = state['telemetry']
    values = {}
    for field in FIELDS[1:]:
        value = float(telemetry[field])
        if not math.isfinite(value):
            raise ValueError(f'Invalid telemetry field: {field}')
        values[field] = value
    return values


def record_geometry(destination, seconds, interval):
    if seconds <= 0 or interval <= 0:
        raise ValueError('Duration and interval must be positive')

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    next_sample = started
    count = 0

    with destination.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()

        try:
            while time.monotonic() - started < seconds:
                remaining = next_sample - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

                if time.monotonic() - started >= seconds:
                    break

                telemetry = read_telemetry()
                elapsed = round(time.monotonic() - started, 6)
                writer.writerow({'t': elapsed, **telemetry})
                stream.flush()
                count += 1
                next_sample += interval
                if next_sample < time.monotonic():
                    next_sample = time.monotonic()
        except KeyboardInterrupt:
            pass

    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--output', type=Path, default=Path('flight_geometry_raw.csv'))
    args = parser.parse_args()
    count = record_geometry(args.output, args.seconds, args.interval)
    print(f'Recorded {count} samples to {args.output}')
