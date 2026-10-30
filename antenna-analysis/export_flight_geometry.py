import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = ['t', 'east', 'north', 'height', 'roll', 'pitch']


def extract_geometry(recording):
    rows = []
    previous_time = -math.inf

    for index, record in enumerate(recording['records']):
        telemetry = record['telemetry']
        row = {'t': float(record['elapsed'])}

        for field in FIELDS[1:]:
            row[field] = float(telemetry[field])

        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f'Non-finite measurement at record {index}')
        if row['t'] <= previous_time:
            raise ValueError(f'Non-increasing timestamp at record {index}')

        previous_time = row['t']
        rows.append(row)

    if not rows:
        raise ValueError('The recording contains no samples')
    return rows


def export_recording(source, destination):
    recording = json.loads(Path(source).read_text(encoding='utf-8'))
    rows = extract_geometry(recording)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, default=Path('flight_geometry_raw.csv'))
    args = parser.parse_args()
    count = export_recording(args.recording, args.output)
    print(f'Exported {count} local-coordinate samples to {args.output}')
