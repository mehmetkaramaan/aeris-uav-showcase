from pathlib import Path
import sys, json, time
import cv2, numpy as np
from ultralytics import YOLO
root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perception import landmarks
model = YOLO(str(root / 'runs/markers/weights/best.pt'))
names = ['fire_marker', 'water_station']
stats = {n: dict(tp=0, fp=0, fn=0, latencies=[]) for n in ['classical', 'yolo']}

def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    z = x * y
    return z / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - z)
for path in sorted((root / 'dataset/images/test').glob('*.jpg')):
    frame = cv2.imread(str(path))
    h, w = frame.shape[:2]
    truth = []
    for line in (root / 'dataset/labels/test' / path.with_suffix('.txt').name).read_text().splitlines():
        c, x, y, bw, bh = map(float, line.split())
        truth.append((names[int(c)], [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    start = time.perf_counter()
    classical = landmarks(frame.copy(), {'height': 7}, lambda *a: {'east': 0, 'north': 0})
    classical_ms = (time.perf_counter() - start) * 1000
    pred = [(names[0] if d['label'] == 'Fire' else names[1], [d['bbox'][0], d['bbox'][1], d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]]) for d in classical]
    start = time.perf_counter()
    result = model.predict(frame, imgsz=320, conf=0.45, iou=0.5, device='cpu', verbose=False)[0]
    yolo_ms = (time.perf_counter() - start) * 1000
    yp = [(model.names[int(c)], b) for c, b in zip(result.boxes.cls.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]
    for method, predictions, latency in [('classical', pred, classical_ms), ('yolo', yp, yolo_ms)]:
        used = set()
        r = stats[method]
        r['latencies'].append(latency)
        for label, box in predictions:
            options = [(iou(box, b), i) for i, (c, b) in enumerate(truth) if c == label and i not in used]
            best = max(options, default=(0, -1))
            if best[0] >= 0.5:
                r['tp'] += 1
                used.add(best[1])
            else:
                r['fp'] += 1
        r['fn'] += len(truth) - len(used)
for r in stats.values():
    r['precision'] = r['tp'] / max(1, r['tp'] + r['fp'])
    r['recall'] = r['tp'] / max(1, r['tp'] + r['fn'])
    r['median_ms'] = float(np.median(r.pop('latencies')))
(root / 'detector_comparison.json').write_text(json.dumps({'scope': '48 independently generated procedural test images; IoU >=0.5 and YOLO confidence >=0.45; includes model call overhead, first inference included; not camera/Gazebo/real-fire validation', 'results': stats}, indent=2))
print(json.dumps(stats, indent=2))
model.predict(str(root / 'dataset/images/test/0003.jpg'), imgsz=320, conf=0.45, device='cpu', save=True, project=str(root / 'runs'), name='preview', exist_ok=True, verbose=False)
