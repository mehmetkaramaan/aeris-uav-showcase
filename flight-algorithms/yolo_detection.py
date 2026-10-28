import base64
import collections
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import cv2
import numpy as np
from coverage import rotation
LABELS = {'fire_marker': 'Fire', 'water_station': 'Station'}

def project_detections(boxes, shape, st, project, front=False):
    height, width = shape[:2]
    out = []
    for box in boxes:
        label = LABELS.get(box.get('class'))
        try:
            confidence = float(box.get('confidence', 0))
            coords = [float(v) for v in box.get('xyxy', [])]
        except (TypeError, ValueError):
            continue
        if label is None or len(coords) != 4 or (not math.isfinite(confidence)) or (confidence < 0.45):
            continue
        if not all((math.isfinite(float(v)) for v in coords)):
            continue
        x1, y1, x2, y2 = coords
        x1, x2 = (max(0.0, x1), min(float(width), x2))
        y1, y2 = (max(0.0, y1), min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        u, v = ((x1 + x2) / 2, (y1 + y2) / 2)
        if front:
            f = width / (2 * math.tan(0.85))
            ray = rotation(st) @ np.array([1.0, (u - width / 2) / f, (v - height / 2) / f])
            if ray[2] < 0.16:
                continue
            scale = (st.get('height', 0) + 0.3 - (0.65 if label == 'Fire' else 0.06)) / ray[2]
            if not 0 < scale < 25:
                continue
            position = {'east': st.get('east', 0) + ray[1] * scale, 'north': st.get('north', 0) + ray[0] * scale}
        else:
            position = project(u, v, max(0.1, st.get('height', 0) - 0.1 - (1.3 if label == 'Fire' else 0.06)), st)
        if not all((math.isfinite(float(position[k])) for k in ('east', 'north'))):
            continue
        out.append(dict(label=label, u=u, v=v, bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), confidence=confidence, detector='yolo', **position))
    return out

class YoloBridge:

    def __init__(self, python=None, weights=None):
        root = Path(__file__).resolve().parent
        self.python = python or os.environ.get('AERIS_YOLO_PYTHON', '')
        self.weights = Path(weights or os.environ.get('AERIS_YOLO_WEIGHTS', str(root / 'models/yolo/markers.pt')))
        self.condition = threading.Condition()
        self.pending = collections.OrderedDict()
        self.results = {}
        self.status = 'unavailable'
        self.error = ''
        self.process = None
        self.closed = False
        self.frames = 0
        if self.python and Path(self.python).is_file() and self.weights.is_file():
            self.status = 'starting'
            threading.Thread(target=self._run, name='aeris-yolo', daemon=True).start()
        else:
            self.error = 'Optional Python environment or trained marker weights are missing.'

    def submit(self, channel, frame, telemetry):
        if self.status not in {'starting', 'ready'}:
            return
        with self.condition:
            self.pending[channel] = (frame.copy(), dict(telemetry), time.monotonic())
            self.condition.notify()

    def _read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError('YOLO worker response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('YOLO worker exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def _run(self):
        try:
            self.process = subprocess.Popen([self.python, '-u', str(Path(__file__).with_name('yolo_worker.py')), '--weights', str(self.weights)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            if not self._read(90).get('ready'):
                raise RuntimeError('YOLO worker initialization failed')
            self.status = 'ready'
            while not self.closed:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closed)
                    if self.closed:
                        break
                    channel, (frame, telemetry, captured) = self.pending.popitem(last=False)
                if time.monotonic() - captured > 0.8:
                    continue
                ok, image = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                self.process.stdin.write(json.dumps({'image': base64.b64encode(image).decode()}) + '\n')
                self.process.stdin.flush()
                result = self._read(5)
                with self.condition:
                    self.results[channel] = dict(result, captured=captured, telemetry=telemetry, shape=frame.shape)
                    self.frames += 1
        except Exception as exc:
            self.status = 'error'
            self.error = str(exc)
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

    def snapshot(self, channel, now=None):
        now = time.monotonic() if now is None else now
        with self.condition:
            result = self.results.get(channel)
            if result is None or not 0 <= now - result['captured'] <= 0.8:
                return None
            return dict(result)

    def state(self):
        return dict(status=self.status, mode='shadow', frames=self.frames, error=self.error, model='YOLOv8n AERIS synthetic marker pilot', mission_authority=False)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
