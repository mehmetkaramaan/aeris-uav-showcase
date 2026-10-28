import argparse
import base64
import contextlib
import json
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
        model = YOLO(args.weights)
    names = set(model.names.values())
    if names != {'fire_marker', 'water_station'}:
        raise ValueError('Expected an AERIS two-class marker detector; generic COCO weights are incompatible.')
    print(json.dumps({'ready': True, 'names': model.names}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(request['image']), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid image')
            start = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                result = model.predict(frame, imgsz=320, conf=0.45, iou=0.5, device='cpu', verbose=False)[0]
            boxes = []
            for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                boxes.append({'class': model.names[int(cls)], 'confidence': conf, 'xyxy': xyxy})
            print(json.dumps({'detections': boxes, 'inference_ms': (time.monotonic() - start) * 1000}), flush=True)
        except Exception as exc:
            print(json.dumps({'error': str(exc)}), flush=True)
if __name__ == '__main__':
    main()
