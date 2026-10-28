import math
import cv2, numpy as np
from coverage import rotation

def landmarks(frame, st, project, front=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    out = []
    masks = [('Fire', ((h > 3) & (h < 22) & (s > 165) & (v > 150)).astype('uint8') * 255), ('Station', ((h > 79) & (h < 100) & (s > 150) & (v > 140)).astype('uint8') * 255)]
    for label, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (35 if label == 'Fire' else 80):
                continue
            x, y, w, hh = cv2.boundingRect(contour)
            if label == 'Station':
                roi = hsv[y:y + hh, x:x + w]
                purple = ((roi[:, :, 0] > 135) & (roi[:, :, 0] < 172) & (roi[:, :, 1] > 140) & (roi[:, :, 2] > 130)).sum()
                if purple < max(6, area * 0.005) or (not front and (not 0.6 < w / max(1, hh) < 1.6)):
                    continue
            u = x + w / 2
            vv = y + hh / 2
            if front:
                f = 640 / (2 * math.tan(0.85))
                ray = rotation(st) @ np.array([1.0, (u - 320) / f, (vv - 180) / f])
                down = ray[2]
                if down < 0.16:
                    continue
                scale = (st.get('height', 0) + 0.3 - (0.65 if label == 'Fire' else 0.06)) / down
                if not 0 < scale < 25:
                    continue
                p = {'east': st.get('east', 0) + ray[1] * scale, 'north': st.get('north', 0) + ray[0] * scale}
            else:
                p = project(u, vv, max(0.1, st.get('height', 0) - 0.1 - (1.3 if label == 'Fire' else 0.06)), st)
            det = {'label': label, 'u': u, 'v': vv, 'bbox': [x, y, w, hh], 'area': float(area), **p}
            out.append(det)
            color = (0, 170, 255) if label == 'Fire' else (220, 220, 220)
            cv2.rectangle(frame, (x, y), (x + w, y + hh), color, 2)
            cv2.putText(frame, label + (' / On tespit' if front else ''), (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out
