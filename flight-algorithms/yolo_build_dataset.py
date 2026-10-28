import json
from pathlib import Path
import cv2
import numpy as np
root = Path(__file__).resolve().parent / 'dataset'
for split, count, seed in [('train', 240, 42), ('val', 48, 10042), ('test', 48, 20042)]:
    rng = np.random.default_rng(seed)
    for kind in ('images', 'labels'):
        (root / kind / split).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = np.full((320, 320, 3), rng.integers(25, 105), np.uint8)
        noise = rng.normal(0, rng.uniform(2, 14), image.shape)
        image = np.clip(image + noise, 0, 255).astype('uint8')
        for _ in range(int(rng.integers(0, 8))):
            x, y = rng.integers(0, 270, 2)
            cv2.rectangle(image, (x, y), (x + int(rng.integers(10, 50)), y + int(rng.integers(10, 50))), tuple(map(int, rng.integers(40, 160, 3))), -1)
        labels = []
        locations = [(70, 70), (235, 235)]
        rng.shuffle(locations)
        for cls, (cx, cy) in enumerate(locations):
            if rng.random() < 0.2:
                continue
            cx += int(rng.integers(-30, 30))
            cy += int(rng.integers(-30, 30))
            size = int(rng.integers(14, 80))
            x, y = (cx - size // 2, cy - size // 2)
            tile = np.zeros((size, size, 3), np.uint8)
            if cls == 0:
                tile[:] = (int(rng.integers(0, 30)), int(rng.integers(65, 140)), int(rng.integers(210, 256)))
                cv2.rectangle(tile, (0, 0), (size // 3, size - 1), (0, 65, 190), -1)
            else:
                tile[:] = (230, 230, 5)
                cv2.line(tile, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), (220, 5, 240), max(2, size // 9))
                cv2.line(tile, (size // 4, 3 * size // 4), (3 * size // 4, size // 4), (220, 5, 240), max(2, size // 9))
            image[y:y + size, x:x + size] = tile
            labels.append(f'{cls} {cx / 320:.7f} {cy / 320:.7f} {size / 320:.7f} {size / 320:.7f}')
        image = np.clip(image.astype(float) * rng.uniform(0.6, 1.25), 0, 255).astype('uint8')
        if rng.random() < 0.35:
            image = cv2.GaussianBlur(image, (3, 3), 0.6)
        cv2.imwrite(str(root / 'images' / split / f'{index:04}.jpg'), image)
        (root / 'labels' / split / f'{index:04}.txt').write_text('\n'.join(labels))
(root / 'data.yaml').write_text(f'path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fire_marker\n  1: water_station\n')
(root / 'provenance.json').write_text(json.dumps({'scope': 'Procedurally drawn synthetic colored simulation markers; not real fire photographs or Gazebo-rendered validation', 'split_seeds': {'train': 42, 'val': 10042, 'test': 20042}, 'counts': {'train': 240, 'val': 48, 'test': 48}}, indent=2))
