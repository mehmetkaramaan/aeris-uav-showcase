from pathlib import Path
import json
from ultralytics import YOLO
if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    model = YOLO('yolov8n.pt')
    model.train(data=str(root / 'dataset/data.yaml'), epochs=18, imgsz=320, batch=16, workers=0, device='mps', project=str(root / 'runs'), name='markers', exist_ok=True, seed=42, deterministic=True, plots=True, verbose=False, patience=6, degrees=25, perspective=0.0005, fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.25, hsv_v=0.3, mosaic=0.5)
    best = YOLO(str(root / 'runs/markers/weights/best.pt'))
    metrics = best.val(data=str(root / 'dataset/data.yaml'), split='test', imgsz=320, device='cpu', workers=0, plots=True, project=str(root / 'runs'), name='heldout', exist_ok=True, verbose=False)
    (root / 'pilot_metrics.json').write_text(json.dumps({'scope': 'Procedural marker held-out images only; not Gazebo mission or real-fire validation', 'metrics': metrics.results_dict, 'speed_ms': metrics.speed}, indent=2))
