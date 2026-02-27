from pathlib import Path
from ultralytics import YOLO

_BASE = Path(__file__).resolve().parent
MODEL_PATH = _BASE / "runs/detect/yolov8_kamikaze_960/weights/best.pt"
VIDEO_PATH = _BASE / "video/kamikaze.mp4"
OUTPUT_DIR = _BASE / "video"
TRACKER_CFG = _BASE / "cfg/trackers/bytetrack_kamikaze.yaml"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video bulunamadı: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))
    model.model.names = {0: "Kamikaze"}

    results = model.track(
        source=str(VIDEO_PATH),
        show=True,
        save=True,
        project=str(OUTPUT_DIR.resolve()),
        name="kamikaze_track",
        exist_ok=True,
        imgsz=960,

        conf=0.12,        # Düşük = daha az "no detections", iz kopması azalır → aynı ID
        iou=0.5,

        tracker=str(TRACKER_CFG) if TRACKER_CFG.exists() else "bytetrack.yaml",
        persist=True
    )

    print("\nTracking tamamlandı.")

if __name__ == "__main__":
    main()