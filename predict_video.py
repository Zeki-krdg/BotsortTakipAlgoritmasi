"""
Eğitilmiş kamikaze modeli ile video üzerinde tespit.
Tespit edilen kamikaze çerçeve içinde gösterilir, üstünde isim yazar.
"""
from pathlib import Path
from ultralytics import YOLO

# Eğitim çıktısı: runs/detect/yolov8_kamikaze_960/weights/best.pt
# Farklı run kullandıysan aşağıdaki yolu değiştir
MODEL_PATH = Path("runs/detect/yolov8_kamikaze_960/weights/best.pt")
VIDEO_PATH = Path("video/kamikaze.mp4")
# Çıktı videoyu kaydetmek için (None = sadece pencerede göster)
OUTPUT_DIR = Path("video")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}\n"
            "Eğitimi bitirdikten sonra bu yolu kontrol et. "
            "Run adı farklıysa (örn. yolov8_kamikaze_9602) MODEL_PATH'i güncelle."
        )
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video bulunamadı: {VIDEO_PATH}")

    model = YOLO(str(MODEL_PATH))
    
    model.model.names = {0: "Kamikaze"}

    
    results = model.predict(
        source=str(VIDEO_PATH),
        show=True,               
        save=True,
        save_txt=False,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name="kamikaze_pred",
        exist_ok=True,
        imgsz=960,
        conf=0.25,
        iou=0.5,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )

    out_dir = OUTPUT_DIR / "kamikaze_pred"
    print(f"\nÇıktı kaydedildi: {out_dir}")
    if out_dir.exists():
        for f in out_dir.glob("*.mp4"):
            print(f"  Video: {f}")


if __name__ == "__main__":
    main()
