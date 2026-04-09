"""
YOLOv8 ile kamikaze (fixed-wing UAV) tespiti eğitimi.
Veri seti: Roboflow Kamikaze UAV Dataset (dataset/data.yaml)
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolov8s.pt")  # GPU uygunsa yolov8s de düşünebiliriz

    model.train(
        data="dataset/data.yaml",  # Kamikaze UAV veri seti (dataset klasöründeki data.yaml)

        # ===== TEMEL AYARLAR =====
        epochs=80,              # Tiny object için daha uzun eğitim
        imgsz=960,               # Yüksek çözünürlük (küçük drone için)
        batch=12,                 # 960'da 8 daha stabil (GPU'ya göre 8–16)
        device=0,

        # ===== STABİLİTE =====
        workers=2,
        cache=True,              # RAM'e cache → daha hızlı eğitim
        amp=True,                # Mixed precision → hız + daha az VRAM

        # ===== OVERFITTING KONTROL =====
        patience=25,             # Early stopping
        cos_lr=True,             # Cosine learning rate decay
        mosaic=0.0,

        # ===== OPTİMİZASYON =====
        optimizer="AdamW",        # Tiny object için genelde daha stabil
        lr0=0.0008,               # Biraz düşük başlangıç LR
        weight_decay=0.0005,

        # ===== DENEY İSMİ (kamikaze tespiti) =====
        name="yolov8_kamikaze_960",
    )


if __name__ == "__main__":
    freeze_support()
    main()
