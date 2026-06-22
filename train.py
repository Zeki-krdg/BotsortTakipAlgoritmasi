"""
YOLO11 ile kamikaze (fixed-wing UAV) tespiti eğitimi.
Veri seti: Roboflow Kamikaze UAV Dataset (dataset/data.yaml)
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    # Model: YOLO11s (Hız ve doğruluk dengesi için Small model)
    model = YOLO("yolo11s.pt") 

    model.train(
        data="dataset/data.yaml",  # Kamikaze UAV veri seti

        # ===== ÖZEL ÇIKTI AYARLARI =====
        project="yolov11_egitimleri", # Artık 'runs' yerine bu klasöre kaydedilecek
        name="kamikaze_uav_640",      # Eğitimin özel klasör adı

        # ===== TEMEL AYARLAR =====
        epochs=100,              # Sabaha kadar sürecek kaliteli bir eğitim için
        imgsz=640,               # İstek üzerine 640 çözünürlük
        batch=16,                # 640'ta 16-32 arası GPU zorlamaz
        device=0,                # GPU kullanımı

        # ===== STABİLİTE =====
        workers=4,
        cache=True,              # RAM'e cache
        amp=True,                # Mixed precision

        # ===== OPTİMİZASYON & DENETİM =====
        patience=30,             # Gelişme yoksa durdurma sınırı
        cos_lr=True,             # Cosine learning rate decay
        optimizer="AdamW",
        lr0=0.001,
        
        # ===== VERİ ARTIRMA =====
        mosaic=1.0,              # İHA tespiti için küçük nesne yakalama kabiliyetini artırır
    )


if __name__ == "__main__":
    freeze_support()
    main()
