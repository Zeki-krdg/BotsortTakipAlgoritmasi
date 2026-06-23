"""
BOT-SORT Tracker — Kamikaze İHA Takip Sistemi
===============================================

Çalışma mantığı:
  SEARCH modu: YOLO her karede çalışır, hedef aranır.
  TRACK modu:  Kalman her karede tahmin yapar, YOLO aralıklarla düzeltir.
               YOLO tespiti SADECE Kalman tahminine yakınsa (IoU > eşik) kabul edilir.
               Uzaktaki tespitler GÖRMEZDEN GELİNİR (false positive koruması).

Kullanım:
    python botsort_tracker.py
"""

import cv2
import time
from pathlib import Path
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from botsort_kalman import KalmanBoxTracker, iou

# ── Ayarlar ───────────────────────────────────────────────────────────────
MODEL_PATH = Path("runs/detect/yolov8_kamikaze_960/weights/best.pt")
VIDEO_PATH = Path("video/kamikaze.mp4")

YOLO_CONF = 0.15             # YOLO güven eşiği (daha çok tespit, daha iyi kilitlenme)
DETECT_INTERVAL = 2           # TRACK modunda kaç karede bir YOLO çalıştır (Daha sık kontrol = Kopmaz kilit)
MATCH_DIST_THRESH = 100.0     # YOLO tespitini kabul etmek için max merkez mesafesi (Hızlı SİHA için genişlettik)
MAX_MISSED_DETECTIONS = 30    # Kaç YOLO döngüsü hedefi bulamazsa SEARCH'e dön (Hemen pes etme)


def bbox_center(bbox):
    """[x1,y1,x2,y2] → (cx, cy)"""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def match_detection(results, predicted_bbox, max_dist):
    """YOLO sonuçlarından Kalman tahminine en yakın tespiti bul.

    Merkez mesafesi kullanır (IoU yerine). Küçük nesneler için çok daha güvenilir.
    SADECE mesafe < max_dist olan tespitler kabul edilir.

    Returns:
        (bbox_xyxy, conf) veya (None, 0.0)
    """
    if len(results[0].boxes) == 0:
        return None, 0.0

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    pred_cx, pred_cy = bbox_center(predicted_bbox)

    best_dist = float('inf')
    best_idx = -1

    for i, box in enumerate(boxes):
        det_cx, det_cy = bbox_center(box)
        dist = np.sqrt((pred_cx - det_cx) ** 2 + (pred_cy - det_cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # Sadece Kalman tahminine YAKIN olan tespitleri kabul et
    if best_idx >= 0 and best_dist <= max_dist:
        return boxes[best_idx], float(confs[best_idx])

    return None, 0.0


def find_any_detection(results):
    """SEARCH modunda en yüksek güvenli tespiti seç."""
    if len(results[0].boxes) == 0:
        return None, 0.0

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    return boxes[best_idx], float(confs[best_idx])


def main():
    if not MODEL_PATH.exists():
        print(f"Hata: Model bulunamadı: {MODEL_PATH}")
        return
    if not VIDEO_PATH.exists():
        print(f"Hata: Video bulunamadı: {VIDEO_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Hata: Video açılamadı.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n--- BOT-SORT TRACKER (Kalman + YOLO) ---\n")

    # ── Takip Durumu ──────────────────────────────────────────────────────
    STATE = "SEARCH"           # SEARCH veya TRACK
    kalman = None              # KalmanBoxTracker instance
    missed_detections = 0      # Ardışık YOLO kaçırma sayısı
    frame_count = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_h, img_w = frame.shape[:2]
        draw_bbox = None

        # ==============================================================
        #  SEARCH MODU: Her karede YOLO çalıştır, hedef ara
        # ==============================================================
        if STATE == "SEARCH":
            results = model.predict(frame, conf=YOLO_CONF, verbose=False)
            det_box, det_conf = find_any_detection(results)

            if det_box is not None:
                # Hedef bulundu! Kalman başlat, TRACK moduna geç
                kalman = KalmanBoxTracker(det_box)
                STATE = "TRACK"
                missed_detections = 0
                draw_bbox = det_box
                print(f"[Kare {frame_count:04d}] HEDEF BULUNDU → TRACK moduna geçildi (conf: {det_conf:.2f})")

        # ==============================================================
        #  TRACK MODU: Kalman tahmin + aralıklı YOLO düzeltme
        # ==============================================================
        elif STATE == "TRACK":
            # Her karede Kalman tahmini (çok hızlı, ~0ms)
            predicted = kalman.predict()
            draw_bbox = predicted

            # Belirli aralıklarla YOLO çalıştır ve eşleştir
            if frame_count % DETECT_INTERVAL == 0:
                results = model.predict(frame, conf=YOLO_CONF, verbose=False)

                # SADECE Kalman tahminine yakın tespitleri kabul et!
                matched_box, matched_conf = match_detection(
                    results, predicted, MATCH_DIST_THRESH
                )

                if matched_box is not None:
                    # Eşleşme başarılı → Kalman güncelle
                    kalman.update(matched_box)
                    draw_bbox = kalman.get_bbox()
                    missed_detections = 0
                else:
                    # YOLO hedefi bulamadı veya uzak bir şey tespit etti
                    # → Kalman tahminiyle devam et, atlama!
                    missed_detections += 1

                # Çok uzun süre YOLO hedefi bulamazsa veya güven %50'nin altına düşerse → SEARCH'e dön
                if missed_detections >= MAX_MISSED_DETECTIONS or kalman.get_confidence() < 0.50:
                    STATE = "SEARCH"
                    kalman = None
                    missed_detections = 0
                    draw_bbox = None
                    print(f"[Kare {frame_count:04d}] KİLİT ZAYIFLADI (<%50) → SEARCH moduna geçildi")

        # ==============================================================
        #  ÇİZİM
        # ==============================================================
        if draw_bbox is not None and STATE == "TRACK":
            x1 = max(0, min(int(draw_bbox[0]), img_w - 1))
            y1 = max(0, min(int(draw_bbox[1]), img_h - 1))
            x2 = max(0, min(int(draw_bbox[2]), img_w - 1))
            y2 = max(0, min(int(draw_bbox[3]), img_h - 1))

            conf = kalman.get_confidence()
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"BOT-SORT {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS ve durum
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"STATE: {STATE}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if STATE == "TRACK" else (0, 0, 255), 2)

        if frame_count % 50 == 0:
            print(f"[Kare {frame_count:04d}/{total_frames}] {STATE} | FPS: {fps:.1f} | Miss: {missed_detections}")

        cv2.imshow("BOT-SORT Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nToplam {frame_count} kare işlendi.")


if __name__ == "__main__":
    main()
