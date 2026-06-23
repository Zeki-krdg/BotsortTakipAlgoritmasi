"""
BOT-SORT Test — Detaylı Performans Analizi
============================================

siamcar_test.py dosyasının BOT-SORT karşılığı.
YOLO tespit + BOT-SORT takip performansını detaylı loglarla test eder.

Özellikler:
  • Kare-kare detaylı terminal çıktısı
  • Takip istatistikleri (kayıp süresi, toplam iz sayısı, vb.)
  • Ayarlanabilir YOLO confidence ve tracker parametreleri
  • Oynatma hızı kontrolü

Kullanım:
    python botsort_test.py
"""

import cv2
import time
import numpy as np
from pathlib import Path
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ──────────────────── CONFIG (Sadece burayı düzenle) ─────────────────────
VIDEO_PATH = Path("video/kamikaze.mp4")
YOLO_WEIGHTS = Path("runs/detect/yolov11_egitimleri/kamikaze_uav_640/weights/best.pt")
TRACKER_CONFIG = "botsort_kamikaze.yaml"

YOLO_CONF = 0.22
PLAYBACK_SPEED = 1.0
ENABLE_TERMINAL_LOGS = True
# ─────────────────────────────────────────────────────────────────────────

# Renk Paleti
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
]


def get_color(track_id):
    return COLORS[track_id % len(COLORS)]


def main():
    if not VIDEO_PATH.exists():
        print(f"Error: Video not found: {VIDEO_PATH}")
        return
    if not YOLO_WEIGHTS.exists():
        print(f"Error: YOLO weights not found: {YOLO_WEIGHTS}")
        return

    model = YOLO(str(YOLO_WEIGHTS))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 1e-6:
        video_fps = 30.0
    target_fps = max(1.0, video_fps * max(0.1, PLAYBACK_SPEED))
    wait_ms = max(1, int(round(1000.0 / target_fps)))

    # ── İstatistikler ─────────────────────────────────────────────────────
    frame_count = 0
    frames_with_track = 0
    frames_without_track = 0
    total_tracks_seen = set()
    max_concurrent_tracks = 0
    fps_list = []

    print(f"\n{'=' * 65}")
    print(f"  BOT-SORT TEST — Detaylı Performans Analizi")
    print(f"{'=' * 65}")
    print(f"  YOLO Model  : {YOLO_WEIGHTS}")
    print(f"  Video       : {VIDEO_PATH}")
    print(f"  Tracker     : BOT-SORT (botsort_kamikaze.yaml)")
    print(f"  Video FPS   : {video_fps:.2f}")
    print(f"  Display FPS : {target_fps:.2f}")
    print(f"  Toplam Kare : {total_frames}")
    print(f"{'=' * 65}")
    print("  ESC veya 'q' ile çıkış\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sonu.")
            break

        frame_count += 1
        start_time = time.time()

        # ── BOT-SORT Takip ────────────────────────────────────────────────
        results = model.track(
            frame,
            tracker=TRACKER_CONFIG,
            persist=True,
            conf=YOLO_CONF,
            verbose=False,
        )

        # ── Sonuçları İşle ────────────────────────────────────────────────
        active_tracks = 0
        current_boxes = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            active_tracks = len(track_ids)
            max_concurrent_tracks = max(max_concurrent_tracks, active_tracks)

            for box, track_id, conf in zip(boxes, track_ids, confs):
                total_tracks_seen.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                color = get_color(track_id)

                current_boxes.append({
                    'id': track_id, 'conf': conf,
                    'bbox': (x1, y1, w, h),
                })

                # Bounding box çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                # Etiket
                label = f"BOT-SORT ID:{track_id} | {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Boyut bilgisi
                size_label = f"{w}x{h}px"
                cv2.putText(frame, size_label, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if ENABLE_TERMINAL_LOGS:
                    print(
                        f"[Kare {frame_count:04d}] TRACKING | "
                        f"ID:{track_id} | Score:{conf:.2f} | "
                        f"BBox:[{x1},{y1},{w},{h}]"
                    )

        if active_tracks > 0:
            frames_with_track += 1
        else:
            frames_without_track += 1
            if ENABLE_TERMINAL_LOGS:
                print(f"[Kare {frame_count:04d}] SEARCHING | Tespit=0")

        # ── FPS Hesapla ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        fps = 1.0 / (elapsed + 1e-5)
        fps_list.append(fps)

        # ── Bilgi Paneli ──────────────────────────────────────────────────
        panel_h = 120
        cv2.rectangle(frame, (0, 0), (400, panel_h), (0, 0, 0), -1)

        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} (proc) | {target_fps:.1f} (display)", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        status = "TRACKING" if active_tracks > 0 else "SEARCHING"
        status_color = (0, 255, 0) if active_tracks > 0 else (0, 165, 255)
        cv2.putText(frame, f"Mode: {status} ({active_tracks} target)", (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        track_pct = (frames_with_track / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(frame, f"Track Rate: {track_pct:.1f}% | IDs Seen: {len(total_tracks_seen)}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Göster ────────────────────────────────────────────────────────
        cv2.imshow("BOT-SORT Test", frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Final İstatistikleri ──────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  BOT-SORT TEST — Sonuç Raporu")
    print(f"{'=' * 65}")
    print(f"  İşlenen Kare       : {frame_count}")
    print(f"  Takipli Kare       : {frames_with_track} ({frames_with_track / max(1, frame_count) * 100:.1f}%)")
    print(f"  Takipsiz Kare      : {frames_without_track} ({frames_without_track / max(1, frame_count) * 100:.1f}%)")
    print(f"  Toplam Farklı ID   : {len(total_tracks_seen)}")
    print(f"  Maks Eşzamanlı İz  : {max_concurrent_tracks}")
    if fps_list:
        print(f"  Ortalama FPS       : {np.mean(fps_list):.1f}")
        print(f"  Min / Maks FPS     : {np.min(fps_list):.1f} / {np.max(fps_list):.1f}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
