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

import sys

# ── Ayarlar ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"runs\detect\yolov11_egitimleri\best_yolo11n.pt")

# Eğer komut satırından video verilmişse onu kullan, yoksa varsayılanı kullan
if len(sys.argv) > 1:
    VIDEO_PATH = Path(sys.argv[1])
else:
    VIDEO_PATH = Path("video/kamikaze.mp4")

YOLO_CONF = 0.30             # 0.15 idi  YOLO güven eşiği (daha çok tespit, daha iyi kilitlenme)
DETECT_INTERVAL = 2        # TRACK modunda kaç karede bir YOLO çalıştır (Daha sık kontrol = Kopmaz kilit)
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

    # Pencerenin oranları bozulmadan büyütülüp küçültülebilmesini sağlar
    cv2.namedWindow("BOT-SORT Tracker", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # ── Takip Durumu ──────────────────────────────────────────────────────
    STATE = "SEARCH"           # SEARCH veya TRACK
    kalman = None              # KalmanBoxTracker instance
    missed_detections = 0      # Ardışık YOLO kaçırma sayısı
    frame_count = 0
    prev_time = time.time()
    lock_start_time = None
    ignored_trackers = []      # 4 sn kilitlenilen hedefleri saklar
    show_success_text = False  # Kilitlenme başarılı yazısının kalıcılığını tutar

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img_h, img_w = frame.shape[:2]
        draw_bbox = None

        # Hedef Vuruş Alanı (Av) tanımı
        av_x1 = int(img_w * 0.25)
        av_y1 = int(img_h * 0.10)
        av_x2 = int(img_w * 0.75)
        av_y2 = int(img_h * 0.90)

        # YOLO çalıştırılacak mı?
        run_yolo = (STATE == "SEARCH") or (STATE == "TRACK" and frame_count % DETECT_INTERVAL == 0)
        if run_yolo:
            results = model.predict(frame, conf=YOLO_CONF, verbose=False)
        else:
            results = None

        # Ignored hedefleri güncelle
        active_ignored = []
        ignored_boxes_to_exclude = []
        ignored_boxes_to_draw = []
        for ign in ignored_trackers:
            ign_k = ign['kalman']
            ign_pred = ign_k.predict()
            
            if results is not None:
                matched_box, matched_conf = match_detection(results, ign_pred, MATCH_DIST_THRESH)
                if matched_box is not None:
                    ign_k.update(matched_box)
                    ign['missed'] = 0
                else:
                    ign['missed'] += 1
                    
            if ign['missed'] < MAX_MISSED_DETECTIONS:
                active_ignored.append(ign)
                ign_bbox = ign_k.get_bbox()
                ignored_boxes_to_exclude.append(ign_bbox)
                ignored_boxes_to_draw.append(ign_bbox)
                
        ignored_trackers = active_ignored

        # ==============================================================
        #  SEARCH MODU: Her karede YOLO çalıştır, hedef ara
        # ==============================================================
        if STATE == "SEARCH" and results is not None:
            best_conf_inside = -1.0
            best_box_inside = None
            best_conf_outside = -1.0
            best_box_outside = None

            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    bcx, bcy = bbox_center(box)
                    is_ignored = False
                    for ign_box in ignored_boxes_to_exclude:
                        icx, icy = bbox_center(ign_box)
                        dist = np.sqrt((bcx - icx)**2 + (bcy - icy)**2)
                        if dist < MATCH_DIST_THRESH:
                            is_ignored = True
                            break
                    
                    if not is_ignored:
                        bx1, by1, bx2, by2 = box
                        is_inside_av = (bx1 >= av_x1) and (bx2 <= av_x2) and (by1 >= av_y1) and (by2 <= av_y2)
                        
                        if is_inside_av:
                            if confs[i] > best_conf_inside:
                                best_conf_inside = float(confs[i])
                                best_box_inside = box
                        else:
                            if confs[i] > best_conf_outside:
                                best_conf_outside = float(confs[i])
                                best_box_outside = box

            # Öncelik: Sarı kare içi
            best_box = None
            best_conf = -1.0
            if best_box_inside is not None:
                best_box = best_box_inside
                best_conf = best_conf_inside
            elif best_box_outside is not None:
                best_box = best_box_outside
                best_conf = best_conf_outside

            if best_box is not None:
                kalman = KalmanBoxTracker(best_box)
                STATE = "TRACK"
                missed_detections = 0
                draw_bbox = best_box
                last_yolo_conf = best_conf
                print(f"[Kare {frame_count:04d}] HEDEF BULUNDU → TRACK moduna geçildi (conf: {best_conf:.2f})")

        # ==============================================================
        #  TRACK MODU: Kalman tahmin + aralıklı YOLO düzeltme
        # ==============================================================
        elif STATE == "TRACK":
            predicted = kalman.predict()
            draw_bbox = predicted

            if results is not None:
                matched_box, matched_conf = match_detection(
                    results, predicted, MATCH_DIST_THRESH
                )

                if matched_box is not None:
                    kalman.update(matched_box)
                    draw_bbox = kalman.get_bbox()
                    last_yolo_conf = matched_conf
                    missed_detections = 0
                else:
                    missed_detections += 1

            if missed_detections >= MAX_MISSED_DETECTIONS or kalman.get_confidence() < 0.50:
                STATE = "SEARCH"
                kalman = None
                missed_detections = 0
                draw_bbox = None
                lock_start_time = None
                print(f"[Kare {frame_count:04d}] KİLİT ZAYIFLADI (<%50) → SEARCH moduna geçildi")

        # ==============================================================
        #  ÇİZİM
        # ==============================================================
        win_w, win_h = img_w, img_h
        try:
            rect = cv2.getWindowImageRect("BOT-SORT Tracker")
            if rect[2] > 0 and rect[3] > 0:
                win_w, win_h = rect[2], rect[3]
        except:
            pass

        if win_w != img_w or win_h != img_h:
            frame = cv2.resize(frame, (win_w, win_h))
            
        scale_x = win_w / img_w
        scale_y = win_h / img_h

        # Ignored boxes çizimi
        for ign_bbox in ignored_boxes_to_draw:
            ix1, iy1, ix2, iy2 = map(int, ign_bbox)
            icx = (ix1 + ix2) / 2.0
            icy = (iy1 + iy2) / 2.0
            iw = ix2 - ix1
            ih = iy2 - iy1

            ix1 = max(0, min(int(icx - iw / 2.0), img_w - 1))
            iy1 = max(0, min(int(icy - ih / 2.0), img_h - 1))
            ix2 = max(0, min(int(icx + iw / 2.0), img_w - 1))
            iy2 = max(0, min(int(icy + ih / 2.0), img_h - 1))

            s_ix1, s_iy1 = int(ix1 * scale_x), int(iy1 * scale_y)
            s_ix2, s_iy2 = int(ix2 * scale_x), int(iy2 * scale_y)
            s_icx, s_icy = int(icx * scale_x), int(icy * scale_y)

            color = (0, 165, 255) # Turuncu
            cv2.rectangle(frame, (s_ix1, s_iy1), (s_ix2, s_iy2), color, 2)
            cv2.putText(frame, "Ayni Siha", (s_ix1, max(20, s_iy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, "HH", (s_icx - 10, s_icy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, "AK: Kamera Gorus Alani", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        s_av_x1, s_av_y1 = int(av_x1 * scale_x), int(av_y1 * scale_y)
        s_av_x2, s_av_y2 = int(av_x2 * scale_x), int(av_y2 * scale_y)

        cv2.rectangle(frame, (s_av_x1, s_av_y1), (s_av_x2, s_av_y2), (0, 255, 255), 2)
        cv2.putText(frame, "AV: Hedef Vurus Alani", (s_av_x1, s_av_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if draw_bbox is not None and STATE == "TRACK":
            x1, y1, x2, y2 = draw_bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            x1 = max(0, min(int(x1), img_w - 1))
            y1 = max(0, min(int(y1), img_h - 1))
            x2 = max(0, min(int(x2), img_w - 1))
            y2 = max(0, min(int(y2), img_h - 1))

            s_x1, s_y1 = int(x1 * scale_x), int(y1 * scale_y)
            s_x2, s_y2 = int(x2 * scale_x), int(y2 * scale_y)
            s_cx, s_cy = int(cx * scale_x), int(cy * scale_y)

            conf = kalman.get_confidence()
            
            min_req_w = img_w * 0.05
            min_req_h = img_h * 0.05
            is_large_enough = (w >= min_req_w) or (h >= min_req_h)
            
            is_inside_av = (x1 >= av_x1) and (x2 <= av_x2) and (y1 >= av_y1) and (y2 <= av_y2)
            
            if conf > 0.5 and is_inside_av and is_large_enough:
                color = (0, 0, 255)       
                text_ah = f"AH: Kilitlenme Dortgeni (Gecerli: {conf:.2f})"
                if lock_start_time is None:
                    lock_start_time = time.time()
                    show_success_text = False

            else:
                color = (0, 165, 255)     
                text_ah = f"AH: Kilitlenme Dortgeni (Gecersiz: {conf:.2f})"
                lock_start_time = None

            cv2.rectangle(frame, (s_x1, s_y1), (s_x2, s_y2), color, 2)
            cv2.putText(frame, text_ah, (s_x1, s_y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            
            cv2.putText(frame, f"YOLO Conf: {last_yolo_conf:.2f}", (s_x1, s_y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.putText(frame, "HH", (s_cx - 10, s_cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            lock_start_time = None

        if lock_start_time is not None:
            elapsed = time.time() - lock_start_time
            if elapsed >= 4.0:
                snap_name = f"kilit_siha_{frame_count}.jpg"
                cv2.imwrite(snap_name, frame)
                print(f"[BİLGİ] 4 saniye kilitlenildi. Görüntü kaydedildi: {snap_name}")
                
                ignored_trackers.append({'kalman': kalman, 'missed': 0})
                STATE = "SEARCH"
                kalman = None
                draw_bbox = None
                lock_start_time = None
                show_success_text = True
            else:
                counter_text = f"{elapsed:.1f} sn"
                text_color = (0, 0, 255)
                cv2.putText(frame, counter_text, (s_av_x2 - 180, s_av_y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                            
        if show_success_text:
            cv2.putText(frame, "Kilitlenme Basarili!", (s_av_x2 - 250, s_av_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
