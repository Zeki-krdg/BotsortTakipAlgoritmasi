import cv2
import time
from ultralytics import YOLO
from pathlib import Path
from ostrackAlgorithm import OptimizedOSTrack

# Ayarlar
MODEL_PATH = Path("runs/detect/yolov8_kamikaze_960/weights/best.pt")
VIDEO_PATH = Path("video/kamikaze.mp4")

# Parametreler 
# 🚨 FIX 1: Orijinal 0.50'den 0.65'e çıkarıldı. (Arka plan eşleşmesi önlendi)
TRACK_SCORE_THRESHOLD = 0.62
YOLO_CONF = 0.25  
MAX_LOST = 5 

def main():
    if not MODEL_PATH.exists() or not VIDEO_PATH.exists():
        print("Hata: Model veya Video dosyası bulunamadı!")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    tracker = OptimizedOSTrack()
    
    frame_count = 0
    lost_counter = 0
    STATE = "DETECT" 

    print("\n--- PRO LEVEL HIBRIT TRACKER BAŞLATILDI ---")
    print("Mimarideki saplantılı takip (Drift) düzeltildi!\n")

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- STATE: TRACKING ---
        if STATE == "TRACKING":
            # 🚨 FIX 2: 2 kareden fazla kayıpsa GLOBAL SEARCH (Tüm Frame'de Ara) başlar!
            full_frame_search = (lost_counter > 2)
            bbox, score = tracker.track(frame, full_frame_search=full_frame_search)
            
            # 🚨 FIX 5: YOLO'ya ACİL KAÇIŞ (FAILSAFE)
            # Eğer skor dibi gördüyse (0.55 altı) bekleme, direkt at. Uçak büyük ihtimal kadrajdan çıktı veya arkasına geçti.
            if score < 0.55:
                STATE = "DETECT"
                print(f"[Kare {frame_count:04d}] FAILSAFE AKTİF! Kesin Koptu. (Score: {score:.2f})")
            else:
                if score < TRACK_SCORE_THRESHOLD:
                    lost_counter += 1
                else:
                    lost_counter = 0 # Hedef iyi, sayacı sıfırlıyoruz

                if lost_counter > MAX_LOST:
                    STATE = "DETECT"
                else:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"OSTrack: {score:.2f}", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    if full_frame_search:
                         cv2.putText(frame, "GLOBAL SEARCH", (x, y - 25), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # --- STATE: DETECT ---
        if STATE == "DETECT":
            results = model.predict(frame, conf=YOLO_CONF, verbose=False)
            
            if len(results[0].boxes) > 0:
                top_box = results[0].boxes[0]
                conf_score = float(top_box.conf[0])
                bbox = top_box.xyxy[0].cpu().numpy() 
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                if tracker.init(frame, (x1, y1, w, h)):
                    STATE = "TRACKING" 
                    lost_counter = 0 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"YOLO: {conf_score:.2f}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # --- EKRAN VE LOG BİLGİLERİ ---
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-5) 
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"STATE: {STATE}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if STATE=="DETECT" else (0, 255, 0), 2)

        # Temiz Terminal Çıktısı
        print(f"[Kare {frame_count:04d}] {STATE:<8} | LSTS: {lost_counter} | FPS: {fps:.1f}")

        # Canlı Göster
        cv2.imshow("Hibrit Takip: YOLO + OSTRACK", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nİşlem sonlandırıldı.")

if __name__ == "__main__":
    main()
