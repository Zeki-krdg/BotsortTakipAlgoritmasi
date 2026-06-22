import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ostrackAlgorithm import OptimizedOSTrack

# ----------------------- CONFIG (edit here only) -----------------------
VIDEO_PATH = Path("video/kamikaze.mp4")
YOLO_WEIGHTS = Path("runs/detect/yolov11_egitimleri/kamikaze_uav_640/weights/best.pt")

# Leave as empty string to disable OSTrack model and use YOLO-first tracking.
OSTRACK_WEIGHTS = ""
OSTRACK_MODEL = "base"  # "base" or "small"

YOLO_CONF = 0.22
YOLO_EVERY = 1
ASSOCIATION_DIST = 260.0
SMOOTH_ALPHA = 0.75
MIN_BOX_AREA = 80.0
PLAYBACK_SPEED = 1.0

LOST_TH = 0.48
RECOVER_TH = 0.62
MAX_LOST = 8
ENABLE_TERMINAL_LOGS = True
# ----------------------------------------------------------------------


def _xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([x1, y1, max(2.0, x2 - x1), max(2.0, y2 - y1)], dtype=np.float32)


def _box_center(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)


def _clip_box(box_xywh, frame_shape):
    h, w = frame_shape[:2]
    x, y, bw, bh = box_xywh
    x = float(np.clip(x, 0, w - 2))
    y = float(np.clip(y, 0, h - 2))
    bw = float(np.clip(bw, 2, w - x))
    bh = float(np.clip(bh, 2, h - y))
    return np.array([x, y, bw, bh], dtype=np.float32)


def _select_detection(dets_xywh, dets_conf, ref_box, max_center_dist):
    """
    Select best detection with confidence + proximity scoring.
    """
    if len(dets_xywh) == 0:
        return None, 0.0

    if ref_box is None:
        best_i = int(np.argmax(dets_conf))
        return dets_xywh[best_i], float(dets_conf[best_i])

    ref_c = _box_center(ref_box)
    best_score = -1e9
    best_box = None
    best_conf = 0.0
    for box, conf in zip(dets_xywh, dets_conf):
        c = _box_center(box)
        dist = float(np.linalg.norm(c - ref_c))
        if dist > max_center_dist:
            continue
        score = 1.2 * float(conf) - 0.0035 * dist
        if score > best_score:
            best_score = score
            best_box = box
            best_conf = float(conf)
    return best_box, best_conf

def main():
    if not VIDEO_PATH.exists():
        print(f"Error: Video not found: {VIDEO_PATH}")
        return
    if not YOLO_WEIGHTS.exists():
        print(f"Error: YOLO weights not found: {YOLO_WEIGHTS}")
        return

    # If OSTrack weights are unavailable, fall back to YOLO-first tracking mode.
    use_ostrack = bool(OSTRACK_WEIGHTS)
    tracker = OptimizedOSTrack(model_type=OSTRACK_MODEL, weight_path=OSTRACK_WEIGHTS) if use_ostrack else None
    detector = YOLO(str(YOLO_WEIGHTS))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 1e-6:
        video_fps = 30.0
    target_fps = max(1.0, video_fps * max(0.1, PLAYBACK_SPEED))
    wait_ms = max(1, int(round(1000.0 / target_fps)))

    frame_count = 0
    low_score_streak = 0
    tracking_active = False
    last_box = None
    last_velocity = np.zeros(2, dtype=np.float32)
    
    print(f"\n--- AUTO SEARCH + TRACKING STARTED --- (Video FPS: {video_fps:.2f}, Display FPS: {target_fps:.2f})")
    print("Press 'ESC' or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        frame_count += 1
        
        run_yolo_now = (frame_count % max(1, YOLO_EVERY) == 0)
        status_text = "SEARCHING"
        full_frame_search = False

        # SEARCH mode: keep scanning with YOLO until a target appears.
        if not tracking_active:
            if run_yolo_now:
                init_results = detector.predict(frame, conf=YOLO_CONF, verbose=False)
                if len(init_results[0].boxes) > 0:
                    boxes_xyxy = init_results[0].boxes.xyxy.cpu().numpy()
                    boxes_conf = init_results[0].boxes.conf.cpu().numpy()
                    dets_xywh = [_xyxy_to_xywh(b) for b in boxes_xyxy if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA]
                    dets_conf = [float(c) for b, c in zip(boxes_xyxy, boxes_conf) if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA]
                    best_box, best_conf = _select_detection(dets_xywh, dets_conf, last_box, ASSOCIATION_DIST)
                    if best_box is not None and (not use_ostrack or tracker.init(frame, tuple(best_box.tolist()))):
                        tracking_active = True
                        low_score_streak = 0
                        status_text = "TRACKING"
                        last_box = _clip_box(best_box, frame.shape)
                        cv2.putText(frame, f"YOLO INIT ({best_conf:.2f})", (int(last_box[0]), max(20, int(last_box[1]) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        status_text = "SEARCHING"
                else:
                    if ENABLE_TERMINAL_LOGS:
                        print(f"[Frame {frame_count:04d}] YOLO_SEARCH | detections=0 | tracking_active={tracking_active}")
            if not tracking_active:
                cv2.putText(frame, "SEARCHING TARGET WITH YOLO...", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # TRACK mode
        if tracking_active:
            if use_ostrack:
                full_frame_search = low_score_streak >= 2
                tracked_bbox, score = tracker.track(frame, full_frame_search=full_frame_search)
                if not tracked_bbox:
                    tracking_active = False
                    low_score_streak = 0
                    status_text = "SEARCHING"
                    tracked_box = None
                    if ENABLE_TERMINAL_LOGS:
                        print(f"[Frame {frame_count:04d}] TRACKER_LOST | source=OSTrack | score=NA | low_streak={low_score_streak}")
                else:
                    tracked_box = np.array(tracked_bbox, dtype=np.float32)
            else:
                # YOLO-first tracking without OSTrack weights:
                # predict next center by simple velocity model, then associate best detection.
                tracked_box = None
                score = 0.0
                predicted_ref = None
                if last_box is not None:
                    predicted_ref = last_box.copy()
                    predicted_ref[0] += float(last_velocity[0])
                    predicted_ref[1] += float(last_velocity[1])
                if run_yolo_now:
                    tr_results = detector.predict(frame, conf=YOLO_CONF, verbose=False)
                    if len(tr_results[0].boxes) > 0:
                        boxes_xyxy = tr_results[0].boxes.xyxy.cpu().numpy()
                        boxes_conf = tr_results[0].boxes.conf.cpu().numpy()
                        dets_xywh = [_xyxy_to_xywh(b) for b in boxes_xyxy if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA]
                        dets_conf = [float(c) for b, c in zip(boxes_xyxy, boxes_conf) if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_BOX_AREA]
                        best_box, best_conf = _select_detection(dets_xywh, dets_conf, predicted_ref, ASSOCIATION_DIST)
                        if best_box is not None:
                            score = best_conf
                            if predicted_ref is None:
                                tracked_box = best_box
                            else:
                                tracked_box = SMOOTH_ALPHA * best_box + (1.0 - SMOOTH_ALPHA) * predicted_ref
                if tracked_box is None:
                    low_score_streak += 1
                    score = 0.0
                    if ENABLE_TERMINAL_LOGS:
                        print(f"[Frame {frame_count:04d}] TRACKER_MISS | source=YOLO-first | low_streak={low_score_streak}")
                    if low_score_streak > MAX_LOST:
                        tracking_active = False
                        low_score_streak = 0
                        status_text = "SEARCHING"
                else:
                    low_score_streak = 0
                    tracked_box = _clip_box(tracked_box, frame.shape)
                    if last_box is not None:
                        last_velocity = _box_center(tracked_box) - _box_center(last_box)
                    last_box = tracked_box.copy()
                    status_text = "TRACKING"
                    if ENABLE_TERMINAL_LOGS:
                        print(f"[Frame {frame_count:04d}] TRACKING | source=YOLO-first | score={score:.2f} | low_streak={low_score_streak}")

            if tracked_box is not None:
                status_text = "TRACKING"
                x, y, w, h = tracked_box.astype(np.int32).tolist()
                if score < LOST_TH:
                    low_score_streak += 1
                elif score > RECOVER_TH:
                    low_score_streak = max(0, low_score_streak - 2)
                else:
                    low_score_streak = max(0, low_score_streak - 1)

                is_lost = low_score_streak > MAX_LOST
                color = (0, 0, 255) if is_lost else ((0, 255, 0) if score > 0.55 else (0, 165, 255))

                # Draw bbox only when coordinates exist.
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"OSTrack Score: {score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"LowScoreStreak: {low_score_streak}", (x, y - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if full_frame_search:
                    cv2.putText(frame, "GLOBAL SEARCH", (x, y - 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                if is_lost:
                    cv2.putText(frame, "TRACK UNSTABLE", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Recovery: if unstable, switch to search and let YOLO reacquire.
                    tracking_active = False
                    low_score_streak = 0
                    status_text = "SEARCHING"
                    last_box = None
                    last_velocity = np.zeros(2, dtype=np.float32)
                    if ENABLE_TERMINAL_LOGS:
                        print(f"[Frame {frame_count:04d}] SWITCH_TO_SEARCH | reason=unstable | source={'OSTrack' if use_ostrack else 'YOLO-first'}")
        
        # Display Info
        cv2.putText(frame, f"Frame: {frame_count}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Mode: {status_text}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if status_text == "TRACKING" else (0, 165, 255), 2)
        if ENABLE_TERMINAL_LOGS:
            print(f"[Frame {frame_count:04d}] MODE={status_text} | run_yolo_now={run_yolo_now} | use_ostrack={use_ostrack}")

        cv2.imshow("OSTrack Run", frame)
        
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
