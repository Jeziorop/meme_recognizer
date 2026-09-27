"""Interactive standalone tool to record true webcam data for meme gestures and train on GPU."""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from ml.features import FEATURE_DIM, MEME_CLASSES, extract_pose_features
from ml.train import train_and_export_classifier
from ultralytics import YOLO

# Upper-body bones connecting keypoints (nose=0, eyes=1,2, ears=3,4, shoulders=5,6, elbows=7,8, wrists=9,10)
UPPER_BODY_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Facial features
    (5, 6),                          # Shoulder span
    (5, 7), (7, 9),                  # Left arm
    (6, 8), (8, 10),                 # Right arm
]


def draw_upper_body_skeleton(frame: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.35) -> None:
    for a, b in UPPER_BODY_BONES:
        if keypoints[a, 2] >= min_conf and keypoints[b, 2] >= min_conf:
            pt_a = (int(keypoints[a, 0]), int(keypoints[a, 1]))
            pt_b = (int(keypoints[b, 0]), int(keypoints[b, 1]))
            cv2.line(frame, pt_a, pt_b, (20, 20, 25), 6, cv2.LINE_AA)
            cv2.line(frame, pt_a, pt_b, (0, 235, 175), 3, cv2.LINE_AA)

    for i in range(11):
        if keypoints[i, 2] >= min_conf:
            pt = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            color = (40, 180, 255) if i in (9, 10) else (255, 215, 60)
            radius = 9 if i in (9, 10) else 6
            cv2.circle(frame, pt, radius + 2, (15, 15, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record true webcam data for meme gestures and train on GPU.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--samples", type=int, default=100, help="Frames to capture per recording burst (default: 100)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "webcam_dataset"
    kp_dir = data_dir / "keypoints"
    frames_dir = data_dir / "frames"
    kp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("  MEME GESTURE RECOGNIZER - WEBCAM DATASET RECORDER & GPU TRAINER")
    print("=" * 76)
    print("  Controls:")
    print("    [1 - 8]  Select target meme gesture to record")
    print("    [SPACE]  Start 3-second countdown and record pose burst")
    print("    [T]      Train PyTorch model on GPU (Quadro P620) & export ONNX")
    print("    [Q/ESC]  Quit")
    print("=" * 76)

    # Initialize YOLOv8n-pose
    model = YOLO(str(project_root / "yolov8n-pose.pt"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    selected_class_idx = 0
    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start = 0.0
    recorded_features = []
    recorded_kps = []
    status_msg = "Ready. Select gesture (1-8) and press SPACE to record."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror camera horizontally so left/right feels natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display = frame.copy()

        # Run pose estimation
        results = model(frame, imgsz=384, device=device, verbose=False)
        best_kp = None

        if results and len(results) > 0 and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data.cpu().numpy()
            if len(kps_data) > 0:
                best_kp = kps_data[0]  # First detected person
                draw_upper_body_skeleton(display, best_kp)

        current_time = time.time()
        cls_info = MEME_CLASSES[selected_class_idx]

        if state == "COUNTDOWN":
            elapsed = current_time - countdown_start
            remaining = 3.0 - elapsed
            if remaining > 0.0:
                cd_text = f"HOLD POSE IN: {int(np.ceil(remaining))}s"
                cv2.rectangle(display, (0, 0), (w, 80), (15, 23, 42), -1)
                cv2.putText(display, cd_text, (w // 2 - 180, 55), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 220, 255), 3, cv2.LINE_AA)
            else:
                state = "RECORDING"
                recorded_features = []
                recorded_kps = []
                print(f"[Record] Started burst capture for '{cls_info['title']}'...")

        elif state == "RECORDING":
            if best_kp is not None:
                feat = extract_pose_features(best_kp[:, :2])
                recorded_features.append(feat)
                recorded_kps.append(best_kp)

            rec_count = len(recorded_features)
            cv2.rectangle(display, (0, 0), (w, 80), (180, 40, 20), -1)
            cv2.putText(display, f"RECORDING '{cls_info['title']}': {rec_count}/{args.samples}", (24, 52), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            if rec_count >= args.samples:
                state = "IDLE"
                save_path = kp_dir / f"{cls_info['id']}_{int(time.time())}.npz"
                np.savez_compressed(
                    save_path,
                    features=np.asarray(recorded_features, dtype=np.float32),
                    keypoints=np.asarray(recorded_kps, dtype=np.float32),
                    class_index=selected_class_idx,
                    class_id=cls_info["id"],
                )
                status_msg = f"Saved {rec_count} samples for '{cls_info['title']}'! Press T to train."
                print(f"[Record] {status_msg} -> {save_path}")

        else:
            # Top HUD bar
            cv2.rectangle(display, (0, 0), (w, 90), (15, 23, 42), -1)
            cv2.putText(display, f"Target: [{selected_class_idx + 1}] {cls_info['title']}", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 185, 40), 2, cv2.LINE_AA)
            cv2.putText(display, cls_info["hint"], (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 215, 235), 1, cv2.LINE_AA)

            # Bottom Status Bar
            cv2.rectangle(display, (0, h - 45), (w, h), (15, 23, 42), -1)
            cv2.putText(display, status_msg, (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 240, 160), 1, cv2.LINE_AA)

        cv2.imshow("Meme Gesture Recorder - Upper Body Tracking", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif ord('1') <= key <= ord('8'):
            selected_class_idx = key - ord('1')
            status_msg = f"Selected: {MEME_CLASSES[selected_class_idx]['title']}. Press SPACE to record."
        elif key == 32:  # SPACE
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
        elif key in (ord('t'), ord('T')):
            print("[Train] Launching GPU training on recorded webcam samples...")
            status_msg = "Training model on GPU (Quadro P620)... please wait."
            train_and_export_classifier(project_root)
            status_msg = "GPU Training complete! Updated ONNX & CUDA models exported."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
