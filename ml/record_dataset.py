"""Interactive standalone tool to record true webcam data for meme gestures and train on GPU."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2
import numpy as np
import torch

from ml.features import FEATURE_DIM, MEME_CLASSES, extract_pose_features, get_canonical_pose
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


def draw_pose_hint_overlay(
    frame: np.ndarray,
    class_idx: int,
    meme_thumb: np.ndarray | None,
) -> None:
    """Draws a semi-transparent target canonical skeleton guide and meme thumbnail hint."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    kp = get_canonical_pose(class_idx, mirror=False)

    # Draw ghost torso and head guide
    nose = (int(kp[0, 0]), int(kp[0, 1]))
    cv2.circle(overlay, nose, 32, (255, 200, 80), 2, cv2.LINE_AA)

    for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]:
        pt_a = (int(kp[a, 0]), int(kp[a, 1]))
        pt_b = (int(kp[b, 0]), int(kp[b, 1]))
        cv2.line(overlay, pt_a, pt_b, (255, 190, 50), 4, cv2.LINE_AA)

    for i in (5, 6, 7, 8, 9, 10):
        pt = (int(kp[i, 0]), int(kp[i, 1]))
        radius = 10 if i in (9, 10) else 6
        color = (50, 220, 255) if i in (9, 10) else (255, 190, 50)
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)

    # Draw meme thumbnail in top-right below the HUD bar
    if meme_thumb is not None:
        th, tw, _ = meme_thumb.shape
        x0 = w - tw - 12
        y0 = 98
        if x0 >= 0 and y0 + th <= h - 45:
            cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + tw + 2, y0 + th + 2), (255, 185, 40), 2)
            frame[y0 : y0 + th, x0 : x0 + tw] = meme_thumb


def load_meme_thumbnails(assets_dir: Path, thumb_size: int = 110) -> list[np.ndarray | None]:
    thumbs: list[np.ndarray | None] = []
    for meta in MEME_CLASSES:
        img_path = assets_dir / meta.get("image", f"{meta['id']}.png")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if img_path.exists() else None
        if img is not None:
            thumbs.append(cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA))
        else:
            thumbs.append(None)
    return thumbs


def detect_working_cameras(max_probe: int = 10) -> List[Tuple[int, str]]:
    """Scans system for active video capture devices and returns list of (index, label)."""
    candidates: List[Tuple[int, str]] = []
    sysfs_v4l = Path("/sys/class/video4linux")
    if sysfs_v4l.exists():
        for p in sorted(sysfs_v4l.glob("video*"), key=lambda x: int(x.name.replace("video", "") or 0)):
            try:
                idx = int(p.name.replace("video", ""))
                name_file = p / "name"
                name = name_file.read_text().strip() if name_file.exists() else f"Camera {idx}"
                candidates.append((idx, f"/dev/{p.name} ({name})"))
            except ValueError:
                continue
    else:
        candidates = [(i, f"Camera {i}") for i in range(max_probe)]

    working: List[Tuple[int, str]] = []
    for idx, label in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working.append((idx, label))
            cap.release()
    return working


def open_camera_stream(device_idx: int) -> Optional[cv2.VideoCapture]:
    """Configures and opens video capture for the given camera index."""
    cap = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_camera_arg(arg_val: Optional[str]) -> Optional[int]:
    """Parses integer or /dev/videoX string into an integer index."""
    if arg_val is None or arg_val.strip() == "":
        return None
    val = arg_val.strip()
    if val.startswith("/dev/video"):
        try:
            return int(val.replace("/dev/video", ""))
        except ValueError:
            return None
    try:
        return int(val)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record true webcam data for meme gestures and train on GPU.")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera device index (e.g. 0, 2) or path (e.g. /dev/video2). If omitted, automatically detects active camera.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List all detected camera devices and exit")
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

    # Detect all available cameras
    detected = detect_working_cameras()
    if args.list_cameras:
        print("[Cameras] Detected working capture devices:")
        if not detected:
            print("  (None found)")
        for idx, label in detected:
            print(f"  Index {idx}: {label}")
        return

    # Determine initial camera index
    chosen_idx: Optional[int] = parse_camera_arg(args.camera)
    if chosen_idx is None:
        if detected:
            chosen_idx = detected[0][0]
            print(f"[Camera] Auto-selected camera index {chosen_idx}: {detected[0][1]}")
        else:
            chosen_idx = 0
            print("[Camera] Warning: No working camera detected by probe; attempting default index 0...")
    else:
        label = next((lbl for idx, lbl in detected if idx == chosen_idx), f"Index {chosen_idx}")
        print(f"[Camera] Using manually specified camera: {label}")

    cap = open_camera_stream(chosen_idx)
    if cap is None:
        print(f"\n[Error] Failed to open camera index {chosen_idx}!")
        if detected:
            print("[Camera] Working cameras detected on your system:")
            for idx, label in detected:
                print(f"  --camera {idx}   ({label})")
        return

    current_cam_idx = chosen_idx
    meme_thumbs = load_meme_thumbnails(project_root / "assets" / "memes")
    num_classes = len(MEME_CLASSES)

    print("  Controls:")
    print(f"    [1 - {num_classes}]  Select target meme gesture to record")
    print("    [H]      Toggle visual target pose skeleton & meme hint overlay")
    print("    [C]      Cycle to next available camera device")
    print("    [SPACE]  Start 3-second countdown and record pose burst")
    print("    [T]      Train PyTorch model on GPU & export ONNX/CUDA")
    print("    [Q/ESC]  Quit")
    print("=" * 76)

    # Initialize YOLOv8n-pose
    model = YOLO(str(project_root / "yolov8n-pose.pt"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    selected_class_idx = 0
    show_hint_overlay = True
    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start = 0.0
    recorded_features = []
    recorded_kps = []
    status_msg = f"Ready. Select gesture (1-{num_classes}) and press SPACE to record."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Lost camera feed, waiting...")
            time.sleep(0.05)
            continue

        # Mirror camera horizontally so left/right feels natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        display = frame.copy()

        if show_hint_overlay:
            draw_pose_hint_overlay(display, selected_class_idx, meme_thumbs[selected_class_idx])

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
                cv2.rectangle(display, (0, 0), (w, 90), (15, 23, 42), -1)
                cv2.putText(display, cd_text, (w // 2 - 180, 42), cv2.FONT_HERSHEY_DUPLEX, 1.15, (0, 220, 255), 3, cv2.LINE_AA)
                cv2.putText(display, cls_info["hint"], (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 215, 235), 1, cv2.LINE_AA)
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
            cv2.rectangle(display, (0, 0), (w, 90), (180, 40, 20), -1)
            cv2.putText(display, f"RECORDING '{cls_info['title']}': {rec_count}/{args.samples}", (24, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, cls_info["hint"], (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 245), 1, cv2.LINE_AA)

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
            cv2.putText(
                display,
                f"Target [{selected_class_idx + 1}/{num_classes}]: {cls_info['title']}",
                (20, 34),
                cv2.FONT_HERSHEY_DUPLEX,
                0.80,
                (255, 185, 40),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(display, cls_info["hint"], (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 215, 235), 1, cv2.LINE_AA)

            # Bottom Status Bar with Camera label
            cam_label = f"Cam: /dev/video{current_cam_idx} [C: Switch]"
            cv2.rectangle(display, (0, h - 45), (w, h), (15, 23, 42), -1)
            cv2.putText(display, status_msg, (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (110, 240, 160), 1, cv2.LINE_AA)
            cv2.putText(display, cam_label, (w - 230, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 230), 1, cv2.LINE_AA)

        cv2.imshow("Meme Gesture Recorder - Upper Body Tracking", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif ord('1') <= key <= ord(str(min(9, num_classes))):
            selected_class_idx = key - ord('1')
            status_msg = f"Selected [{selected_class_idx + 1}]: {MEME_CLASSES[selected_class_idx]['title']}. Press SPACE to record."
        elif key in (ord('h'), ord('H')):
            show_hint_overlay = not show_hint_overlay
        elif key in (ord('c'), ord('C')):
            # Cycle to next camera
            if len(detected) > 1:
                cur_pos = next((i for i, (idx, _) in enumerate(detected) if idx == current_cam_idx), 0)
                next_pos = (cur_pos + 1) % len(detected)
                next_idx = detected[next_pos][0]
                new_cap = open_camera_stream(next_idx)
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    current_cam_idx = next_idx
                    status_msg = f"Switched to camera index {current_cam_idx}."
                    print(f"[Camera] Switched to {detected[next_pos][1]}")
                else:
                    status_msg = f"Failed to switch to camera index {next_idx}."
            else:
                status_msg = f"Only 1 camera active (/dev/video{current_cam_idx})."
        elif key == 32:  # SPACE
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
        elif key in (ord('t'), ord('T')):
            print("[Train] Launching GPU training on recorded webcam samples...")
            status_msg = "Training model on GPU... please wait."
            train_and_export_classifier(project_root)
            status_msg = "GPU Training complete! Updated ONNX & CUDA models exported."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
