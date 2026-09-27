"""Downloads official pretrained YOLOv8n-pose weights and exports models/yolov8n-pose.onnx."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def export_yolov8_pose(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "yolov8n-pose.onnx"

    print("[PoseBackbone] Loading official pretrained YOLOv8n-pose model...")
    model = YOLO("yolov8n-pose.pt")

    print("[PoseBackbone] Exporting to ONNX format (input 1x3x384x384)...")
    exported_file = model.export(
        format="onnx",
        imgsz=(384, 384),
        opset=17,
        dynamic=False,
        simplify=True,
    )

    exported_path = Path(exported_file)
    if exported_path != onnx_path:
        exported_path.replace(onnx_path)

    # Verify with OpenCV DNN
    cv_net = cv2.dnn.readNetFromONNX(str(onnx_path))
    dummy = np.zeros((1, 3, 384, 384), dtype=np.float32)
    cv_net.setInput(dummy)
    out = cv_net.forward()
    print(f"[PoseBackbone] Successfully verified {onnx_path} with OpenCV DNN (output shape={out.shape})")
    return onnx_path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    export_yolov8_pose(project_root / "models")


if __name__ == "__main__":
    main()
