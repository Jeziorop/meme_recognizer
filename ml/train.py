"""GPU-accelerated PyTorch training script incorporating real recorded webcam datasets."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.features import (
    FEATURE_DIM,
    MEME_CLASSES,
    extract_pose_features,
    get_canonical_pose,
)
from ml.model import MemePoseClassifier, export_model_artifacts


def _augment_pose(base_kp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    kp = base_kp.copy()
    scale = float(rng.uniform(0.70, 1.35))
    angle_rad = float(np.deg2rad(rng.uniform(-10.0, 10.0)))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    center = np.array([320.0, 250.0], dtype=np.float32)
    rel = (kp - center) * scale
    rot_x = rel[:, 0] * cos_a - rel[:, 1] * sin_a
    rot_y = rel[:, 0] * sin_a + rel[:, 1] * cos_a
    kp[:, 0] = rot_x + center[0] + float(rng.uniform(-60.0, 60.0))
    kp[:, 1] = rot_y + center[1] + float(rng.uniform(-40.0, 40.0))

    joint_std = np.array(
        [3.0, 2.5, 2.5, 3.0, 3.0, 4.0, 4.0, 8.0, 8.0, 10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        dtype=np.float32,
    ) * scale
    noise = rng.normal(0.0, 1.0, size=kp.shape).astype(np.float32) * joint_std[:, None]
    kp += noise
    return kp


def load_real_webcam_dataset(data_dir: Path) -> Tuple[list[np.ndarray], list[int]]:
    webcam_feats = []
    webcam_labels = []
    kp_dir = data_dir / "keypoints"

    if kp_dir.exists():
        for npz_file in kp_dir.glob("*.npz"):
            try:
                data = np.load(npz_file)
                feats = data["features"]
                cls_idx = int(data["class_index"])
                for f in feats:
                    webcam_feats.append(f)
                    webcam_labels.append(cls_idx)
            except Exception as e:
                print(f"[Dataset] Warning loading {npz_file}: {e}")

    return webcam_feats, webcam_labels


def build_pose_dataset(
    project_root: Path,
    samples_per_class: int = 2000,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes = len(MEME_CLASSES)

    syn_feats = []
    syn_labels = []
    for c in range(num_classes):
        for s in range(samples_per_class):
            mirror = bool(s % 2 == 1)
            base_kp = get_canonical_pose(c, mirror=mirror)
            aug_kp = _augment_pose(base_kp, rng) if s > 0 else base_kp
            syn_feats.append(extract_pose_features(aug_kp))
            syn_labels.append(c)

    # Load real webcam recordings if gathered
    webcam_feats, webcam_labels = load_real_webcam_dataset(project_root / "data" / "webcam_dataset")

    all_feats = list(syn_feats)
    all_labels = list(syn_labels)

    if webcam_feats:
        print(f"[Dataset] Loaded {len(webcam_feats)} REAL webcam frames from data/webcam_dataset/keypoints/!")
        # Replicate real webcam samples to give them high weight in training
        multiplier = max(1, 1500 // max(1, len(webcam_feats)))
        for _ in range(multiplier):
            for wf, wl in zip(webcam_feats, webcam_labels):
                jitter = rng.normal(0.0, 0.015, size=wf.shape).astype(np.float32)
                all_feats.append(wf + jitter)
                all_labels.append(wl)

    return np.asarray(all_feats, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)


def train_and_export_classifier(project_root: Path) -> dict:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"[MemeClassifier] Training device: {device} ({gpu_name})")

    features, labels = build_pose_dataset(project_root)

    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(labels))
    features = features[perm]
    labels = labels[perm]

    split = int(0.85 * len(labels))
    x_train = torch.from_numpy(features[:split]).to(device)
    y_train = torch.from_numpy(labels[:split]).to(device)
    x_val = torch.from_numpy(features[split:]).to(device)
    y_val = torch.from_numpy(labels[split:]).to(device)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = MemePoseClassifier(in_features=FEATURE_DIM, hidden_dim=128, num_classes=len(MEME_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_val_acc = 0.0
    for epoch in range(1, 36):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(by)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = float((val_preds == y_val).float().mean().item())
            best_val_acc = max(best_val_acc, val_acc)

    models_dir = project_root / "models"
    onnx_path = models_dir / "meme_gesture_classifier.onnx"
    cuda_bin_path = models_dir / "meme_gesture_classifier.bin"
    export_model_artifacts(model, onnx_path, cuda_bin_path)

    # Verify ONNX numerical equivalence with OpenCV DNN
    cv_net = cv2.dnn.readNetFromONNX(str(onnx_path))
    model.eval()
    max_diff = 0.0
    for c_idx, meta in enumerate(MEME_CLASSES):
        kp = get_canonical_pose(c_idx, mirror=False)
        feat = extract_pose_features(kp).reshape(1, FEATURE_DIM)
        with torch.no_grad():
            pt_logits = model(torch.from_numpy(feat)).numpy()
        cv_net.setInput(feat)
        cv_logits = cv_net.forward()
        diff = float(np.max(np.abs(pt_logits - cv_logits)))
        max_diff = max(max_diff, diff)

    metrics = {
        "device": str(device),
        "gpu_name": gpu_name,
        "val_accuracy": round(best_val_acc, 5),
        "onnx_cv_dnn_max_abs_diff": max_diff,
        "onnx_model_path": str(onnx_path.relative_to(project_root)),
        "cuda_bin_path": str(cuda_bin_path.relative_to(project_root)),
    }
    with open(models_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[MemeClassifier] Exported ONNX & CUDA models | Val Acc: {best_val_acc * 100:.2f}% | "
        f"PyTorch vs cv::dnn max diff: {max_diff:.2e}"
    )
    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_and_export_classifier(project_root)
