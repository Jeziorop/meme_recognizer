"""Utility to record or fine-tune custom meme gesture pose samples and retrain on GPU."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from ml.features import MEME_CLASSES, extract_pose_features, get_canonical_pose
from ml.train import train_and_export_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Add pose calibration samples and retrain ONNX/CUDA models on GPU.")
    parser.add_argument(
        "--class-id",
        type=str,
        default="absolute_cinema",
        choices=[m["id"] for m in MEME_CLASSES],
        help="Target meme class ID to calibrate",
    )
    parser.add_argument("--samples", type=int, default=120, help="Number of calibration samples to append")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cls_idx = next(i for i, m in enumerate(MEME_CLASSES) if m["id"] == args.class_id)

    rng = np.random.default_rng()
    feats = []
    labels = []
    for i in range(args.samples):
        kp = get_canonical_pose(cls_idx, mirror=(i % 2 == 1))
        kp += rng.normal(0.0, 5.0, size=kp.shape).astype(np.float32)
        feats.append(extract_pose_features(kp))
        labels.append(cls_idx)

    custom_path = project_root / "models" / "custom_samples.npz"
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        custom_path,
        features=np.asarray(feats, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )
    print(f"Saved {len(feats)} calibration samples for '{args.class_id}' to {custom_path}")
    train_and_export_classifier(project_root)


if __name__ == "__main__":
    main()
