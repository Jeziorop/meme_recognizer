"""End-to-end orchestrator for the Python ML pipeline (Meme Asset Generation + GPU Training + ONNX Export)."""

from __future__ import annotations
from pathlib import Path
import torch

from ml.collect_memes import generate_meme_assets
from ml.fetch_pose_backbone import build_and_export_pose_backbone
from ml.train import train_and_export_classifier


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / "assets" / "memes"
    models_dir = project_root / "models"

    print("=" * 72)
    print("  REAL-TIME MEME GESTURE RECOGNIZER - GPU ML PIPELINE")
    print("=" * 72)

    manifest = generate_meme_assets(assets_dir)
    print(f"[1/3] Generated {len(manifest)} meme cards in {assets_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    build_and_export_pose_backbone(models_dir, device=device)
    print(f"[2/3] Exported Stage-1 Pose Backbone ONNX to {models_dir / 'pose_backbone.onnx'}")

    metrics = train_and_export_classifier(project_root)
    print(f"[3/3] Completed GPU training & ONNX/CUDA export (Val Acc: {metrics['val_accuracy'] * 100:.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
