"""Synchronizes assets/memes/manifest.json with the curated real meme images in assets/memes/."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from ml.features import MEME_CLASSES


def sync_meme_manifest(output_dir: Path) -> List[Dict[str, str]]:
    """Writes manifest.json for the real meme images without generating or overwriting image files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, str]] = []

    for idx, meta in enumerate(MEME_CLASSES):
        img_filename = meta.get("image", f"{meta['id']}.png")
        img_path = output_dir / img_filename
        if not img_path.exists():
            print(f"[MemeManifest] Warning: Expected real meme image not found at {img_path}")

        entry = dict(meta)
        entry["index"] = idx
        entry["image"] = img_filename
        manifest_entries.append(entry)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"classes": manifest_entries}, f, indent=2)

    return manifest_entries


# Alias kept for pipeline compatibility
generate_meme_assets = sync_meme_manifest


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    entries = sync_meme_manifest(project_root / "assets" / "memes")
    print(f"Synchronized manifest.json with {len(entries)} real meme classes.")
