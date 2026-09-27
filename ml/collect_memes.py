"""Curates and renders the 8 high-resolution meme artwork cards and manifest.json."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np

from ml.features import MEME_CLASSES, get_canonical_pose


def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _draw_stylized_character(canvas: np.ndarray, class_idx: int, accent_bgr: tuple[int, int, int]) -> None:
    h, w, _ = canvas.shape
    kp = get_canonical_pose(class_idx, mirror=False).copy()

    kp[:, 0] = (kp[:, 0] - 320.0) * 1.35 + (w * 0.5)
    kp[:, 1] = (kp[:, 1] - 260.0) * 1.35 + (h * 0.54)

    yy, xx = np.ogrid[:h, :w]
    cx, cy = w * 0.5, h * 0.50
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    spotlight = np.clip(1.0 - dist / 380.0, 0.0, 1.0) ** 1.8
    for c in range(3):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c].astype(np.float32) + spotlight * (accent_bgr[c] * 0.42 + 35.0),
            0,
            255,
        ).astype(np.uint8)

    l_sh = tuple(kp[5].astype(int))
    r_sh = tuple(kp[6].astype(int))
    l_hip = tuple(kp[11].astype(int))
    r_hip = tuple(kp[12].astype(int))
    torso_pts = np.array([l_sh, r_sh, r_hip, l_hip], dtype=np.int32)

    suit_color = (38, 38, 42) if class_idx == 0 else (45, 52, 68)
    cv2.fillConvexPoly(canvas, torso_pts, suit_color, cv2.LINE_AA)
    cv2.polylines(canvas, [torso_pts], True, accent_bgr, 3, cv2.LINE_AA)

    sh_mid = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
    chest_mid = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2 + 85)
    if class_idx == 0:
        shirt_pts = np.array(
            [[sh_mid[0] - 28, sh_mid[1]], [sh_mid[0] + 28, sh_mid[1]], [chest_mid[0], chest_mid[1]]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, shirt_pts, (235, 235, 240), cv2.LINE_AA)
        cv2.circle(canvas, (sh_mid[0], sh_mid[1] + 18), 8, (25, 25, 30), -1, cv2.LINE_AA)

    nose = tuple(kp[0].astype(int))
    head_center = (nose[0], nose[1] - 8)
    skin_color = (210, 210, 215) if class_idx == 0 else (155, 190, 230)
    cv2.circle(canvas, head_center, 54, skin_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, 54, accent_bgr, 3, cv2.LINE_AA)

    if class_idx == 0:
        cv2.ellipse(canvas, (head_center[0], head_center[1] - 15), (55, 44), 0, 180, 360, (220, 220, 225), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (head_center[0] - 38, head_center[1] - 12), (head_center[0] - 8, head_center[1] + 10), (20, 20, 20), 4, cv2.LINE_AA)
        cv2.rectangle(canvas, (head_center[0] + 8, head_center[1] - 12), (head_center[0] + 38, head_center[1] + 10), (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(canvas, (head_center[0] - 8, head_center[1] - 2), (head_center[0] + 8, head_center[1] - 2), (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(canvas, (head_center[0] - 38, head_center[1] - 22), (head_center[0] - 8, head_center[1] - 18), (40, 40, 40), 6, cv2.LINE_AA)
        cv2.line(canvas, (head_center[0] + 8, head_center[1] - 18), (head_center[0] + 38, head_center[1] - 22), (40, 40, 40), 6, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (head_center[0] - 18, head_center[1] - 4), 6, (25, 25, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, (head_center[0] + 18, head_center[1] - 4), 6, (25, 25, 30), -1, cv2.LINE_AA)
        if class_idx == 5:
            cv2.ellipse(canvas, (head_center[0], head_center[1] + 24), (14, 18), 0, 0, 360, (30, 30, 35), -1, cv2.LINE_AA)
        elif class_idx in (1, 2, 7):
            cv2.ellipse(canvas, (head_center[0], head_center[1] + 18), (18, 10), 0, 10, 170, (30, 30, 35), 4, cv2.LINE_AA)
        else:
            cv2.line(canvas, (head_center[0] - 14, head_center[1] + 22), (head_center[0] + 14, head_center[1] + 22), (30, 30, 35), 4, cv2.LINE_AA)

    arm_chains = [(5, 7, 9), (6, 8, 10)]
    for sh_i, el_i, wr_i in arm_chains:
        p_sh = tuple(kp[sh_i].astype(int))
        p_el = tuple(kp[el_i].astype(int))
        p_wr = tuple(kp[wr_i].astype(int))

        cv2.line(canvas, p_sh, p_el, accent_bgr, 26, cv2.LINE_AA)
        cv2.line(canvas, p_el, p_wr, accent_bgr, 24, cv2.LINE_AA)
        cv2.line(canvas, p_sh, p_el, (65, 72, 86), 18, cv2.LINE_AA)
        cv2.line(canvas, p_el, p_wr, (80, 88, 104), 16, cv2.LINE_AA)

        cv2.circle(canvas, p_el, 11, accent_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_wr, 20, skin_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_wr, 20, accent_bgr, 3, cv2.LINE_AA)

        if class_idx in (0, 3, 5):
            for angle_deg in (-50, -25, 0, 25, 50):
                rad = np.deg2rad(angle_deg - 90)
                fx = int(p_wr[0] + 30 * np.cos(rad))
                fy = int(p_wr[1] + 30 * np.sin(rad))
                cv2.line(canvas, p_wr, (fx, fy), skin_color, 6, cv2.LINE_AA)


def generate_meme_assets(output_dir: Path) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, str]] = []

    for idx, meta in enumerate(MEME_CLASSES):
        w, h = 800, 800
        canvas = np.full((h, w, 3), (18, 20, 26), dtype=np.uint8)
        accent_bgr = _hex_to_bgr(meta["accent_hex"])

        if idx == 0:
            canvas[:] = (16, 16, 18)

        _draw_stylized_character(canvas, idx, accent_bgr)

        if idx == 0:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(canvas, (0, 0), (w, 108), (10, 11, 15), -1)
        cv2.rectangle(canvas, (0, h - 135), (w, h), (10, 11, 15), -1)
        cv2.line(canvas, (0, 108), (w, 108), accent_bgr, 3)
        cv2.line(canvas, (0, h - 135), (w, h - 135), accent_bgr, 3)

        badge_text = f"MEME #{idx + 1}  |  POSE MATCHED"
        cv2.putText(
            canvas,
            badge_text,
            (36, 66),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            accent_bgr,
            2,
            cv2.LINE_AA,
        )

        title = meta["title"]
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.65 if len(title) <= 16 else 1.30
        thickness = 4
        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = max(24, (w - tw) // 2)
        ty = h - 68

        cv2.putText(canvas, title, (tx + 3, ty + 3), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(canvas, title, (tx, ty), font, font_scale, (250, 250, 252), thickness, cv2.LINE_AA)

        sub = meta["subtitle"]
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        sx = max(20, (w - sw) // 2)
        cv2.putText(canvas, sub, (sx, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (185, 195, 210), 1, cv2.LINE_AA)

        img_filename = f"{meta['id']}.png"
        img_path = output_dir / img_filename
        cv2.imwrite(str(img_path), canvas)

        entry = dict(meta)
        entry["index"] = idx
        entry["image"] = img_filename
        manifest_entries.append(entry)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"classes": manifest_entries}, f, indent=2)

    return manifest_entries


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    generate_meme_assets(project_root / "assets" / "memes")
