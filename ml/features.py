"""54-D scale- and translation-invariant geometric pose feature extractor.

Mathematically identical to src/PoseFeatureExtractor.cpp in C++20.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

FEATURE_DIM: int = 54
NUM_COCO_KEYPOINTS: int = 17

MEME_CLASSES: List[Dict[str, str]] = [
    {
        "id": "absolute_cinema",
        "title": "ABSOLUTE CINEMA",
        "subtitle": "Martin Scorsese - Both hands raised open at head height",
        "hint": "Raise both hands open near head/shoulder height with elbows bent",
        "accent_hex": "#F59E0B",
        "image": "absolute_cinema.png",
    },
    {
        "id": "thinking_monkey",
        "title": "THINKING MONKEY",
        "subtitle": "Thinking Monkey - One hand resting on chin, other arm relaxed",
        "hint": "Place one hand on your chin while keeping the other arm relaxed down",
        "accent_hex": "#3B82F6",
        "image": "thinking_monkey.png",
    },
    {
        "id": "elon",
        "title": "ELON SALUTE",
        "subtitle": "Elon Musk - One arm stretched straight up and forward",
        "hint": "Stretch one arm straight up and forward diagonally while keeping the other arm down",
        "accent_hex": "#EF4444",
        "image": "elon.png",
    },
    {
        "id": "kamehameha",
        "title": "KAMEHAMEHA",
        "subtitle": "Goku - Both arms stretched forward in front next to each other",
        "hint": "Stretch both arms forward in front of your torso with wrists next to each other",
        "accent_hex": "#06B6D4",
        "image": "kamehameha.png",
    },
    {
        "id": "trade",
        "title": "TRADE OFFER",
        "subtitle": "Trade Offer - Both hands pressed together in a praying posture",
        "hint": "Bring both hands together in front of your chest like praying",
        "accent_hex": "#8B5CF6",
        "image": "trade.png",
    },
    {
        "id": "gagri_gagri",
        "title": "GAGRI GAGRI",
        "subtitle": "Gagri Gagri - Scratching the top of your head with both hands",
        "hint": "Place both hands right on the top of your head as if scratching it",
        "accent_hex": "#EC4899",
        "image": "gagri_gagri.png",
    },
    {
        "id": "spongebob",
        "title": "SPONGEBOB BICEPS",
        "subtitle": "SpongeBob Flex - Flexing arm at 90 degrees to show biceps",
        "hint": "Extend one arm out to the side and bend your elbow 90 degrees upward to show your bicep",
        "accent_hex": "#EAB308",
        "image": "spongebob.png",
    },
    {
        "id": "t_pose",
        "title": "T-POSE DOMINANCE",
        "subtitle": "Assert Dominance - Both arms stretched straight horizontally",
        "hint": "Extend both arms straight out horizontally at shoulder level",
        "accent_hex": "#10B981",
        "image": "t_pose.png",
    },
    {
        "id": "chill_guy",
        "title": "CHILL GUY",
        "subtitle": "Just a Chill Guy - Relaxed posture with arms resting down",
        "hint": "Stand in a relaxed posture with both arms resting down at your sides",
        "accent_hex": "#64748B",
        "image": "chill_guy.png",
    },
]


def _unit_vec(ax: float, ay: float, bx: float, by: float) -> Tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    norm = math.hypot(dx, dy)
    inv = 1.0 / max(norm, 1e-6)
    return dx * inv, dy * inv, norm


def extract_pose_features(keypoints_xy: np.ndarray) -> np.ndarray:
    kp = np.asarray(keypoints_xy, dtype=np.float32)
    assert kp.shape[0] >= 11 and kp.shape[1] >= 2

    l_sh_x, l_sh_y = float(kp[5, 0]), float(kp[5, 1])
    r_sh_x, r_sh_y = float(kp[6, 0]), float(kp[6, 1])

    center_x = 0.5 * (l_sh_x + r_sh_x)
    center_y = 0.5 * (l_sh_y + r_sh_y)
    shoulder_dist = math.hypot(l_sh_x - r_sh_x, l_sh_y - r_sh_y)
    scale = max(shoulder_dist, 1e-4)

    nx = np.zeros(11, dtype=np.float32)
    ny = np.zeros(11, dtype=np.float32)
    for i in range(11):
        nx[i] = float(np.clip((float(kp[i, 0]) - center_x) / scale, -4.0, 4.0))
        ny[i] = float(np.clip((float(kp[i, 1]) - center_y) / scale, -4.0, 4.0))

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)

    for i in range(11):
        feat[2 * i] = nx[i]
        feat[2 * i + 1] = ny[i]

    segments = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (0, 9),
        (0, 10),
    ]
    unit_vecs: List[Tuple[float, float]] = []
    for idx, (a, b) in enumerate(segments):
        ux, uy, _ = _unit_vec(nx[a], ny[a], nx[b], ny[b])
        feat[22 + 2 * idx] = ux
        feat[22 + 2 * idx + 1] = uy
        unit_vecs.append((ux, uy))

    sh_r_to_l_x, sh_r_to_l_y, _ = _unit_vec(nx[6], ny[6], nx[5], ny[5])
    sh_l_to_r_x, sh_l_to_r_y = -sh_r_to_l_x, -sh_r_to_l_y

    angle_pairs = [
        ((sh_r_to_l_x, sh_r_to_l_y), unit_vecs[0]),
        (unit_vecs[0], unit_vecs[1]),
        ((sh_l_to_r_x, sh_l_to_r_y), unit_vecs[2]),
        (unit_vecs[2], unit_vecs[3]),
    ]
    for idx, ((u1x, u1y), (u2x, u2y)) in enumerate(angle_pairs):
        cos_t = float(np.clip(u1x * u2x + u1y * u2y, -1.0, 1.0))
        sin_t = float(np.clip(u1x * u2y - u1y * u2x, -1.0, 1.0))
        feat[34 + 2 * idx] = cos_t
        feat[34 + 2 * idx + 1] = sin_t

    feat[42] = math.hypot(nx[9] - nx[0], ny[9] - ny[0])
    feat[43] = math.hypot(nx[10] - nx[0], ny[10] - ny[0])
    feat[44] = math.hypot(nx[9] - nx[3], ny[9] - ny[3])
    feat[45] = math.hypot(nx[10] - nx[4], ny[10] - ny[4])
    feat[46] = math.hypot(nx[9] - nx[6], ny[9] - ny[6])
    feat[47] = math.hypot(nx[10] - nx[5], ny[10] - ny[5])
    feat[48] = math.hypot(nx[9] - nx[10], ny[9] - ny[10])
    feat[49] = ny[9] - ny[0]
    feat[50] = ny[10] - ny[0]
    feat[51] = abs(nx[9] - nx[10])
    feat[52] = 0.5 * (ny[7] + ny[8])
    feat[53] = float(np.clip((nx[9] - nx[5]) * (nx[10] - nx[6]), -4.0, 4.0))

    return feat


def get_canonical_pose(class_idx: int, mirror: bool = False) -> np.ndarray:
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[0] = [320.0, 175.0]
    kp[1] = [334.0, 162.0]
    kp[2] = [306.0, 162.0]
    kp[3] = [348.0, 170.0]
    kp[4] = [292.0, 170.0]
    kp[5] = [370.0, 250.0]
    kp[6] = [270.0, 250.0]
    kp[11] = [355.0, 410.0]
    kp[12] = [285.0, 410.0]
    kp[13] = [355.0, 465.0]
    kp[14] = [285.0, 465.0]
    kp[15] = [355.0, 478.0]
    kp[16] = [285.0, 478.0]

    if class_idx == 0:  # absolute_cinema: Both hands raised open at head height
        kp[7] = [425.0, 265.0]
        kp[8] = [215.0, 265.0]
        kp[9] = [415.0, 182.0]
        kp[10] = [225.0, 182.0]
    elif class_idx == 1:  # thinking_monkey: One hand on chin, other arm relaxed
        kp[7] = [395.0, 305.0]
        kp[9] = [330.0, 210.0]
        kp[8] = [252.0, 325.0]
        kp[10] = [255.0, 390.0]
    elif class_idx == 2:  # elon: One arm stretched straight up and diagonal, other arm down
        kp[7] = [435.0, 185.0]
        kp[9] = [495.0, 120.0]
        kp[8] = [252.0, 325.0]
        kp[10] = [255.0, 390.0]
    elif class_idx == 3:  # kamehameha: Two arms stretched in front next to each other
        kp[7] = [350.0, 265.0]
        kp[8] = [290.0, 265.0]
        kp[9] = [326.0, 285.0]
        kp[10] = [314.0, 285.0]
    elif class_idx == 4:  # trade: Praying posture, elbows wide, hands together in front of chest
        kp[7] = [410.0, 325.0]
        kp[8] = [230.0, 325.0]
        kp[9] = [326.0, 225.0]
        kp[10] = [314.0, 225.0]
    elif class_idx == 5:  # gagri_gagri: Scratching top of head with both hands
        kp[7] = [430.0, 190.0]
        kp[8] = [210.0, 190.0]
        kp[9] = [342.0, 135.0]
        kp[10] = [298.0, 135.0]
    elif class_idx == 6:  # spongebob: Flexing biceps (one arm horizontal out, forearm bent 90 deg up)
        kp[7] = [455.0, 250.0]
        kp[9] = [455.0, 170.0]
        kp[8] = [255.0, 325.0]
        kp[10] = [255.0, 395.0]
    elif class_idx == 7:  # t_pose: Both arms stretched straight horizontally
        kp[7] = [445.0, 250.0]
        kp[8] = [195.0, 250.0]
        kp[9] = [525.0, 250.0]
        kp[10] = [115.0, 250.0]
    else:  # chill_guy: Relaxed posture
        kp[7] = [382.0, 325.0]
        kp[8] = [258.0, 325.0]
        kp[9] = [375.0, 395.0]
        kp[10] = [265.0, 395.0]

    if mirror:
        kp[:, 0] = 640.0 - kp[:, 0]
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for a, b in swap_pairs:
            kp[[a, b]] = kp[[b, a]]

    return kp
