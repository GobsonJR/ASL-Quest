"""Geometric features from MediaPipe's 21 hand landmarks.

The point of the representation: a letter should look the same wherever the hand is in the frame, at any distance,
in either hand, and (optionally) at any in-plane rotation. Appearance (skin, face, room) is thrown away entirely.
That also discards cues the image model can use (finger contact, hidden thumbs for A/S/E/M/N/T), so whether this
helps is decided by measurement, not assumed.

Landmark indices: 0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20 (MCP, PIP, DIP, TIP).
"""

from __future__ import annotations

import numpy as np

WRIST = 0
TIPS = (4, 8, 12, 16, 20)
MCPS = (1, 5, 9, 13, 17)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
THUMB = (1, 2, 3, 4)
N_LANDMARKS = 21


class LandmarkError(ValueError):
    pass


def as_points(landmarks) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (N_LANDMARKS, 3):
        raise LandmarkError(f"expected 21 landmarks with x, y, z, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise LandmarkError("landmarks contain non-finite values")
    return pts


def canonicalize(landmarks, hand: str | None = "right", *, rotate: bool = True) -> np.ndarray:
    """Wrist at the origin, palm length 1, every hand mapped to a right hand, optionally palm pointing up.

    `hand` is the person's real hand in an UNMIRRORED frame ("left" is mirrored so all samples look alike).
    Unknown (None) is treated as right.
    """
    pts = as_points(landmarks).copy()
    if hand == "left":
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    palm = float(np.linalg.norm(pts[9, :2]))  # wrist -> middle-finger MCP, in the image plane
    if palm < 1e-9:
        raise LandmarkError("degenerate hand (zero palm length)")
    pts /= palm
    if rotate:
        # rotate in the image plane so the wrist->middle-MCP direction points up (negative y)
        angle = np.arctan2(-pts[9, 0], -pts[9, 1])
        c, s = np.cos(angle), np.sin(angle)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = c * x - s * y
        pts[:, 1] = s * x + c * y
    return pts


def _angle_cos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """cos of the angle at b between segments b->a and b->c (1 = straight back on itself, -1 = fully straight)."""
    u, v = a - b, c - b
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def feature_vector(landmarks, hand: str | None = "right", *, rotate: bool = True) -> np.ndarray:
    p = canonicalize(landmarks, hand, rotate=rotate)
    feats: list[float] = list(p[1:].flatten())  # 60 normalised coordinates (the wrist is always the origin)
    tips = p[list(TIPS)]
    for i in range(len(TIPS)):  # tip-to-tip distances (10)
        for j in range(i + 1, len(TIPS)):
            feats.append(float(np.linalg.norm(tips[i] - tips[j])))
    for tip in TIPS:  # tip-to-wrist distance (5)
        feats.append(float(np.linalg.norm(p[tip])))
    for mcp in MCPS[1:]:  # thumb tip to each finger MCP (4): finger contact / curl cues
        feats.append(float(np.linalg.norm(p[4] - p[mcp])))
    for tip in TIPS[1:]:  # thumb tip to each fingertip is already above; add thumb tip to each finger PIP (4)
        pip = {8: 6, 12: 10, 16: 14, 20: 18}[tip]
        feats.append(float(np.linalg.norm(p[4] - p[pip])))
    for chain in FINGERS.values():  # flexion angles at PIP and DIP (8)
        mcp, pip, dip, tip = chain
        feats.append(_angle_cos(p[mcp], p[pip], p[dip]))
        feats.append(_angle_cos(p[pip], p[dip], p[tip]))
    feats.append(_angle_cos(p[1], p[2], p[3]))  # thumb MCP and IP flexion (2)
    feats.append(_angle_cos(p[2], p[3], p[4]))
    return np.asarray(feats, dtype=np.float64)


FEATURE_DIM = len(feature_vector(np.random.default_rng(0).random((N_LANDMARKS, 3)) + np.arange(N_LANDMARKS)[:, None] * 0.01))


def augment_landmarks(landmarks, rng: np.random.Generator, *, max_rotation_deg: float = 15.0, noise: float = 0.01, scale: float = 0.1) -> np.ndarray:
    """Training-time jitter that keeps the letter: small in-plane rotation, scale change and landmark noise."""
    pts = as_points(landmarks).copy()
    centre = pts[WRIST]
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    c, s = np.cos(angle), np.sin(angle)
    rel = pts - centre
    x, y = rel[:, 0].copy(), rel[:, 1].copy()
    rel[:, 0], rel[:, 1] = c * x - s * y, s * x + c * y
    rel *= 1.0 + rng.uniform(-scale, scale)
    palm = np.linalg.norm(rel[9, :2]) + 1e-9
    rel += rng.normal(0.0, noise * palm, rel.shape)
    return rel + centre
