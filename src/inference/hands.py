from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import HAND_LANDMARKER_PATH


# Minimum side (pixels) of the square window handed to the classifier. Guards against a degenerate
# landmark box (e.g. all points collapsing) producing a 1-pixel "crop".
MIN_CROP_SIDE = 32


@dataclass
class HandCrop:
    image: Image.Image
    # Square crop window actually given to the model, clipped to the frame (for drawing an overlay).
    box: tuple[int, int, int, int] | None
    detected: bool
    # Raw MediaPipe landmark bounding box (before padding/squaring), for diagnostics only.
    hand_box: tuple[float, float, float, float] | None = None


@dataclass
class HandObservation:
    """Full MediaPipe result for one hand, for the landmark classifier and for dataset export."""

    landmarks: list[tuple[float, float, float]]  # 21 points, x/y normalised to the image, z relative depth
    handedness: str | None  # MediaPipe's raw label "Left"/"Right"
    handedness_score: float | None

    @property
    def actual_hand(self) -> str | None:
        """"left"/"right" for the person's real hand in an UNMIRRORED camera frame.

        MediaPipe labels hands as if the image were a mirrored selfie, so for the camera-native (unmirrored)
        frames this app uses, its "Left" is the person's right hand and vice versa.
        """
        if self.handedness is None:
            return None
        return {"Left": "right", "Right": "left"}.get(self.handedness)


def square_window(
    box: tuple[float, float, float, float], padding: float, min_side: int = MIN_CROP_SIDE
) -> tuple[int, int, int, int]:
    """Square, aspect-preserving window centred on a hand box.

    The classifier was trained on whole, roughly square photos and then resizes its input to a
    square. Cropping to the landmark rectangle and stretching it to 224x224 distorts the hand
    (measured: 69.9% vs 92.9% accuracy on the same held-out images, see
    scripts/diagnose_crop_strategies.py). `padding` is the margin added on every side as a fraction
    of the larger box side. The returned window may extend past the frame; crop_square() fills that.
    """
    x1, y1, x2, y2 = box
    side = max(max(x2 - x1, y2 - y1) * (1 + 2 * padding), float(min_side))
    half = int(round(side / 2))
    cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
    return cx - half, cy - half, cx + half, cy + half


def crop_square(rgb: np.ndarray, window: tuple[int, int, int, int]) -> Image.Image:
    """Crop `window` from `rgb`; any part outside the frame is edge-replicated (never black bars)."""
    left, top, right, bottom = window
    h, w = rgb.shape[:2]
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - w), max(0, bottom - h)
    if pad_l or pad_t or pad_r or pad_b:
        rgb = np.pad(rgb, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return Image.fromarray(np.ascontiguousarray(rgb[top + pad_t : bottom + pad_t, left + pad_l : right + pad_l]))


class HandDetector:
    """Crop a hand region using MediaPipe when a compatible runtime is available."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: Path | None = None,
    ) -> None:
        self._legacy_hands = None
        self._landmarker = None
        self._mp_image_cls = None
        self._image_format = None
        self._video_mode = False
        self._timestamp_ms = 0
        self.error: str | None = None
        model_path = model_path or HAND_LANDMARKER_PATH
        try:
            self._init_legacy(max_hands, detection_confidence, static_image_mode)
            if self._legacy_hands is None:
                self._init_tasks(max_hands, detection_confidence, static_image_mode, model_path)
        except Exception as exc:
            self._legacy_hands = None
            self._landmarker = None
            self.error = str(exc)

    def _init_legacy(self, max_hands: int, detection_confidence: float, static_image_mode: bool) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            return
        self._legacy_hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )

    def _init_tasks(
        self,
        max_hands: int,
        detection_confidence: float,
        static_image_mode: bool,
        model_path: Path,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe Hands Tasks model not found at {model_path}. "
                "Download hand_landmarker.task into models/."
            )
        from mediapipe.tasks.python.core import base_options as base_options_module
        from mediapipe.tasks.python.vision import hand_landmarker
        from mediapipe.tasks.python.vision.core import image as image_module
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

        running_mode = (
            running_mode_module.VisionTaskRunningMode.IMAGE
            if static_image_mode
            else running_mode_module.VisionTaskRunningMode.VIDEO
        )
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
        self._mp_image_cls = image_module.Image
        self._image_format = image_module.ImageFormat.SRGB
        self._video_mode = running_mode == running_mode_module.VisionTaskRunningMode.VIDEO

    @property
    def available(self) -> bool:
        return self._legacy_hands is not None or self._landmarker is not None

    def crop_bgr(self, frame_bgr: np.ndarray, padding: float = 0.30) -> HandCrop:
        rgb = frame_bgr[:, :, ::-1]
        return self.crop_rgb(rgb, padding=padding)

    def crop_pil(self, image: Image.Image, padding: float = 0.30) -> HandCrop:
        rgb = np.array(image.convert("RGB"))
        return self.crop_rgb(rgb, padding=padding)

    def landmark_box(self, rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        """Pixel bounding box (x1, y1, x2, y2) of the detected hand's landmarks, or None."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        landmarks = self._landmarks(rgb)
        if not landmarks:
            return None
        h, w, _ = rgb.shape
        xs = [point[0] * w for point in landmarks]
        ys = [point[1] * h for point in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    def crop_rgb(self, rgb: np.ndarray, padding: float = 0.30, observation: HandObservation | None = None) -> HandCrop:
        """Square crop around the hand. Pass `observation` (from observe()) to reuse one detection for crop and landmarks."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        pil_full = Image.fromarray(rgb)
        if observation is not None:
            h, w = rgb.shape[:2]
            xs = [p[0] * w for p in observation.landmarks]
            ys = [p[1] * h for p in observation.landmarks]
            hand_box = (min(xs), min(ys), max(xs), max(ys))
        else:
            hand_box = self.landmark_box(rgb)
        if hand_box is None:
            return HandCrop(image=pil_full, box=None, detected=False)

        h, w, _ = rgb.shape
        window = square_window(hand_box, padding)
        left, top, right, bottom = window
        clipped = (max(0, left), max(0, top), min(w, right), min(h, bottom))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            # The hand box lies entirely outside the frame: treat as not detected.
            return HandCrop(image=pil_full, box=None, detected=False)
        return HandCrop(image=crop_square(rgb, window), box=clipped, detected=True, hand_box=hand_box)

    def _landmarks(self, rgb: np.ndarray) -> list[tuple[float, float]] | None:
        if self._legacy_hands is not None:
            result = self._legacy_hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            return [(lm.x, lm.y) for lm in result.multi_hand_landmarks[0].landmark]
        if self._landmarker is None or self._mp_image_cls is None or self._image_format is None:
            return None
        mp_image = self._mp_image_cls(self._image_format, rgb)
        if self._video_mode:
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        return [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

    def observe(self, rgb: np.ndarray) -> HandObservation | None:
        """Landmarks (x, y, z) and handedness of the most confident hand, or None."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self._legacy_hands is not None:
            result = self._legacy_hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            points = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
            label = score = None
            if result.multi_handedness:
                label = result.multi_handedness[0].classification[0].label
                score = float(result.multi_handedness[0].classification[0].score)
            return HandObservation(points, label, score)
        if self._landmarker is None or self._mp_image_cls is None or self._image_format is None:
            return None
        mp_image = self._mp_image_cls(self._image_format, rgb)
        if self._video_mode:
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        label = score = None
        if result.handedness and result.handedness[0]:
            label = result.handedness[0][0].category_name
            score = float(result.handedness[0][0].score)
        return HandObservation([(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]], label, score)

    def close(self) -> None:
        if self._legacy_hands is not None:
            self._legacy_hands.close()
            self._legacy_hands = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
