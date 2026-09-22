from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import HAND_LANDMARKER_PATH


@dataclass
class HandCrop:
    image: Image.Image
    box: tuple[int, int, int, int] | None
    detected: bool


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

    def crop_rgb(self, rgb: np.ndarray, padding: float = 0.30) -> HandCrop:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        pil_full = Image.fromarray(rgb)
        landmarks = self._landmarks(rgb)
        if not landmarks:
            return HandCrop(image=pil_full, box=None, detected=False)

        h, w, _ = rgb.shape
        xs = [point[0] * w for point in landmarks]
        ys = [point[1] * h for point in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x1 = max(0, int(x1 - bw * padding))
        y1 = max(0, int(y1 - bh * padding))
        x2 = min(w, int(x2 + bw * padding))
        y2 = min(h, int(y2 + bh * padding))
        if x2 <= x1 or y2 <= y1:
            return HandCrop(image=pil_full, box=None, detected=False)
        crop = pil_full.crop((x1, y1, x2, y2))
        return HandCrop(image=crop, box=(x1, y1, x2, y2), detected=True)

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

    def close(self) -> None:
        if self._legacy_hands is not None:
            self._legacy_hands.close()
            self._legacy_hands = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
