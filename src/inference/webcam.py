from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
from src.inference.engine import Predictor
from src.inference.hands import HandDetector
from src.utils.smoothing import TemporalSmoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time ASL webcam recognition")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--window", type=int, default=SMOOTHING_WINDOW)
    parser.add_argument("--no-hands", action="store_true", help="Classify the full frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model else (BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else None)
    predictor = Predictor(model_path=model_path)
    detector = HandDetector(static_image_mode=False) if not args.no_hands else None
    smoother = TemporalSmoother(window=args.window, threshold=args.threshold)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera {args.camera} is unavailable. Close other apps using the webcam and try again."
        )

    prev = time.perf_counter()
    fps = 0.0
    print("Press Q to quit. Dynamic J/Z motion is not supported; each frame is a static A-Z class.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.")
                break
            # Classify the camera-native (unmirrored) frame so left/right signs stay valid.
            # Mirror only the preview.
            display = cv2.flip(frame, 1)
            width = frame.shape[1]
            detector_unavailable = detector is not None and not detector.available
            if detector is not None:
                crop = detector.crop_bgr(frame)
                image = crop.image
                box = crop.box
                hand_found = crop.detected
            else:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                box = None
                hand_found = True

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            if box:
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (width - x2, y1), (width - x1, y2), (0, 200, 180), 2)

            if detector_unavailable:
                label = "Hand detector unavailable"
                conf_text = detector.error or "Run with --no-hands to classify full frames."
                smoother.reset()
            elif not hand_found:
                label = "No hand detected"
                conf_text = ""
                smoother.reset()
            else:
                result = predictor.predict_pil(image)
                stable, confidence, status = smoother.update(
                    str(result["prediction"]), float(result["confidence"])
                )
                if stable is None:
                    label = "Low confidence"
                    conf_text = f"{result['confidence'] * 100:.1f}%"
                else:
                    label = f"Prediction: {stable}"
                    conf_text = f"Confidence: {confidence * 100:.1f}%  ({status})"

            cv2.putText(display, label, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
            if conf_text:
                cv2.putText(display, conf_text, (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 210), 2)
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (24, display.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.imshow("ASL Recognition", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
