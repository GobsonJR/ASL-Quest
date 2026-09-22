from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BEST_MODEL_PATH
from src.inference.engine import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an ASL letter from an image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model if args.model else (BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else None)
    predictor = Predictor(model_path=model_path)
    result = predictor.predict_path(args.image, top_k=args.top_k)
    print(f"Predicted Letter: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    if result["low_confidence"]:
        print("Note: Low confidence")
    print("\nTop Predictions:")
    for i, item in enumerate(result["top_predictions"], start=1):
        print(f"{i}. {item['label']} - {item['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
