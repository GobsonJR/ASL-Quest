"""Safe loading of the EXPERIMENTAL landmark recognizer used for live side-by-side comparison with the ResNet.

Never raises: a missing, corrupt or incompatible file only produces an error message, so the app (and the production
ResNet) keeps working. .joblib files are pickles and loading one can run code, so only files under the project's
`models/` folder are accepted, and the model must match the live feature schema before it is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"


def load_experimental_landmarks(path_text: str, allowed_root: Path = MODELS_ROOT) -> tuple[Any | None, str | None]:
    """Return (classifier, None) on success, (None, None) when the feature is off, or (None, message) when it cannot be used."""
    if not path_text or not path_text.strip():
        return None, None
    try:
        path = Path(path_text.strip())
        if not path.is_absolute():
            path = allowed_root.parent / path
        path = path.resolve()
        if path.suffix.lower() != ".joblib":
            return None, f"experimental landmark model must be a .joblib file, got {path.name!r}"
        if allowed_root.resolve() not in path.parents:
            return None, f"experimental landmark model must live under {allowed_root} (loading a pickle from elsewhere is refused)"
        if not path.is_file():
            return None, f"experimental landmark model not found: {path}"
        from src.landmarks.classifier import LandmarkClassifier
        from src.landmarks.features import FEATURE_DIM

        classifier = LandmarkClassifier(path)  # raises if the class order is not A-Z
        n_features = getattr(classifier.pipeline[0], "n_features_in_", None)
        if n_features != FEATURE_DIM:
            return None, f"{path.name}: it expects {n_features} features but the live pipeline produces {FEATURE_DIM}"
        return classifier, None
    except Exception as exc:  # noqa: BLE001 - any failure must leave the app running
        return None, f"could not load the experimental landmark model: {exc}"


def describe(classifier: Any | None, error: str | None) -> dict[str, Any]:
    """Small, safe description for /health and the UI label."""
    if classifier is None:
        return {"loaded": False, "model_file": None, "model": None, "trained_on": None, "error": error}
    meta = getattr(classifier, "meta", {}) or {}
    trained_on = meta.get("trained_on") or {}
    return {
        "loaded": True,
        "model_file": classifier.path.name,
        "model": meta.get("model"),
        "trained_on": {"rooms": trained_on.get("rooms"), "n_samples": trained_on.get("n_samples")},
        "error": None,
    }
