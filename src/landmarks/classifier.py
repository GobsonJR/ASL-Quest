"""Train, select and run a landmark-based A-Z classifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONFIDENCE_THRESHOLD
from src.landmarks.features import LandmarkError, augment_landmarks, feature_vector

LETTERS = [chr(65 + i) for i in range(26)]

# MediaPipe labels hands as if the image were a mirrored selfie; this app classifies UNMIRRORED frames, so its "Left" is
# the person's right hand. The same mapping must be used when training, evaluating and running live: a model trained with
# one hand convention and run with another sees every hand mirrored (a failure the smoke test exposed).
MEDIAPIPE_TO_ACTUAL_HAND = {"Left": "right", "Right": "left"}


def with_deployed_hand(rows: list[dict[str, Any]], source: str = "mediapipe") -> list[dict[str, Any]]:
    """Rows whose `hand` is the one the live pipeline will use (MediaPipe's, corrected), falling back to the label.

    source="label" keeps the person's own label instead (only sensible if you also deploy with that, which you cannot).
    """
    if source == "label":
        return rows
    out = []
    for row in rows:
        mapped = MEDIAPIPE_TO_ACTUAL_HAND.get(row.get("handedness_raw") or "")
        out.append({**row, "hand": mapped or row.get("hand")})
    return out


def make_xy(rows: list[dict[str, Any]], *, rotate: bool, augment: int = 0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features and labels for rows with `landmarks`, `hand`, `letter`. Augmented copies are TRAINING-only."""
    rng = np.random.default_rng(seed)
    xs, ys, groups = [], [], []
    for row in rows:
        if not row.get("landmarks"):
            continue
        label = LETTERS.index(row["letter"])
        try:
            xs.append(feature_vector(row["landmarks"], row.get("hand"), rotate=rotate))
        except LandmarkError:
            continue
        ys.append(label)
        groups.append(row.get("room", "?"))
        for _ in range(augment):
            try:
                xs.append(feature_vector(augment_landmarks(row["landmarks"], rng), row.get("hand"), rotate=rotate))
            except LandmarkError:
                continue
            ys.append(label)
            groups.append(row.get("room", "?"))
    return np.asarray(xs), np.asarray(ys), groups


def candidate_models(seed: int = 0) -> dict[str, Callable[[], Pipeline]]:
    return {
        "logreg": lambda: Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=2.0))]),
        "mlp": lambda: Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier((128, 64), max_iter=600, early_stopping=True, random_state=seed))]),
        "hgb": lambda: Pipeline([("scale", StandardScaler()), ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed))]),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=sorted(set(y_true.tolist())), average="macro", zero_division=0))


def select_and_train(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    augment: int = 4,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose (feature config x model) by validation macro-F1, then refit on train (+ val when it was only used for selection).

    Honest selection needs data the candidates have not seen from a different room: an unseen-room validation set, or
    at least two training rooms to cross-validate across. Otherwise this refuses instead of guessing.
    """
    rooms = sorted({r.get("room", "?") for r in train_rows if r.get("landmarks")})
    if not val_rows and len(rooms) < 2:
        raise ValueError(
            "model selection needs an unseen-room validation set or at least 2 training rooms; "
            "record another room instead of selecting on data from the one you trained on"
        )
    scores: dict[str, float] = {}
    for rotate in (True, False):
        for name, build in candidate_models(seed).items():
            key = f"{name}/{'rotate' if rotate else 'no-rotate'}"
            if val_rows:
                x_tr, y_tr, _ = make_xy(train_rows, rotate=rotate, augment=augment, seed=seed)
                x_va, y_va, _ = make_xy(val_rows, rotate=rotate)
                scores[key] = macro_f1(y_va, build().fit(x_tr, y_tr).predict(x_va))
            else:
                folds = []
                rows_by_room = {r: [x for x in train_rows if x.get("room") == r] for r in rooms}
                splitter = GroupKFold(n_splits=min(3, len(rooms)))
                idx = np.arange(len(rooms))
                for tr_i, te_i in splitter.split(idx, groups=idx):
                    tr = [x for i in tr_i for x in rows_by_room[rooms[i]]]
                    te = [x for i in te_i for x in rows_by_room[rooms[i]]]
                    x_tr, y_tr, _ = make_xy(tr, rotate=rotate, augment=augment, seed=seed)
                    x_te, y_te, _ = make_xy(te, rotate=rotate)
                    if len(x_te):
                        folds.append(macro_f1(y_te, build().fit(x_tr, y_tr).predict(x_te)))
                scores[key] = float(np.mean(folds)) if folds else 0.0
    best = max(scores, key=scores.get)
    name, cfg = best.split("/")
    rotate = cfg == "rotate"
    final_rows = train_rows + ([] if not val_rows else val_rows)  # val was only used to choose; refit on both
    x, y, _ = make_xy(final_rows, rotate=rotate, augment=augment, seed=seed)
    pipeline = candidate_models(seed)[name]().fit(x, y)
    model = {
        "pipeline": pipeline,
        "classes": LETTERS,
        "rotate": rotate,
        "model": name,
        "trained_on": {"rooms": sorted({r.get("room", "?") for r in final_rows}), "n_samples": int(len(y) / (augment + 1)), "augment_copies": augment},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report = {"selection_scores_macro_f1": scores, "selected": best, "selection_data": "unseen-room validation" if val_rows else "leave-room-out cross-validation"}
    return model, report


def save_model(model: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


class LandmarkClassifier:
    """Runtime wrapper: landmarks in, the same prediction dict the image classifier returns."""

    def __init__(self, path: Path) -> None:
        data = joblib.load(path)
        if data["classes"] != LETTERS:
            raise ValueError(f"{path}: class order is not A-Z")
        self.path = Path(path)
        self.pipeline = data["pipeline"]
        self.rotate = bool(data["rotate"])
        self.meta = {k: data.get(k) for k in ("model", "trained_on", "created_at")}

    def predict_points(self, landmarks, hand: str | None, top_k: int = 3) -> dict[str, Any]:
        features = feature_vector(landmarks, hand, rotate=self.rotate)[None, :]
        probs = self.pipeline.predict_proba(features)[0]
        classes = list(self.pipeline.classes_)
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"label": LETTERS[int(classes[i])], "confidence": float(probs[i])} for i in order]
        return {
            "prediction": top[0]["label"],
            "confidence": top[0]["confidence"],
            "top_predictions": top,
            "low_confidence": top[0]["confidence"] < CONFIDENCE_THRESHOLD,
        }

    def predict_observation(self, observation, top_k: int = 3) -> dict[str, Any] | None:
        if observation is None:
            return None
        return self.predict_points(observation.landmarks, observation.actual_hand, top_k=top_k)
