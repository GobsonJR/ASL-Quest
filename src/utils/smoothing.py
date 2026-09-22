from __future__ import annotations

from collections import Counter, deque

import torch.nn.functional as F
from torch import Tensor


class TemporalSmoother:
    """Majority vote over recent high-confidence predictions."""

    def __init__(self, window: int = 7, threshold: float = 0.70) -> None:
        self.window = window
        self.threshold = threshold
        self._labels: deque[str] = deque(maxlen=window)

    def update(self, label: str, confidence: float) -> tuple[str | None, float, str]:
        if confidence < self.threshold:
            self.reset()
            return None, confidence, "Low confidence"
        self._labels.append(label)
        counts = Counter(self._labels)
        stable, _ = counts.most_common(1)[0]
        status = "Stable" if counts[stable] >= max(2, len(self._labels) // 2 + 1) else "Updating"
        return stable, confidence, status

    def reset(self) -> None:
        self._labels.clear()


def softmax_topk(logits: Tensor, class_names: list[str], k: int = 3) -> list[dict[str, float | str]]:
    probs = F.softmax(logits, dim=1)[0]
    values, indices = probs.topk(min(k, len(class_names)))
    results: list[dict[str, float | str]] = []
    for conf, idx in zip(values.tolist(), indices.tolist()):
        results.append({"label": class_names[idx], "confidence": float(conf)})
    return results
