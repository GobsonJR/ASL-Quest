"""Training and evaluation metrics."""

from __future__ import annotations

import torch


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    if logits.numel() == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = sum(int(labels[i] in topk[i]) for i in range(labels.size(0)))
    return correct / labels.size(0)


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
