"""ResNet18 frame encoder + GRU/LSTM temporal classifier for WLASL100."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.word_model.config import WordModelConfig, default_config

BackboneTrainMode = Literal["frozen", "layer4", "full"]


def _build_resnet18_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        net = models.resnet18(weights=weights)
    except Exception:
        net = models.resnet18(weights=None)
    feature_dim = int(net.fc.in_features)
    net.fc = nn.Identity()
    return net, feature_dim


class WordTemporalModel(nn.Module):
    """Frame-wise 2D CNN features followed by a temporal RNN head."""

    def __init__(
        self,
        num_classes: int = 100,
        backbone: str = "resnet18",
        rnn_type: str = "gru",
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        backbone_train_mode: BackboneTrainMode = "frozen",
    ) -> None:
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone '{backbone}'. Phase 6A supports resnet18 only.")
        self.backbone_name = backbone
        self.encoder, self.feature_dim = _build_resnet18_backbone(pretrained=pretrained)
        self.backbone_train_mode: BackboneTrainMode = "frozen"
        self.freeze_backbone = freeze_backbone
        self.set_backbone_train_mode("frozen" if freeze_backbone else backbone_train_mode)

        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()

    def encoder_requires_grad(self) -> bool:
        return any(parameter.requires_grad for parameter in self.encoder.parameters())

    def set_backbone_train_mode(self, mode: BackboneTrainMode) -> None:
        """Configure which ResNet18 blocks can receive gradients."""
        if mode not in ("frozen", "layer4", "full"):
            raise ValueError(f"Unsupported backbone_train_mode: {mode}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if mode == "layer4":
            for parameter in self.encoder.layer4.parameters():
                parameter.requires_grad = True
        elif mode == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        self.backbone_train_mode = mode
        self.freeze_backbone = mode == "frozen"

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.encoder_requires_grad():
            self.encoder.eval()
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, C, H, W] -> [B, T, feature_dim]."""
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        if self.encoder_requires_grad():
            features = self.encoder(flat)
        else:
            with torch.no_grad():
                features = self.encoder(flat)
        return features.view(batch, time, self.feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.encode_frames(frames)
        outputs, _hidden = self.rnn(features)
        pooled = outputs[:, -1, :]
        return self.classifier(pooled)

    def forward_debug(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_frames(frames)
        outputs, _hidden = self.rnn(features)
        pooled = outputs[:, -1, :]
        logits = self.classifier(pooled)
        return {
            "input": frames,
            "features": features,
            "temporal": pooled,
            "logits": logits,
        }


def build_word_model(cfg: WordModelConfig | None = None, pretrained: bool = True) -> WordTemporalModel:
    cfg = cfg or default_config()
    mode: BackboneTrainMode = "frozen" if cfg.freeze_backbone else "full"
    return WordTemporalModel(
        num_classes=cfg.num_classes,
        backbone=cfg.backbone,
        rnn_type=cfg.rnn_type,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=pretrained,
        backbone_train_mode=mode,
    )


def trainable_parameter_count(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
