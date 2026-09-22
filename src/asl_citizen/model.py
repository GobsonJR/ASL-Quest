"""ResNet18 + GRU temporal classifier for ASL Citizen."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.asl_citizen.config import ASLCitizenConfig, default_config

BackboneTrainMode = Literal["frozen", "layer4", "full"]


def _build_resnet18_backbone(pretrained: bool) -> tuple[nn.Module, int, bool]:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    pretrained_loaded = False
    try:
        net = models.resnet18(weights=weights)
        pretrained_loaded = pretrained and weights is not None
    except Exception:
        net = models.resnet18(weights=None)
        pretrained_loaded = False
    feature_dim = int(net.fc.in_features)
    net.fc = nn.Identity()
    return net, feature_dim, pretrained_loaded


class ASLCitizenResNet18GRU(nn.Module):
    """Video clip classifier: [B,T,3,H,W] -> logits [B,num_classes]."""

    def __init__(
        self,
        num_classes: int = 100,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        backbone_train_mode: BackboneTrainMode | None = None,
    ) -> None:
        super().__init__()
        self.encoder, self.feature_dim, self.pretrained_loaded = _build_resnet18_backbone(pretrained=pretrained)
        if backbone_train_mode is None:
            backbone_train_mode = "frozen" if freeze_backbone else "full"
        self.set_backbone_train_mode(backbone_train_mode)

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def encoder_requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.encoder.parameters())

    def set_backbone_train_mode(self, mode: BackboneTrainMode) -> None:
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
        if self.backbone_train_mode == "frozen":
            self.encoder.eval()
        elif self.backbone_train_mode == "layer4":
            self.encoder.eval()
            self.encoder.layer4.train(mode)
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
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
        outputs, _hidden = self.gru(features)
        pooled = outputs[:, -1, :]
        return self.classifier(pooled)


def resolve_backbone_mode(cfg: ASLCitizenConfig) -> BackboneTrainMode:
    mode = str(getattr(cfg, "backbone_train_mode", "") or "").strip().lower()
    if mode in ("frozen", "layer4", "full"):
        return mode  # type: ignore[return-value]
    return "frozen" if cfg.freeze_backbone else "full"


def build_model(cfg: ASLCitizenConfig | None = None, pretrained: bool = True) -> ASLCitizenResNet18GRU:
    cfg = cfg or default_config()
    mode = resolve_backbone_mode(cfg)
    return ASLCitizenResNet18GRU(
        num_classes=cfg.num_classes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        freeze_backbone=mode == "frozen",
        pretrained=pretrained,
        backbone_train_mode=mode,
    )


def parameter_counts(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def encoder_trainability_report(model: ASLCitizenResNet18GRU) -> dict[str, Any]:
    def _trainable(module: nn.Module) -> bool:
        params = list(module.parameters())
        return any(p.requires_grad for p in params) if params else False

    return {
        "backbone_train_mode": model.backbone_train_mode,
        "pretrained_loaded": model.pretrained_loaded,
        "stem_trainable": _trainable(model.encoder.conv1) or _trainable(model.encoder.bn1),
        "layer1_trainable": _trainable(model.encoder.layer1),
        "layer2_trainable": _trainable(model.encoder.layer2),
        "layer3_trainable": _trainable(model.encoder.layer3),
        "layer4_trainable": _trainable(model.encoder.layer4),
        "gru_trainable": _trainable(model.gru),
        "classifier_trainable": _trainable(model.classifier),
        **parameter_counts(model),
    }


def build_optimizer(
    model: ASLCitizenResNet18GRU,
    *,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Discriminative AdamW: backbone params vs GRU/classifier."""
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in list(model.gru.parameters()) + list(model.classifier.parameters()) if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def gradient_flow_report(model: ASLCitizenResNet18GRU) -> dict[str, Any]:
    def _stats(params) -> dict[str, Any]:
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return {"has_grad": False, "mean_abs": 0.0, "finite": True, "numel": 0}
        stacked = torch.cat([g.detach().flatten().float() for g in grads])
        return {
            "has_grad": True,
            "mean_abs": float(stacked.abs().mean()),
            "max_abs": float(stacked.abs().max()),
            "finite": bool(torch.isfinite(stacked).all()),
            "numel": int(stacked.numel()),
        }

    return {
        "layer4": _stats(model.encoder.layer4.parameters()),
        "gru": _stats(model.gru.parameters()),
        "classifier": _stats(model.classifier.parameters()),
        "frozen_layer1_requires_grad": any(p.requires_grad for p in model.encoder.layer1.parameters()),
        "layer4_requires_grad": any(p.requires_grad for p in model.encoder.layer4.parameters()),
        "gru_requires_grad": any(p.requires_grad for p in model.gru.parameters()),
        "classifier_requires_grad": any(p.requires_grad for p in model.classifier.parameters()),
    }
