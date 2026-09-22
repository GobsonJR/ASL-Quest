"""Configuration for the ASL Citizen 100-class temporal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT

# External dataset (read-only; never modified by this pipeline).
ASL_CITIZEN_ROOT = Path(r"D:\ASL_Citizen\ASL_Citizen")
ASL_CITIZEN_SPLITS_DIR = ASL_CITIZEN_ROOT / "splits"
ASL_CITIZEN_VIDEOS_DIR = ASL_CITIZEN_ROOT / "videos"

# Project-local manifest and outputs (100-class subset).
DATA_DIR = PROJECT_ROOT / "data" / "asl_citizen_100"
CLASSES_CSV_PATH = DATA_DIR / "classes.csv"
TRAIN_MANIFEST_PATH = DATA_DIR / "train.csv"
VAL_MANIFEST_PATH = DATA_DIR / "val.csv"
TEST_MANIFEST_PATH = DATA_DIR / "test.csv"
CLASS_TO_IDX_PATH = DATA_DIR / "class_to_idx.json"
IDX_TO_CLASS_PATH = DATA_DIR / "idx_to_class.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "asl_citizen_100"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "asl_citizen_100_resnet18_gru_best.pth"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "asl_citizen_100_resnet18_gru_last.pth"

DATA_DIR_20 = PROJECT_ROOT / "data" / "asl_citizen_20"
OUTPUTS_DIR_20 = PROJECT_ROOT / "outputs" / "asl_citizen_20"

NUM_CLASSES = 100
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
SEED = 42
EARLY_STOPPING_PATIENCE = 5
FREEZE_BACKBONE = True
BACKBONE_TRAIN_MODE: Literal["frozen", "layer4", "full"] = "frozen"
HEAD_LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 1e-4
TRAINING_HISTORY_PATH = METRICS_DIR / "training_history.json"
TRAIN_JITTER_FRAMES = 2  # max +/- frame jitter around uniform samples during training


@dataclass(frozen=True)
class ASLCitizenConfig:
    dataset_root: Path = ASL_CITIZEN_ROOT
    video_root: Path = ASL_CITIZEN_VIDEOS_DIR
    data_dir: Path = DATA_DIR
    train_manifest: Path = TRAIN_MANIFEST_PATH
    val_manifest: Path = VAL_MANIFEST_PATH
    test_manifest: Path = TEST_MANIFEST_PATH
    class_to_idx_path: Path = CLASS_TO_IDX_PATH
    idx_to_class_path: Path = IDX_TO_CLASS_PATH
    outputs_dir: Path = OUTPUTS_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    num_classes: int = NUM_CLASSES
    num_frames: int = NUM_FRAMES
    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_epochs: int = NUM_EPOCHS
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    seed: int = SEED
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    freeze_backbone: bool = FREEZE_BACKBONE
    backbone_train_mode: str = BACKBONE_TRAIN_MODE
    head_learning_rate: float = HEAD_LEARNING_RATE
    backbone_learning_rate: float = BACKBONE_LEARNING_RATE
    train_jitter_frames: int = TRAIN_JITTER_FRAMES
    experiment_name: str = "asl_citizen_100"
    classes_csv_path: Path = CLASSES_CSV_PATH
    reports_dir: Path = REPORTS_DIR
    metrics_dir: Path = METRICS_DIR
    logs_dir: Path = LOGS_DIR
    training_history_path: Path = TRAINING_HISTORY_PATH
    best_checkpoint_path: Path = BEST_CHECKPOINT_PATH
    last_checkpoint_path: Path = LAST_CHECKPOINT_PATH
    consistent_clip_transforms: bool = True
    imagenet_mean: tuple[float, float, float] = IMAGENET_MEAN
    imagenet_std: tuple[float, float, float] = IMAGENET_STD


def default_config(**overrides) -> ASLCitizenConfig:
    return ASLCitizenConfig(**overrides)


def experiment_config(experiment: str = "asl_citizen_100", **overrides) -> ASLCitizenConfig:
    """Build config for an isolated experiment. Never overwrites another experiment's paths."""
    name = str(experiment).strip().lower().replace("-", "_")
    if name in {"100", "asl_citizen_100"}:
        return default_config(**overrides)
    if name in {"20", "asl_citizen_20"}:
        data_dir = DATA_DIR_20
        outputs = OUTPUTS_DIR_20
        checkpoints = outputs / "checkpoints"
        metrics = outputs / "metrics"
        base = {
            "experiment_name": "asl_citizen_20",
            "data_dir": data_dir,
            "train_manifest": data_dir / "train.csv",
            "val_manifest": data_dir / "val.csv",
            "test_manifest": data_dir / "test.csv",
            "class_to_idx_path": data_dir / "class_to_idx.json",
            "idx_to_class_path": data_dir / "idx_to_class.json",
            "classes_csv_path": data_dir / "classes.csv",
            "outputs_dir": outputs,
            "checkpoints_dir": checkpoints,
            "reports_dir": outputs / "reports",
            "metrics_dir": metrics,
            "logs_dir": outputs / "logs",
            "training_history_path": metrics / "training_history.json",
            "best_checkpoint_path": checkpoints / "asl_citizen_20_resnet18_gru_best.pth",
            "last_checkpoint_path": checkpoints / "asl_citizen_20_resnet18_gru_last.pth",
            "num_classes": 20,
            "consistent_clip_transforms": True,
        }
        base.update(overrides)
        return ASLCitizenConfig(**base)
    raise ValueError(f"Unknown ASL Citizen experiment: {experiment}")
