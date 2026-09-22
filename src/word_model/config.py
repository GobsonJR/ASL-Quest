"""Dedicated configuration for the WLASL100 word-level pipeline.

This module is independent of the alphabet ResNet18 settings in ``src.config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT

WORDS_RAW_ROOT = PROJECT_ROOT / "dataset" / "asl_words_train"
VIDEO_ROOT = WORDS_RAW_ROOT / "videos"
NSLT_100_PATH = WORDS_RAW_ROOT / "nslt_100.json"
WLASL_JSON_PATH = WORDS_RAW_ROOT / "WLASL_v0.3.json"
CLASS_LIST_PATH = WORDS_RAW_ROOT / "wlasl_class_list.txt"

WLASL100_DIR = PROJECT_ROOT / "dataset" / "wlasl100"
MANIFEST_PATH = WLASL100_DIR / "manifest.csv"
CLASSES_PATH = WLASL100_DIR / "classes.txt"

WORD_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "wlasl100"
DATASET_SUMMARY_PATH = WORD_OUTPUTS_DIR / "dataset_summary.json"
VALIDATION_REPORT_PATH = WORD_OUTPUTS_DIR / "validation_report.json"
TRAINING_HISTORY_PATH = WORD_OUTPUTS_DIR / "training_history.json"
EVALUATION_RESULTS_PATH = WORD_OUTPUTS_DIR / "evaluation_results.json"
CONFUSION_MATRIX_CSV_PATH = WORD_OUTPUTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_PNG_PATH = WORD_OUTPUTS_DIR / "confusion_matrix.png"
WORD_MODELS_DIR = PROJECT_ROOT / "models" / "word"
WORD_BEST_MODEL_PATH = WORD_MODELS_DIR / "wlasl100_resnet18_gru_best.pth"
WORD_LAST_MODEL_PATH = WORD_MODELS_DIR / "wlasl100_resnet18_gru_last.pth"

NUM_CLASSES = 100
FRAME_COUNT = 16
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3
BACKBONE = "resnet18"
RNN_TYPE = "gru"
FREEZE_BACKBONE = True
WEIGHT_DECAY = 1e-4
SEED = 42
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0
FEW_VIDEOS_THRESHOLD = 5

ALLOWED_FRAME_COUNTS = (8, 16, 24, 32)


@dataclass(frozen=True)
class WordModelConfig:
    dataset_root: Path = WORDS_RAW_ROOT
    manifest_path: Path = MANIFEST_PATH
    classes_path: Path = CLASSES_PATH
    video_root: Path = VIDEO_ROOT
    outputs_dir: Path = WORD_OUTPUTS_DIR
    models_dir: Path = WORD_MODELS_DIR
    num_classes: int = NUM_CLASSES
    frame_count: int = FRAME_COUNT
    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    learning_rate: float = LEARNING_RATE
    num_epochs: int = NUM_EPOCHS
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    backbone: str = BACKBONE
    rnn_type: str = RNN_TYPE
    freeze_backbone: bool = FREEZE_BACKBONE
    weight_decay: float = WEIGHT_DECAY
    seed: int = SEED
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = EARLY_STOPPING_MIN_DELTA
    imagenet_mean: tuple[float, float, float] = IMAGENET_MEAN
    imagenet_std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.frame_count not in ALLOWED_FRAME_COUNTS:
            raise ValueError(f"frame_count must be one of {ALLOWED_FRAME_COUNTS}, got {self.frame_count}")
        if self.num_classes != NUM_CLASSES:
            raise ValueError(f"Phase 6A expects {NUM_CLASSES} WLASL100 classes")


def default_config(**overrides) -> WordModelConfig:
    return WordModelConfig(**overrides)
