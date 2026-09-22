"""Project-wide defaults. Paths are relative to the repository root."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "asl_alphabet_train"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LAST_MODEL_PATH = MODELS_DIR / "last_model.pth"
V2_BEST_MODEL_PATH = MODELS_DIR / "asl_resnet18_v2_best.pth"
V2_LAST_MODEL_PATH = MODELS_DIR / "asl_resnet18_v2_last.pth"
LEGACY_MODEL_PATH = PROJECT_ROOT / "asl_resnet18_best.pth"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
SPLITS_PATH = OUTPUTS_DIR / "splits.json"

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42
EARLY_STOP_PATIENCE = 3
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7

ARCHITECTURE = "resnet18"
