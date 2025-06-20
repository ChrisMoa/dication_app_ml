# config.py
# Configuration file for dictation_app_ml

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
CONVERTED_MODELS_DIR = MODELS_DIR / "converted"
MOBILE_MODELS_DIR = MODELS_DIR / "mobile"
OPTIMIZED_MODELS_DIR = MODELS_DIR / "optimized"

# Default model name (configurable)
DEFAULT_MODEL_NAME = "dictation_gec_model"  # Changed from german_gec_mt5

# Training configuration
TRAINING_CONFIG = {
    "model_name": "google/mt5-small",
    "output_dir": str(TRAINED_MODELS_DIR),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "learning_rate": 5e-5,
    "max_length": 64,
}

# Server configuration
SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8001,
    "model_path": str(TRAINED_MODELS_DIR / "final_model"),
}

# Create directories
for directory in [MODELS_DIR, DATA_DIR, TRAINED_MODELS_DIR, 
                  CONVERTED_MODELS_DIR, MOBILE_MODELS_DIR, OPTIMIZED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
