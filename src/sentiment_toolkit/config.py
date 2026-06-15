"""Application configuration."""

from pathlib import Path

# Project root (repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local fine-tuned model directory (created by notebooks/train_and_evaluate.ipynb)
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "sentiment-model"

# Hugging Face fallback when no local model is available
DEFAULT_HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Labels for the default SST-2 binary classifier
BINARY_LABELS = ("NEGATIVE", "POSITIVE")

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
