"""Sentiment inference using Hugging Face transformers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_toolkit.config import BINARY_LABELS, DEFAULT_HF_MODEL, LOCAL_MODEL_DIR


@dataclass(frozen=True)
class SentimentResult:
    """Prediction for a single text input."""

    text: str
    label: str
    confidence: float
    scores: dict[str, float]


def resolve_model_path(model_path: str | Path | None = None) -> str:
    """Return local model directory when it exists, otherwise the HF model id."""
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return str(path)
        return str(model_path)

    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "config.json").exists():
        return str(LOCAL_MODEL_DIR)

    return DEFAULT_HF_MODEL


class SentimentAnalyzer:
    """Load a transformer model and run sentiment classification."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        resolved = resolve_model_path(model_path)
        self.model_id = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForSequenceClassification.from_pretrained(resolved)
        self.model.eval()
        self.id2label = self._build_label_map()

    def _build_label_map(self) -> dict[int, str]:
        if getattr(self.model.config, "id2label", None):
            return {
                int(key): str(value).upper()
                for key, value in self.model.config.id2label.items()
            }
        return {index: label for index, label in enumerate(BINARY_LABELS)}

    def predict(self, text: str) -> SentimentResult:
        """Classify a single text and return label with confidence scores."""
        text = text.strip()
        if not text:
            raise ValueError("Text must not be empty.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        probabilities = torch.softmax(logits, dim=-1).tolist()
        predicted_id = int(torch.argmax(logits).item())
        label = self.id2label.get(predicted_id, BINARY_LABELS[predicted_id])

        scores = {
            self.id2label.get(index, f"LABEL_{index}"): round(probability, 4)
            for index, probability in enumerate(probabilities)
        }

        return SentimentResult(
            text=text,
            label=label,
            confidence=round(probabilities[predicted_id], 4),
            scores=scores,
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classify multiple texts sequentially."""
        return [self.predict(text) for text in texts if text.strip()]


@lru_cache(maxsize=1)
def get_analyzer(model_path: str | None = None) -> SentimentAnalyzer:
    """Return a cached analyzer instance for API and UI reuse."""
    return SentimentAnalyzer(model_path=model_path)
