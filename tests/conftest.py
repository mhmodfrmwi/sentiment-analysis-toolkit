"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def mock_model_bundle():
    """Patch Hugging Face model loading for fast, offline tests."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    model = MagicMock()
    logits = torch.tensor([[0.2, 0.8]])
    model.return_value = MagicMock(logits=logits)
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.eval = MagicMock()

    with (
        patch("sentiment_toolkit.inference.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "sentiment_toolkit.inference.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ),
    ):
        yield tokenizer, model
