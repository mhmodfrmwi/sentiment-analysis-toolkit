"""Tests for sentiment inference."""

import pytest

from sentiment_toolkit.inference import SentimentAnalyzer, resolve_model_path


def test_resolve_model_path_prefers_existing_local_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "sentiment-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sentiment_toolkit.inference.LOCAL_MODEL_DIR",
        model_dir,
    )
    assert resolve_model_path() == str(model_dir)


def test_predict_returns_label_and_confidence(mock_model_bundle):
    analyzer = SentimentAnalyzer(model_path="mock-model")
    result = analyzer.predict("I love this")

    assert result.label == "POSITIVE"
    assert result.confidence > 0.5
    assert "POSITIVE" in result.scores
    assert "NEGATIVE" in result.scores


def test_predict_rejects_empty_text(mock_model_bundle):
    analyzer = SentimentAnalyzer(model_path="mock-model")
    with pytest.raises(ValueError, match="must not be empty"):
        analyzer.predict("   ")
