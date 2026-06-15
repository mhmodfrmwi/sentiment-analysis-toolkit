"""Tests for the FastAPI application."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sentiment_toolkit.inference import SentimentResult


@pytest.fixture
def client(mock_model_bundle):
    from api.main import app

    with patch("api.main._analyzer") as analyzer_factory:
        analyzer = analyzer_factory.return_value
        analyzer.model_id = "mock-model"
        analyzer.predict.return_value = SentimentResult(
            text="I love this",
            label="POSITIVE",
            confidence=0.91,
            scores={"NEGATIVE": 0.09, "POSITIVE": 0.91},
        )
        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_id"] == "mock-model"


def test_analyze_endpoint(client):
    response = client.post("/analyze", json={"text": "I love this"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "POSITIVE"
    assert payload["confidence"] == 0.91


def test_analyze_batch_endpoint(client):
    response = client.post(
        "/analyze/batch",
        json={"texts": ["Great", "Terrible"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["summary"]["POSITIVE"] == 2


def test_analyze_file_endpoint(client):
    csv_content = "text\nAmazing product\n"
    response = client.post(
        "/analyze/file?text_column=text",
        files={"file": ("reviews.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows_analyzed"] == 1
    assert "csv" in payload
