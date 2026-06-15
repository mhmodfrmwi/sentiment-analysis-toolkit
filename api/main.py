"""FastAPI application for sentiment analysis."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from sentiment_toolkit.inference import SentimentAnalyzer, get_analyzer
from sentiment_toolkit.preprocessing import read_csv_texts, records_to_csv, split_sentences

app = FastAPI(
    title="Sentiment Analysis API",
    description="REST API for transformer-based sentiment classification.",
    version="1.0.0",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify.")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to classify.")


class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float
    scores: dict[str, float]


class BatchSentimentResponse(BaseModel):
    results: list[SentimentResponse]
    summary: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_id: str


@lru_cache(maxsize=1)
def _analyzer() -> SentimentAnalyzer:
    return get_analyzer()


def _to_response(result) -> SentimentResponse:
    return SentimentResponse(
        text=result.text,
        label=result.label,
        confidence=result.confidence,
        scores=result.scores,
    )


def _build_summary(results: list[SentimentResponse]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in results:
        summary[item.label] = summary.get(item.label, 0) + 1
    return summary


def _build_summary_from_labels(labels: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for label in labels:
        summary[label] = summary.get(label, 0) + 1
    return summary


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    analyzer = _analyzer()
    return HealthResponse(status="ok", model_id=analyzer.model_id)


@app.post("/analyze", response_model=SentimentResponse)
def analyze(request: AnalyzeRequest) -> SentimentResponse:
    try:
        result = _analyzer().predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(result)


@app.post("/analyze/sentences", response_model=BatchSentimentResponse)
def analyze_sentences(request: AnalyzeRequest) -> BatchSentimentResponse:
    sentences = split_sentences(request.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found in text.")

    results = [_to_response(_analyzer().predict(sentence)) for sentence in sentences]
    return BatchSentimentResponse(results=results, summary=_build_summary(results))


@app.post("/analyze/batch", response_model=BatchSentimentResponse)
def analyze_batch(request: BatchAnalyzeRequest) -> BatchSentimentResponse:
    cleaned = [text.strip() for text in request.texts if text.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="No non-empty texts provided.")

    results = [_to_response(_analyzer().predict(text)) for text in cleaned]
    return BatchSentimentResponse(results=results, summary=_build_summary(results))


@app.post("/analyze/file")
async def analyze_file(
  file: Annotated[UploadFile, File(...)],
  text_column: str = Query(default="text", description="CSV column containing text."),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        records = read_csv_texts(content, text_column=text_column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analyzer = _analyzer()
    output_rows: list[dict] = []
    for record in records:
        result = analyzer.predict(record.text)
        output_rows.append(
            {
                "row_index": record.index,
                "text": result.text,
                "label": result.label,
                "confidence": result.confidence,
            }
        )

    csv_content = records_to_csv(
        output_rows,
        fieldnames=["row_index", "text", "label", "confidence"],
    )
    return {
        "filename": file.filename,
        "rows_analyzed": len(output_rows),
        "summary": _build_summary_from_labels([row["label"] for row in output_rows]),
        "csv": csv_content,
    }
