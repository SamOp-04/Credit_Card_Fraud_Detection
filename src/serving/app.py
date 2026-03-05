"""FastAPI application for serving fraud detection predictions."""

import time
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.utils.logger import logger


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TransactionRequest(BaseModel):
    """Single transaction for prediction."""

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Transaction feature vector: Time, V1-V28, Amount (30 features)",
    )

    model_config = {"json_schema_extra": {"examples": [{"features": [0.0] * 30}]}}


class BatchRequest(BaseModel):
    """Batch of transactions for prediction."""

    transactions: list[TransactionRequest] = Field(..., max_length=1000)


class PredictionResponse(BaseModel):
    """Prediction result."""

    is_fraud: bool
    fraud_probability: float
    threshold: float


class BatchResponse(BaseModel):
    """Batch prediction result."""

    predictions: list[PredictionResponse]
    total_transactions: int
    flagged_fraud: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_type: str | None
    uptime_seconds: float


# ── Application state ─────────────────────────────────────────────────────────


class AppState:
    model = None
    scaler = None
    config: dict = {}
    start_time: float = 0.0
    prediction_count: int = 0


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup."""
    state.config = load_config()
    state.start_time = time.time()

    model_path = state.config["serving"]["model_path"]
    scaler_path = "models/scaler.joblib"

    if Path(model_path).exists():
        state.model = joblib.load(model_path)
        logger.info("model_loaded", path=model_path)
    else:
        logger.warning("model_not_found", path=model_path)

    if Path(scaler_path).exists():
        state.scaler = joblib.load(scaler_path)
        logger.info("scaler_loaded", path=scaler_path)

    yield

    logger.info("shutting_down", total_predictions=state.prediction_count)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection for credit card transactions",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_type = type(state.model).__name__ if state.model else None
    return HealthResponse(
        status="healthy" if state.model else "model_not_loaded",
        model_loaded=state.model is not None,
        model_type=model_type,
        uptime_seconds=round(time.time() - state.start_time, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    """Predict fraud for a single transaction."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(request.features).reshape(1, -1)
    threshold = state.config["evaluation"]["threshold"]

    proba = float(state.model.predict_proba(features)[0][1])
    is_fraud = proba >= threshold

    state.prediction_count += 1

    if is_fraud:
        logger.warning(
            "fraud_detected",
            probability=round(proba, 4),
            prediction_id=state.prediction_count,
        )

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 6),
        threshold=threshold,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict fraud for a batch of transactions."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([t.features for t in request.transactions])
    threshold = state.config["evaluation"]["threshold"]

    probas = state.model.predict_proba(features)[:, 1]
    predictions = []
    flagged = 0

    for proba in probas:
        is_fraud = float(proba) >= threshold
        if is_fraud:
            flagged += 1
        predictions.append(
            PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=round(float(proba), 6),
                threshold=threshold,
            )
        )

    state.prediction_count += len(predictions)
    logger.info("batch_prediction", count=len(predictions), flagged=flagged)

    return BatchResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        flagged_fraud=flagged,
    )


@app.get("/metrics")
async def get_metrics():
    """Return basic serving metrics."""
    return {
        "total_predictions": state.prediction_count,
        "uptime_seconds": round(time.time() - state.start_time, 2),
        "model_loaded": state.model is not None,
    }
