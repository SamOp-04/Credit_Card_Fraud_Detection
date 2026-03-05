"""Integration tests for the FastAPI serving endpoint."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

from src.serving.app import app, state


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictions."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1]])
    return model


@pytest.fixture
def mock_model_fraud():
    """Create a mock model that returns fraud predictions."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    return model


@pytest.mark.asyncio
async def test_health_check_no_model():
    """Health check should report model not loaded."""
    state.model = None
    state.start_time = 1000.0
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "model_not_loaded"
    assert data["model_loaded"] is False


@pytest.mark.asyncio
async def test_health_check_with_model(mock_model):
    """Health check should report healthy with model loaded."""
    state.model = mock_model
    state.start_time = 1000.0
    state.config = {"evaluation": {"threshold": 0.5}}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_predict_no_model():
    """Predict should return 503 when model not loaded."""
    state.model = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"features": [0.0] * 29})
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_predict_legitimate(mock_model):
    """Predict should return not fraud for low probability."""
    state.model = mock_model
    state.config = {"evaluation": {"threshold": 0.5}}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"features": [0.0] * 29})
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5


@pytest.mark.asyncio
async def test_predict_fraud(mock_model_fraud):
    """Predict should return fraud for high probability."""
    state.model = mock_model_fraud
    state.config = {"evaluation": {"threshold": 0.5}}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"features": [0.0] * 29})
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] > 0.5


@pytest.mark.asyncio
async def test_predict_invalid_features():
    """Predict should reject invalid feature length."""
    state.model = MagicMock()
    state.config = {"evaluation": {"threshold": 0.5}}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={"features": [0.0] * 5})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict(mock_model):
    """Batch prediction should process multiple transactions."""
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    state.model = mock_model
    state.config = {"evaluation": {"threshold": 0.5}}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/predict/batch",
            json={"transactions": [{"features": [0.0] * 29}, {"features": [1.0] * 29}]},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert data["flagged_fraud"] == 1


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Metrics endpoint should return prediction count."""
    state.prediction_count = 42
    state.model = True
    state.start_time = 1000.0
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 42
