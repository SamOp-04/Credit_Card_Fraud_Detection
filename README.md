# Credit Card Fraud Detection — MLOps Pipeline

[![CI/CD](https://github.com/<your-username>/Credit_Card_Fraud_Detection/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/<your-username>/Credit_Card_Fraud_Detection/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)

Production-grade MLOps pipeline for detecting fraudulent credit card transactions. Built as an end-to-end **Build → Test → Deploy** project demonstrating real-world ML engineering practices.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Raw Data    │───▶│  Data        │───▶│  Model      │───▶│  Model       │
│  (CSV)       │    │  Pipeline    │    │  Training   │    │  Evaluation  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
                   ┌──────────────┐    ┌─────────────┐            │
                   │  Monitoring  │◀───│  FastAPI     │◀───────────┘
                   │  (Evidently) │    │  Serving     │
                   └──────────────┘    └─────────────┘
                                            │
                   ┌──────────────┐    ┌─────────────┐
                   │  Docker      │◀───│  CI/CD       │
                   │  Container   │    │  (GitHub     │
                   └──────────────┘    │   Actions)   │
                                       └─────────────┘
```

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions with 492 frauds (0.17% positive rate). Features V1–V28 are PCA-transformed for confidentiality.

## Project Structure

```
├── configs/
│   └── config.yaml              # Pipeline configuration
├── Data/
│   └── creditcard.csv           # Raw dataset
├── src/
│   ├── data/
│   │   └── data_pipeline.py     # Loading, validation, preprocessing, SMOTE
│   ├── models/
│   │   ├── trainer.py           # Training with MLflow tracking
│   │   └── evaluator.py         # Metrics, threshold optimization
│   ├── serving/
│   │   └── app.py               # FastAPI REST API
│   ├── monitoring/
│   │   └── drift_monitor.py     # Evidently data drift detection
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── logger.py            # Structured logging
├── tests/
│   ├── unit/                    # Unit tests (data, model)
│   └── integration/             # API integration tests
├── .github/workflows/
│   └── ci-cd.yaml               # GitHub Actions CI/CD
├── Dockerfile                   # Production container
├── docker-compose.yaml          # Multi-service orchestration
├── train.py                     # Main training entrypoint
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Project metadata & pytest config
```

## Key Features

| Category | Details |
|---|---|
| **Data Pipeline** | Automated loading, validation, StandardScaler, SMOTE/undersampling for class imbalance |
| **Models** | XGBoost (default), Random Forest, Logistic Regression — configurable via YAML |
| **Experiment Tracking** | MLflow integration — parameters, metrics, artifacts logged per run |
| **Serving** | FastAPI with single + batch prediction endpoints, health checks, Pydantic validation |
| **Monitoring** | Evidently data drift detection, performance degradation alerts |
| **Testing** | pytest with unit tests (data, model) + integration tests (API), coverage reports |
| **CI/CD** | GitHub Actions — lint → test → Docker build → push to GHCR → deploy |
| **Containerization** | Multi-stage Docker, docker-compose with API + MLflow services |

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Train Model

```bash
python train.py
```

This runs the full pipeline: data loading → preprocessing → SMOTE → training → evaluation → model saved to `models/model.joblib`.

### 3. Serve API

```bash
uvicorn src.serving.app:app --reload
```

API available at `http://localhost:8000` — interactive docs at `/docs`.

### 4. Run Tests

```bash
pytest                          # All tests with coverage
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
```

### 5. Docker

```bash
# Build and run all services
docker compose up --build

# API only
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 1000) |
| `GET` | `/metrics` | Serving metrics |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [-1.36, -0.07, 2.54, 1.38, -0.34, -0.47, -0.08, 0.09, 0.36, -0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, -0.17, -0.01, -0.05, -0.05, 0.01, 0.01, 0.01, 149.62]}'
```

## Configuration

All pipeline parameters are in `configs/config.yaml`:

- **Model type** — switch between `xgboost`, `random_forest`, `logistic_regression`
- **Sampling strategy** — `smote`, `undersampling`, or `none`
- **Evaluation threshold** — adjustable classification threshold
- **Monitoring alerts** — configure precision/recall drop thresholds

## MLOps Practices Demonstrated

- **Version Control** — Git-tracked code, configs, and pipeline definitions
- **Experiment Tracking** — MLflow logging of parameters, metrics, and models
- **Automated Testing** — Unit + integration tests in CI pipeline
- **Containerization** — Docker for reproducible deployment
- **CI/CD** — GitHub Actions for automated lint → test → build → deploy
- **Model Monitoring** — Evidently drift detection + performance tracking
- **Configuration Management** — YAML-driven pipeline, no hardcoded values
- **Structured Logging** — Production-grade observability with structlog
- **API Design** — RESTful endpoints with validation, health checks, batch support

## Tech Stack

Python 3.11 · XGBoost · scikit-learn · FastAPI · MLflow · Evidently · Docker · GitHub Actions · pytest · structlog