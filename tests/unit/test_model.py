"""Unit tests for model trainer and evaluator."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


@pytest.fixture
def config():
    return {
        "project": {"random_seed": 42},
        "model": {
            "type": "logistic_regression",
            "logistic_regression": {
                "class_weight": "balanced",
                "max_iter": 100,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
            },
        },
        "training": {
            "sampling_strategy": "none",
            "smote_ratio": 0.5,
            "cross_validation_folds": 3,
        },
        "evaluation": {
            "threshold": 0.5,
            "metrics": ["precision", "recall", "f1_score"],
        },
        "serving": {"model_path": "models/test_model.joblib"},
    }


@pytest.fixture
def synthetic_data():
    """Generate synthetic binary classification data."""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
    # Create a linearly separable problem
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    split = int(n * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


class TestModelTrainer:
    """Tests for ModelTrainer."""

    def test_build_model_logistic(self, config):
        trainer = ModelTrainer(config)
        model = trainer._build_model()
        assert model.__class__.__name__ == "LogisticRegression"

    def test_build_model_xgboost(self, config):
        config["model"]["type"] = "xgboost"
        trainer = ModelTrainer(config)
        model = trainer._build_model()
        assert model.__class__.__name__ == "XGBClassifier"

    def test_build_model_invalid(self, config):
        config["model"]["type"] = "invalid_model"
        trainer = ModelTrainer(config)
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._build_model()

    @patch("src.models.trainer.mlflow")
    def test_train_returns_results(self, mock_mlflow, config, synthetic_data):
        X_train, X_test, y_train, y_test = synthetic_data
        mock_mlflow.active_run.return_value = None
        mock_mlflow.start_run.return_value.__enter__ = lambda s: type(
            "Run", (), {"info": type("Info", (), {"run_id": "test123"})()}
        )()
        mock_mlflow.start_run.return_value.__exit__ = lambda *a: None

        trainer = ModelTrainer(config)
        results = trainer.train(X_train, y_train, X_test, y_test)

        assert "run_id" in results
        assert "cv_avg_precision_mean" in results
        assert trainer.model is not None

    def test_save_model_without_training(self, config):
        trainer = ModelTrainer(config)
        with pytest.raises(RuntimeError, match="No trained model"):
            trainer.save_model()

    def test_feature_importance_without_training(self, config):
        trainer = ModelTrainer(config)
        with pytest.raises(RuntimeError, match="No trained model"):
            trainer.get_feature_importance(["f1", "f2"])


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_evaluate_returns_metrics(self, config, synthetic_data):
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = synthetic_data
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator(config)
        with patch("src.models.evaluator.mlflow") as mock_mlflow:
            mock_mlflow.active_run.return_value = None
            metrics = evaluator.evaluate(model, X_test, y_test, log_to_mlflow=False)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_find_optimal_threshold(self, config, synthetic_data):
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = synthetic_data
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator(config)
        result = evaluator.find_optimal_threshold(model, X_test, y_test)

        assert "optimal_threshold" in result
        assert 0 <= result["optimal_threshold"] <= 1

    def test_generate_report(self, config, synthetic_data):
        from sklearn.linear_model import LogisticRegression

        X_train, X_test, y_train, y_test = synthetic_data
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator(config)
        report = evaluator.generate_report(model, X_test, y_test)

        assert "Legitimate" in report
        assert "Fraud" in report
        assert "precision" in report
