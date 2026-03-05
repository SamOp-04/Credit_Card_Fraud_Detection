"""Model training pipeline with MLflow experiment tracking."""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib

from src.utils.logger import logger


class ModelTrainer:
    """Handles model training, hyperparameter management, and experiment tracking."""

    MODEL_REGISTRY = {
        "xgboost": XGBClassifier,
        "random_forest": RandomForestClassifier,
        "logistic_regression": LogisticRegression,
    }

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_type = config["model"]["type"]

    def _build_model(self):
        """Instantiate the model from config."""
        if self.model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {self.model_type}. "
                f"Choose from: {list(self.MODEL_REGISTRY.keys())}"
            )

        model_params = self.config["model"].get(self.model_type, {})
        model_class = self.MODEL_REGISTRY[self.model_type]

        logger.info("building_model", model_type=self.model_type, params=model_params)
        return model_class(**model_params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_name: str = "fraud-detection",
    ) -> dict:
        """Train model with MLflow experiment tracking."""
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{self.model_type}_run") as run:
            self.model = self._build_model()

            # Log parameters
            model_params = self.config["model"].get(self.model_type, {})
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param(
                "sampling_strategy",
                self.config["training"]["sampling_strategy"],
            )
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            # Train
            logger.info("training_started", model_type=self.model_type)

            if self.model_type == "xgboost":
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False,
                )
            else:
                self.model.fit(X_train, y_train)

            logger.info("training_complete")

            # Cross-validation score
            cv_folds = self.config["training"]["cross_validation_folds"]
            cv_scores = cross_val_score(
                self._build_model(),
                X_train,
                y_train,
                cv=cv_folds,
                scoring="average_precision",
            )
            mlflow.log_metric("cv_avg_precision_mean", cv_scores.mean())
            mlflow.log_metric("cv_avg_precision_std", cv_scores.std())

            logger.info(
                "cross_validation",
                mean_avg_precision=round(cv_scores.mean(), 4),
                std=round(cv_scores.std(), 4),
            )

            # Log model
            if self.model_type == "xgboost":
                mlflow.xgboost.log_model(self.model, "model")
            else:
                mlflow.sklearn.log_model(self.model, "model")

            return {
                "run_id": run.info.run_id,
                "model_type": self.model_type,
                "cv_avg_precision_mean": round(cv_scores.mean(), 4),
                "cv_avg_precision_std": round(cv_scores.std(), 4),
            }

    def save_model(self, path: str | None = None) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        save_path = path or self.config["serving"]["model_path"]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info("model_saved", path=save_path)
        return save_path

    @staticmethod
    def load_model(path: str):
        """Load a saved model from disk."""
        model = joblib.load(path)
        logger.info("model_loaded", path=path)
        return model

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        fi = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return fi
