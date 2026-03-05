"""Model evaluation and metrics computation."""

import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from src.utils.logger import logger


class ModelEvaluator:
    """Evaluate model performance with comprehensive metrics."""

    def __init__(self, config: dict):
        self.config = config
        self.threshold = config["evaluation"]["threshold"]

    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, log_to_mlflow: bool = True
    ) -> dict:
        """Run full evaluation suite."""
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
            "threshold": self.threshold,
        }

        cm = confusion_matrix(y_test, y_pred)
        metrics["true_negatives"] = int(cm[0][0])
        metrics["false_positives"] = int(cm[0][1])
        metrics["false_negatives"] = int(cm[1][0])
        metrics["true_positives"] = int(cm[1][1])

        # Log to MLflow if in active run
        if log_to_mlflow and mlflow.active_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

        logger.info("evaluation_complete", **{k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def find_optimal_threshold(
        self, model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict:
        """Find threshold that maximizes F1 score."""
        y_proba = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

        # Compute F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

        result = {
            "optimal_threshold": round(optimal_threshold, 4),
            "precision_at_optimal": round(float(precisions[optimal_idx]), 4),
            "recall_at_optimal": round(float(recalls[optimal_idx]), 4),
            "f1_at_optimal": round(float(f1_scores[optimal_idx]), 4),
        }

        logger.info("optimal_threshold_found", **result)
        return result

    def generate_report(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Generate a human-readable classification report."""
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        return classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])
