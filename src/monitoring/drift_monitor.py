"""Data drift and model performance monitoring using Evidently."""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

from src.utils.logger import logger


class ModelMonitor:
    """Monitor model performance and data drift in production."""

    def __init__(self, config: dict, reference_data: pd.DataFrame):
        self.config = config
        self.reference_data = reference_data
        self.prediction_log: list[dict] = []

    def log_prediction(self, features: list[float], prediction: int, probability: float):
        """Log a single prediction for monitoring."""
        self.prediction_log.append(
            {"features": features, "prediction": prediction, "probability": probability}
        )

    def check_data_drift(self, current_data: pd.DataFrame) -> dict:
        """Detect data drift between reference and current data."""
        logger.info("checking_data_drift", current_rows=len(current_data))

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)

        result = report.as_dict()
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]

        logger.info("drift_check_complete", drift_detected=drift_detected)
        return {
            "drift_detected": drift_detected,
            "report": result,
        }

    def check_performance(
        self, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series | None = None
    ) -> dict:
        """Check if model performance has degraded."""
        from sklearn.metrics import precision_score, recall_score

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        thresholds = self.config["monitoring"]["alert_threshold"]
        alerts = []

        if precision < (1 - thresholds["precision_drop"]):
            alerts.append(f"Precision dropped to {precision:.4f}")

        if recall < (1 - thresholds["recall_drop"]):
            alerts.append(f"Recall dropped to {recall:.4f}")

        result = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "alerts": alerts,
            "status": "degraded" if alerts else "healthy",
        }

        if alerts:
            logger.warning("performance_degradation", **result)
        else:
            logger.info("performance_check_ok", **result)

        return result

    def generate_drift_report(self, current_data: pd.DataFrame, output_path: str):
        """Generate and save an HTML drift report."""
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)
        report.save_html(output_path)
        logger.info("drift_report_saved", path=output_path)

    def get_prediction_stats(self) -> dict:
        """Get statistics from logged predictions."""
        if not self.prediction_log:
            return {"total": 0}

        predictions = [p["prediction"] for p in self.prediction_log]
        probabilities = [p["probability"] for p in self.prediction_log]

        return {
            "total": len(predictions),
            "fraud_flagged": sum(predictions),
            "fraud_rate": round(sum(predictions) / len(predictions), 4),
            "avg_probability": round(np.mean(probabilities), 4),
            "max_probability": round(max(probabilities), 4),
        }
