"""Main training pipeline - orchestrates data, training, and evaluation."""

import sys
from pathlib import Path

from src.utils.config import load_config, get_project_root
from src.utils.logger import logger
from src.data.data_pipeline import DataPipeline
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


def main(config_path: str = "configs/config.yaml"):
    """Run the full ML training pipeline."""
    logger.info("pipeline_started")

    # 1. Load config
    config = load_config(config_path)

    # 2. Data pipeline
    data_pipeline = DataPipeline(config)
    X_train, X_test, y_train, y_test = data_pipeline.run_pipeline()
    data_pipeline.save_scaler()

    data_stats = data_pipeline.get_data_stats()
    logger.info("data_stats", **data_stats)

    # 3. Train model
    trainer = ModelTrainer(config)
    train_results = trainer.train(X_train, y_train, X_test, y_test)
    logger.info("training_results", **train_results)

    # 4. Evaluate
    evaluator = ModelEvaluator(config)
    metrics = evaluator.evaluate(trainer.model, X_test, y_test)

    # 5. Find optimal threshold
    threshold_results = evaluator.find_optimal_threshold(trainer.model, X_test, y_test)
    logger.info("threshold_optimization", **threshold_results)

    # 6. Print classification report
    report = evaluator.generate_report(trainer.model, X_test, y_test)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # 7. Feature importance
    fi = trainer.get_feature_importance(list(X_train.columns))
    if not fi.empty:
        print("\nTOP 10 FEATURES:")
        print(fi.head(10).to_string(index=False))

    # 8. Save model
    model_path = trainer.save_model()

    logger.info(
        "pipeline_complete",
        model_path=model_path,
        roc_auc=metrics["roc_auc"],
        avg_precision=metrics["average_precision"],
    )

    return metrics


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(config_file)
