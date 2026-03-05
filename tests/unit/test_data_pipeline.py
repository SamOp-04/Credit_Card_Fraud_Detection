"""Unit tests for the data pipeline."""

import pytest
import pandas as pd
import numpy as np

from src.data.data_pipeline import DataPipeline


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "project": {"random_seed": 42},
        "data": {"raw_path": "Data/creditcard.csv", "processed_dir": "Data/processed", "test_size": 0.2, "val_size": 0.1},
        "features": {"target_column": "Class", "drop_columns": [], "scale_columns": ["Amount"]},
        "training": {"sampling_strategy": "none", "smote_ratio": 0.5, "cross_validation_folds": 3},
    }


@pytest.fixture
def sample_df():
    """Create a small synthetic fraud dataset."""
    np.random.seed(42)
    n = 1000
    fraud_count = 50

    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Amount"] = np.random.exponential(100, n)
    data["Time"] = np.arange(n, dtype=float)
    data["Class"] = [0] * (n - fraud_count) + [1] * fraud_count

    return pd.DataFrame(data)


@pytest.fixture
def pipeline(sample_config):
    return DataPipeline(sample_config)


class TestDataPipeline:
    """Tests for DataPipeline class."""

    def test_validate_data_valid(self, pipeline, sample_df):
        """Valid data should pass validation without errors."""
        pipeline._validate_data(sample_df)  # Should not raise

    def test_validate_data_missing_target(self, pipeline, sample_df):
        """Missing target column should raise ValueError."""
        df = sample_df.drop(columns=["Class"])
        with pytest.raises(ValueError, match="Target column"):
            pipeline._validate_data(df)

    def test_validate_data_invalid_target(self, pipeline, sample_df):
        """Non-binary target should raise ValueError."""
        sample_df["Class"] = np.random.randint(0, 3, len(sample_df))
        with pytest.raises(ValueError, match="binary"):
            pipeline._validate_data(sample_df)

    def test_preprocess_scales_columns(self, pipeline, sample_df):
        """Amount column should be standardized after preprocessing."""
        original_mean = sample_df["Amount"].mean()
        processed = pipeline.preprocess(sample_df.copy())
        # Scaled column should have ~0 mean
        assert abs(processed["Amount"].mean()) < 0.01

    def test_split_data_proportions(self, pipeline, sample_df):
        """Split should respect configured test size."""
        X_train, X_test, y_train, y_test = pipeline.split_data(sample_df)
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert abs(test_ratio - 0.2) < 0.02

    def test_split_data_stratification(self, pipeline, sample_df):
        """Fraud ratio should be preserved in both sets."""
        X_train, X_test, y_train, y_test = pipeline.split_data(sample_df)
        original_ratio = sample_df["Class"].mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - original_ratio) < 0.02
        assert abs(test_ratio - original_ratio) < 0.02

    def test_handle_imbalance_smote(self, sample_config, sample_df):
        """SMOTE should increase minority class count."""
        sample_config["training"]["sampling_strategy"] = "smote"
        pipeline = DataPipeline(sample_config)
        X = sample_df.drop(columns=["Class"])
        y = sample_df["Class"]

        X_res, y_res = pipeline.handle_imbalance(X, y)
        assert y_res.sum() > y.sum()  # More fraud samples after SMOTE

    def test_handle_imbalance_none(self, pipeline, sample_df):
        """No sampling should return data unchanged."""
        X = sample_df.drop(columns=["Class"])
        y = sample_df["Class"]

        X_res, y_res = pipeline.handle_imbalance(X, y)
        assert len(X_res) == len(X)

    def test_get_data_stats_empty(self, pipeline):
        """Stats should be empty before loading data."""
        assert pipeline.get_data_stats() == {}
