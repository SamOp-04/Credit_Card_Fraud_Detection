"""Data loading, validation, and preprocessing pipeline."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

from src.utils.logger import logger


class DataPipeline:
    """End-to-end data pipeline for credit card fraud detection."""

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self._raw_data: pd.DataFrame | None = None

    def load_data(self, data_path: str | None = None) -> pd.DataFrame:
        """Load raw CSV data with validation."""
        path = data_path or self.config["data"]["raw_path"]
        logger.info("loading_data", path=path)

        df = pd.read_csv(path)
        self._raw_data = df.copy()

        self._validate_data(df)
        logger.info("data_loaded", rows=len(df), columns=len(df.columns))
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data schema and quality."""
        target = self.config["features"]["target_column"]

        # Check target column exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        # Check for required value range
        unique_targets = set(df[target].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"Target column must be binary (0/1), got: {unique_targets}"
            )

        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(
                "null_values_found", columns=null_counts[null_counts > 0].to_dict()
            )

        # Log class distribution
        class_dist = df[target].value_counts().to_dict()
        fraud_rate = class_dist.get(1, 0) / len(df) * 100
        logger.info(
            "class_distribution",
            distribution=class_dist,
            fraud_rate_pct=round(fraud_rate, 4),
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features and handle preprocessing."""
        logger.info("preprocessing_data")

        # Drop configured columns
        drop_cols = self.config["features"].get("drop_columns", [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Scale specified columns
        scale_cols = self.config["features"]["scale_columns"]
        existing_scale_cols = [c for c in scale_cols if c in df.columns]
        if existing_scale_cols:
            df[existing_scale_cols] = self.scaler.fit_transform(df[existing_scale_cols])

        logger.info("preprocessing_complete", scaled_columns=existing_scale_cols)
        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split into train/test sets."""
        target = self.config["features"]["target_column"]
        X = df.drop(columns=[target])
        y = df[target]

        test_size = self.config["data"]["test_size"]
        seed = self.config["project"]["random_seed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        logger.info(
            "data_split",
            train_size=len(X_train),
            test_size=len(X_test),
            train_fraud_rate=round(y_train.mean() * 100, 4),
            test_fraud_rate=round(y_test.mean() * 100, 4),
        )
        return X_train, X_test, y_train, y_test

    def handle_imbalance(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply sampling strategy to handle class imbalance."""
        strategy = self.config["training"]["sampling_strategy"]
        seed = self.config["project"]["random_seed"]

        if strategy == "smote":
            ratio = self.config["training"]["smote_ratio"]
            sampler = SMOTE(sampling_strategy=ratio, random_state=seed)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            logger.info(
                "smote_applied",
                original_size=len(X_train),
                resampled_size=len(X_resampled),
            )
        elif strategy == "undersampling":
            sampler = RandomUnderSampler(random_state=seed)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            logger.info(
                "undersampling_applied",
                original_size=len(X_train),
                resampled_size=len(X_resampled),
            )
        else:
            logger.info("no_sampling_applied")
            return X_train, y_train

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled
        )

    def run_pipeline(
        self, data_path: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute the full data pipeline."""
        df = self.load_data(data_path)
        df = self.preprocess(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        X_train, y_train = self.handle_imbalance(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: str = "models/scaler.joblib") -> None:
        """Persist the fitted scaler."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info("scaler_saved", path=path)

    def get_data_stats(self) -> dict:
        """Return summary statistics of loaded data."""
        if self._raw_data is None:
            return {}
        target = self.config["features"]["target_column"]
        df = self._raw_data
        return {
            "total_transactions": len(df),
            "total_features": len(df.columns) - 1,
            "fraud_count": int(df[target].sum()),
            "legitimate_count": int((df[target] == 0).sum()),
            "fraud_rate_pct": round(df[target].mean() * 100, 4),
            "null_values": int(df.isnull().sum().sum()),
        }
