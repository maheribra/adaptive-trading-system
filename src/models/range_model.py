import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional, Dict

# ── 1. Fix: Project Root Injection ─────────────────────────────────────────
# This allows running the file directly without "attempted relative import" errors
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler

# Use absolute imports now that project root is in sys.path
from src.models.features.range_features import build_range_features, RANGE_FEATURE_COLS

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL  = "next_bar_up"
TRAIN_RATIO = 0.80
MIN_BARS    = 300


class RangeModel:
    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.params = dict(
            C            = C,
            max_iter     = max_iter,
            random_state = random_state,
            solver       = "lbfgs",
        )
        self._model    = None
        self._scaler   = StandardScaler()
        self.is_fitted = False
        self.feature_importance_: Optional[pd.Series] = None

    def _prepare(self, df: pd.DataFrame):
        """Build features if not already built, return X and y."""
        # Ensure we have the required columns
        if not all(col in df.columns for col in RANGE_FEATURE_COLS):
            df = build_range_features(df)

        X = df[RANGE_FEATURE_COLS].copy()
        X = X.clip(-10, 10) # Prevent outlier explosion
        
        y = None
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].values
            
        return X, y

    def fit(self, df: pd.DataFrame) -> "RangeModel":
        X, y = self._prepare(df)
        if y is None:
            raise ValueError(f"Target column '{TARGET_COL}' missing from DataFrame.")

        if len(X) < MIN_BARS:
            raise ValueError(f"Need ≥{MIN_BARS} bars. Got {len(X)}.")

        split_idx = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scaler fit on train only
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled  = self._scaler.transform(X_test)

        logger.info(f"Training RangeModel — train: {len(X_train):,} | test: {len(X_test):,}")

        self._model = LogisticRegression(**self.params)
        self._model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        self.feature_importance_ = pd.Series(
            np.abs(self._model.coef_[0]),
            index=RANGE_FEATURE_COLS,
        ).sort_values(ascending=False)

        self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
        return self

    def _evaluate(self, X_train, y_train, X_test, y_test):
        train_pred = self._model.predict(X_train)
        test_pred  = self._model.predict(X_test)

        logger.info("── RangeModel Evaluation ──────────────────────────────")
        logger.info(f"  Train accuracy : {accuracy_score(y_train, train_pred):.4f}")
        logger.info(f"  Test  accuracy : {accuracy_score(y_test, test_pred):.4f}")
        logger.info(f"  Test F1        : {f1_score(y_test, test_pred, zero_division=0):.4f}")
        logger.info("── Top 5 Features ─────────────────────────────────────")
        for feat, imp in self.feature_importance_.head(5).items():
            logger.info(f"  {feat:25s}: {imp:.4f}")
        logger.info("───────────────────────────────────────────────────────")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self._prepare(df)
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"RangeModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RangeModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.success(f"RangeModel loaded ← {path}")
        return obj


# ── 2. Fix: Execution Logic ────────────────────────────────────────────────
if __name__ == "__main__":
    # This block runs when you execute the file directly
    DATA_PATH = root / "data" / "regimes" / "ranging_data.parquet"
    MODEL_PATH = root / "data" / "models" / "range_model.pkl"

    if not DATA_PATH.exists():
        logger.error(f"No data found at {DATA_PATH}. Run data preparation first.")
    else:
        logger.info(f"Loading training data from {DATA_PATH}...")
        df_train = pd.read_parquet(DATA_PATH)
        
        # Train and Save
        model = RangeModel(C=0.05) # Slightly more regularization
        model.fit(df_train)
        model.save(str(MODEL_PATH))