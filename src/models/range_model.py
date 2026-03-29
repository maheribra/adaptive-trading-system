"""
Range Regime ML Model (Mean Reversion)
=======================================
Trains a Logistic Regression classifier on RANGING regime data only.
Predicts next bar direction: 1 = up, 0 = down.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler

from .features.range_features import build_range_features, RANGE_FEATURE_COLS

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
        self._model   = None
        self._scaler  = StandardScaler()
        self.is_fitted = False
        self.feature_importance_: Optional[pd.Series] = None

    def _prepare(self, df: pd.DataFrame):
        """Build features if not already built, return X and y."""
        if TARGET_COL not in df.columns:
            df = build_range_features(df)

        X = df[RANGE_FEATURE_COLS].copy()

        # Clip extreme values — fixes the dist_from_vwap 1e+11 issue
        X = X.clip(-10, 10)

        y = df[TARGET_COL].values
        return X, y

    def fit(self, df: pd.DataFrame) -> "RangeModel":
        """
        Train on ranging regime data.
        Expects raw price df OR already-featured df.
        Uses chronological train/test split internally.
        """
        X, y = self._prepare(df)

        if len(X) < MIN_BARS:
            raise ValueError(f"Need ≥{MIN_BARS} bars. Got {len(X)}.")

        # Chronological split
        split_idx = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit scaler on train only
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled  = self._scaler.transform(X_test)

        logger.info(
            f"Training RangeModel — "
            f"train: {len(X_train):,} | test: {len(X_test):,}"
        )

        self._model = LogisticRegression(**self.params)
        self._model.fit(X_train_scaled, y_train)

        self.is_fitted = True

        # Feature importance (coefficients for logistic regression)
        self.feature_importance_ = pd.Series(
            np.abs(self._model.coef_[0]),
            index=RANGE_FEATURE_COLS,
        ).sort_values(ascending=False)

        # Evaluate
        self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

        return self

    def _evaluate(self, X_train, y_train, X_test, y_test):
        train_pred = self._model.predict(X_train)
        test_pred  = self._model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        logger.info("── RangeModel Evaluation ──────────────────────────────")
        logger.info(f"  Train accuracy : {train_acc:.4f}")
        logger.info(f"  Test  accuracy : {test_acc:.4f}")
        logger.info(f"  Gap            : {abs(train_acc - test_acc):.4f}")
        logger.info(
            f"  Test Precision : "
            f"{precision_score(y_test, test_pred, zero_division=0):.4f}"
        )
        logger.info(
            f"  Test Recall    : "
            f"{recall_score(y_test, test_pred, zero_division=0):.4f}"
        )
        logger.info(
            f"  Test F1        : "
            f"{f1_score(y_test, test_pred, zero_division=0):.4f}"
        )
        logger.info("── Feature Importance (top 10) ────────────────────────")
        for feat, imp in self.feature_importance_.head(10).items():
            logger.info(f"  {feat:25s}: {imp:.4f}")
        logger.info("───────────────────────────────────────────────────────")

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Time-series cross validation.
        Returns mean accuracy and std across folds.
        """
        X, y = self._prepare(df)
        X_scaled = self._scaler.transform(X.clip(-10, 10))

        tscv   = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            m = LogisticRegression(**self.params)
            m.fit(X_scaled[train_idx], y[train_idx])
            preds  = m.predict(X_scaled[test_idx])
            acc    = accuracy_score(y[test_idx], preds)
            scores.append(acc)
            logger.info(f"  Fold {fold+1}/{n_splits} accuracy: {acc:.4f}")

        mean_acc = float(np.mean(scores))
        std_acc  = float(np.std(scores))
        logger.info(f"  Walk-forward mean: {mean_acc:.4f} ± {std_acc:.4f}")
        return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "folds": scores}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self._prepare(df)
        X_scaled = self._scaler.transform(X.clip(-10, 10))
        return self._model.predict(X_scaled)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self._prepare(df)
        X_scaled = self._scaler.transform(X.clip(-10, 10))
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