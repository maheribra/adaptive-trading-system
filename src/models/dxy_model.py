"""
DXY Correlation ML Model
=========================
Trains an XGBoost classifier on RANGING regime data using DXY features.
Predicts next bar direction: 1 = up, 0 = down.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional, Dict

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from .features.dxy_features import build_dxy_features, DXY_FEATURE_COLS

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL  = "next_bar_up"
TRAIN_RATIO = 0.80
MIN_BARS    = 300


class DXYModel:

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            random_state     = random_state,
            eval_metric      = "logloss",
        )
        self._model   = None
        self._scaler  = StandardScaler()
        self.is_fitted = False
        self.feature_importance_: Optional[pd.Series] = None
        self._dxy_df: Optional[pd.DataFrame] = None

    def _prepare(self, df: pd.DataFrame, dxy_df: Optional[pd.DataFrame] = None):
        if TARGET_COL not in df.columns or "dxy_pct_change" not in df.columns:
            if dxy_df is None:
                raise ValueError("dxy_df required when features not yet built.")
            df = build_dxy_features(df, dxy_df)
        X = df[DXY_FEATURE_COLS].copy().clip(-10, 10)
        y = df[TARGET_COL].values
        return X, y

    def fit(
        self,
        df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
    ) -> "DXYModel":
        # Store dxy_df for walk-forward use
        if dxy_df is not None:
            self._dxy_df = dxy_df

        X, y = self._prepare(df, dxy_df)

        if len(X) < MIN_BARS:
            raise ValueError(f"Need ≥{MIN_BARS} bars. Got {len(X)}.")

        split_idx = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled  = self._scaler.transform(X_test)

        logger.info(
            f"Training DXYModel — "
            f"train: {len(X_train):,} | test: {len(X_test):,}"
        )

        self._model = xgb.XGBClassifier(**self.params)
        self._model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
        self.is_fitted = True

        self.feature_importance_ = pd.Series(
            self._model.feature_importances_,
            index=DXY_FEATURE_COLS,
        ).sort_values(ascending=False)

        self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
        return self

    def _evaluate(self, X_train, y_train, X_test, y_test):
        train_pred = self._model.predict(X_train)
        test_pred  = self._model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        logger.info("── DXYModel Evaluation ────────────────────────────────")
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

        cm = confusion_matrix(y_test, test_pred)
        logger.info("── Confusion Matrix (Test) ────────────────────────────")
        logger.info(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        logger.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        logger.info("── Feature Importance (top 10) ────────────────────────")
        for feat, imp in self.feature_importance_.head(10).items():
            logger.info(f"  {feat:28s}: {imp:.4f}")
        logger.info("───────────────────────────────────────────────────────")

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        dxy = dxy_df if dxy_df is not None else self._dxy_df
        X, y = self._prepare(df, dxy)
        X_scaled = self._scaler.transform(X.clip(-10, 10))

        tscv   = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            m = xgb.XGBClassifier(**self.params)
            m.fit(X_scaled[train_idx], y[train_idx], verbose=False)
            preds = m.predict(X_scaled[test_idx])
            acc   = accuracy_score(y[test_idx], preds)
            scores.append(acc)
            logger.info(f"  Fold {fold+1}/{n_splits} accuracy: {acc:.4f}")

        mean_acc = float(np.mean(scores))
        std_acc  = float(np.std(scores))
        logger.info(f"  Walk-forward mean: {mean_acc:.4f} ± {std_acc:.4f}")
        return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "folds": scores}

    def predict(self, df: pd.DataFrame, dxy_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        self._check_fitted()
        dxy = dxy_df if dxy_df is not None else self._dxy_df
        X, _ = self._prepare(df, dxy)
        return self._model.predict(self._scaler.transform(X.clip(-10, 10)))

    def predict_proba(self, df: pd.DataFrame, dxy_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        self._check_fitted()
        dxy = dxy_df if dxy_df is not None else self._dxy_df
        X, _ = self._prepare(df, dxy)
        return self._model.predict_proba(self._scaler.transform(X.clip(-10, 10)))

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"DXYModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DXYModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.success(f"DXYModel loaded ← {path}")
        return obj