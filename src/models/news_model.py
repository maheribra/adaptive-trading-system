"""
News Regime ML Model (Gradient Boosting)
=========================================
Trains a GradientBoostingClassifier on NEWS_DRIVEN regime data only.
Predicts next bar direction: 1 = up, 0 = down.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional, Dict

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from .features.news_features import build_news_features, NEWS_FEATURE_COLS

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL  = "next_bar_up"
TRAIN_RATIO = 0.80
MIN_BARS    = 300


class NewsModel:

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            learning_rate = learning_rate,
            subsample     = subsample,
            random_state  = random_state,
        )
        self._model   = None
        self._scaler  = StandardScaler()
        self.is_fitted = False
        self.feature_importance_: Optional[pd.Series] = None

    def _prepare(self, df: pd.DataFrame):
        if TARGET_COL not in df.columns:
            df = build_news_features(df)
        X = df[NEWS_FEATURE_COLS].copy().clip(-10, 10)
        y = df[TARGET_COL].values
        return X, y

    def fit(self, df: pd.DataFrame) -> "NewsModel":
        X, y = self._prepare(df)

        if len(X) < MIN_BARS:
            raise ValueError(f"Need ≥{MIN_BARS} bars. Got {len(X)}.")

        split_idx = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled  = self._scaler.transform(X_test)

        logger.info(
            f"Training NewsModel — "
            f"train: {len(X_train):,} | test: {len(X_test):,}"
        )

        self._model = GradientBoostingClassifier(**self.params)
        self._model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        self.feature_importance_ = pd.Series(
            self._model.feature_importances_,
            index=NEWS_FEATURE_COLS,
        ).sort_values(ascending=False)

        self._evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
        return self

    def _evaluate(self, X_train, y_train, X_test, y_test):
        train_pred = self._model.predict(X_train)
        test_pred  = self._model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        logger.info("── NewsModel Evaluation ───────────────────────────────")
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

        # Confusion matrix
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
            n_splits: int = 5,
    ) -> Dict[str, float]:
        X, y = self._prepare(df)
        X_raw = X.values  # unscaled — scaler fit fresh per fold

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_raw)):
            fold_scaler = StandardScaler()
            X_fold_train = fold_scaler.fit_transform(X_raw[train_idx])
            X_fold_test = fold_scaler.transform(X_raw[test_idx])

            m = GradientBoostingClassifier(**self.params)
            m.fit(X_fold_train, y[train_idx])
            preds = m.predict(X_fold_test)
            acc = accuracy_score(y[test_idx], preds)
            scores.append(acc)
            logger.info(f"  Fold {fold + 1}/{n_splits} accuracy: {acc:.4f}")

        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        logger.info(f"  Walk-forward mean: {mean_acc:.4f} ± {std_acc:.4f}")
        return {"mean_accuracy": mean_acc, "std_accuracy": std_acc, "folds": scores}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self._prepare(df)
        return self._model.predict(self._scaler.transform(X.clip(-10, 10)))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X, _ = self._prepare(df)
        return self._model.predict_proba(self._scaler.transform(X.clip(-10, 10)))

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"NewsModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NewsModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.success(f"NewsModel loaded ← {path}")
        return obj