import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional


# ── Project Root Injection ────────────────────────────────────────────────
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from sklearn.ensemble import RandomForestClassifier # Or LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Absolute import
from src.models.features.trend_features import build_trend_features, TREND_FEATURE_COLS

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_COL  = "next_bar_up"
TRAIN_RATIO = 0.80
MIN_BARS    = 300

class TrendModel:
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        self._scaler = StandardScaler()
        self.is_fitted = False

    def _prepare(self, df: pd.DataFrame):
        if not all(col in df.columns for col in TREND_FEATURE_COLS):
            df = build_trend_features(df)
        X = df[TREND_FEATURE_COLS].copy().fillna(0)
        y = df[TARGET_COL].values if TARGET_COL in df.columns else None
        return X, y

    def fit(self, df: pd.DataFrame):
        X, y = self._prepare(df)
        split = int(len(X) * TRAIN_RATIO)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model.fit(X_train_scaled, y_train)
        importances = pd.Series(
            self._model.feature_importances_,
            index=TREND_FEATURE_COLS
        ).sort_values(ascending=False)

        logger.info("── Top 5 Trend Features ──────────────────")
        for feat, val in importances.head(5).items():
            logger.info(f"  {feat:25s}: {val:.4f}")
        self.is_fitted = True
        
        # Simple evaluation
        acc = accuracy_score(y_test, self._model.predict(X_test_scaled))
        logger.info(f"TrendModel Trained. Test Accuracy: {acc:.4f}")
        return self

    def predict_proba(self, df: pd.DataFrame):
        X, _ = self._prepare(df)
        return self._model.predict_proba(self._scaler.transform(X))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"TrendModel saved → {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    DATA_PATH = root / "data" / "regimes" / "trending_data.parquet"
    MODEL_PATH = root / "data" / "models" / "trend_model.pkl"

    if DATA_PATH.exists():
        df_train = pd.read_parquet(DATA_PATH)
        model = TrendModel()
        model.fit(df_train)
        model.save(str(MODEL_PATH))
    else:
        logger.error("Trending data not found. Run pipeline first.")