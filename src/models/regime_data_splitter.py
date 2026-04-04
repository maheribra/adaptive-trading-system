"""
Regime Data Splitter
====================
Fixed: Added Project Root Injection and Absolute Imports.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

# ── 1. Fix: Project Root Injection ─────────────────────────────────────────
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# ── 2. Fix: Absolute Imports ───────────────────────────────────────────────
try:
    from src.models.hmm_regime_classifier import (
        HMMRegimeClassifier,
        engineer_features,
        FEATURE_COLS,
        REGIME_LABELS,
    )
    # Import the feature builders to ensure the splitter uses the NEW logic
    from src.models.features.trend_features import build_trend_features
    from src.models.features.range_features import build_range_features
except ImportError:
    from hmm_regime_classifier import HMMRegimeClassifier, engineer_features, FEATURE_COLS, REGIME_LABELS
    from features.trend_features import build_trend_features
    from features.range_features import build_range_features

# ── Constants ──────────────────────────────────────────────────────────────
MIN_SAMPLES_PER_REGIME = 100
OUTPUT_DIR = root / "data" / "regimes"

class RegimeDataSplitter:
    def __init__(self, classifier: HMMRegimeClassifier):
        self.clf = classifier

    def run_on_featured(self, featured_df: pd.DataFrame, save: bool = True) -> Dict[str, pd.DataFrame]:
        logger.info("=== Regime Data Splitter Started (pre-featured) ===")
        return self._predict_and_split(featured_df.copy(), save)

    def _predict_and_split(self, featured_df: pd.DataFrame, save: bool) -> Dict[str, pd.DataFrame]:
        # Extract feature matrix for HMM
        X = featured_df[FEATURE_COLS].values.astype(np.float32)

        # Predict regimes
        logger.info("Predicting regimes on full dataset...")
        regimes = self.clf.predict(X)
        featured_df["regime"] = regimes
        featured_df["regime_label"] = [REGIME_LABELS[r] for r in regimes]
        featured_df["confidence"] = self.clf.confidence(X)

        # ── CRITICAL: Apply the NEW Feature Engineering ────────────────────
        # We split first, then run the specific regime-based feature builders
        splits = {}
        for regime_id, label in REGIME_LABELS.items():
            subset = featured_df[featured_df["regime"] == regime_id].copy()

            if label == "TRENDING":
                logger.info("Building NEW Trend Features (ADX/ROC)...")
                subset = build_trend_features(subset)
            elif label == "RANGING":
                logger.info("Building Range Features...")
                subset = build_range_features(subset)

            splits[label] = subset

        self._validate(splits)
        self._report(featured_df, splits)

        if save:
            self._save(splits, featured_df)

        logger.success("=== Regime Data Splitter Complete ===")
        return splits

    def _validate(self, splits: Dict[str, pd.DataFrame]):
        for label, df in splits.items():
            count = len(df)
            status = "✓" if count >= MIN_SAMPLES_PER_REGIME else "✗"
            logger.info(f"  {label:12s}: {count:,} samples {status}")

    def _report(self, full_df: pd.DataFrame, splits: Dict[str, pd.DataFrame]):
        total = len(full_df)
        logger.info("── Regime Distribution Report ─────────────────────")
        for label, df in splits.items():
            pct = 100 * len(df) / total if total > 0 else 0
            logger.info(f"  {label:12s}: {len(df):6,} bars ({pct:5.1f}%)")

    def _save(self, splits: Dict[str, pd.DataFrame], full_df: pd.DataFrame):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for label, df in splits.items():
            path = OUTPUT_DIR / f"{label.lower()}_data.parquet"
            df.to_parquet(path, index=False)
            logger.success(f"Saved {label} → {path}")

# ── 3. Execution Block ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # 1. Project Root & Directory Setup
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    RAW_DATA_DIR = root / "data" / "raw"
    HMM_PATH = root / "data" / "models" / "hmm_regime_classifier.pkl"

    # 2. Safety Checks
    if not RAW_DATA_DIR.exists():
        logger.error(f"Directory not found: {RAW_DATA_DIR}")
        sys.exit(1)

    # List files to help you if the name is slightly different
    available_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.csv', '.parquet'))]
    logger.info(f"Available files in data/raw: {available_files}")

    # 3. Load Data (Adjust filename if needed)
    TARGET_FILE = "AUDUSD_1h.csv"  # Ensure this matches your folder exactly
    PRICE_PATH = RAW_DATA_DIR / TARGET_FILE

    if not PRICE_PATH.exists():
        logger.error(f"File {TARGET_FILE} not found. Check 'available_files' log above.")
        sys.exit(1)

    logger.info(f"Loading raw data from {PRICE_PATH}...")
    df_raw = pd.read_csv(PRICE_PATH)

    # 4. Column Normalization (Fixes KeyError 'timestamp' and 'volume')
    # Standardize to lowercase
    df_raw.columns = [c.lower() for c in df_raw.columns]

    # Map time variations
    time_map = {'date': 'timestamp', 'datetime': 'timestamp', 'time': 'timestamp'}
    df_raw = df_raw.rename(columns=time_map)

    # Create Synthetic Volume (Crucial for AUD/USD data)
    if 'volume' not in df_raw.columns:
        logger.warning("Volume missing. Creating synthetic volume from Price Range (Volatility Proxy).")
        # Formula: (High - Low) * 10,000 (standardized for FX pips)
        df_raw['volume'] = (df_raw['high'] - df_raw['low']) * 10000 + 1.0

        # 5. Execute Splitter
    if not HMM_PATH.exists():
        logger.error(f"HMM Model not found at {HMM_PATH}. Please train HMM first.")
    else:
        # Load Classifier
        try:
            from src.models.hmm_regime_classifier import HMMRegimeClassifier, engineer_features

            hmm_clf = HMMRegimeClassifier.load(str(HMM_PATH))
            splitter = RegimeDataSplitter(hmm_clf)

            # Step 1: Run HMM Feature Engineering
            logger.info("Running HMM feature engineering...")
            feat_df = engineer_features(df_raw)

            # Step 2: Split and build regime-specific features (ADX, ROC, etc.)
            logger.info("Splitting data and building regime features...")
            splitter.run_on_featured(feat_df, save=True)

            logger.success("Pipeline complete. You can now run trend_model.py and range_model.py")

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")