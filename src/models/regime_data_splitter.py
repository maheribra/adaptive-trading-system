"""
Regime Data Splitter
====================
After the HMM labels every bar with a regime, this module:
  1. Attaches regime labels + confidence to the price DataFrame
  2. Splits the data into 3 regime-specific DataFrames
  3. Validates minimum sample counts (≥100 per regime)
  4. Produces a distribution report
  5. Saves splits to data/regimes/ as Parquet files
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

from .hmm_regime_classifier import (
    HMMRegimeClassifier,
    engineer_features,
    FEATURE_COLS,
    REGIME_LABELS,
)

MIN_SAMPLES_PER_REGIME = 100
OUTPUT_DIR = Path("data/regimes")


class RegimeDataSplitter:

    def __init__(self, classifier: HMMRegimeClassifier):
        self.clf = classifier

    def run(
        self,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame = None,
        dxy_df: pd.DataFrame = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Original method — engineers features internally.
        Only use this if you have NOT already engineered features externally.
        For the fixed pipeline, use run_on_featured() instead.
        """
        logger.info("=== Regime Data Splitter Started ===")

        featured_df = engineer_features(price_df, news_df, dxy_df)
        return self._predict_and_split(featured_df, save)

    def run_on_featured(
        self,
        featured_df: pd.DataFrame,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Use this when features are already engineered externally (preferred).
        Avoids re-engineering features on the full dataset which causes
        rolling-window leakage from future data into the past.
        """
        logger.info("=== Regime Data Splitter Started (pre-featured) ===")
        return self._predict_and_split(featured_df.copy(), save)

    def _predict_and_split(
        self,
        featured_df: pd.DataFrame,
        save: bool,
    ) -> Dict[str, pd.DataFrame]:

        # Extract feature matrix
        X = featured_df[FEATURE_COLS].values.astype(np.float32)

        # Predict regimes + confidence
        logger.info("Predicting regimes on full dataset...")
        regimes    = self.clf.predict(X)
        confidence = self.clf.confidence(X)
        probs      = self.clf.predict_proba(X)

        featured_df["regime"]           = regimes
        featured_df["regime_label"]     = [REGIME_LABELS[r] for r in regimes]
        featured_df["confidence"]       = confidence
        featured_df["prob_ranging"]     = probs[:, 0]
        featured_df["prob_trending"]    = probs[:, 1]
        featured_df["prob_news_driven"] = probs[:, 2]

        # Split by regime
        splits = self._split(featured_df)

        # Validate
        self._validate(splits)

        # Distribution report
        self._report(featured_df, splits)

        # Save
        if save:
            self._save(splits, featured_df)

        logger.success("=== Regime Data Splitter Complete ===")
        return splits

    def _split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        splits = {}
        for regime_id, label in REGIME_LABELS.items():
            subset = df[df["regime"] == regime_id].copy().reset_index(drop=True)
            splits[label] = subset
            logger.info(f"  {label:12s}: {len(subset):,} bars")
        return splits

    def _validate(self, splits: Dict[str, pd.DataFrame]):
        logger.info("Validating regime sample counts...")
        all_ok = True
        for label, df in splits.items():
            count  = len(df)
            status = "✓" if count >= MIN_SAMPLES_PER_REGIME else "✗ INSUFFICIENT"
            logger.info(
                f"  {label:12s}: {count:,} samples  {status}"
                f"  (minimum required: {MIN_SAMPLES_PER_REGIME})"
            )
            if count < MIN_SAMPLES_PER_REGIME:
                all_ok = False
                logger.warning(
                    f"  ↳ {label} has fewer than {MIN_SAMPLES_PER_REGIME} samples. "
                    f"Consider: (1) more history, (2) tuning HMM, (3) merging rare regimes."
                )
        if all_ok:
            logger.success("All regimes have sufficient samples.")

    def _report(self, full_df: pd.DataFrame, splits: Dict[str, pd.DataFrame]):
        total = len(full_df)
        logger.info("── Regime Distribution Report ─────────────────────")
        for label, df in splits.items():
            pct      = 100 * len(df) / total if total > 0 else 0
            avg_conf = df["confidence"].mean() if len(df) > 0 else 0
            logger.info(
                f"  {label:12s}: {len(df):6,} bars  "
                f"({pct:5.1f}%)  avg confidence: {avg_conf:.3f}"
            )

        transitions  = (full_df["regime"] != full_df["regime"].shift(1)).sum()
        avg_duration = total / (transitions + 1)
        logger.info(f"  Total transitions: {transitions:,}")
        logger.info(f"  Avg regime duration: {avg_duration:.1f} bars")
        logger.info("───────────────────────────────────────────────────")

    def _save(self, splits: Dict[str, pd.DataFrame], full_df: pd.DataFrame):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for label, df in splits.items():
            path = OUTPUT_DIR / f"{label.lower()}_data.parquet"
            df.to_parquet(path, index=False)
            logger.success(f"Saved {label} → {path}")

        full_path = OUTPUT_DIR / "full_labelled_dataset.parquet"
        full_df.to_parquet(full_path, index=False)
        logger.success(f"Saved full labelled dataset → {full_path}")