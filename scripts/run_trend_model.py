import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from src.models.features.trend_features import build_trend_features, TREND_FEATURE_COLS
from src.models.trend_model import TrendModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
TRENDING_DATA   = BASE_DIR / "data" / "regimes" / "trending_data.parquet"
MODEL_SAVE_PATH = BASE_DIR / "data" / "models" / "trend_model.pkl"


def run():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║       Trend Model Training Pipeline          ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Load trending regime data ───────────────────────────────────────
    logger.info(f"Loading trending data from: {TRENDING_DATA}")
    assert TRENDING_DATA.exists(), f"File not found: {TRENDING_DATA}. Run run_regime_detection.py first."
    raw_df = pd.read_parquet(TRENDING_DATA)
    logger.info(f"Loaded {len(raw_df):,} trending bars")

    # ── 2. Build trend features ────────────────────────────────────────────
    logger.info("Building ICT trend features...")
    featured_df = build_trend_features(raw_df)
    logger.info(f"Featured dataset: {len(featured_df):,} bars × {len(TREND_FEATURE_COLS)} features")

    # ── 3. Check target balance ────────────────────────────────────────────
    dist = featured_df["next_bar_up"].value_counts(normalize=True)
    logger.info(f"Target distribution — Up: {dist.get(1, 0):.1%} | Down: {dist.get(0, 0):.1%}")

    # ── 4. Train model ─────────────────────────────────────────────────────
    logger.info("Training TrendModel (XGBoost)...")
    model = TrendModel(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
    )
    model.fit(featured_df)

    # ── 5. Walk-forward validation ─────────────────────────────────────────
    logger.info("Running walk-forward validation (5 folds)...")
    wf_results = model.walk_forward_validate(featured_df, n_splits=5)

    # ── 6. Save model ──────────────────────────────────────────────────────
    model.save(str(MODEL_SAVE_PATH))

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║           TREND MODEL SUMMARY                ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  Training bars  : {len(featured_df):>8,}              ║")
    logger.info(f"║  Features       : {len(TREND_FEATURE_COLS):>8,}              ║")
    logger.info(f"║  WF Mean Acc    : {wf_results['mean_accuracy']:>8.4f}              ║")
    logger.info(f"║  WF Std Acc     : {wf_results['std_accuracy']:>8.4f}              ║")
    logger.info(f"║  Model saved    : {str(MODEL_SAVE_PATH.name):<28s}║")
    logger.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    run()