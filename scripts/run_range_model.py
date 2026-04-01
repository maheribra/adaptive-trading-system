import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from src.models.features.range_features import build_range_features, RANGE_FEATURE_COLS
from src.models.range_model import RangeModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]
RANGING_DATA   = BASE_DIR / "data" / "regimes" / "ranging_data.parquet"
MODEL_SAVE_PATH = BASE_DIR / "data" / "models" / "range_model.pkl"


def run():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║       Range Model Training Pipeline          ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Load ranging regime data ────────────────────────────────────────
    logger.info(f"Loading ranging data from: {RANGING_DATA}")
    assert RANGING_DATA.exists(), f"File not found: {RANGING_DATA}. Run run_regime_detection.py first."
    raw_df = pd.read_parquet(RANGING_DATA)
    logger.info(f"Loaded {len(raw_df):,} ranging bars")

    # ── 2. Build range features ────────────────────────────────────────────
    logger.info("Building mean reversion features...")
    featured_df = build_range_features(raw_df)
    logger.info(f"Featured dataset: {len(featured_df):,} bars × {len(RANGE_FEATURE_COLS)} features")

    # ── 3. Check target balance ────────────────────────────────────────────
    dist = featured_df["next_bar_up"].value_counts(normalize=True)
    logger.info(f"Target distribution — Up: {dist.get(1, 0):.1%} | Down: {dist.get(0, 0):.1%}")

    # ── 4. Train model ─────────────────────────────────────────────────────
    logger.info("Training RangeModel (Logistic Regression)...")
    model = RangeModel(
        C        = 0.1,
        max_iter = 1000,
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
    logger.info("║           RANGE MODEL SUMMARY                ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  Training bars  : {len(featured_df):>8,}              ║")
    logger.info(f"║  Features       : {len(RANGE_FEATURE_COLS):>8,}              ║")
    logger.info(f"║  WF Mean Acc    : {wf_results['mean_accuracy']:>8.4f}              ║")
    logger.info(f"║  WF Std Acc     : {wf_results['std_accuracy']:>8.4f}              ║")
    logger.info(f"║  Model saved    : {str(MODEL_SAVE_PATH.name):<28s}║")
    logger.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    run()