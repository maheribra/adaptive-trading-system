import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from src.models.features.dxy_features import build_dxy_features, DXY_FEATURE_COLS
from src.models.dxy_model import DXYModel

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[1]
RANGING_DATA    = BASE_DIR / "data" / "regimes" / "ranging_data.parquet"
DXY_CSV         = BASE_DIR / "data" / "raw" / "dxy_index.csv"
MODEL_SAVE_PATH = BASE_DIR / "data" / "models" / "dxy_model.pkl"


def load_dxy(path: Path) -> pd.DataFrame:
    dxy = pd.read_csv(path)
    dxy.columns = [c.lower() for c in dxy.columns]
    if "time" in dxy.columns:
        dxy = dxy.rename(columns={"time": "timestamp"})
    dxy["timestamp"] = pd.to_datetime(
        dxy["timestamp"]
    ).dt.tz_localize(None).astype("datetime64[ns]")
    return dxy


def run():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║       DXY Model Training Pipeline            ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Load ranging regime data ────────────────────────────────────────
    logger.info(f"Loading ranging data from: {RANGING_DATA}")
    assert RANGING_DATA.exists(), f"File not found: {RANGING_DATA}."
    raw_df = pd.read_parquet(RANGING_DATA)
    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"]
    ).dt.tz_localize(None).astype("datetime64[ns]")
    logger.info(f"Loaded {len(raw_df):,} ranging bars")

    # ── 2. Load DXY data ───────────────────────────────────────────────────
    logger.info(f"Loading DXY data from: {DXY_CSV}")
    assert DXY_CSV.exists(), f"File not found: {DXY_CSV}."
    dxy_df = load_dxy(DXY_CSV)
    logger.info(f"Loaded {len(dxy_df):,} DXY bars")

    # ── 3. Build DXY features ──────────────────────────────────────────────
    logger.info("Building DXY correlation features...")
    featured_df = build_dxy_features(raw_df, dxy_df)
    logger.info(f"Featured dataset: {len(featured_df):,} bars × {len(DXY_FEATURE_COLS)} features")

    # ── 4. Check target balance ────────────────────────────────────────────
    dist = featured_df["next_bar_up"].value_counts(normalize=True)
    logger.info(f"Target distribution — Up: {dist.get(1, 0):.1%} | Down: {dist.get(0, 0):.1%}")

    # ── 5. Train model ─────────────────────────────────────────────────────
    logger.info("Training DXYModel (XGBoost)...")
    model = DXYModel(
        n_estimators     = 200,
        max_depth        = 3,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
    )
    model.fit(featured_df, dxy_df)

    # ── 6. Walk-forward validation ─────────────────────────────────────────
    logger.info("Running walk-forward validation (5 folds)...")
    wf_results = model.walk_forward_validate(featured_df, n_splits=5)

    # ── 7. Save model ──────────────────────────────────────────────────────
    model.save(str(MODEL_SAVE_PATH))

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║           DXY MODEL SUMMARY                  ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  Training bars  : {len(featured_df):>8,}              ║")
    logger.info(f"║  Features       : {len(DXY_FEATURE_COLS):>8,}              ║")
    logger.info(f"║  WF Mean Acc    : {wf_results['mean_accuracy']:>8.4f}              ║")
    logger.info(f"║  WF Std Acc     : {wf_results['std_accuracy']:>8.4f}              ║")
    logger.info(f"║  Model saved    : {str(MODEL_SAVE_PATH.name):<28s}║")
    logger.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    run()