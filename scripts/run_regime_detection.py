import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

from src.database import get_connection, create_schema
from src.fetchers.yahoo_fetcher import load_dxy_from_csv
from src.models.hmm_regime_classifier import (
    HMMRegimeClassifier,
    engineer_features,
    FEATURE_COLS,
    REGIME_LABELS,
)
from src.models.regime_data_splitter import RegimeDataSplitter

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "data/models/hmm_regime_classifier.pkl"
TRAIN_RATIO     = 0.70
SYMBOL          = "EURUSD=X"

N_REGIMES = 3


def load_price_data(con) -> pd.DataFrame:
    logger.info(f"Loading price data for {SYMBOL}...")
    df = con.execute(f"""
        SELECT timestamp, open, high, low, close, volume
        FROM prices
        WHERE symbol = '{SYMBOL}'
        ORDER BY timestamp
    """).df()
    logger.info(f"Loaded {len(df):,} price bars")
    return df


def load_news_data(con) -> pd.DataFrame:
    logger.info("Loading news events...")
    try:
        df = con.execute("""
            SELECT event_time, title, currency, impact
            FROM news_events
            ORDER BY event_time
        """).df()
        logger.info(f"Loaded {len(df):,} news events")
        return df
    except Exception as e:
        logger.warning(f"No news data: {e}")
        return pd.DataFrame()


def load_dxy_data(con) -> pd.DataFrame:
    logger.info("Loading DXY data...")
    dxy_df = load_dxy_from_csv(con)
    if dxy_df.empty:
        logger.warning("DXY data is empty — DXY features will be zeroed out")
    else:
        logger.info(f"Loaded {len(dxy_df):,} DXY bars")
    return dxy_df


def diagnose_features(featured_df: pd.DataFrame):
    logger.info("── Feature Statistics ─────────────────────────────────────")
    stats = featured_df[FEATURE_COLS].describe().round(4)
    for col in FEATURE_COLS:
        s = stats[col]
        logger.info(
            f"  {col:20s}  mean={s['mean']:7.3f}  "
            f"std={s['std']:6.3f}  "
            f"min={s['min']:7.3f}  "
            f"max={s['max']:7.3f}"
        )
    logger.info("────────────────────────────────────────────────────────────")


def diagnose_regime_means(featured_df: pd.DataFrame):
    logger.info("── Mean Feature Values per Regime ─────────────────────────")
    header = f"{'Feature':20s}" + "".join(f"{REGIME_LABELS[i]:>14s}" for i in range(3))
    logger.info(header)
    for col in FEATURE_COLS:
        row = f"{col:20s}"
        for regime_id in range(3):
            subset = featured_df[featured_df["regime"] == regime_id][col]
            val    = subset.mean() if len(subset) > 0 else float("nan")
            row   += f"{val:>14.4f}"
        logger.info(row)
    logger.info("────────────────────────────────────────────────────────────")


def walk_forward_validate(
    clf: HMMRegimeClassifier,
    featured_train: pd.DataFrame,
    featured_test: pd.DataFrame,
) -> dict:
    logger.info("Running walk-forward validation...")
    train_ll = clf.log_likelihood(featured_train)
    test_ll  = clf.log_likelihood(featured_test)
    gap      = abs(train_ll - test_ll)
    logger.info(f"  Train log-likelihood / bar: {train_ll:.6f}")
    logger.info(f"  Test  log-likelihood / bar: {test_ll:.6f}")
    logger.info(f"  Gap: {gap:.6f}  {'⚠ check for overfitting' if gap > 0.5 else '✓ ok'}")
    return {"train_ll": train_ll, "test_ll": test_ll, "gap": gap}


def analyse_regime_transitions(labelled_df: pd.DataFrame):
    logger.info("── Regime Transition Matrix ───────────────────────────────")
    regimes = labelled_df["regime"].values
    matrix  = np.zeros((N_REGIMES, N_REGIMES), dtype=int)
    for i in range(len(regimes) - 1):
        matrix[regimes[i], regimes[i + 1]] += 1

    header = f"{'':20s}" + "".join(f"{REGIME_LABELS[j]:>14s}" for j in range(N_REGIMES))
    logger.info(header)
    for i in range(N_REGIMES):
        total   = matrix[i].sum()
        row_pct = 100 * matrix[i] / (total + 1e-9)
        row     = f"{REGIME_LABELS[i]:20s}" + "".join(
            f"{row_pct[j]:>13.1f}%" for j in range(N_REGIMES)
        )
        logger.info(row)
    logger.info("───────────────────────────────────────────────────────────")


def run():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   HMM Regime Detection Pipeline v4  ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # ── 1. Connect ─────────────────────────────────────────────────────────
    con = get_connection()
    create_schema(con)

    # ── 2. Load data ───────────────────────────────────────────────────────
    price_df = load_price_data(con)
    news_df  = load_news_data(con)
    dxy_df   = load_dxy_data(con)

    if len(price_df) < 600:
        logger.error(f"Only {len(price_df)} bars. Need ≥600. Run pipeline.py first.")
        return

    # ── 3. Feature engineering ─────────────────────────────────────────────
    logger.info("Step 3: Engineering features (v4)...")
    featured_df = engineer_features(price_df, news_df, dxy_df)
    diagnose_features(featured_df)

    # ── 4. Walk-forward split ──────────────────────────────────────────────
    split_idx      = int(len(featured_df) * TRAIN_RATIO)
    featured_train = featured_df.iloc[:split_idx].copy()
    featured_test  = featured_df.iloc[split_idx:].copy()
    logger.info(f"Split → Train: {len(featured_train):,} | Test: {len(featured_test):,}")

    # ── 5. Train HMM ───────────────────────────────────────────────────────
    logger.info("Step 5: Training Gaussian HMM...")
    clf = HMMRegimeClassifier(
        n_components=3,
        n_iter=200,
        covariance_type="diag",
        n_restarts=5,
    )
    clf.fit(featured_train)

    # ── 6. Walk-forward validation ─────────────────────────────────────────
    logger.info("Step 6: Walk-forward validation...")
    val = walk_forward_validate(clf, featured_train, featured_test)

    # ── 7. Label full dataset ──────────────────────────────────────────────
    logger.info("Step 7: Labelling full dataset...")
    splitter      = RegimeDataSplitter(clf)
    regime_splits = splitter.run(price_df, news_df, dxy_df=dxy_df, save=True)

    # ── 8. Diagnose regime means ───────────────────────────────────────────
    labelled_df = pd.read_parquet("data/regimes/full_labelled_dataset.parquet")
    diagnose_regime_means(labelled_df)
    analyse_regime_transitions(labelled_df)

    # ── 9. Store regime labels in DuckDB ───────────────────────────────────
    logger.info("Step 9: Writing regime labels to DuckDB...")
    regime_records = labelled_df[["timestamp", "regime_label", "confidence"]].copy()
    regime_records.columns = ["timestamp", "regime", "confidence"]
    con.execute("DELETE FROM market_regimes")
    con.execute("INSERT INTO market_regimes SELECT timestamp, regime, confidence FROM regime_records")
    stored = con.execute("SELECT COUNT(*) FROM market_regimes").fetchone()[0]
    logger.success(f"Stored {stored:,} regime labels in DuckDB")

    # ── 10. Save model ─────────────────────────────────────────────────────
    clf.save(MODEL_SAVE_PATH)

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(labelled_df)
    logger.info("")
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║              PIPELINE SUMMARY  v4            ║")
    logger.info("╠══════════════════════════════════════════════╣")
    logger.info(f"║  Total bars labelled : {total:>8,}              ║")
    for label, df in regime_splits.items():
        pct = 100 * len(df) / total
        status = "✓" if len(df) >= 100 else "✗"
        logger.info(f"║  {label:12s}  {status}  {len(df):>8,}  ({pct:5.1f}%)        ║")
    logger.info(f"║  Train LL / bar     : {val['train_ll']:>10.4f}              ║")
    logger.info(f"║  Test  LL / bar     : {val['test_ll']:>10.4f}              ║")
    logger.info(f"║  Model saved        : {MODEL_SAVE_PATH:<28s}║")
    logger.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    run()