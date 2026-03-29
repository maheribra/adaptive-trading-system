"""
News Regime Feature Engineering
================================
Features designed for NEWS_DRIVEN market conditions.
Focus: volatility spikes, event timing, price reaction magnitude.

Target: next bar direction — 1 = price goes up, 0 = price goes down
"""

import pandas as pd
import numpy as np
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────
NEWS_FEATURE_COLS = [
    "volatility_spike",
    "volatility",
    "hl_range_norm",
    "atr_ratio",
    "vol_regime",
    "spike_magnitude",
    "spike_direction",
    "bars_since_spike",
    "post_spike_drift",
    "pre_spike_vol",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "momentum_3",
    "momentum_10",
    "vol_acceleration",
    "range_expansion",
    "candle_direction",
    "mean_reversion_pressure",
]


def build_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a price DataFrame with HMM features already computed
    (from news_driven_data.parquet) and adds news-specific features.

    Target (next_bar_up):
        1 = next bar closes higher than current close
        0 = next bar closes lower or equal
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    candle_range = (df["high"] - df["low"]).replace(0, 1e-9)
    atr          = candle_range.rolling(14).mean()

    # ── 1. Spike magnitude: how big is this bar vs recent average ──────────
    df["spike_magnitude"] = (candle_range / (atr + 1e-9)).clip(0, 10)

    # ── 2. Spike direction: which way did the spike go ─────────────────────
    # +1 if bullish spike (close > open), -1 if bearish
    df["spike_direction"] = np.where(df["close"] >= df["open"], 1.0, -1.0)
    # Only meaningful during spikes
    df["spike_direction"] = df["spike_direction"] * df["volatility_spike"]

    # ── 3. Bars since last spike ───────────────────────────────────────────
    bars_since = []
    count = 999
    for spike in df["volatility_spike"]:
        if spike == 1:
            count = 0
        else:
            count += 1
        bars_since.append(min(count, 50))  # cap at 50
    df["bars_since_spike"] = bars_since

    # ── 4. Post-spike drift: cumulative return since last spike ────────────
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    post_spike_drift = []
    cumret = 0.0
    for i, spike in enumerate(df["volatility_spike"]):
        if spike == 1:
            cumret = 0.0
        else:
            cumret += log_ret.iloc[i]
        post_spike_drift.append(cumret)
    df["post_spike_drift"] = np.clip(post_spike_drift, -0.05, 0.05)

    # ── 5. Pre-spike volatility: rolling std before spike ─────────────────
    df["pre_spike_vol"] = log_ret.rolling(5).std().shift(1).fillna(0)

    # ── 6. Candle structure ────────────────────────────────────────────────
    body                   = (df["close"] - df["open"]).abs()
    df["body_ratio"]       = (body / candle_range).clip(0, 1)
    df["upper_wick_ratio"] = (
        (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    ).clip(0, 1)
    df["lower_wick_ratio"] = (
        (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    ).clip(0, 1)
    df["close_position"]   = ((df["close"] - df["low"]) / candle_range).clip(0, 1)
    df["candle_direction"]  = (df["close"] >= df["open"]).astype(int)

    # ── 7. Short-term momentum ─────────────────────────────────────────────
    df["momentum_3"]  = df["close"].pct_change(3).clip(-0.05, 0.05)
    df["momentum_10"] = df["close"].pct_change(10).clip(-0.05, 0.05)

    # ── 8. Volatility acceleration: is vol increasing or decreasing ────────
    raw_vol              = log_ret.rolling(5).std()
    raw_vol_prev         = log_ret.rolling(5).std().shift(3)
    df["vol_acceleration"] = (raw_vol - raw_vol_prev).clip(-0.01, 0.01)

    # ── 9. Range expansion vs recent average ──────────────────────────────
    avg_range            = candle_range.rolling(20).mean()
    df["range_expansion"] = (candle_range / (avg_range + 1e-9)).clip(0, 5)

    # ── 10. Mean reversion pressure after spike ────────────────────────────
    ma20 = df["close"].rolling(20).mean()
    df["mean_reversion_pressure"] = (
        (df["close"] - ma20) / (ma20 + 1e-9)
    ).clip(-0.05, 0.05)

    # ── Target label ───────────────────────────────────────────────────────
    df["next_bar_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Drop NaNs ──────────────────────────────────────────────────────────
    df = df.dropna(subset=NEWS_FEATURE_COLS + ["next_bar_up"]).reset_index(drop=True)

    logger.info(
        f"News features built: {len(df):,} bars × {len(NEWS_FEATURE_COLS)} features"
    )
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    BASE_DIR = Path(__file__).resolve().parents[3]
    PARQUET  = BASE_DIR / "data" / "regimes" / "news_driven_data.parquet"

    print(f"Looking for: {PARQUET}")
    assert PARQUET.exists(), f"File not found: {PARQUET}"

    sample = pd.read_parquet(PARQUET)
    result = build_news_features(sample)
    print(result[NEWS_FEATURE_COLS + ["next_bar_up"]].head())
    print(f"\nTarget distribution:\n{result['next_bar_up'].value_counts(normalize=True)}")