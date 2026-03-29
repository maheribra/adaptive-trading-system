"""
Trend Regime Feature Engineering (ICT-Based)
=============================================
Features designed for TRENDING market conditions.
Inspired by ICT concepts: liquidity, sessions, market structure.

Target: next bar direction — 1 = price goes up, 0 = price goes down
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────
LONDON_OPEN_H  = 8    # UTC
LONDON_CLOSE_H = 12
NY_OPEN_H      = 13
NY_CLOSE_H     = 17

TREND_FEATURE_COLS = [
    "prev_high_dist",
    "prev_low_dist",
    "high_20_dist",
    "low_20_dist",
    "session_london",
    "session_ny",
    "session_overlap",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "momentum_5",
    "momentum_20",
    "atr_norm",
    "close_position",
    "structure_break_up",
    "structure_break_down",
    "liquidity_sweep_high",
    "liquidity_sweep_low",
    "higher_high",
    "lower_low",
]


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a price DataFrame (timestamp, open, high, low, close, volume)
    and returns a DataFrame with ICT-inspired trend features + target label.

    Target (next_bar_up):
        1 = next bar closes higher than current close
        0 = next bar closes lower or equal
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Session flags ──────────────────────────────────────────────────────
    hour = df["timestamp"].dt.hour
    df["session_london"]  = ((hour >= LONDON_OPEN_H) & (hour < LONDON_CLOSE_H)).astype(int)
    df["session_ny"]      = ((hour >= NY_OPEN_H)     & (hour < NY_CLOSE_H)).astype(int)
    df["session_overlap"] = ((hour >= NY_OPEN_H)     & (hour < LONDON_CLOSE_H)).astype(int)

    # ── Previous day high/low distance ────────────────────────────────────
    daily_high = df["high"].rolling(24).max().shift(1)
    daily_low  = df["low"].rolling(24).min().shift(1)
    df["prev_high_dist"] = (df["close"] - daily_high) / (df["close"] + 1e-9)
    df["prev_low_dist"]  = (df["close"] - daily_low)  / (df["close"] + 1e-9)

    # ── 20-bar swing high/low distance ────────────────────────────────────
    swing_high = df["high"].rolling(20).max()
    swing_low  = df["low"].rolling(20).min()
    df["high_20_dist"] = (df["close"] - swing_high) / (df["close"] + 1e-9)
    df["low_20_dist"]  = (df["close"] - swing_low)  / (df["close"] + 1e-9)

    # ── Candle structure ───────────────────────────────────────────────────
    candle_range         = (df["high"] - df["low"]).replace(0, 1e-9)
    body                 = (df["close"] - df["open"]).abs()
    df["body_ratio"]     = body / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    # ── Momentum ───────────────────────────────────────────────────────────
    df["momentum_5"]  = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)

    # ── ATR normalised ─────────────────────────────────────────────────────
    atr             = candle_range.rolling(14).mean()
    df["atr_norm"]  = candle_range / (atr + 1e-9)

    # ── Close position within bar range ───────────────────────────────────
    df["close_position"] = (df["close"] - df["low"]) / candle_range

    # ── Market structure: higher highs / lower lows ────────────────────────
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"]   = (df["low"]  < df["low"].shift(1)).astype(int)

    # ── Structure breaks ───────────────────────────────────────────────────
    # Break of structure up: close above previous 10-bar high
    prev_10_high              = df["high"].rolling(10).max().shift(1)
    prev_10_low               = df["low"].rolling(10).min().shift(1)
    df["structure_break_up"]   = (df["close"] > prev_10_high).astype(int)
    df["structure_break_down"] = (df["close"] < prev_10_low).astype(int)

    # ── Liquidity sweeps ───────────────────────────────────────────────────
    # Sweep high: wick went above previous high but closed below it
    df["liquidity_sweep_high"] = (
        (df["high"] > df["high"].shift(1)) &
        (df["close"] < df["high"].shift(1))
    ).astype(int)

    # Sweep low: wick went below previous low but closed above it
    df["liquidity_sweep_low"] = (
        (df["low"] < df["low"].shift(1)) &
        (df["close"] > df["low"].shift(1))
    ).astype(int)

    # ── Target label ───────────────────────────────────────────────────────
    df["next_bar_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Drop NaNs ─────────────────────────────────────────────────────────
    df = df.dropna(subset=TREND_FEATURE_COLS + ["next_bar_up"]).reset_index(drop=True)

    logger.info(f"Trend features built: {len(df):,} bars × {len(TREND_FEATURE_COLS)} features")
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    # Build absolute path relative to this file's location
    BASE_DIR = Path(__file__).resolve().parents[3]
    PARQUET  = BASE_DIR / "data" / "regimes" / "trending_data.parquet"

    print(f"Looking for: {PARQUET}")  # debug line so we can see exact path
    assert PARQUET.exists(), f"File not found: {PARQUET}"

    sample = pd.read_parquet(PARQUET)
    result = build_trend_features(sample)
    print(result[TREND_FEATURE_COLS + ["next_bar_up"]].head())
    print(f"\nTarget distribution:\n{result['next_bar_up'].value_counts(normalize=True)}")