"""
Trend Regime Feature Engineering (ICT-Based + Momentum)
=======================================================
Features designed for TRENDING market conditions.
Inspired by ICT concepts: liquidity, sessions, market structure.
Added: ADX for trend strength and ROC for velocity.

Target: next bar direction — 1 = price goes up, 0 = price goes down
"""

import pandas as pd
import numpy as np
from loguru import logger

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
    "adx_14",          # NEW: Trend Strength
    "roc_5",           # NEW: Velocity (Short-term)
    "roc_20",          # NEW: Velocity (Medium-term)
]

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Standard Average Directional Index (ADX) calculation."""
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff().clip(upper=0).abs()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(period).mean()
    return adx

def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a price DataFrame and returns ICT + Momentum features.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── 1. Session flags ───────────────────────────────────────────────────
    hour = df["timestamp"].dt.hour
    df["session_london"]  = ((hour >= LONDON_OPEN_H) & (hour < LONDON_CLOSE_H)).astype(int)
    df["session_ny"]      = ((hour >= NY_OPEN_H)     & (hour < NY_CLOSE_H)).astype(int)
    df["session_overlap"] = ((hour >= NY_OPEN_H)     & (hour < LONDON_CLOSE_H)).astype(int)

    # ── 2. Previous day high/low distance ─────────────────────────────────
    daily_high = df["high"].rolling(24).max().shift(1)
    daily_low  = df["low"].rolling(24).min().shift(1)
    df["prev_high_dist"] = (df["close"] - daily_high) / (df["close"] + 1e-9)
    df["prev_low_dist"]  = (df["close"] - daily_low)  / (df["close"] + 1e-9)

    # ── 3. 20-bar swing high/low distance ─────────────────────────────────
    swing_high = df["high"].rolling(20).max()
    swing_low  = df["low"].rolling(20).min()
    df["high_20_dist"] = (df["close"] - swing_high) / (df["close"] + 1e-9)
    df["low_20_dist"]  = (df["close"] - swing_low)  / (df["close"] + 1e-9)

    # ── 4. Candle structure ────────────────────────────────────────────────
    candle_range         = (df["high"] - df["low"]).replace(0, 1e-9)
    body                 = (df["close"] - df["open"]).abs()
    df["body_ratio"]     = body / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    # ── 5. Momentum & Velocity (REFACTORED) ───────────────────────────────
    df["momentum_5"]  = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)
    df["roc_5"]       = df["close"].pct_change(5) * 100
    df["roc_20"]      = df["close"].pct_change(20) * 100
    df["adx_14"]      = calculate_adx(df, 14)

    # ── 6. ATR normalised ──────────────────────────────────────────────────
    atr             = candle_range.rolling(14).mean()
    df["atr_norm"]  = candle_range / (atr + 1e-9)

    # ── 7. Close position ──────────────────────────────────────────────────
    df["close_position"] = (df["close"] - df["low"]) / candle_range

    # ── 8. Market structure ────────────────────────────────────────────────
    df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
    df["lower_low"]   = (df["low"]  < df["low"].shift(1)).astype(int)

    # ── 9. Structure breaks ────────────────────────────────────────────────
    prev_10_high = df["high"].rolling(10).max().shift(1)
    prev_10_low  = df["low"].rolling(10).min().shift(1)
    df["structure_break_up"]   = (df["close"] > prev_10_high).astype(int)
    df["structure_break_down"] = (df["close"] < prev_10_low).astype(int)

    # ── 10. Liquidity sweeps ───────────────────────────────────────────────
    df["liquidity_sweep_high"] = ((df["high"] > df["high"].shift(1)) & (df["close"] < df["high"].shift(1))).astype(int)
    df["liquidity_sweep_low"]  = ((df["low"] < df["low"].shift(1)) & (df["close"] > df["low"].shift(1))).astype(int)

    # ── Target label ───────────────────────────────────────────────────────
    df["next_bar_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Final Cleaning ─────────────────────────────────────────────────────
    df = df.dropna(subset=TREND_FEATURE_COLS + ["next_bar_up"]).reset_index(drop=True)

    logger.info(f"Trend features built: {len(df):,} bars × {len(TREND_FEATURE_COLS)} features")
    return df

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    BASE_DIR = Path(__file__).resolve().parents[3]
    PARQUET  = BASE_DIR / "data" / "regimes" / "trending_data.parquet"

    if PARQUET.exists():
        sample = pd.read_parquet(PARQUET)
        result = build_trend_features(sample)
        print(result[TREND_FEATURE_COLS + ["next_bar_up"]].head())
        print(f"\nTarget distribution:\n{result['next_bar_up'].value_counts(normalize=True)}")
    else:
        print(f"File not found: {PARQUET}")