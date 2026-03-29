"""
Range Regime Feature Engineering (Mean Reversion)
==================================================
Features designed for RANGING market conditions.
Focus: distance from equilibrium, band position, oscillators.

Target: next bar direction — 1 = price goes up, 0 = price goes down
"""

import pandas as pd
import numpy as np
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────
RANGE_FEATURE_COLS = [
    "dist_from_ma20",
    "dist_from_ma50",
    "bb_position",
    "bb_width",
    "rsi_14",
    "rsi_overbought",
    "rsi_oversold",
    "dist_from_support",
    "dist_from_resistance",
    "price_percentile_20",
    "price_percentile_50",
    "mean_reversion_signal",
    "body_ratio",
    "candle_direction",
    "volume_ratio",
    "close_position",
    "atr_norm",
    "consecutive_up",
    "consecutive_down",
    "dist_from_vwap",
]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a price DataFrame (timestamp, open, high, low, close, volume)
    and returns a DataFrame with mean-reversion features + target label.

    Target (next_bar_up):
        1 = next bar closes higher than current close
        0 = next bar closes lower or equal
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Moving averages ────────────────────────────────────────────────────
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    df["dist_from_ma20"] = (df["close"] - ma20) / (ma20 + 1e-9)
    df["dist_from_ma50"] = (df["close"] - ma50) / (ma50 + 1e-9)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    std20          = df["close"].rolling(20).std()
    bb_upper       = ma20 + 2 * std20
    bb_lower       = ma20 - 2 * std20
    bb_range       = (bb_upper - bb_lower).replace(0, 1e-9)
    df["bb_position"] = (df["close"] - bb_lower) / bb_range   # 0=lower band, 1=upper band
    df["bb_width"]    = bb_range / (ma20 + 1e-9)              # band width normalised

    # ── RSI ────────────────────────────────────────────────────────────────
    df["rsi_14"]       = _rsi(df["close"], 14)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

    # ── Support / Resistance (20-bar rolling min/max) ──────────────────────
    support    = df["low"].rolling(20).min()
    resistance = df["high"].rolling(20).max()
    df["dist_from_support"]    = (df["close"] - support)    / (df["close"] + 1e-9)
    df["dist_from_resistance"] = (df["close"] - resistance) / (df["close"] + 1e-9)

    # ── Price percentile within recent range ──────────────────────────────
    roll20_min = df["close"].rolling(20).min()
    roll20_max = df["close"].rolling(20).max()
    roll50_min = df["close"].rolling(50).min()
    roll50_max = df["close"].rolling(50).max()
    df["price_percentile_20"] = (df["close"] - roll20_min) / (roll20_max - roll20_min + 1e-9)
    df["price_percentile_50"] = (df["close"] - roll50_min) / (roll50_max - roll50_min + 1e-9)

    # ── Mean reversion signal: composite ──────────────────────────────────
    # +1 when price is low (oversold, below MA, near support) = expect bounce up
    # -1 when price is high (overbought, above MA, near resistance) = expect drop
    df["mean_reversion_signal"] = (
        -df["dist_from_ma20"]           # negative = below MA = bullish
        - df["bb_position"] * 2         # low bb_position = bullish
        + df["dist_from_support"]       # far from support = less bullish
        + df["dist_from_resistance"]    # close to resistance = bearish
    )

    # ── Candle structure ───────────────────────────────────────────────────
    candle_range         = (df["high"] - df["low"]).replace(0, 1e-9)
    body                 = (df["close"] - df["open"]).abs()
    df["body_ratio"]     = body / candle_range
    df["candle_direction"] = (df["close"] > df["open"]).astype(int)
    df["close_position"] = (df["close"] - df["low"]) / candle_range

    # ── ATR normalised ─────────────────────────────────────────────────────
    atr            = candle_range.rolling(14).mean()
    df["atr_norm"] = candle_range / (atr + 1e-9)

    # ── Volume ratio ───────────────────────────────────────────────────────
    vol_ma          = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (vol_ma + 1e-9)

    # ── Consecutive up/down bars ───────────────────────────────────────────
    direction = (df["close"] > df["close"].shift(1)).astype(int)
    consec_up   = []
    consec_down = []
    up_count = down_count = 0
    for d in direction:
        if d == 1:
            up_count += 1
            down_count = 0
        else:
            down_count += 1
            up_count = 0
        consec_up.append(up_count)
        consec_down.append(down_count)
    df["consecutive_up"]   = consec_up
    df["consecutive_down"] = consec_down

    # ── VWAP distance (approximated with rolling typical price / volume) ───
    typical_price    = (df["high"] + df["low"] + df["close"]) / 3
    vwap             = (typical_price * df["volume"]).rolling(24).sum() / (df["volume"].rolling(24).sum() + 1e-9)
    df["dist_from_vwap"] = (df["close"] - vwap) / (vwap + 1e-9)

    # ── Target label ───────────────────────────────────────────────────────
    df["next_bar_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Drop NaNs ──────────────────────────────────────────────────────────
    df = df.dropna(subset=RANGE_FEATURE_COLS + ["next_bar_up"]).reset_index(drop=True)

    logger.info(f"Range features built: {len(df):,} bars × {len(RANGE_FEATURE_COLS)} features")
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    BASE_DIR = Path(__file__).resolve().parents[3]
    PARQUET  = BASE_DIR / "data" / "regimes" / "ranging_data.parquet"

    print(f"Looking for: {PARQUET}")
    assert PARQUET.exists(), f"File not found: {PARQUET}"

    sample = pd.read_parquet(PARQUET)
    result = build_range_features(sample)
    print(result[RANGE_FEATURE_COLS + ["next_bar_up"]].head())
    print(f"\nTarget distribution:\n{result['next_bar_up'].value_counts(normalize=True)}")