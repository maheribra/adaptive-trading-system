"""
DXY Correlation Feature Engineering
=====================================
Features for DXY-based correlation trading.
Used on RANGING regime data where DXY/price correlation is most stable.

Target: next bar direction — 1 = price goes up, 0 = price goes down
"""

import pandas as pd
import numpy as np
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────
DXY_FEATURE_COLS = [
    "dxy_pct_change",
    "dxy_ma5_dist",
    "dxy_ma20_dist",
    "dxy_trend",
    "dxy_bullish",
    "dxy_acceleration",
    "corr_10",
    "corr_20",
    "corr_50",
    "divergence_10",
    "divergence_20",
    "price_vs_dxy_momentum",
    "dxy_vol",
    "price_vol",
    "vol_ratio",
    "dxy_range_norm",
    "price_range_norm",
    "dxy_close_position",
    "price_close_position",
    "regime_confidence",
]


def build_dxy_features(
    df: pd.DataFrame,
    dxy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Takes a price DataFrame (from ranging_data.parquet) and a DXY DataFrame,
    merges them on timestamp, and builds DXY correlation features.

    dxy_df must have columns: [timestamp, dxy] or [timestamp, close]

    IMPORTANT — regime_confidence dependency:
        This function uses a 'confidence' column from df as the
        regime_confidence feature. This column is produced by the HMM
        classifier (HMMRegimeClassifier.confidence()).

        At training time: automatically present in ranging_data.parquet.
        At inference time: the meta-controller MUST run the HMM first
            and add a 'confidence' column to the bar DataFrame before
            calling this function or dxy_model.predict().

        Inference call order:
            1. engineer_features(raw_bars)
            2. hmm.predict() + hmm.confidence()  → adds 'confidence' col
            3. build_dxy_features(bars_with_confidence, dxy_df)
            4. dxy_model.predict()
    """
    df  = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # ── Prepare DXY ────────────────────────────────────────────────────────
    dxy = dxy_df.copy()
    dxy["timestamp"] = pd.to_datetime(dxy["timestamp"]).dt.tz_localize(None)

    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    dxy["timestamp"] = dxy["timestamp"].astype("datetime64[ns]")

    # Normalise column name
    if "dxy" in dxy.columns:
        dxy = dxy.rename(columns={"dxy": "dxy_close"})
    elif "close" in dxy.columns:
        dxy = dxy.rename(columns={"close": "dxy_close"})
    else:
        raise ValueError("dxy_df must have a 'dxy' or 'close' column")

    dxy = dxy[["timestamp", "dxy_close"]].drop_duplicates("timestamp")

    # ── Merge price + DXY on nearest timestamp ─────────────────────────────
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        dxy.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )

    missing_dxy = df["dxy_close"].isna().sum()
    if missing_dxy > 0:
        logger.warning(f"{missing_dxy} rows have no DXY match — forward filling")
        df["dxy_close"] = df["dxy_close"].ffill().bfill()

    # ── DXY momentum features ──────────────────────────────────────────────
    dxy_ret                = df["dxy_close"].pct_change()
    df["dxy_pct_change"]   = dxy_ret.clip(-0.02, 0.02)

    dxy_ma5                = df["dxy_close"].rolling(5).mean()
    dxy_ma20               = df["dxy_close"].rolling(20).mean()
    df["dxy_ma5_dist"]     = (df["dxy_close"] - dxy_ma5)  / (dxy_ma5  + 1e-9)
    df["dxy_ma20_dist"]    = (df["dxy_close"] - dxy_ma20) / (dxy_ma20 + 1e-9)
    df["dxy_trend"]        = (dxy_ma5 > dxy_ma20).astype(int)
    df["dxy_bullish"]      = (df["dxy_close"] > dxy_ma5).astype(int)
    df["dxy_acceleration"] = dxy_ret.diff().clip(-0.01, 0.01)

    # ── Rolling correlation: price vs DXY ──────────────────────────────────
    price_ret = df["close"].pct_change()
    df["corr_10"] = price_ret.rolling(10).corr(dxy_ret).clip(-1, 1)
    df["corr_20"] = price_ret.rolling(20).corr(dxy_ret).clip(-1, 1)
    df["corr_50"] = price_ret.rolling(50).corr(dxy_ret).clip(-1, 1)

    # ── Divergence: price and DXY moving in opposite directions ───────────
    price_ma10 = df["close"].rolling(10).mean()
    price_ma20 = df["close"].rolling(20).mean()
    price_trend_10 = (df["close"] > price_ma10).astype(int)
    price_trend_20 = (df["close"] > price_ma20).astype(int)

    # Divergence = 1 when price and DXY trend in opposite directions
    df["divergence_10"] = (price_trend_10 != df["dxy_trend"]).astype(int)
    df["divergence_20"] = (price_trend_20 != df["dxy_trend"]).astype(int)

    # ── Momentum divergence: price momentum vs DXY momentum ───────────────
    price_mom = df["close"].pct_change(10)
    dxy_mom   = df["dxy_close"].pct_change(10)
    df["price_vs_dxy_momentum"] = (price_mom - dxy_mom).clip(-0.05, 0.05)

    # ── Volatility comparison ──────────────────────────────────────────────
    df["dxy_vol"]   = dxy_ret.rolling(20).std().clip(0, 0.01)
    df["price_vol"] = price_ret.rolling(20).std().clip(0, 0.01)
    df["vol_ratio"] = (df["price_vol"] / (df["dxy_vol"] + 1e-9)).clip(0, 10)

    # ── Range normalisation ────────────────────────────────────────────────
    dxy_range              = (df["dxy_close"].rolling(14).max() - df["dxy_close"].rolling(14).min())
    price_range            = (df["high"] - df["low"])
    atr                    = price_range.rolling(14).mean()
    df["dxy_range_norm"]   = (dxy_range / (df["dxy_close"].rolling(14).mean() + 1e-9)).clip(0, 0.05)
    df["price_range_norm"] = (price_range / (atr + 1e-9)).clip(0, 5)

    # ── Close position within bar ──────────────────────────────────────────
    candle_range               = price_range.replace(0, 1e-9)
    df["price_close_position"] = ((df["close"] - df["low"]) / candle_range).clip(0, 1)
    dxy_14_low                 = df["dxy_close"].rolling(14).min()
    dxy_14_high                = df["dxy_close"].rolling(14).max()
    dxy_14_range               = (dxy_14_high - dxy_14_low).replace(0, 1e-9)
    df["dxy_close_position"]   = ((df["dxy_close"] - dxy_14_low) / dxy_14_range).clip(0, 1)

    # ── Regime confidence (already in parquet) ─────────────────────────────
    # ── Regime confidence (injected by meta-controller or from parquet) ────
    # At training time: comes from the HMM confidence column in
    #   ranging_data.parquet (produced by regime_data_splitter.py).
    # At inference time: the meta-controller must run the HMM first and
    #   inject the confidence score into the DataFrame before calling
    #   build_dxy_features() or dxy_model.predict().
    # If missing, we fall back to 0.5 (neutral) and log a warning —
    #   this should never happen in production.
    if "confidence" in df.columns:
        df["regime_confidence"] = df["confidence"].clip(0, 1)
    else:
        logger.warning(
            "regime_confidence: 'confidence' column not found in DataFrame. "
            "Falling back to 0.5 (neutral). "
            "At inference time the meta-controller must inject HMM confidence "
            "before calling build_dxy_features() or dxy_model.predict(). "
            "This should not happen in production."
        )
        df["regime_confidence"] = 0.5

    # ── Target label ───────────────────────────────────────────────────────
    df["next_bar_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ── Drop NaNs ──────────────────────────────────────────────────────────
    df = df.dropna(subset=DXY_FEATURE_COLS + ["next_bar_up"]).reset_index(drop=True)

    logger.info(
        f"DXY features built: {len(df):,} bars × {len(DXY_FEATURE_COLS)} features"
    )
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    BASE_DIR = Path(__file__).resolve().parents[3]
    RANGING  = BASE_DIR / "data" / "regimes" / "ranging_data.parquet"
    DXY_CSV  = BASE_DIR / "data" / "raw" / "dxy_index.csv"

    print(f"Loading ranging data: {RANGING}")
    assert RANGING.exists(), f"Not found: {RANGING}"

    print(f"Loading DXY data: {DXY_CSV}")
    assert DXY_CSV.exists(), f"Not found: {DXY_CSV}"

    price_df = pd.read_parquet(RANGING)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")

    dxy_df = pd.read_csv(DXY_CSV)
    dxy_df.columns = [c.lower() for c in dxy_df.columns]
    dxy_df = dxy_df.rename(columns={"time": "timestamp"})
    dxy_df["timestamp"] = pd.to_datetime(dxy_df["timestamp"]).dt.tz_localize(None).astype("datetime64[ns]")

    result = build_dxy_features(price_df, dxy_df)
    print(result[DXY_FEATURE_COLS + ["next_bar_up"]].head())
    print(f"\nTarget distribution:\n{result['next_bar_up'].value_counts(normalize=True)}")