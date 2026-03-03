"""
Tests for HMM Regime Classifier
================================
Run with:
    python -m pytest tests/test_regime_classifier.py -v
    or just:
    python tests/test_regime_classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from loguru import logger


# ── Helpers ────────────────────────────────────────────────────────────────

def make_synthetic_price_data(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data that contains all 3 regimes:
      - First third  : ranging  (low vol, low trend)
      - Second third : trending (strong uptrend)
      - Final third  : news spike (sudden volatility burst)
    """
    rng   = np.random.default_rng(seed)
    n3    = n_bars // 3
    dates = pd.date_range("2022-01-01", periods=n_bars, freq="1h", tz="UTC")

    # Ranging segment
    r1 = rng.normal(0, 0.0003, n3).cumsum()
    # Trending segment
    r2 = rng.normal(0.0005, 0.0004, n3).cumsum()
    # News-driven segment (big spikes)
    r3_base  = rng.normal(0, 0.0003, n3).cumsum()
    r3_spikes = np.zeros(n3)
    spike_idx = rng.choice(n3, size=20, replace=False)
    r3_spikes[spike_idx] = rng.choice([-1, 1], size=20) * rng.uniform(0.005, 0.015, 20)
    r3 = r3_base + r3_spikes

    returns = np.concatenate([r1, r2, r3])
    close   = 1.1000 * np.exp(np.cumsum(returns))
    spread  = rng.uniform(0.0001, 0.0005, n_bars)
    high    = close * (1 + spread)
    low     = close * (1 - spread)
    open_   = np.roll(close, 1)
    open_[0] = close[0]
    volume  = rng.uniform(500, 5000, n_bars)
    # Higher volume during news spikes
    volume[np.concatenate([np.zeros(2 * n3, dtype=bool),
                           np.isin(np.arange(n3), spike_idx)])] *= 4

    return pd.DataFrame({
        "timestamp": dates,
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "volume":    volume,
    })


def make_synthetic_news_data(price_df: pd.DataFrame, n_events: int = 30) -> pd.DataFrame:
    """Place news events in the final third of the price data."""
    rng   = np.random.default_rng(0)
    n     = len(price_df)
    idx   = rng.choice(np.arange(n * 2 // 3, n), size=n_events, replace=False)
    times = price_df["timestamp"].iloc[idx].values
    return pd.DataFrame({"event_time": times, "title": ["CPI Release"] * n_events})


# ══════════════════════════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_feature_engineering():
    logger.info("TEST: Feature engineering")
    from src.models.hmm_regime_classifier import engineer_features, FEATURE_COLS

    price_df = make_synthetic_price_data(1500)
    news_df  = make_synthetic_news_data(price_df)
    featured = engineer_features(price_df, news_df)

    # All feature columns must exist
    for col in FEATURE_COLS:
        assert col in featured.columns, f"Missing feature: {col}"

    # No NaNs in feature columns
    assert featured[FEATURE_COLS].isnull().sum().sum() == 0, "NaNs in features"

    # News flag should be > 0 somewhere
    assert featured["news_flag"].sum() > 0, "News flag never triggered"

    logger.success(f"  Feature engineering ok: {len(featured):,} bars")


def test_hmm_fit_predict():
    logger.info("TEST: HMM fit + predict")
    from src.models.hmm_regime_classifier import (
        HMMRegimeClassifier, engineer_features, FEATURE_COLS, REGIME_LABELS
    )

    price_df = make_synthetic_price_data(2000)
    featured = engineer_features(price_df)
    X        = featured[FEATURE_COLS].values.astype(np.float32)

    clf     = HMMRegimeClassifier(n_iter=50)   # fast for testing
    clf.fit(X)

    regimes = clf.predict(X)
    assert regimes.shape == (len(X),),  "Wrong output shape"
    assert set(regimes).issubset({0, 1, 2}), "Unknown regime id"

    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 3), "Wrong proba shape"
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    conf = clf.confidence(X)
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence out of [0,1]"

    logger.success(f"  Regime counts: { {REGIME_LABELS[r]: (regimes==r).sum() for r in [0,1,2]} }")


def test_all_regimes_detected():
    logger.info("TEST: All 3 regimes detected")
    from src.models.hmm_regime_classifier import (
        HMMRegimeClassifier, engineer_features, FEATURE_COLS
    )

    price_df = make_synthetic_price_data(3000)
    featured = engineer_features(price_df)
    X        = featured[FEATURE_COLS].values.astype(np.float32)

    clf = HMMRegimeClassifier(n_iter=100)
    clf.fit(X)
    regimes = clf.predict(X)

    for regime_id in [0, 1, 2]:
        count = (regimes == regime_id).sum()
        assert count >= 100, (
            f"Regime {regime_id} has only {count} samples (need ≥ 100)"
        )
    logger.success("  All 3 regimes have ≥ 100 samples")


def test_regime_splitter():
    logger.info("TEST: RegimeDataSplitter")
    from src.models.hmm_regime_classifier import HMMRegimeClassifier, engineer_features, FEATURE_COLS
    from src.models.regime_data_splitter import RegimeDataSplitter

    price_df = make_synthetic_price_data(3000)
    news_df  = make_synthetic_news_data(price_df)
    featured = engineer_features(price_df, news_df)
    X        = featured[FEATURE_COLS].values.astype(np.float32)

    clf = HMMRegimeClassifier(n_iter=100)
    clf.fit(X)

    splitter = RegimeDataSplitter(clf)
    splits   = splitter.run(price_df, news_df, save=False)

    assert set(splits.keys()) == {"RANGING", "TRENDING", "NEWS_DRIVEN"}
    total = sum(len(v) for v in splits.values())
    assert total > 0, "No data in splits"

    # Check required columns exist in each split
    required = ["timestamp", "close", "regime", "regime_label", "confidence"]
    for label, df in splits.items():
        for col in required:
            assert col in df.columns, f"{label} missing column: {col}"
    logger.success("  Splitter ok")


def test_model_save_load(tmp_path):
    logger.info("TEST: Model serialisation")
    from src.models.hmm_regime_classifier import (
        HMMRegimeClassifier, engineer_features, FEATURE_COLS
    )

    price_df = make_synthetic_price_data(1500)
    featured = engineer_features(price_df)
    X        = featured[FEATURE_COLS].values.astype(np.float32)

    clf = HMMRegimeClassifier(n_iter=50)
    clf.fit(X)

    path = str(tmp_path / "test_model.pkl")
    clf.save(path)

    clf2    = HMMRegimeClassifier.load(path)
    result1 = clf.predict(X)
    result2 = clf2.predict(X)
    np.testing.assert_array_equal(result1, result2)
    logger.success("  Save/load roundtrip ok")


# ── Run directly ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    tests = [
        test_feature_engineering,
        test_hmm_fit_predict,
        test_all_regimes_detected,
        test_regime_splitter,
    ]

    # For save/load test we need a tmp dir
    with tempfile.TemporaryDirectory() as tmp:
        tests.append(lambda: test_model_save_load(Path(tmp)))

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAILED {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed | {failed} failed")
    print(f"{'='*50}")