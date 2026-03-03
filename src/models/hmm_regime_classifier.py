"""
HMM Market Regime Classifier  v3
==================================
Regimes:
    0 = RANGING     - low volatility, choppy sideways movement
    1 = TRENDING    - directional momentum, sustained price moves
    2 = NEWS_DRIVEN - sharp volatility spikes, large candles

v3 fixes:
    - Removed volume_zscore  (FX CSVs have no volume — was all zeros)
    - Removed return_autocorr (rolling corr was silently returning 0)
    - Replaced with: atr_ratio, bb_position, momentum_consistency
    - These 3 are pure price-based and cleanly separate all 3 regimes
    - Regime scoring rewritten to use only reliable features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    logger.error("hmmlearn not installed. Run: pip install hmmlearn==0.3.2")
    HMMLEARN_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────
N_REGIMES      = 3
REGIME_LABELS  = {0: "RANGING", 1: "TRENDING", 2: "NEWS_DRIVEN"}
REGIME_COLORS  = {0: "#f0ad4e", 1: "#5cb85c", 2: "#d9534f"}
MIN_TRAIN_BARS = 500
N_RESTARTS     = 8

# ── Feature columns (all pure price-based, no volume dependency) ───────────
FEATURE_COLS = [
    "volatility",           # z-scored rolling std of log returns
    "vol_regime",           # 1 if current vol > 100-bar median
    "trend_strength",       # z-scored abs 50-bar momentum
    "trend_direction",      # sign of 20-bar momentum (+1/0/-1)
    "hl_range_norm",        # (H-L) / 14-bar ATR
    "volatility_spike",     # 1 if vol > 2x rolling median (news proxy)
    "atr_ratio",            # current ATR / 50-bar avg ATR  (expansion detector)
    "bb_position",          # where close sits in Bollinger Bands [-1, +1]
    "momentum_consistency", # fraction of last 10 bars moving in same direction
]


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  v3
# ══════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV data (volume not required).

    Parameters
    ----------
    df      : DataFrame with [timestamp, open, high, low, close, volume]
    news_df : Optional DataFrame with [event_time] column

    Returns
    -------
    df with FEATURE_COLS appended, NaN rows dropped
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["log_ret"] = log_ret

    # ── 1. Volatility (z-scored rolling std) ──────────────────────────────
    raw_vol  = log_ret.rolling(20).std()
    vol_mean = raw_vol.rolling(200, min_periods=20).mean()
    vol_std  = raw_vol.rolling(200, min_periods=20).std()
    df["volatility"] = (raw_vol - vol_mean) / (vol_std + 1e-9)

    # ── 2. Vol regime: binary above/below 100-bar median ──────────────────
    vol_median    = raw_vol.rolling(100, min_periods=20).median()
    df["vol_regime"] = (raw_vol > vol_median).astype(float)

    # ── 3. Trend strength (z-scored abs 50-bar momentum) ──────────────────
    mom    = df["close"].pct_change(50).abs()
    mom_mu = mom.rolling(200, min_periods=50).mean()
    mom_sd = mom.rolling(200, min_periods=50).std()
    df["trend_strength"] = (mom - mom_mu) / (mom_sd + 1e-9)

    # ── 4. Trend direction: sign of 20-bar momentum ────────────────────────
    df["trend_direction"] = np.sign(df["close"].pct_change(20)).fillna(0)

    # ── 5. Normalised HL range: (H-L) / 14-bar ATR ────────────────────────
    hl  = df["high"] - df["low"]
    atr = hl.rolling(14).mean()
    df["hl_range_norm"] = (hl / (atr + 1e-9)).clip(0, 5)

    # ── 6. Volatility spike: vol > 2× rolling median ──────────────────────
    vol_spike = (raw_vol > 2.0 * vol_median).astype(float)
    if news_df is not None and not news_df.empty and "event_time" in news_df.columns:
        news_times = pd.to_datetime(
            news_df["event_time"], utc=True, errors="coerce"
        ).dropna()
        df_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for nt in news_times:
            mask = (
                (df_ts >= nt - pd.Timedelta(hours=1)) &
                (df_ts <= nt + pd.Timedelta(hours=4))
            )
            vol_spike[mask] = 1.0
    df["volatility_spike"] = vol_spike

    # ── 7. ATR ratio: current ATR vs 50-bar average ATR ───────────────────
    atr_avg = atr.rolling(50, min_periods=14).mean()
    df["atr_ratio"] = (atr / (atr_avg + 1e-9)).clip(0, 4)

    # ── 8. Bollinger Band position ─────────────────────────────────────────
    # Where is close relative to its 20-bar Bollinger Band?
    # +1 = at upper band, -1 = at lower band, 0 = at middle
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = ((df["close"] - bb_mid) / (bb_range / 2 + 1e-9)).clip(-2, 2)

    # ── 9. Momentum consistency ────────────────────────────────────────────
    # Fraction of last 10 bars where return has same sign as current bar
    # High value = persistent trend, Low = choppy
    sign_ret = np.sign(log_ret)
    def consistency(x):
        if len(x) < 2:
            return 0.0
        current_sign = x.iloc[-1]
        if current_sign == 0:
            return 0.0
        return float((x == current_sign).sum()) / len(x)

    df["momentum_consistency"] = (
        sign_ret.rolling(10, min_periods=5)
        .apply(consistency, raw=False)
        .fillna(0.5)
    )

    # ── Drop NaN rows ──────────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # ── Sanity check: warn about any dead features ─────────────────────────
    for col in FEATURE_COLS:
        std = df[col].std()
        if std < 0.001:
            logger.warning(
                f"Feature '{col}' has near-zero std ({std:.6f}) — "
                f"will not help the HMM discriminate regimes."
            )

    logger.info(
        f"Feature engineering complete: {len(df):,} bars × {len(FEATURE_COLS)} features"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════
#  HMM CLASSIFIER  v3
# ══════════════════════════════════════════════════════════════════════════

class HMMRegimeClassifier:
    """
    Gaussian HMM — 3 hidden states → RANGING / TRENDING / NEWS_DRIVEN.

    Regime signatures in scaled feature space:
    ┌─────────────────────┬──────────┬──────────┬─────────────┐
    │ Feature             │ RANGING  │ TRENDING │ NEWS_DRIVEN │
    ├─────────────────────┼──────────┼──────────┼─────────────┤
    │ volatility          │ low      │ moderate │ HIGH        │
    │ vol_regime          │ low      │ moderate │ HIGH        │
    │ trend_strength      │ low      │ HIGH     │ moderate    │
    │ trend_direction     │ ~0       │ +1 or -1 │ ~0          │
    │ hl_range_norm       │ low      │ moderate │ HIGH        │
    │ volatility_spike    │ 0        │ 0        │ 1           │
    │ atr_ratio           │ low      │ moderate │ HIGH        │
    │ bb_position         │ ~0       │ +1 or -1 │ extreme     │
    │ momentum_consistency│ low      │ HIGH     │ low         │
    └─────────────────────┴──────────┴──────────┴─────────────┘
    """

    def __init__(
        self,
        n_components: int = N_REGIMES,
        n_iter: int = 300,
        covariance_type: str = "full",
        n_restarts: int = N_RESTARTS,
        random_state: int = 42,
    ):
        if not HMMLEARN_AVAILABLE:
            raise ImportError("pip install hmmlearn==0.3.2")
        self.n_components    = n_components
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.n_restarts      = n_restarts
        self.random_state    = random_state
        self._model: Optional[GaussianHMM] = None
        self._scaler         = StandardScaler() if SKLEARN_AVAILABLE else None
        self._state_map: Dict[int, int] = {}
        self.is_fitted       = False

    # ── Array helper ────────────────────────────────────────────────────────
    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X[FEATURE_COLS].values
        else:
            arr = np.array(X)
        return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(self, X) -> "HMMRegimeClassifier":
        raw = self._to_array(X)
        if len(raw) < MIN_TRAIN_BARS:
            raise ValueError(f"Need ≥{MIN_TRAIN_BARS} bars. Got {len(raw)}.")

        # Scale
        if self._scaler is not None:
            scaled = self._scaler.fit_transform(raw)
        else:
            self._mu  = raw.mean(0)
            self._sd  = raw.std(0) + 1e-9
            scaled    = (raw - self._mu) / self._sd

        logger.info(
            f"Training HMM: {len(scaled):,} bars | "
            f"{self.n_components} states | "
            f"{self.n_iter} iters | "
            f"{self.n_restarts} restarts"
        )

        best_model, best_score = None, -np.inf
        for i in range(self.n_restarts):
            try:
                m = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state + i,
                    verbose=False,
                )
                m.fit(scaled)
                s = m.score(scaled)
                logger.debug(f"  Restart {i+1}/{self.n_restarts} score: {s:.2f}")
                if s > best_score:
                    best_score, best_model = s, m
            except Exception as e:
                logger.warning(f"  Restart {i+1} failed: {e}")

        if best_model is None:
            raise RuntimeError("All HMM restarts failed.")

        self._model    = best_model
        self._state_map = self._resolve_mapping()
        self.is_fitted  = True

        logger.success(
            f"HMM trained (best score: {best_score:.2f}). Mapping: " +
            ", ".join(
                f"state{k}→{REGIME_LABELS[v]}"
                for k, v in sorted(self._state_map.items())
            )
        )
        return self

    def _resolve_mapping(self) -> Dict[int, int]:
        """
        Map HMM states to regimes via emission means.

        FEATURE_COLS index:
            0:volatility  1:vol_regime  2:trend_strength  3:trend_direction
            4:hl_range_norm  5:volatility_spike  6:atr_ratio
            7:bb_position  8:momentum_consistency
        """
        means  = self._model.means_   # (n_states, n_features)
        scores = np.zeros((self.n_components, N_REGIMES))

        for s in range(self.n_components):
            m = means[s]
            # idx:  0=vol  1=vol_regime  2=trend_str  3=trend_dir
            #       4=hl   5=spike       6=atr_ratio
            #       7=bb   8=consistency

            # RANGING: low vol, low trend, low consistency, bb near centre
            scores[s, 0] = (
                -m[0]          # low volatility
                -m[1]          # low vol_regime
                -m[2]          # low trend_strength
                -abs(m[3])     # trend_direction near 0
                -m[8]          # low momentum_consistency
                -abs(m[7])     # bb_position near 0
            )

            # TRENDING: high trend_strength, high consistency, strong direction
            scores[s, 1] = (
                 m[2]          # high trend_strength
                + m[8]         # high momentum_consistency
                + abs(m[3])    # strong trend_direction either way
                + abs(m[7])    # bb_position near extremes
                - m[5]         # low volatility_spike (not news)
            )

            # NEWS_DRIVEN: high spike, high vol, high ATR, high hl_range
            scores[s, 2] = (
                 m[5]          # high volatility_spike
                + m[0]         # high volatility
                + m[6]         # high atr_ratio
                + m[4]         # high hl_range_norm
                + m[1]         # high vol_regime
            )

        mapping: Dict[int, int] = {}
        used: set = set()
        for idx in np.argsort(-scores, axis=None):
            state, regime = divmod(int(idx), N_REGIMES)
            if state not in mapping and regime not in used:
                mapping[state] = regime
                used.add(regime)
            if len(mapping) == self.n_components:
                break
        return mapping

    # ── Scale for inference ──────────────────────────────────────────────────
    def _scale(self, raw: np.ndarray) -> np.ndarray:
        if self._scaler is not None and hasattr(self._scaler, 'mean_'):
            return self._scaler.transform(raw)
        return (raw - self._mu) / self._sd

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, X) -> np.ndarray:
        """Viterbi-decoded regime sequence. Returns int array (T,)."""
        self._check_fitted()
        scaled = self._scale(self._to_array(X))
        raw    = self._model.predict(scaled)
        return np.array([self._state_map.get(int(s), 0) for s in raw])

    def predict_proba(self, X) -> np.ndarray:
        """Posterior probabilities. Shape (T, 3) → [RANGING, TRENDING, NEWS_DRIVEN]."""
        self._check_fitted()
        scaled        = self._scale(self._to_array(X))
        _, posteriors = self._model.score_samples(scaled)
        out           = np.zeros((len(scaled), N_REGIMES))
        for raw_state, regime_id in self._state_map.items():
            out[:, regime_id] += posteriors[:, raw_state]
        return out

    def confidence(self, X) -> np.ndarray:
        return self.predict_proba(X).max(axis=1)

    def log_likelihood(self, X) -> float:
        self._check_fitted()
        try:
            scaled = self._scale(self._to_array(X))
            score = float(self._model.score(scaled)) / len(scaled)
            # Catch numerical overflow from unseen feature combinations
            if not np.isfinite(score) or score < -1e6:
                logger.warning(f"Log-likelihood overflow detected ({score:.2f}) — clamping to -999")
                return -999.0
            return score
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}")
            return -999.0

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "HMMRegimeClassifier":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.success(f"Model loaded ← {path}")
        return obj


