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
N_RESTARTS     = 10

# ── Feature columns — only features proven to discriminate regimes ─────────
FEATURE_COLS = [
    "volatility",        # RANGING=low, TRENDING=mid, NEWS=high
    "vol_regime",        # RANGING=0,   TRENDING=1,   NEWS=1
    "volatility_spike",  # RANGING=0,   TRENDING=0,   NEWS=1
    "atr_ratio",         # RANGING=low, TRENDING=high, NEWS=mid
    "trend_strength",    # RANGING=low, TRENDING=high, NEWS=mid
    "hl_range_norm",     # RANGING=mid, TRENDING=mid,  NEWS=high
]


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  v5
# ══════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None,
    dxy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["log_ret"] = log_ret

    # ── 1. Volatility (z-scored rolling std) ──────────────────────────────
    raw_vol  = log_ret.rolling(20).std()
    vol_mean = raw_vol.rolling(200, min_periods=20).mean()
    vol_std  = raw_vol.rolling(200, min_periods=20).std()
    df["volatility"] = (raw_vol - vol_mean) / (vol_std + 1e-9)

    # ── 2. Vol regime: binary above/below 100-bar median ──────────────────
    vol_median = raw_vol.rolling(100, min_periods=20).median()
    df["vol_regime"] = (raw_vol > vol_median).astype(float)

    # ── 3. Volatility spike: news proxy ───────────────────────────────────
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

    # ── 4. ATR ratio ───────────────────────────────────────────────────────
    hl      = df["high"] - df["low"]
    atr     = hl.rolling(14).mean()
    atr_avg = atr.rolling(50, min_periods=14).mean()
    df["atr_ratio"] = (atr / (atr_avg + 1e-9)).clip(0, 4)

    # ── 5. Trend strength (z-scored abs 50-bar momentum) ──────────────────
    mom    = df["close"].pct_change(50).abs()
    mom_mu = mom.rolling(200, min_periods=50).mean()
    mom_sd = mom.rolling(200, min_periods=50).std()
    df["trend_strength"] = (mom - mom_mu) / (mom_sd + 1e-9)

    # ── 6. Normalised HL range ─────────────────────────────────────────────
    df["hl_range_norm"] = (hl / (atr + 1e-9)).clip(0, 5)

    # ── Drop NaN rows ──────────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

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
#  HMM CLASSIFIER  v5
# ══════════════════════════════════════════════════════════════════════════

class HMMRegimeClassifier:

    def __init__(
        self,
        n_components: int = N_REGIMES,
        n_iter: int = 200,
        covariance_type: str = "diag",
        n_restarts: int = N_RESTARTS,
        random_state: int = 42,
        tol: float = 1e-4,
    ):
        if not HMMLEARN_AVAILABLE:
            raise ImportError("pip install hmmlearn==0.3.2")
        self.n_components    = n_components
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.n_restarts      = n_restarts
        self.random_state    = random_state
        self.tol             = tol
        self._model: Optional[GaussianHMM] = None
        self._scaler         = StandardScaler() if SKLEARN_AVAILABLE else None
        self._state_map: Dict[int, int] = {}
        self.is_fitted       = False

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X[FEATURE_COLS].values
        else:
            arr = np.array(X)
        return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X) -> "HMMRegimeClassifier":
        raw = self._to_array(X)
        if len(raw) < MIN_TRAIN_BARS:
            raise ValueError(f"Need ≥{MIN_TRAIN_BARS} bars. Got {len(raw)}.")

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
            f"{self.n_restarts} restarts | "
            f"cov={self.covariance_type}"
        )

        best_model, best_score = None, -np.inf
        for i in range(self.n_restarts):
            try:
                m = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
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

        self._model     = best_model
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
        means  = self._model.means_
        scores = np.zeros((self.n_components, N_REGIMES))

        # FEATURE_COLS index:
        # 0=volatility  1=vol_regime  2=volatility_spike
        # 3=atr_ratio   4=trend_strength  5=hl_range_norm

        for s in range(self.n_components):
            m = means[s]

            # RANGING: low vol, low vol_regime, no spike, low atr, low trend
            scores[s, 0] = (
                -m[0]   # low volatility
                -m[1]   # low vol_regime
                -m[2]   # no spike
                -m[3]   # low atr_ratio
                -m[4]   # low trend_strength
            )

            # TRENDING: high trend, high atr, high vol_regime, no spike
            scores[s, 1] = (
                 m[4]   # high trend_strength
                + m[3]  # high atr_ratio
                + m[1]  # high vol_regime
                - m[2]  # no spike
                + m[0]  # elevated volatility
            )

            # NEWS_DRIVEN: spike=1, high vol, high hl_range
            scores[s, 2] = (
                 m[2]   # high volatility_spike
                + m[0]  # high volatility
                + m[5]  # high hl_range_norm
                + m[1]  # high vol_regime
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

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        if self._scaler is not None and hasattr(self._scaler, 'mean_'):
            return self._scaler.transform(raw)
        return (raw - self._mu) / self._sd

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        scaled = self._scale(self._to_array(X))
        raw    = self._model.predict(scaled)
        return np.array([self._state_map.get(int(s), 0) for s in raw])

    def predict_proba(self, X) -> np.ndarray:
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
            raw     = self._to_array(X)
            scaled  = self._scale(raw)
            total   = self._model.score(scaled)
            per_bar = total / max(len(scaled), 1)
            if not np.isfinite(per_bar) or per_bar < -1e4:
                logger.warning(f"Log-likelihood overflow ({per_bar:.2f}) — clamping to -999")
                return -999.0
            return float(per_bar)
        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}")
            return -999.0

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

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