"""
Meta-Controller
===============
Loads all trained models, detects the current regime every bar,
routes to the correct ML model, and outputs a structured signal.

Signal output:
    {
        "signal":          "BUY" | "SELL" | "HOLD",
        "confidence":      float  (0–1, HMM regime confidence),
        "model_prob":      float  (0–1, routed model's probability for predicted class),
        "regime":          "RANGING" | "TRENDING" | "NEWS_DRIVEN",
        "regime_id":       int    (0 | 1 | 2),
        "bars_in_regime":  int    (consecutive bars confirmed in current regime),
        "confirmed":       bool   (True once ≥ REGIME_CONFIRM_BARS consecutive bars seen),
        "dxy_confirm":     bool | None  (DXY agreement in RANGING, else None),
        "timestamp":       str    (last bar timestamp),
    }

Regime routing:
    RANGING    → RangeModel (primary) + DXYModel (confirmation filter)
    TRENDING   → TrendModel
    NEWS_DRIVEN→ NewsModel

Regime transition policy:
    A regime change is only acted on after REGIME_CONFIRM_BARS (3)
    consecutive bars predict the new regime. Until confirmed, the
    controller holds the last confirmed regime and emits HOLD.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional
from dataclasses import dataclass, field

from src.models.hmm_regime_classifier import (
    HMMRegimeClassifier,
    engineer_features,
    REGIME_LABELS,
)
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel
from src.models.features.dxy_features import build_dxy_features

# ── Constants ──────────────────────────────────────────────────────────────
REGIME_CONFIRM_BARS  = 3      # consecutive bars before a transition is accepted
MIN_CONFIDENCE       = 0.45   # below this HMM confidence → HOLD regardless
MIN_MODEL_PROB       = 0.52   # below this model probability → HOLD regardless
DXY_CORR_THRESHOLD   = 0.0    # corr_20 must be non-zero for DXY to be useful
                               # (confirmation filter: both models must agree)

REGIME_RANGING      = 0
REGIME_TRENDING     = 1
REGIME_NEWS_DRIVEN  = 2

# ── Default model paths ────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "hmm":   _BASE / "data" / "models" / "hmm_regime_classifier.pkl",
    "range": _BASE / "data" / "models" / "range_model.pkl",
    "trend": _BASE / "data" / "models" / "trend_model.pkl",
    "news":  _BASE / "data" / "models" / "news_model.pkl",
    "dxy":   _BASE / "data" / "models" / "dxy_model.pkl",
}


# ══════════════════════════════════════════════════════════════════════════
#  Signal dataclass
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    signal:         str             # "BUY" | "SELL" | "HOLD"
    confidence:     float           # HMM regime confidence
    model_prob:     float           # routed model's winning class probability
    regime:         str             # "RANGING" | "TRENDING" | "NEWS_DRIVEN"
    regime_id:      int
    bars_in_regime: int             # consecutive bars in current regime
    confirmed:      bool            # whether regime is confirmed
    dxy_confirm:    Optional[bool]  # None unless RANGING
    timestamp:      str

    def to_dict(self) -> dict:
        return {
            "signal":         self.signal,
            "confidence":     round(self.confidence, 4),
            "model_prob":     round(self.model_prob, 4),
            "regime":         self.regime,
            "regime_id":      self.regime_id,
            "bars_in_regime": self.bars_in_regime,
            "confirmed":      self.confirmed,
            "dxy_confirm":    self.dxy_confirm,
            "timestamp":      self.timestamp,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Regime transition tracker
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _RegimeState:
    """Tracks consecutive bar count for regime confirmation."""
    confirmed_regime:   int  = REGIME_RANGING  # last confirmed regime
    candidate_regime:   int  = REGIME_RANGING  # regime being accumulated
    consecutive_count:  int  = 0

    def update(self, new_regime: int) -> bool:
        """
        Feed one bar's regime prediction.
        Returns True when a new regime becomes confirmed.
        """
        if new_regime == self.candidate_regime:
            self.consecutive_count += 1
        else:
            # Reset accumulator for the new candidate
            self.candidate_regime  = new_regime
            self.consecutive_count = 1

        if self.consecutive_count >= REGIME_CONFIRM_BARS:
            if self.candidate_regime != self.confirmed_regime:
                logger.info(
                    f"Regime transition confirmed: "
                    f"{REGIME_LABELS[self.confirmed_regime]} → "
                    f"{REGIME_LABELS[self.candidate_regime]} "
                    f"(after {self.consecutive_count} bars)"
                )
                self.confirmed_regime = self.candidate_regime
                return True  # transition just confirmed
        return False


# ══════════════════════════════════════════════════════════════════════════
#  Meta-Controller
# ══════════════════════════════════════════════════════════════════════════

class MetaController:

    def __init__(
        self,
        hmm_path:   str = None,
        range_path: str = None,
        trend_path: str = None,
        news_path:  str = None,
        dxy_path:   str = None,
    ):
        self._hmm:   HMMRegimeClassifier = None
        self._range: RangeModel          = None
        self._trend: TrendModel          = None
        self._news:  NewsModel           = None
        self._dxy:   DXYModel            = None

        self._paths = {
            "hmm":   hmm_path   or str(DEFAULT_MODELS["hmm"]),
            "range": range_path or str(DEFAULT_MODELS["range"]),
            "trend": trend_path or str(DEFAULT_MODELS["trend"]),
            "news":  news_path  or str(DEFAULT_MODELS["news"]),
            "dxy":   dxy_path   or str(DEFAULT_MODELS["dxy"]),
        }

        self._state       = _RegimeState()
        self.is_loaded    = False

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self) -> "MetaController":
        """Load all five models from disk. Must be called before predict()."""
        logger.info("Loading all models...")

        self._hmm   = HMMRegimeClassifier.load(self._paths["hmm"])
        self._range = RangeModel.load(self._paths["range"])
        self._trend = TrendModel.load(self._paths["trend"])
        self._news  = NewsModel.load(self._paths["news"])
        self._dxy   = DXYModel.load(self._paths["dxy"])

        self.is_loaded = True
        logger.success("All models loaded. MetaController ready.")
        return self

    # ── Main entry point ───────────────────────────────────────────────────

    def predict(
        self,
        bars_df:    pd.DataFrame,
        dxy_df:     Optional[pd.DataFrame] = None,
        news_df:    Optional[pd.DataFrame] = None,
    ) -> Signal:
        """
        Generate a trading signal for the latest bar.

        Args:
            bars_df:  Price DataFrame with columns:
                      timestamp, open, high, low, close, volume.
                      Must contain enough history for feature windows
                      (≥ 200 bars recommended).
            dxy_df:   DXY price DataFrame. Required for RANGING regime
                      DXY confirmation. If None, DXY confirmation is skipped.
            news_df:  News events DataFrame (optional, improves HMM spikes).

        Returns:
            Signal dataclass with full routing metadata.
        """
        self._check_loaded()

        # ── Step 1: Engineer HMM features ─────────────────────────────────
        featured = engineer_features(bars_df, news_df=news_df)
        if len(featured) == 0:
            logger.warning("No bars after feature engineering — returning HOLD")
            return self._hold_signal(
                "RANGING", 0, 0.0, 0, ""
            )

        # ── Step 2: HMM — predict regime + confidence ──────────────────────
        raw_regime  = int(self._hmm.predict(featured)[-1])
        confidence  = float(self._hmm.confidence(featured)[-1])
        proba_all   = self._hmm.predict_proba(featured)[-1]  # shape (3,)

        # Inject confidence into featured df for DXY model downstream
        featured["confidence"] = self._hmm.confidence(featured)

        timestamp = str(featured["timestamp"].iloc[-1])

        # ── Step 3: Regime transition tracking ────────────────────────────
        self._state.update(raw_regime)
        confirmed_regime  = self._state.confirmed_regime
        bars_in_regime    = self._state.consecutive_count
        is_confirmed      = bars_in_regime >= REGIME_CONFIRM_BARS

        regime_label = REGIME_LABELS[confirmed_regime]

        logger.info(
            f"[{timestamp}] raw_regime={REGIME_LABELS[raw_regime]} "
            f"confirmed={regime_label} "
            f"bars_streak={bars_in_regime} "
            f"hmm_conf={confidence:.3f}"
        )

        # ── Step 4: Low-confidence gate ────────────────────────────────────
        if confidence < MIN_CONFIDENCE or not is_confirmed:
            reason = (
                f"low HMM confidence ({confidence:.3f} < {MIN_CONFIDENCE})"
                if confidence < MIN_CONFIDENCE
                else f"regime not yet confirmed ({bars_in_regime}/{REGIME_CONFIRM_BARS} bars)"
            )
            logger.info(f"HOLD — {reason}")
            return self._hold_signal(
                regime_label, confirmed_regime, confidence, bars_in_regime, timestamp
            )

        # ── Step 5: Merge HMM features back onto original bars ─────────────
        # Sub-models need both raw OHLCV columns and HMM-derived columns.
        bars_with_hmm = self._merge_hmm_features(bars_df, featured)

        # ── Step 6: Route to regime model ─────────────────────────────────
        if confirmed_regime == REGIME_TRENDING:
            return self._route_trend(
                bars_with_hmm, confidence, bars_in_regime, timestamp
            )

        elif confirmed_regime == REGIME_NEWS_DRIVEN:
            return self._route_news(
                bars_with_hmm, confidence, bars_in_regime, timestamp
            )

        else:  # RANGING
            return self._route_ranging(
                bars_with_hmm, dxy_df, confidence, bars_in_regime, timestamp
            )

    # ── Routing methods ────────────────────────────────────────────────────

    def _route_trend(
        self,
        df: pd.DataFrame,
        confidence: float,
        bars_in_regime: int,
        timestamp: str,
    ) -> Signal:
        proba      = self._trend.predict_proba(df)[-1]  # [prob_down, prob_up]
        prob_up    = float(proba[1])
        model_prob = max(prob_up, 1 - prob_up)

        signal = self._prob_to_signal(prob_up, model_prob)

        logger.info(
            f"TRENDING → {signal} | prob_up={prob_up:.3f} | "
            f"model_prob={model_prob:.3f} | hmm_conf={confidence:.3f}"
        )

        return Signal(
            signal         = signal,
            confidence     = confidence,
            model_prob     = model_prob,
            regime         = "TRENDING",
            regime_id      = REGIME_TRENDING,
            bars_in_regime = bars_in_regime,
            confirmed      = True,
            dxy_confirm    = None,
            timestamp      = timestamp,
        )

    def _route_news(
        self,
        df: pd.DataFrame,
        confidence: float,
        bars_in_regime: int,
        timestamp: str,
    ) -> Signal:
        proba      = self._news.predict_proba(df)[-1]
        prob_up    = float(proba[1])
        model_prob = max(prob_up, 1 - prob_up)

        signal = self._prob_to_signal(prob_up, model_prob)

        logger.info(
            f"NEWS_DRIVEN → {signal} | prob_up={prob_up:.3f} | "
            f"model_prob={model_prob:.3f} | hmm_conf={confidence:.3f}"
        )

        return Signal(
            signal         = signal,
            confidence     = confidence,
            model_prob     = model_prob,
            regime         = "NEWS_DRIVEN",
            regime_id      = REGIME_NEWS_DRIVEN,
            bars_in_regime = bars_in_regime,
            confirmed      = True,
            dxy_confirm    = None,
            timestamp      = timestamp,
        )

    def _route_ranging(
        self,
        df: pd.DataFrame,
        dxy_df: Optional[pd.DataFrame],
        confidence: float,
        bars_in_regime: int,
        timestamp: str,
    ) -> Signal:
        # Primary: RangeModel
        range_proba = self._range.predict_proba(df)[-1]
        prob_up     = float(range_proba[1])
        model_prob  = max(prob_up, 1 - prob_up)
        range_signal = self._prob_to_signal(prob_up, model_prob)

        # Confirmation: DXYModel (if dxy_df available)
        dxy_confirm = None
        if dxy_df is not None:
            try:
                dxy_proba    = self._dxy.predict_proba(df, dxy_df)[-1]
                dxy_prob_up  = float(dxy_proba[1])
                dxy_signal   = self._prob_to_signal(
                    dxy_prob_up, max(dxy_prob_up, 1 - dxy_prob_up)
                )
                dxy_confirm  = (dxy_signal == range_signal)

                logger.info(
                    f"RANGING | RangeModel={range_signal} (p={prob_up:.3f}) | "
                    f"DXYModel={dxy_signal} (p={dxy_prob_up:.3f}) | "
                    f"agree={dxy_confirm}"
                )

                # Confirmation filter: HOLD if models disagree
                if not dxy_confirm and range_signal != "HOLD":
                    logger.info(
                        "HOLD — RangeModel and DXYModel disagree "
                        f"({range_signal} vs {dxy_signal})"
                    )
                    return Signal(
                        signal         = "HOLD",
                        confidence     = confidence,
                        model_prob     = model_prob,
                        regime         = "RANGING",
                        regime_id      = REGIME_RANGING,
                        bars_in_regime = bars_in_regime,
                        confirmed      = True,
                        dxy_confirm    = False,
                        timestamp      = timestamp,
                    )

            except Exception as e:
                logger.warning(f"DXYModel prediction failed: {e}. Proceeding without DXY confirmation.")
                dxy_confirm = None

        else:
            logger.info("No dxy_df supplied — skipping DXY confirmation filter.")

        logger.info(
            f"RANGING → {range_signal} | prob_up={prob_up:.3f} | "
            f"model_prob={model_prob:.3f} | dxy_confirm={dxy_confirm} | "
            f"hmm_conf={confidence:.3f}"
        )

        return Signal(
            signal         = range_signal,
            confidence     = confidence,
            model_prob     = model_prob,
            regime         = "RANGING",
            regime_id      = REGIME_RANGING,
            bars_in_regime = bars_in_regime,
            confirmed      = True,
            dxy_confirm    = dxy_confirm,
            timestamp      = timestamp,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _prob_to_signal(self, prob_up: float, model_prob: float) -> str:
        """Convert probability to BUY/SELL/HOLD using MIN_MODEL_PROB gate."""
        if model_prob < MIN_MODEL_PROB:
            return "HOLD"
        return "BUY" if prob_up >= MIN_MODEL_PROB else "SELL"

    def _hold_signal(
        self,
        regime_label: str,
        regime_id: int,
        confidence: float,
        bars_in_regime: int,
        timestamp: str = "",
    ) -> Signal:
        return Signal(
            signal         = "HOLD",
            confidence     = confidence,
            model_prob     = 0.0,
            regime         = regime_label,
            regime_id      = regime_id,
            bars_in_regime = bars_in_regime,
            confirmed      = bars_in_regime >= REGIME_CONFIRM_BARS,
            dxy_confirm    = None,
            timestamp      = timestamp,
        )

    def _merge_hmm_features(
        self,
        bars_df: pd.DataFrame,
        featured: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge HMM-derived columns back onto the original bars DataFrame
        so sub-models have access to both OHLCV and HMM features.
        Aligns on timestamp.
        """
        hmm_cols = [
            "timestamp", "volatility", "vol_regime", "volatility_spike",
            "atr_ratio", "trend_strength", "hl_range_norm", "confidence",
        ]
        available = [c for c in hmm_cols if c in featured.columns]
        hmm_slice = featured[available].copy()

        merged = pd.merge_asof(
            bars_df.sort_values("timestamp"),
            hmm_slice.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )
        return merged.reset_index(drop=True)

    def _check_loaded(self):
        if not self.is_loaded:
            raise RuntimeError("Call .load() before .predict().")