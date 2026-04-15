import pandas as pd
import numpy as np
import sys
import pickle
import __main__
from pathlib import Path
from loguru import logger
from typing import Optional, Dict
from dataclasses import dataclass
from src.models.fvg_model import FVGModel

# ── 1. Pathing Logic ──────────────────────────────────────────────────────
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# ── 2. Imports ─────────────────────────────────────────────────────────────
from src.models.hmm_regime_classifier import (
    HMMRegimeClassifier,
    engineer_features,
    REGIME_LABELS,
)

# ── 3. Constants ───────────────────────────────────────────────────────────
REGIME_CONFIRM_BARS = 1
MIN_CONFIDENCE = 0.45
MIN_MODEL_PROB = 0.51

REGIME_RANGING = 0
REGIME_TRENDING = 1
REGIME_NEWS_DRIVEN = 2

DEFAULT_MODELS = {
    "hmm": root / "data" / "models" / "hmm_regime_classifier.pkl",
    "range": root / "data" / "models" / "range_model.pkl",
    "trend": root / "data" / "models" / "trend_model.pkl",
    "news": root / "data" / "models" / "news_model.pkl",
    "dxy": root / "data" / "models" / "dxy_model.pkl",
}


@dataclass
class Signal:
    signal: str
    confidence: float
    model_prob: float
    regime: str
    regime_id: int
    bars_in_regime: int
    confirmed: bool
    dxy_confirm: Optional[bool]
    timestamp: str

    # ADD THIS METHOD BELOW:
    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "model_prob": round(self.model_prob, 4),
            "regime": self.regime,
            "regime_id": self.regime_id,
            "bars_in_regime": self.bars_in_regime,
            "confirmed": self.confirmed,
            "dxy_confirm": self.dxy_confirm,
            "timestamp": self.timestamp,
        }


# ── 4. Regime State Management ──────────────────────────────────────────────
class _RegimeState:
    def __init__(self):
        self.confirmed_regime = REGIME_RANGING
        self.candidate_regime = REGIME_RANGING
        self.consecutive_count = 0

    def update(self, new_regime: int) -> bool:
        if new_regime == self.candidate_regime:
            self.consecutive_count += 1
        else:
            self.candidate_regime = new_regime
            self.consecutive_count = 1

        if self.consecutive_count >= REGIME_CONFIRM_BARS:
            if self.candidate_regime != self.confirmed_regime:
                logger.info(f"Regime Transition: {REGIME_LABELS[self.candidate_regime]}")
                self.confirmed_regime = self.candidate_regime
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════
#  Meta-Controller
# ══════════════════════════════════════════════════════════════════════════

class MetaController:
    def __init__(self, paths: Dict[str, str] = None, mode: str = "Hybrid"):
        self._hmm = None
        self._range = None
        self._trend = None
        self._news = None
        self._dxy = None
        self._fvg = None
        self._paths = paths or {k: str(v) for k, v in DEFAULT_MODELS.items()}
        self._state = _RegimeState()
        self.mode = mode  # "HMM", "FVG", or "Hybrid"
        self.is_loaded = False

    def load(self):
        from src.models.range_model import RangeModel
        from src.models.trend_model import TrendModel
        from src.models.news_model import NewsModel
        from src.models.dxy_model import DXYModel
        from src.models.fvg_model import FVGModel

        for m in [RangeModel, TrendModel, NewsModel, DXYModel]:
            setattr(__main__, m.__name__, m)

        self._hmm = HMMRegimeClassifier.load(self._paths["hmm"])
        self._range = RangeModel.load(self._paths["range"])
        self._trend = TrendModel.load(self._paths["trend"])
        self._news = NewsModel.load(self._paths["news"])
        self._dxy = DXYModel.load(self._paths["dxy"])
        self._fvg = FVGModel()

        self.is_loaded = True
        logger.success(f"MetaController Online | MODE: {self.mode}")
        return self

    def _is_price_in_fvg(self, df: pd.DataFrame, context_df: pd.DataFrame = None) -> bool:
        if self._fvg is None:
            return False

        # Pass the context_df to the FVG model.
        # If it's None, the FVG model just does a standard single-TF check.
        result = self._fvg.predict(df, context_df=context_df)

        return result.signal != "NEUTRAL"

    def predict(
            self,
            bars_df: pd.DataFrame,
            dxy_df: pd.DataFrame = None,
            news_df: pd.DataFrame = None,
            context_df: pd.DataFrame = None  # <--- Added at the end with default None
    ) -> Signal:
        """
        Predicts signal based on regime.
        Old scripts calling mc.predict(df, dxy, news) will still work perfectly.
        """
        if not self.is_loaded:
            raise RuntimeError("MetaController not loaded.")

        # 1. Feature Engineering (This uses bars_df and news_df as usual)
        featured = engineer_features(bars_df, news_df=news_df)
        if len(featured) == 0:
            return self._hold_signal("RANGING", 0, 0.0, 0, "")

        raw_regime = int(self._hmm.predict(featured)[-1])
        confidence = float(self._hmm.confidence(featured)[-1])
        timestamp = str(featured["timestamp"].iloc[-1])

        self._state.update(raw_regime)
        conf_regime = self._state.confirmed_regime
        label = REGIME_LABELS[conf_regime]

        # 2. Hybrid Checklist
        in_fvg = self._is_price_in_fvg(bars_df, context_df=context_df)

        # 3. Decision Gate based on Mode
        if self.mode == "Hybrid":
            if not in_fvg:
                # If using Hybrid Mode and not in FVG, we HOLD regardless of HMM
                return self._hold_signal(f"{label} (Outside FVG)", conf_regime, confidence,
                                         self._state.consecutive_count, timestamp)

        if self.mode == "FVG" and not in_fvg:
            return self._hold_signal("NO FVG", conf_regime, confidence, self._state.consecutive_count, timestamp)

        # 4. Standard Routing
        df_input = self._merge_hmm_features(bars_df, featured)
        if conf_regime == REGIME_TRENDING:
            return self._route_trend(df_input, confidence, self._state.consecutive_count, timestamp)
        elif conf_regime == REGIME_NEWS_DRIVEN:
            return self._route_news(df_input, confidence, self._state.consecutive_count, timestamp)
        else:
            return self._route_ranging(df_input, dxy_df, confidence, self._state.consecutive_count, timestamp)

    # ── Specialized Routing with Debugging ──────────────────────────────────

    def _route_trend(self, df, confidence, streak, ts) -> Signal:
        proba = self._trend.predict_proba(df)[-1]
        prob_up = float(proba[1])
        model_prob = max(prob_up, 1 - prob_up)

        # DEBUG Output
        print(f"  DEBUG [Trend]: SELL {1 - prob_up:.2f} | BUY {prob_up:.2f} | Threshold: {MIN_MODEL_PROB}")

        signal = "HOLD"
        if model_prob >= MIN_MODEL_PROB:
            signal = "BUY" if prob_up > 0.5 else "SELL"
        return Signal(signal, confidence, model_prob, "TRENDING", REGIME_TRENDING, streak, True, None, ts)

    def _route_ranging(self, df, dxy_df, confidence, streak, ts) -> Signal:
        proba = self._range.predict_proba(df)[-1]
        prob_up = float(proba[1])
        model_prob = max(prob_up, 1 - prob_up)
        range_sig = "BUY" if prob_up > 0.5 else "SELL"

        # DEBUG Output
        print(f"  DEBUG [Range]: SELL {1 - prob_up:.2f} | BUY {prob_up:.2f} | Threshold: {MIN_MODEL_PROB}")

        dxy_confirm = None
        if dxy_df is not None:
            try:
                d_proba = self._dxy.predict_proba(df, dxy_df)[-1]
                dxy_sig = "BUY" if d_proba[1] > 0.5 else "SELL"
                dxy_confirm = (range_sig == dxy_sig)
            except:
                dxy_confirm = None

        final_sig = "HOLD"
        if model_prob >= MIN_MODEL_PROB and dxy_confirm is not False:
            final_sig = range_sig
        return Signal(final_sig, confidence, model_prob, "RANGING", REGIME_RANGING, streak, True, dxy_confirm, ts)

    def _route_news(self, df, confidence, streak, ts) -> Signal:
        proba = self._news.predict_proba(df)[-1]
        prob_up = float(proba[1])
        model_prob = max(prob_up, 1 - prob_up)

        print(f"  DEBUG [News]: SELL {1 - prob_up:.2f} | BUY {prob_up:.2f} | Threshold: {MIN_MODEL_PROB}")

        signal = "HOLD"
        if model_prob >= MIN_MODEL_PROB:
            signal = "BUY" if prob_up > 0.5 else "SELL"
        return Signal(signal, confidence, model_prob, "NEWS_DRIVEN", REGIME_NEWS_DRIVEN, streak, True, None, ts)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _hold_signal(self, label, rid, conf, streak, ts) -> Signal:
        return Signal("HOLD", conf, 0.0, label, rid, streak, streak >= REGIME_CONFIRM_BARS, None, ts)

    def _merge_hmm_features(self, bars_df: pd.DataFrame, featured: pd.DataFrame) -> pd.DataFrame:
        # Ensure we don't lose the original OHLC data
        bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
        featured['timestamp'] = pd.to_datetime(featured['timestamp'])

        # We only want to bring in the NEW features from the HMM,
        # not overwrite the existing price columns.
        # Filter featured to only include timestamp + features NOT in bars_df
        new_cols = [c for c in featured.columns if c not in bars_df.columns or c == 'timestamp']
        featured_subset = featured[new_cols]

        merged = pd.merge_asof(
            bars_df.sort_values("timestamp"),
            featured_subset.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1h")
        )

        # FINAL CHECK: If 'high' is missing here, the Trend Model will crash
        if 'high' not in merged.columns:
            logger.error(f"Column 'high' lost during merge! Current columns: {merged.columns}")

        return merged.reset_index(drop=True)


if __name__ == "__main__":
    try:
        mc = MetaController().load()
        logger.success("MetaController self-test passed.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")