import pandas as pd
import numpy as np


class FVGModel:
    def __init__(self):
        # Loosened thresholds
        # min_gap_atr_multiple: The gap must be at least 15% of the average candle size
        self.min_gap_atr_multiple = 0.15

    def get_fvg_details(df):
        """Returns (Type, Top, Bottom, SL_Level) of the most recent FVG."""
        if len(df) < 3: return None

        c0_high, c0_low = df['high'].iloc[-3], df['low'].iloc[-3]
        c2_high, c2_low = df['high'].iloc[-1], df['low'].iloc[-1]

        # Bearish FVG
        if c0_low > c2_high:
            return {
                "type": "SELL",
                "top": c0_low,  # The 'Box' top
                "bottom": c2_high,  # The 'Box' bottom
                "sl": c0_high  # Your rule: SL at the 1st candle high
            }
        return None

    def predict(self, bars_df: pd.DataFrame):
        """
        Analyzes the last 3 candles for a Fair Value Gap.
        Returns a signal object compatible with the MetaController.
        """
        if len(bars_df) < 5:
            return type('Signal', (), {'signal': 'NEUTRAL', 'confidence': 0})

        # 1. Calculate ATR (to keep sensitivity relative to volatility)
        # Standard 14-period range
        recent_range = (bars_df['high'] - bars_df['low']).tail(14)
        atr = recent_range.mean()
        min_gap = atr * self.min_gap_atr_multiple

        # 2. Get the 3-candle sequence (t-2, t-1, t)
        # Candle 0 (Oldest), Candle 1 (Middle), Candle 2 (Current/Newest)
        c0_high, c0_low = bars_df['high'].iloc[-3], bars_df['low'].iloc[-3]
        c2_high, c2_low = bars_df['high'].iloc[-1], bars_df['low'].iloc[-1]

        signal = "NEUTRAL"
        confidence = 0.0

        # 3. Detection Logic
        # Bullish FVG: Low of current candle is higher than High of 2 candles ago
        bullish_gap = c2_low - c0_high

        # Bearish FVG: High of current candle is lower than Low of 2 candles ago
        bearish_gap = c0_low - c2_high

        if bullish_gap > min_gap:
            signal = "BUY"
            confidence = min(1.0, bullish_gap / atr)  # Confidence scales with gap size

        elif bearish_gap > min_gap:
            signal = "SELL"
            confidence = min(1.0, bearish_gap / atr)

        # Return structured object for MetaController
        return type('Signal', (), {'signal': signal, 'confidence': confidence})