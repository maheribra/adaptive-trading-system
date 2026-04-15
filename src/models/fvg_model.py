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

    def predict(self, bars_df: pd.DataFrame, context_df: pd.DataFrame = None):
        """
        Analyzes the last 3 candles for a Fair Value Gap.
        If context_df (4h) is provided, it filters signals based on HTF bias.
        """
        if len(bars_df) < 5:
            return type('Signal', (), {'signal': 'NEUTRAL', 'confidence': 0})

        # 1. HTF Bias Detection (Only if context_df is provided)
        bias = "NEUTRAL"
        if context_df is not None and len(context_df) >= 2:
            # Check if the most recent 4h candle closed higher than the previous
            last_4h = context_df['close'].iloc[-1]
            prev_4h = context_df['close'].iloc[-2]
            bias = "BULLISH" if last_4h > prev_4h else "BEARISH"

        # 2. Standard ATR & Gap Logic (Same as your original)
        recent_range = (bars_df['high'] - bars_df['low']).tail(14)
        atr = recent_range.mean()
        min_gap = atr * self.min_gap_atr_multiple

        c0_high, c0_low = bars_df['high'].iloc[-3], bars_df['low'].iloc[-3]
        c2_high, c2_low = bars_df['high'].iloc[-1], bars_df['low'].iloc[-1]

        signal = "NEUTRAL"
        confidence = 0.0

        # 3. Detection Logic
        bullish_gap = c2_low - c0_high
        bearish_gap = c0_low - c2_high

        # 4. Filter Signals by Bias (If bias is active)
        if bullish_gap > min_gap:
            # If bias is BULLISH or if we have no context, allow the BUY
            if bias in ["BULLISH", "NEUTRAL"]:
                signal = "BUY"
                confidence = min(1.0, bullish_gap / atr)

        elif bearish_gap > min_gap:
            # If bias is BEARISH or if we have no context, allow the SELL
            if bias in ["BEARISH", "NEUTRAL"]:
                signal = "SELL"
                confidence = min(1.0, bearish_gap / atr)

        return type('Signal', (), {'signal': signal, 'confidence': confidence})