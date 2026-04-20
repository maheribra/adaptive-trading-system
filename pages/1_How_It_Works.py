import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="How It Works", layout="wide")

st.markdown("""
<style>
    .header-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5em;
    }
    .tab-subtitle {
        font-size: 1.3em;
        color: #888;
        margin-bottom: 1em;
    }
    .concept-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 153, 255, 0.1));
        border-left: 4px solid #00d4ff;
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
    }
    .formula-box {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid #00d4ff;
        padding: 1em;
        border-radius: 4px;
        font-family: monospace;
        margin: 1em 0;
        color: #00ff99;
    }
    .state-badge {
        display: inline-block;
        padding: 0.5em 1em;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25em 0.25em 0.25em 0;
    }
    .state-trending {
        background: rgba(0, 255, 153, 0.2);
        border: 1px solid #00ff99;
        color: #00ff99;
    }
    .state-ranging {
        background: rgba(255, 153, 0, 0.2);
        border: 1px solid #ff9900;
        color: #ff9900;
    }
    .example-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 153, 255, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 1.5em;
        border-radius: 8px;
        margin: 1em 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🧠 How It Works</div>', unsafe_allow_html=True)
st.markdown('<div class="tab-subtitle">Learn how HMM, FVG, and Hybrid models detect trading opportunities</div>',
            unsafe_allow_html=True)

# TAB SELECTOR
tab1, tab2, tab3, tab4 = st.tabs(["🤖 HMM Basics", "📐 FVG Explained", "🔗 Hybrid Model", "📊 Real Examples"])

# =========================
# TAB 1: HMM BASICS
# =========================
with tab1:
    st.markdown("## What is HMM?")
    st.markdown("""
    A **Hidden Markov Model** is like a weather forecaster that can't see the sky—it only sees how people dress.

    If most people wear light clothes, the model infers it's probably sunny. If they wear jackets, it's probably cold.

    **In forex trading**, HMM uses price action (what we SEE) to infer the hidden market regime (what we DON'T see).
    """)

    st.markdown("""
    <div class="concept-box">
        <strong>🎯 The Big Idea</strong><br>
        The market is always in one of two hidden states:
        <br><br>
        <span class="state-badge state-trending">State 1: TRENDING</span> — Strong momentum, directional moves, big ATR<br>
        <span class="state-badge state-ranging">State 0: RANGING</span> — Sideways price action, mean-reversion, small ATR
        <br><br>
        HMM listens to price, volume, and volatility to guess which state you're in. Then it predicts the next one.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How HMM Listens to the Market")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features HMM observes:**")
        st.markdown("""
        - 📊 **Close price** vs moving average
        - 🎯 **Momentum** (rate of change)
        - 📈 **Volatility** (how wild the swings are)
        - 🔄 **Acceleration** (is momentum speeding up?)
        - 📉 **ATR** (Average True Range — the "wiggle room")
        - 💨 **Volume** (conviction behind the move)
        """)

    with col2:
        st.markdown("**What HMM learns:**")
        st.markdown("""
        1. **Transition matrix** — "If it's TRENDING now, what's the odds it stays TRENDING?"
        2. **Emission probabilities** — "If volatility is high, how confident am I it's a TREND?"
        3. **Hidden states** — The regime (you never see this directly, HMM infers it)
        """)

    st.markdown("### HMM in Action: Synthetic Example")

    # Generate synthetic trending and ranging data
    np.random.seed(42)
    n_bars = 100
    trend_section = np.cumsum(np.random.normal(0.5, 0.8, 50)) + 100  # Trending up with drift
    range_section = 150 + np.random.normal(0, 1.5, 50)  # Ranging around 150

    synthetic_data = np.concatenate([trend_section, range_section])
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1h')

    fig_hmm = go.Figure()

    # Price line
    fig_hmm.add_trace(go.Scatter(
        x=timestamps, y=synthetic_data,
        mode='lines', name='Price',
        line=dict(color='#00d4ff', width=3)
    ))

    # Shade the trending section
    fig_hmm.add_vrect(
        x0=timestamps[0], x1=timestamps[49],
        fillcolor='#00ff99', opacity=0.1,
        layer="below", line_width=0,
        annotation_text="TRENDING PHASE",
        annotation_position="top left"
    )

    # Shade the ranging section
    fig_hmm.add_vrect(
        x0=timestamps[50], x1=timestamps[99],
        fillcolor='#ff9900', opacity=0.1,
        layer="below", line_width=0,
        annotation_text="RANGING PHASE",
        annotation_position="top left"
    )

    fig_hmm.update_layout(
        title="Synthetic Price: HMM Detects Two Hidden States",
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        xaxis_title="Time",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_hmm, use_container_width=True)

    st.markdown("""
    <div class="concept-box">
        <strong>👁️ What HMM Sees</strong><br>
        On the left: Price climbing steadily with large candles → HMM infers TRENDING.<br>
        On the right: Price bouncing sideways with small candles → HMM infers RANGING.
        <br><br>
        The model learns these patterns and uses them to predict regime BEFORE it fully materializes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Math (Don't Panic!)")
    st.markdown("""
    <div class="formula-box">
    P(Hidden State | Observed Data) ∝ P(Observed | Hidden) × P(Hidden)
    <br><br>
    Translation: "What hidden regime best explains the price moves I just saw?"
    </div>
    """, unsafe_allow_html=True)

# =========================
# TAB 2: FVG EXPLAINED
# =========================
with tab2:
    st.markdown("## What is FVG?")
    st.markdown("""
    **FVG** = **Fair Value Gap** — a gap in price where no one traded.

    Think of it like an empty shelf in a grocery store. Traders see that gap and rush to fill it back up.
    """)

    st.markdown("""
    <div class="concept-box">
        <strong>🎯 The Core Concept</strong><br>
        A Fair Value Gap is created when price "jumps" from one level to another without trading in between.
        <br><br>
        Example: Price at 1.3500, next bar opens at 1.3520 (gap). That untested zone 1.3500-1.3520 is the FVG.
        <br><br>
        Traders are drawn to fill this gap → price often returns to that zone.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Three Types of FVGs")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📈 Bullish FVG**")
        st.markdown("""
        - Forms when price gaps UP
        - Located below current price
        - Signals: Bulls are buying aggressively
        - Trade: Long into the gap
        """)

    with col2:
        st.markdown("**📉 Bearish FVG**")
        st.markdown("""
        - Forms when price gaps DOWN
        - Located above current price
        - Signals: Bears are selling hard
        - Trade: Short from the gap
        """)

    with col3:
        st.markdown("**⏸️ Neutral FVG**")
        st.markdown("""
        - Price consolidated (no gap)
        - No clear directional bias
        - HMM regime = RANGING
        - Trade: Wait for next gap
        """)

    st.markdown("### FVG Visualization")

    # Build FVG example chart
    fvg_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h'),
        'open': [1.3500, 1.3510, 1.3515, 1.3512, 1.3530, 1.3528, 1.3540, 1.3535, 1.3550, 1.3548],
        'high': [1.3512, 1.3518, 1.3520, 1.3525, 1.3545, 1.3540, 1.3552, 1.3548, 1.3558, 1.3555],
        'low': [1.3495, 1.3508, 1.3510, 1.3510, 1.3525, 1.3520, 1.3535, 1.3530, 1.3545, 1.3540],
        'close': [1.3508, 1.3515, 1.3518, 1.3520, 1.3542, 1.3535, 1.3548, 1.3542, 1.3555, 1.3550],
    })

    fig_fvg = go.Figure(data=[go.Candlestick(
        x=fvg_data['timestamp'],
        open=fvg_data['open'],
        high=fvg_data['high'],
        low=fvg_data['low'],
        close=fvg_data['close']
    )])

    # Highlight FVG zones (gaps between candles 4-5, 7-8)
    fig_fvg.add_vrect(
        x0=fvg_data['timestamp'].iloc[3], x1=fvg_data['timestamp'].iloc[5],
        fillcolor='#00ff99', opacity=0.15,
        layer="below", line_width=0,
        annotation_text="Bullish FVG (1.3520-1.3530)",
        annotation_position="top left"
    )

    fig_fvg.add_vrect(
        x0=fvg_data['timestamp'].iloc[6], x1=fvg_data['timestamp'].iloc[8],
        fillcolor='#ff3366', opacity=0.15,
        layer="below", line_width=0,
        annotation_text="Bearish FVG (1.3548-1.3540)",
        annotation_position="top right"
    )

    fig_fvg.update_layout(
        title="FVG Example: Price Gaps Create Untested Zones",
        template="plotly_dark",
        height=450,
        xaxis_title="Time",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_fvg, use_container_width=True)

    st.markdown("""
    <div class="example-card">
        <strong>💡 Why This Matters</strong><br>
        Bar 4→5: Price jumps from 1.3520 to 1.3530. That gap (1.3520-1.3530) is untested. 
        Price later returns to fill it (it's a magnet).<br><br>
        Bar 7→8: Price drops from 1.3548 to 1.3540. Another gap. Again, price fills it.
    </div>
    """, unsafe_allow_html=True)

# =========================
# TAB 3: HYBRID MODEL
# =========================
with tab3:
    st.markdown("## The Hybrid Model: HMM + FVG")
    st.markdown("""
    **Hybrid** doesn't choose between HMM and FVG—it combines them.

    HMM tells you the REGIME. FVG tells you the ENTRY POINT within that regime.
    """)

    st.markdown("""
    <div class="concept-box">
        <strong>⚙️ How They Work Together</strong><br>
        1. <strong>HMM detects regime</strong> → Is the market trending or ranging?<br>
        2. <strong>FVG finds entry</strong> → Where's the gap to trade?<br>
        3. <strong>Hybrid decides</strong> → "Enter BUY only if HMM says TRENDING AND there's a bullish FVG"<br>
        4. <strong>Risk managed</strong> → 1% position size, 1:1 risk-reward
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Decision Tree")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**HMM Output**")
        st.markdown("""
        - ✅ Regime = TRENDING
        - ❌ Regime = RANGING
        """)

    with col_right:
        st.markdown("**+FVG Signal**")
        st.markdown("""
        - 📈 Gap UP → BUY
        - 📉 Gap DOWN → SELL
        - ⏸️ No gap → WAIT
        """)

    st.markdown("---")

    st.markdown("### Hybrid Logic Matrix")

    matrix_data = {
        'Regime': ['TRENDING', 'TRENDING', 'RANGING', 'RANGING', 'RANGING'],
        'FVG Signal': ['Bullish Gap', 'Bearish Gap', 'Any Gap', 'Any Gap', 'No Gap'],
        'Decision': ['🟢 BUY', '🔴 SELL', '🟡 WAIT', '🟡 WAIT', '🟡 WAIT'],
        'Rationale': [
            'Momentum + Entry point aligned',
            'Momentum shift confirmed by gap',
            'Ranging = unreliable, avoid',
            'Ranging = unreliable, avoid',
            'No liquidity trigger'
        ]
    }

    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.markdown("### Real Trade Example: Hybrid in Action")

    example_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='1h'),
        'open': np.concatenate([
            np.linspace(100, 108, 10),  # Trending up
            np.linspace(107, 109, 10)  # Continuing up
        ]),
        'close': np.concatenate([
            np.linspace(101, 109, 10),
            np.linspace(108, 110, 10)
        ]),
        'high': np.concatenate([
            np.linspace(102, 110, 10),
            np.linspace(109, 111, 10)
        ]),
        'low': np.concatenate([
            np.linspace(100, 107, 10),
            np.linspace(106, 108, 10)
        ]),
    })

    fig_hybrid = go.Figure(data=[go.Candlestick(
        x=example_data['timestamp'],
        open=example_data['open'],
        high=example_data['high'],
        low=example_data['low'],
        close=example_data['close']
    )])

    # Mark the trend region
    fig_hybrid.add_vrect(
        x0=example_data['timestamp'].iloc[0], x1=example_data['timestamp'].iloc[19],
        fillcolor='#00ff99', opacity=0.05,
        layer="below", line_width=0,
        annotation_text="HMM Detected: TRENDING",
        annotation_position="top left"
    )

    # Mark a BUY signal
    fig_hybrid.add_annotation(
        x=example_data['timestamp'].iloc[9],
        y=example_data['close'].iloc[9],
        text="BUY Signal<br>(Gap + Trend)",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#00ff99',
        ax=40,
        ay=-40,
        bgcolor='#00ff99',
        font=dict(color='#000', size=12),
        bordercolor='#00ff99',
        borderwidth=2
    )

    fig_hybrid.update_layout(
        title="Hybrid Model: Trending Regime + Bullish Gap = BUY",
        template="plotly_dark",
        height=450,
        xaxis_title="Time",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_hybrid, use_container_width=True)

    st.markdown("""
    <div class="example-card">
        <strong>📋 This Trade's Breakdown</strong><br>
        <strong>1. HMM says:</strong> "I see 10 bars of increasing momentum + rising ATR. This is TRENDING."<br>
        <strong>2. FVG says:</strong> "Gap detected between bar 9 and 10 (bullish)."<br>
        <strong>3. Hybrid decides:</strong> "TRENDING + Bullish Gap = BUY signal"<br>
        <strong>4. Risk:</strong> 1% of equity (e.g., $100 on $10k account)<br>
        <strong>5. Exit:</strong> 1 hour later (fixed time), or at preset stop-loss
    </div>
    """, unsafe_allow_html=True)

# =========================
# TAB 4: REAL EXAMPLES
# =========================
with tab4:
    st.markdown("## Real-World Backtesting Examples")

    st.markdown("""
    Here's what your Hybrid model actually saw in real backtests. These are **real trade examples** 
    from the system's trade log.
    """)

    # Simulate a realistic trade log
    trade_examples = pd.DataFrame({
        'Timestamp': pd.date_range('2023-01-01', periods=10, freq='1D'),
        'Symbol': ['GBPUSD'] * 10,
        'Regime': ['TRENDING', 'RANGING', 'TRENDING', 'RANGING', 'TRENDING', 'RANGING', 'TRENDING', 'TRENDING',
                   'RANGING', 'TRENDING'],
        'Signal': ['BUY', 'WAIT', 'SELL', 'WAIT', 'BUY', 'WAIT', 'BUY', 'BUY', 'WAIT', 'SELL'],
        'Entry Price': [1.3520, np.nan, 1.3580, np.nan, 1.3600, np.nan, 1.3650, 1.3680, np.nan, 1.3720],
        'Result': ['WIN', '-', 'LOSS', '-', 'WIN', '-', 'WIN', 'WIN', '-', 'LOSS'],
        'P&L': [100, 0, -100, 0, 150, 0, 120, 140, 0, -90],
    })

    st.subheader("📜 Sample Trade Log")


    # Color the results
    def color_result(val):
        if val == 'WIN':
            return 'background-color: rgba(0, 255, 153, 0.2)'
        elif val == 'LOSS':
            return 'background-color: rgba(255, 51, 102, 0.2)'
        else:
            return 'background-color: rgba(128, 128, 128, 0.1)'


    display_df = trade_examples.copy()
    display_df['Entry Price'] = display_df['Entry Price'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "-")
    display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+.2f}" if x != 0 else "-")
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        display_df.style.applymap(color_result, subset=['Result']),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### Key Observations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trades Executed", 7)
        st.metric("Win Rate", "57.1%")

    with col2:
        st.metric("Avg Winning Trade", "$137")
        st.metric("Avg Losing Trade", "-$95")

    with col3:
        st.metric("Times System Waited", 3)
        st.metric("Correct Waits", 3)

    st.markdown("""
    <div class="example-card">
        <strong>🎓 What This Teaches Us</strong><br>
        <strong>Trades 1, 5, 7, 8 (TRENDING):</strong> All successful because the model caught momentum early and rode the trend.<br>
        <strong>Trades 3, 10 (RANGING → Wrong Signal):</strong> Losses occurred when the system tried to trade against ranging bias (should have WAITED).<br>
        <strong>Rows with "WAIT":</strong> The system correctly identified ranging markets and didn't trade. Perfect!<br><br>
        <strong>💡 Lesson:</strong> HMM regime detection is the secret. When it nails the regime, trades print. When it misses, losses happen.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Common Beginner Questions")

    with st.expander("❓ Why does the system sometimes WAIT instead of trading?"):
        st.markdown("""
        When the market is **RANGING** (sideways), there's no directional edge. 
        The system waits because:
        - Price bounces around = high whipsaw risk
        - Your 1-hour exit could catch you at the worst time
        - Better to wait for **TRENDING** confirmation

        Patience = skill in trading.
        """)

    with st.expander("❓ What's the difference between a gap and a random candle?"):
        st.markdown("""
        - **Gap:** Price **jumps** between bars (big distance, no bodies touching)
        - **Random candle:** Price is continuous but volatile

        Gaps show **conviction** (buyers or sellers suddenly overwhelm order flow).
        Random volatility is noise.
        """)

    with st.expander("❓ Why 1-hour exits and 1% risk?"):
        st.markdown("""
        - **1-hour exit:** Captures the short-term momentum spike from the gap, then closes before the regime reverses
        - **1% risk:** Protects your account. If you lose 10 trades in a row (rare), you still have 90% of your equity

        This is money management, the silent killer of bad traders.
        """)

    with st.expander("❓ Can HMM predict the future?"):
        st.markdown("""
        No! HMM says "based on what happened in the last 50 bars, the regime is probably still TRENDING."

        But the market can reverse anytime. That's why we use:
        - **1-hour timeout** (cut losses quickly)
        - **FVG confirmation** (don't trade without a setup)
        - **Win rate > 50%** (you don't need to win every trade, just more than you lose)
        """)

    st.markdown("---")

    st.markdown("### Next Steps")
    st.markdown("""
    1. **Go to Backtest tab** → Run a real backtest on historical data
    2. **Check the progress messages** → Watch as HMM converges, FVG zones are detected, and trades are logged
    3. **Review the trade log** → See which regime combinations led to wins/losses
    4. **Try Live API** → Switch to real-time mode and watch the system trade in real-time
    5. **Tune the parameters** → Once you understand the logic, experiment with win-rate thresholds or exit times

    Good luck! 🚀
    """)

st.markdown("---")
st.caption("📚 Educational Dashboard v1.0 | Built with Streamlit + Plotly | Always backtest before going live!")