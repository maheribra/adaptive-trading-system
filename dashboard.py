import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from twelvedata import TDClient
import __main__

# --- 1. MODEL SNAPSHOT INJECTION ---
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel

for m in [RangeModel, TrendModel, NewsModel, DXYModel]:
    setattr(__main__, m.__name__, m)

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.meta_controller import MetaController

# --- 2. CONFIG & THE "BOMB-PROOF" CLEANER ---
API_KEY = "7665d57e16444c73b94296b956922044"


def standardize_df(df):
    """The master cleaner: Handles Index-based time and column naming."""
    if df is None or df.empty: return None

    # 1. If 'datetime' is the index (Twelve Data default), move it to a column
    df = df.reset_index()

    # 2. Standardize all headers to lowercase
    df.columns = [str(col).lower().strip() for col in df.columns]

    # 3. Force-map time variations to 'timestamp'
    time_map = {'datetime': 'timestamp', 'date': 'timestamp', 'time': 'timestamp'}
    for old, new in time_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
            break

    # 4. Critical fix for RangeModel/HMM
    if 'timestamp' not in df.columns:
        # Fallback if somehow reset_index didn't give us a time column
        st.error("Data Error: Could not find time column.")
        return None

    if 'volume' not in df.columns: df['volume'] = 0

    # 5. Ensure numeric types
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=60)
def fetch_api_data(symbol, interval="1h"):
    """Robust fetch for Live Market."""
    try:
        api_sym = f"{symbol[:3]}/{symbol[3:]}" if len(symbol) == 6 and "/" not in symbol else symbol
        td = TDClient(apikey=API_KEY)
        # 300 outputsize provides enough lookback for HMM feature engineering
        ts = td.time_series(symbol=api_sym, interval=interval, outputsize=300, order="ASC")
        raw_df = ts.as_pandas()
        return standardize_df(raw_df)
    except Exception as e:
        st.sidebar.error(f"API Error ({interval}): {e}")
        return None


# --- 3. UI SETUP ---
st.set_page_config(page_title="Adaptive Suite Pro", layout="wide")
st.sidebar.title("🎮 System Control")

data_source = st.sidebar.selectbox("Data Source", ["Live API", "Backtest (CSV)"])
mode_selection = st.sidebar.radio("Analysis Mode", ["HMM", "FVG", "Hybrid"])
symbol_input = st.sidebar.text_input("Symbol", "GBPUSD").upper().replace('/', '')

if data_source == "Backtest (CSV)":
    col_s, col_e = st.sidebar.columns(2)
    start_dt = col_s.date_input("Start", value=pd.to_datetime("2023-01-01"))
    end_dt = col_e.date_input("End", value=pd.to_datetime("2023-01-05"))
    run_btn = st.sidebar.button("📊 Run Optimized Backtest")
else:
    run_btn = st.sidebar.button("🚀 Analyze Live Market")

# --- 4. EXECUTION ---
if run_btn:
    mc = MetaController(mode=mode_selection).load()

    if data_source == "Live API":
        with st.spinner("🔄 Fetching MTF Data..."):
            df_main = fetch_api_data(symbol_input, "1h")
            df_htf = fetch_api_data(symbol_input, "4h")

            if df_main is not None and not df_main.empty:
                # MetaController prediction
                sig = mc.predict(bars_df=df_main, context_df=df_htf)
                res = sig.to_dict()

                # UI Layout
                st.title(f"📡 {symbol_input} Live Intelligence")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"{df_main['close'].iloc[-1]:.4f}")
                m2.metric("Regime", res['regime'])
                m3.metric("Signal", res['signal'])
                m4.metric("Last Update", df_main['timestamp'].iloc[-1].strftime('%H:%M'))

                fig = go.Figure(data=[go.Candlestick(
                    x=df_main['timestamp'], open=df_main['open'],
                    high=df_main['high'], low=df_main['low'], close=df_main['close']
                )])
                fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Critical: Live data fetch returned empty. Check Symbol.")

    else:
        # --- 🚀 TURBO BACKTESTER (THE ONE THAT WORKS) ---
        raw_dir = Path("data/raw")
        try:
            df_15m = standardize_df(pd.read_csv(raw_dir / f"{symbol_input}_15m.csv"))
            df_1h = standardize_df(pd.read_csv(raw_dir / f"{symbol_input}_1h.csv"))
            df_4h = standardize_df(pd.read_csv(raw_dir / f"{symbol_input}_4h.csv"))
        except:
            st.error("Missing CSV files in data/raw/")
            st.stop()

        mask = (df_15m['timestamp'].dt.date >= start_dt) & (df_15m['timestamp'].dt.date <= end_dt)
        bt_df = df_15m.loc[mask].copy().reset_index(drop=True)

        trades, wins, losses = [], 0, 0
        last_pred_hour, last_sig = -1, None
        prog = st.progress(0)

        # Account Simulation
        initial_balance = 10000
        balance = initial_balance
        risk_per_trade = 0.01  # 1% risk
        rr = 1  # reward:risk (since fixed 1h exit)
        equity_curve = []

        for i in range(50, len(bt_df) - 4):
            curr_row = bt_df.iloc[i]
            curr_ts = curr_row['timestamp']

            # Predict only on hour change for speed
            if curr_ts.hour != last_pred_hour:
                win_1h = df_1h[df_1h['timestamp'] <= curr_ts].tail(200)
                win_4h = df_4h[df_4h['timestamp'] <= curr_ts].tail(100)
                last_sig = mc.predict(bars_df=win_1h, context_df=win_4h)
                last_pred_hour = curr_ts.hour

            if last_sig and last_sig.signal in ["BUY", "SELL"]:
                # Logic: Enter now, Exit in 1 hour (4 x 15m bars)
                p_in = curr_row['close']
                p_out = bt_df.iloc[i + 4]['close']
                win = (p_out > p_in) if last_sig.signal == "BUY" else (p_out < p_in)

                # Deduplicate: One trade per signal hour
                if not trades or (curr_ts - trades[-1]['timestamp']).total_seconds() >= 3600:
                    risk_amount = balance * risk_per_trade

                    if win:
                        wins += 1
                        profit = risk_amount * rr
                        balance += profit
                    else:
                        losses += 1
                        loss = risk_amount
                        balance -= loss

                    equity_curve.append(balance)
                    trades.append({
                        "timestamp": curr_ts,
                        "price": p_in,
                        "type": last_sig.signal,
                        "result": "WIN" if win else "LOSS",
                        "regime": last_sig.regime,
                        "balance": balance
                    })

            if i % 100 == 0: prog.progress(i / len(bt_df))

        # RESULTS DISPLAY
        st.title(f"📊 {mode_selection} Backtest Results")
        wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        net_return = ((balance - initial_balance) / initial_balance) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trades", len(trades))
        c2.metric("Win Rate", f"{wr:.1f}%")
        c3.metric("Net Change", wins - losses)

        st.markdown("### 💰 Account Performance")

        p1, p2, p3 = st.columns(3)
        p1.metric("Final Value", f"${balance:,.2f}")
        p2.metric("Net Return (%)", f"{net_return:.2f}%")
        p3.metric("Wins / Losses", f"{wins} / {losses}")

        fig = go.Figure(data=[
            go.Candlestick(x=bt_df['timestamp'], open=bt_df['open'], high=bt_df['high'], low=bt_df['low'],
                           close=bt_df['close'])])
        if trades:
            tdf = pd.DataFrame(trades)
            fig.add_trace(go.Scatter(x=tdf[tdf['type'] == 'BUY']['timestamp'], y=tdf[tdf['type'] == 'BUY']['price'],
                                     mode='markers', marker=dict(symbol='triangle-up', color='#00ff99', size=12),
                                     name="BUY"))
            fig.add_trace(go.Scatter(x=tdf[tdf['type'] == 'SELL']['timestamp'], y=tdf[tdf['type'] == 'SELL']['price'],
                                     mode='markers', marker=dict(symbol='triangle-down', color='#ff3366', size=12),
                                     name="SELL"))
        st.plotly_chart(fig, use_container_width=True)

        # 📈 Equity Curve (ADD HERE)
        if trades:
            eq_df = pd.DataFrame(trades)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["timestamp"],
                y=eq_df["balance"],
                mode="lines+markers",
                name="Equity Curve"
            ))

            fig_eq.update_layout(
                title="📈 Equity Curve",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(fig_eq, use_container_width=True)

        # 📜 Trade Log Table (ADD AT VERY BOTTOM)
        if trades:
            st.markdown("### 📜 Trade Log")

            log_df = pd.DataFrame(trades)

            # Clean + format
            log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
            log_df = log_df.sort_values("timestamp")

            log_df = log_df.rename(columns={
                "timestamp": "Time",
                "price": "Entry Price",
                "type": "Position",
                "result": "Result",
                "balance": "Account Balance"
            })

            log_df["Time"] = log_df["Time"].dt.strftime("%Y-%m-%d %H:%M")
            log_df["Account Balance"] = log_df["Account Balance"].map(lambda x: f"${x:,.2f}")

            st.dataframe(log_df, use_container_width=True)