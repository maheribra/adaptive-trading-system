import os
import sys
from pathlib import Path
import warnings
import logging
import pandas as pd
from loguru import logger

# 1. Setup & Pathing
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

logger.remove()
logger.add(sys.stderr, level="ERROR")
logging.getLogger("src.models").setLevel(logging.ERROR)
logging.getLogger("src.meta_controller").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import __main__
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel
from src.meta_controller import MetaController

for m in [RangeModel, TrendModel, NewsModel, DXYModel]:
    setattr(__main__, m.__name__, m)


# 2. Helper for Multi-Timeframe Loading
def load_mtf_data(symbol):
    """Loads 15m, 1h, and 4h CSVs from data/raw."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    data_path = project_root / "data" / "raw"

    def clean_df(file_name):
        full_path = data_path / file_name
        if not full_path.exists():
            print(f" ERROR: File not found at {full_path}")
            return None

        d = pd.read_csv(full_path)
        d.columns = [c.lower() for c in d.columns]

        target_col = next((c for c in d.columns if c in ['timestamp', 'time', 'date']), None)
        if target_col:
            d['timestamp'] = pd.to_datetime(d[target_col]).dt.tz_localize(None)
        else:
            d['timestamp'] = pd.to_datetime(d.index).tz_localize(None)

        if 'volume' not in d.columns: d['volume'] = 0
        return d.sort_values('timestamp').set_index('timestamp')

    base_name = symbol.split('=')[0].strip()
    print(f" Searching in: {data_path}")

    df_15m = clean_df(f"{base_name}_15m.csv")
    df_1h = clean_df(f"{base_name}_1h.csv")
    df_4h = clean_df(f"{base_name}_4h.csv")

    if df_15m is None or df_1h is None or df_4h is None:
        raise FileNotFoundError(f"Missing MTF files for {base_name}")

    return df_15m, df_1h, df_4h


def get_fvg_box(df):
    """Detects a gap and returns the price box boundaries."""
    if len(df) < 3: return None
    c0_high, c0_low = df['high'].iloc[-3], df['low'].iloc[-3]
    c2_high, c2_low = df['high'].iloc[-1], df['low'].iloc[-1]

    if c0_low > c2_high:
        return {"type": "SHORT", "top": c0_low, "bottom": c2_high, "sl": c0_high}
    if c2_low > c0_high:
        return {"type": "LONG", "top": c2_low, "bottom": c0_high, "sl": c0_low}
    return None


# 3. Core Simulation Logic
import sys
import os
import pandas as pd


def run_simulation(mtf_data, mode, start_date=None, end_date=None):
    df_15m, df_1h, df_4h = mtf_data

    # 1. Date Filtering
    if start_date and start_date != 'all':
        sd = pd.to_datetime(start_date)
        df_15m = df_15m[df_15m.index >= sd]
        df_1h = df_1h[df_1h.index >= sd]
        df_4h = df_4h[df_4h.index >= sd]
    if end_date and end_date != 'all':
        ed = pd.to_datetime(end_date)
        df_15m = df_15m[df_15m.index <= ed]
        df_1h = df_1h[df_1h.index <= ed]
        df_4h = df_4h[df_4h.index <= ed]

    # Load MetaController
    mc = MetaController(mode=mode).load()

    # State Variables
    cash, inventory, trade_count = 10000.0, 0, 0
    entry_price, bars_held = 0.0, 0
    position_type, pending_4h_fvg = None, None

    # Metrics tracking
    wins, losses = 0, 0

    # Performance Optimization Variables
    last_h1_ts = None
    cached_signal = None

    start_idx, end_idx = 50, len(df_15m)
    print(f" Running {mode} Simulation (Optimized for Dashboard Sync)...")

    for i in range(start_idx, end_idx):
        ts = df_15m.index[i]
        bar_15m = df_15m.iloc[i]
        price = bar_15m['close']

        # --- 0. SAFETY INITIALIZATION ---
        # Defining these at the start of EVERY bar ensures the "Unbound" error can never happen
        in_zone = False

        # --- PROGRESS BAR ---
        if i % 250 == 0:
            percent = (i - start_idx) / (end_idx - start_idx) * 100
            sys.stdout.write(f"\rProgress: [{int(percent)}%] processing {i}/{end_idx}...")
            sys.stdout.flush()

        # --- 1. EXIT LOGIC (Time-Based) ---
        if position_type:
            bars_held += 1
            if bars_held >= 4:
                if position_type == "LONG":
                    profit = (price - entry_price) * inventory
                    cash += (inventory * entry_price) + profit
                    if price > entry_price:
                        wins += 1
                    else:
                        losses += 1
                else:
                    profit = (entry_price - price) * abs(inventory)
                    cash += profit
                    if price < entry_price:
                        wins += 1
                    else:
                        losses += 1
                inventory, position_type, bars_held = 0, None, 0
                continue

        # --- 2. PREDICTION CACHING ---
        current_h1_ts = ts.floor('H')
        if current_h1_ts != last_h1_ts:
            hist_1h = df_1h[df_1h.index <= ts].tail(200)
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                try:
                    cached_signal = mc.predict(bars_df=hist_1h.reset_index())
                finally:
                    sys.stdout = old_stdout
            last_h1_ts = current_h1_ts

        # --- 3. ENTRY LOGIC ---
        if not position_type:
            risk_pct = 0.01

            # MODE: HMM
            if mode == "HMM" and cached_signal:
                if cached_signal.signal in ["BUY", "SELL"]:
                    entry_price, bars_held, trade_count = price, 0, trade_count + 1
                    risk_dist = entry_price * 0.005
                    qty = (cash * risk_pct) / risk_dist
                    if cached_signal.signal == "BUY":
                        position_type, inventory = "LONG", qty
                        cash -= (qty * entry_price)
                    else:
                        position_type, inventory = "SHORT", -qty

            # MODE: FVG or HYBRID
            elif mode in ["FVG", "Hybrid"]:
                # 1. Look back slightly further for 4H context
                hist_4h = df_4h[df_4h.index <= ts].tail(100)
                found_fvg = get_fvg_box(hist_4h)
                if found_fvg:
                    pending_4h_fvg = found_fvg

                if pending_4h_fvg:
                    # Sync Buffer: 0.0006 (6 pips) to match Dashboard sensitivity
                    buffer = price * 0.0006
                    in_zone = (price <= (pending_4h_fvg['top'] + buffer)) and \
                              (price >= (pending_4h_fvg['bottom'] - buffer))

                    if in_zone:
                        permission = True
                        if mode == "Hybrid" and cached_signal:
                            # DASHBOARD SYNC: Use a more lenient 'Regime' check
                            # Regime 1 = Bullish, Regime 2 = Bearish
                            curr_regime = getattr(cached_signal, 'regime', 0)

                            if pending_4h_fvg['type'] == "LONG":
                                # Permission if BUY signal OR if we are simply in Bullish Regime
                                permission = (cached_signal.signal == "BUY") or (curr_regime == 1)
                            else:
                                # Permission if SELL signal OR if we are simply in Bearish Regime
                                permission = (cached_signal.signal == "SELL") or (curr_regime == 2)

                        if permission:
                            entry_price, bars_held, trade_count = price, 0, trade_count + 1
                            risk_dist = entry_price * 0.005
                            qty = (cash * risk_pct) / risk_dist

                            if pending_4h_fvg['type'] == "LONG":
                                position_type, inventory = "LONG", qty
                                cash -= (qty * entry_price)
                            else:
                                position_type, inventory = "SHORT", -qty

                            # Clear current FVG to prevent immediate re-entry on next 15m bar
                            pending_4h_fvg = None

    # --- FINAL WRAP UP ---
    sys.stdout.write("\n")
    final_val = cash
    # If a trade is still open at the very end of the data, close it at market
    if position_type == "LONG":
        final_val = cash + (inventory * df_15m['close'].iloc[-1])
    elif position_type == "SHORT":
        final_val = cash + abs(inventory) * (entry_price - df_15m['close'].iloc[-1])

    pnl = ((final_val - 10000.0) / 10000.0) * 100
    winrate = (wins / trade_count * 100) if trade_count > 0 else 0

    print(f"\n--- Trade Statistics ---")
    print(f"Wins: {wins} | Losses: {losses} | Winrate: {winrate:.2f}%")

    return (end_idx - start_idx), trade_count, final_val, pnl


def print_backtest_box(symbol, mode, bars, trades, final_val, pnl):
    color = "\033[92m" if pnl >= 0 else "\033[91m"
    reset = "\033[0m"
    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║          BACKTEST RESULTS: {mode:<10}        ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Symbol            : {symbol:<24s}║")
    print(f"║  15m Bars Processed: {bars:<24d}║")
    print(f"║  Total Trades      : {trades:<24d}║")
    print(f"║  Final Value       : ${final_val:<23,.2f}║")
    print(f"║  Net Return (%)    : {color}{pnl:<+23.2f}%{reset}║")
    print(f"╚══════════════════════════════════════════════╝")


def run():
    print("\n" + "═" * 45)
    print("       DATE-DRIVEN MTF BACKTESTER         ")
    print("═" * 45)

    symbol = input("Enter Symbol (EURUSD=X, GBPUSD=X, etc.): ") or "EURUSD=X"
    s_date = input("Start Date (YYYY-MM-DD): ")
    e_date = input("End Date (YYYY-MM-DD): ")

    try:
        mtf_data = load_mtf_data(symbol)
    except Exception as e:
        print(f" Error: {e}");
        return

    seen_modes = set()
    all_modes = {"1": "HMM", "2": "FVG", "3": "Hybrid"}

    while len(seen_modes) < 3:
        remaining = {k: v for k, v in all_modes.items() if v not in seen_modes}
        print("\n--- Select Backtest Mode ---")
        for k, v in remaining.items(): print(f"{k}. {v}")

        choice = input("Choice: ")
        if choice not in remaining: continue
        selected_mode = remaining[choice]
        seen_modes.add(selected_mode)

        bars, trades, f_val, pnl = run_simulation(mtf_data, selected_mode, s_date, e_date)
        print_backtest_box(symbol, selected_mode, bars, trades, f_val, pnl)

        if len(seen_modes) < 3:
            if input(f"\nCompare with others? (y/n): ").lower() != 'y': break

    print("\n Simulation session complete.")


if __name__ == "__main__":
    run()