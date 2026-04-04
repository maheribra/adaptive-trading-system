import sys
from pathlib import Path
import warnings
import logging
import pandas as pd
from loguru import logger

# 1. Project Setup
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# 2. Model Class Injection (Required for Pickle)
from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel
import __main__

for m in [RangeModel, TrendModel, NewsModel, DXYModel]:
    setattr(__main__, m.__name__, m)

# 3. Silence HMM noise
warnings.filterwarnings("ignore", message=".*near-zero std.*")
logging.getLogger("src.models.hmm_regime_classifier").setLevel(logging.ERROR)
logging.getLogger("src.models.features.trend_features").setLevel(logging.ERROR)
logging.getLogger("src.models.features.range_features").setLevel(logging.ERROR)

from src.database import get_connection
from src.meta_controller import MetaController


def get_interactive_config(available_symbols):
    print("\n" + "═" * 45)
    print("      🎯 META-CONTROLLER INTERACTIVE        ")
    print("═" * 45)
    print(f"Available: {', '.join(available_symbols)}")

    symbol = input("\nSelect Symbol (e.g., USDJPY=X): ") or "USDJPY=X"

    print("\nTarget Point:")
    print("1. Latest available data")
    print("2. Specific historical timestamp")
    choice = input("Choice [1/2]: ") or "1"

    target_date = None
    if choice == "2":
        target_date = input("Enter Timestamp (YYYY-MM-DD HH:MM): ")

    num_bars = input("Lookback bars for analysis (Default 500): ")
    num_bars = int(num_bars) if num_bars else 500

    return symbol, target_date, num_bars


def load_data(con, symbol, target_date, n_bars):
    query = f"SELECT * FROM prices WHERE symbol = '{symbol}'"
    if target_date:
        query += f" AND timestamp <= '{target_date}'"
    query += f" ORDER BY timestamp DESC LIMIT {n_bars + 10}"

    df = con.execute(query).df()
    if df.empty:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df


def print_final_box(result: dict, mode: str):
    ts = result['timestamp']
    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║          FINAL SIGNAL ({mode:<7})            ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Timestamp      : {ts_str:<27s}║")
    print(f"║  Signal         : {result['signal']:<27s}║")
    print(f"║  Regime         : {result['regime']:<27s}║")
    print(f"║  Confirmed      : {str(result['confirmed']):<27s}║")
    print(f"║  Bars in regime : {result['bars_in_regime']:<27d}║")
    print(f"║  HMM confidence : {result['confidence']:<27.4f}║")
    print(f"╚══════════════════════════════════════════════╝")


def run():
    con = get_connection()
    symbols = con.execute("SELECT DISTINCT symbol FROM prices").df()['symbol'].tolist()

    # 1. Base Setup
    symbol, target_date, n_bars = get_interactive_config(symbols)
    df = load_data(con, symbol, target_date, n_bars)

    if df is None or len(df) < 200:
        print(f"❌ Error: Not enough data for {symbol}.")
        return

    # 2. Loop Logic for Multiple Mode Selections
    seen_modes = set()
    all_modes = {"1": "HMM", "2": "FVG", "3": "Hybrid"}

    while len(seen_modes) < 3:
        # Filter choices to show only what's left
        remaining = {k: v for k, v in all_modes.items() if v not in seen_modes}

        print("\n--- Select Execution Mode ---")
        for k, v in remaining.items():
            print(f"{k}. {v}")

        choice = input("Choice: ")
        if choice not in remaining:
            print("Invalid choice or already seen.")
            continue

        selected_mode = remaining[choice]
        seen_modes.add(selected_mode)

        # 3. Load & Run MetaController for selected mode
        mc = MetaController(mode=selected_mode).load()

        print(f"\n🔄 Simulating sequence for mode: {selected_mode}...")

        signal_obj = None
        # Simulation loop
        for i in range(4, -1, -1):
            end_idx = len(df) - i
            sim_bars = df.iloc[:end_idx].copy()
            signal_obj = mc.predict(bars_df=sim_bars)

        # Output Box
        print_final_box(signal_obj.to_dict(), selected_mode)

        # 4. Ask to continue
        if len(seen_modes) < 3:
            others = [v for v in all_modes.values() if v not in seen_modes]
            prompt = f"\nWould you like to see signals for {', '.join(others)}? (y/n): "
            cont = input(prompt).lower()
            if cont != 'y':
                break
        else:
            print("\n✅ All modes analyzed. Closing session.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.exception(e)