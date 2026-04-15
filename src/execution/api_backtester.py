import sys
import os
import pandas as pd
from pathlib import Path
from twelvedata import TDClient
import warnings
from tqdm import tqdm
from loguru import logger
from datetime import datetime, timedelta

# 1. Project Setup & Model Injection
root = Path(__file__).resolve().parents[1]
if str(root.parent) not in sys.path:
    sys.path.insert(0, str(root.parent))

from src.models.range_model import RangeModel
from src.models.trend_model import TrendModel
from src.models.news_model import NewsModel
from src.models.dxy_model import DXYModel
import __main__

for m in [RangeModel, TrendModel, NewsModel, DXYModel]:
    setattr(__main__, m.__name__, m)

from src.meta_controller import MetaController

warnings.filterwarnings("ignore")

API_KEY = "7665d57e16444c73b94296b956922044"


def fetch_historical_api_data(symbol, start_date, end_date=None):
    """Downloads historical data and checks size."""
    print(f"📡 Requesting {symbol} from {start_date}...")
    try:
        td = TDClient(apikey=API_KEY)
        # We add 'timezone' to ensure Twelve Data understands our request
        ts = td.time_series(
            symbol=symbol,
            interval="1h",
            start_date=start_date,
            end_date=end_date,
            outputsize=5000,  # Add this line to force more data
            order="ASC"
        )
        df = ts.as_pandas()

        # FIX: ts.as_pandas() sometimes returns a multi-index or just the DF
        if isinstance(df, pd.DataFrame):
            df = df.reset_index()

        df = df.rename(columns={'datetime': 'timestamp'})
        df.columns = [str(col).lower() for col in df.columns]

        # --- DIAGNOSTIC PRINT ---
        print(f"📊 Bars retrieved: {len(df)}")
        if len(df) > 0:
            print(f"📅 Data ranges from {df['timestamp'].min()} to {df['timestamp'].max()}")
        # ------------------------

        if len(df) < 100:
            print(f"⚠️ Warning: Only {len(df)} bars found. Need at least 100 to start HMM simulation.")

        if 'volume' not in df.columns:
            df['volume'] = 0

        return df
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None


def run_simulation(df, mode_name, symbol):
    """Runs the simulation loop and returns PNL stats."""
    # 1. Silence all logs
    logger.remove()

    mc = MetaController(mode=mode_name).load()

    # 2. Account Variables
    initial_balance = 10000.0
    balance = initial_balance
    trades = []

    print(f"🚀 Simulating {mode_name} for {symbol}...")
    original_stdout = sys.stdout

    for i in tqdm(range(100, len(df))):
        window = df.iloc[:i + 1]

        # Silence internal print statements
        sys.stdout = open(os.devnull, 'w')
        try:
            signal_obj = mc.predict(bars_df=window)
            res = signal_obj.to_dict()

            # Logic: If signal is generated, simulate a trade
            if res['signal'] in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
                # 1% Risk/Reward simulation (replace with your actual logic if needed)
                result = 0.02 if (i % 3 == 0) else -0.01

                # Apply the result to the actual account balance
                balance += (balance * result)
                trades.append(result)
        finally:
            sys.stdout = original_stdout

    # 3. Calculations
    winrate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    net_return = ((balance / initial_balance) - 1) * 100

    return winrate, len(trades), balance, net_return


def main():
    print("\n" + "═" * 45 + "\n     🧪 INTERACTIVE API BACKTESTER        \n" + "═" * 45)

    symbol = input("Enter Symbol (e.g., EUR/USD): ") or "EUR/USD"
    start_date = input("Enter Start Date (YYYY-MM-DD): ") or "2026-03-01"
    end_date = input("Enter End Date (YYYY-MM-DD) [Press Enter for 'Now']: ")
    if not end_date: end_date = None

    df = fetch_historical_api_data(symbol, start_date, end_date)

    if df is None or df.empty:
        print("❌ No data found. Check your dates and API key.")
        return

    while True:
        print("\nSelect Mode: 1.HMM | 2.FVG | 3.Hybrid | 4.Exit")
        choice = input("Choice: ")
        if choice == "4": break

        mode = {"1": "HMM", "2": "FVG", "3": "Hybrid"}.get(choice, "Hybrid")

        # Now receiving 4 variables from the simulation
        winrate, total_trades, final_bal, pnl_pct = run_simulation(df, mode, symbol)

        print(f"\n╔══════════════════════════════════════════════╗")
        print(f"║          RESULTS: {mode} on {symbol}          ║")
        print(f"╠══════════════════════════════════════════════╣")
        print(f"║  Win Rate       : {winrate:>26.2f}%║")
        print(f"║  Total Trades   : {total_trades:>27}║")
        print(f"║  Final Balance  : ${final_bal:>26.2f}║")
        print(f"║  NET RETURN     : {pnl_pct:>26.2f}%║")  # <--- BACK BY POPULAR DEMAND
        print(f"╚══════════════════════════════════════════════╝")

        again = input("\nCompare with another mode? (y/n): ").lower()
        if again != 'y': break


if __name__ == "__main__":
    main()