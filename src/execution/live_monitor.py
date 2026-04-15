import sys
import os
import pandas as pd
from pathlib import Path
from twelvedata import TDClient
import warnings
from loguru import logger

# 1. Silence background logs for a cleaner UI
logger.remove()

# 2. Project Setup & Model Injection
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


def get_fvg_box(df):
    if len(df) < 3: return None
    c0_high, c0_low = df['high'].iloc[-3], df['low'].iloc[-3]
    c2_high, c2_low = df['high'].iloc[-1], df['low'].iloc[-1]
    if c0_low > c2_high: return {"type": "SHORT", "top": c0_low, "bottom": c2_high}
    if c2_low > c0_high: return {"type": "LONG", "top": c2_low, "bottom": c0_high}
    return None


def fetch_live_data(symbol):
    print(f"📡 Fetching live data for {symbol}...")
    try:
        td = TDClient(apikey=API_KEY)
        ts = td.time_series(symbol=symbol, interval="1h", outputsize=500, order="ASC")
        df = ts.as_pandas()
        if df is None or df.empty: return None
        df = df.reset_index().rename(columns={'datetime': 'timestamp'})
        df.columns = [str(col).lower() for col in df.columns]
        if 'volume' not in df.columns: df['volume'] = 0
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return None


def run_live_analyst():
    while True:
        print("\n" + "═" * 45 + "\n     🌍 LIVE HYBRID TRADING MONITOR (API)     \n" + "═" * 45)

        sym_input = input("Enter Symbol (e.g., GBP/USD) or 'n' to Exit: ").strip().lower()
        if sym_input in ['n', 'exit', 'quit']:
            print("👋 Closing Monitor...")
            sys.exit()

        symbol = sym_input.upper()
        df = fetch_live_data(symbol)
        if df is None: continue

        while True:
            print(f"\n[ Current Symbol: {symbol} ]")
            print("Select Mode: 1.HMM | 2.FVG | 3.Hybrid | 4.New Symbol | 5.Exit")
            choice = input("Choice: ")

            if choice == "4": break
            if choice == "5": sys.exit()

            mode = {"1": "HMM", "2": "FVG", "3": "Hybrid"}.get(choice)
            if not mode: continue

            # Analysis Logic
            df_4h = df.set_index('timestamp').resample('4H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()

            mc = MetaController(mode=mode).load()

            # Prediction (Internal logs silenced by logger.remove() at top)
            signal_obj = mc.predict(bars_df=df)
            res = signal_obj.to_dict()
            fvg = get_fvg_box(df_4h)
            curr_price = df['close'].iloc[-1]

            final_sig, status = res['signal'], "WAITING"

            if mode == "Hybrid" and fvg:
                in_zone = fvg['bottom'] < curr_price < fvg['top']
                hmm_bull = (res['signal'] == "BUY" or res['regime'] == 1)
                hmm_bear = (res['signal'] == "SELL" or res['regime'] == 2)
                if in_zone and ((fvg['type'] == "LONG" and hmm_bull) or (fvg['type'] == "SHORT" and hmm_bear)):
                    final_sig, status = f"STRONG {fvg['type']}", "TRADE READY"
                else:
                    final_sig = f"WATCHING ({fvg['type']} FVG)"

            # Output Box
            print(f"\n╔══════════════════════════════════════════════╗")
            print(f"║             LIVE ANALYSIS: {symbol:<15s} ║")
            print(f"╠══════════════════════════════════════════════╣")
            print(f"║  Mode           : {mode:<27s}║")
            print(f"║  Price          : {curr_price:<27.5f}║")
            print(f"║  Regime         : {str(res['regime']):<27}║")
            print(f"║  Final Signal   : {final_sig:<27s}║")
            print(f"║  Action Status  : {status:<27s}║")
            print(f"╚══════════════════════════════════════════════╝")

            inner = input(f"\nStay on {symbol}? (y) Try another mode / (n) New Symbol: ").lower()
            if inner != 'y': break


if __name__ == "__main__":
    run_live_analyst()