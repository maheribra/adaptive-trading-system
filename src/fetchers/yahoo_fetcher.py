import yfinance as yf
import polars as pl
from loguru import logger
import duckdb
import time
import requests

PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
    'AUDUSD=X', 'USDCAD=X', 'DX-Y.NYB'
]

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session

def fetch_ohlcv(symbol: str, period='2y', interval='1h') -> pl.DataFrame:
    logger.info(f'Fetching {symbol}')
    try:
        session = get_session()
        ticker = yf.Ticker(symbol, session=session)
        raw = ticker.history(period=period, interval=interval)
        if raw.empty:
            logger.warning(f'No data for {symbol}')
            return pl.DataFrame()
        raw = raw.reset_index()
        raw.columns = [c.strftime('%Y') if hasattr(c, 'strftime') else c for c in raw.columns]
        raw = raw.rename(columns={
            'Datetime': 'timestamp', 'Date': 'timestamp',
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        raw['symbol'] = symbol
        df = pl.from_pandas(raw[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        logger.success(f'Got {len(df)} rows for {symbol}')
        time.sleep(2)
        return df
    except Exception as e:
        logger.error(f'Error fetching {symbol}: {e}')
        return pl.DataFrame()

def fetch_all_pairs(con: duckdb.DuckDBPyConnection):
    for pair in PAIRS:
        df = fetch_ohlcv(pair)
        if not df.is_empty():
            con.execute('INSERT OR REPLACE INTO prices SELECT * FROM df')

if __name__ == '__main__':
    from src.database import get_connection, create_schema
    con = get_connection()
    create_schema(con)
    fetch_all_pairs(con)