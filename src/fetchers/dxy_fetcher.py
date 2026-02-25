import polars as pl
import duckdb
from loguru import logger
import yfinance as yf
import requests
import time

def fetch_dxy_price(con: duckdb.DuckDBPyConnection):
    logger.info('Fetching DXY price data...')
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })
        ticker = yf.Ticker('DX=F', session=session)
        raw = ticker.history(period='2y', interval='1h')
        if raw.empty:
            logger.warning('No DXY price data returned')
            return
        raw = raw.reset_index()
        raw = raw.rename(columns={
            'Datetime': 'timestamp', 'Date': 'timestamp',
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        raw['symbol'] = 'DX=F'
        df = pl.from_pandas(raw[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        con.execute('INSERT OR REPLACE INTO prices SELECT * FROM df')
        logger.success(f'Stored {len(df)} DXY price rows')
    except Exception as e:
        logger.error(f'Error fetching DXY: {e}')

def compute_dxy_signals(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    logger.info('Computing DXY signals...')
    try:
        df = con.execute(
            "SELECT * FROM prices WHERE symbol = 'DX=F' ORDER BY timestamp"
        ).pl()
        if df.is_empty():
            logger.warning('No DXY data in database yet')
            return pl.DataFrame()
        df = df.with_columns([
            pl.col('close').pct_change().alias('dxy_pct_change'),
            pl.col('close').rolling_mean(5).alias('dxy_ma5'),
        ])
        df = df.with_columns([
            (pl.col('close') > pl.col('dxy_ma5')).alias('dxy_bullish')
        ])
        logger.success('DXY signals computed')
        return df
    except Exception as e:
        logger.error(f'Error computing DXY signals: {e}')
        return pl.DataFrame()

if __name__ == '__main__':
    from src.database import get_connection, create_schema
    con = get_connection()
    create_schema(con)
    fetch_dxy_price(con)
    df = compute_dxy_signals(con)
    print(df)