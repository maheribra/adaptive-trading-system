import polars as pl
import duckdb
from loguru import logger

def compute_dxy_signals(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    logger.info('Computing DXY signals...')
    try:
        df = con.execute(
            "SELECT * FROM prices WHERE symbol = 'DX-Y.NYB' ORDER BY timestamp"
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
    from src.database import get_connection
    con = get_connection()
    df = compute_dxy_signals(con)
    print(df)