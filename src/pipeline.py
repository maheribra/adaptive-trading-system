from loguru import logger
from src.database import get_connection, create_schema
from src.fetchers.yahoo_fetcher import load_from_csv, load_dxy_from_csv
from src.fetchers.forexfactory_fetcher import fetch_news_events, load_cpi_from_csv
from src.fetchers.dxy_fetcher import compute_dxy_signals

def run_pipeline():
    logger.info('=== Starting Data Pipeline ===')
    con = get_connection()
    create_schema(con)

    # 1. Load forex pairs from CSVs
    logger.info('--- Loading CSV price data ---')
    load_from_csv(con)

    # 2. Load DXY from CSV (replaces yfinance)
    logger.info('--- Loading DXY from CSV ---')
    load_dxy_from_csv(con)

    # 3. Load CPI events from CSV
    logger.info('--- Loading CPI events from CSV ---')
    cpi_df = load_cpi_from_csv()
    if not cpi_df.is_empty():
        con.register("cpi_df", cpi_df)
        con.execute('INSERT OR REPLACE INTO news_events SELECT * FROM cpi_df')
        logger.success(f'Stored {len(cpi_df)} CPI events')
    else:
        logger.info('No CPI events loaded')

    # 4. Fetch live ForexFactory news (this week only)
    logger.info('--- Fetching ForexFactory news ---')
    news_df = fetch_news_events()
    if not news_df.is_empty():
        con.register("news_df", news_df)
        con.execute('INSERT OR REPLACE INTO news_events SELECT * FROM news_df')
        logger.success(f'Stored {len(news_df)} live news events')
    else:
        logger.info('No live news events this week')

    # 5. Compute DXY signals
    logger.info('--- Computing DXY signals ---')
    dxy_df = compute_dxy_signals(con)
    if not dxy_df.is_empty():
        con.register("dxy_df", dxy_df)
        con.execute("""
            INSERT OR REPLACE INTO dxy_signals (
                timestamp,
                close,
                dxy_pct_change,
                dxy_ma5,
                dxy_bullish
            )
            SELECT
                timestamp,
                close,
                dxy_pct_change,
                dxy_ma5,
                dxy_bullish
            FROM dxy_df
        """)
        logger.success(f'Stored {len(dxy_df)} DXY signal rows')

    # 6. Summary
    logger.info('--- Summary ---')
    for table in ['prices', 'news_events', 'dxy_signals', 'market_regimes']:
        count = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        logger.info(f'{table}: {count:,} rows')

    logger.success('=== Pipeline Complete ===')

if __name__ == '__main__':
    run_pipeline()