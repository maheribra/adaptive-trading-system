from loguru import logger
from src.database import get_connection, create_schema
from src.fetchers.forexfactory_fetcher import fetch_news_events
from src.fetchers.dxy_fetcher import compute_dxy_signals

def run_pipeline():
    logger.info('=== Starting Data Pipeline ===')
    con = get_connection()
    create_schema(con)

    # 1. Fetch high-impact news events
    logger.info('--- Fetching ForexFactory news ---')
    news_df = fetch_news_events()
    if not news_df.is_empty():
        con.execute('INSERT OR REPLACE INTO news_events SELECT * FROM news_df')
        logger.success(f'Stored {len(news_df)} news events')
    else:
        logger.info('No news events to store this week')

    # 2. Compute DXY signals (runs after Yahoo data is loaded)
    logger.info('--- Computing DXY signals ---')
    dxy_df = compute_dxy_signals(con)
    if not dxy_df.is_empty():
        con.execute('INSERT OR REPLACE INTO dxy_signals SELECT * FROM dxy_df')
        logger.success(f'Stored {len(dxy_df)} DXY signal rows')
    else:
        logger.info('No DXY data yet - run yahoo_fetcher first')

    # 3. Summary
    logger.info('--- Summary ---')
    for table in ['prices', 'news_events', 'dxy_signals', 'market_regimes']:
        count = con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        logger.info(f'{table}: {count:,} rows')

    logger.success('=== Pipeline Complete ===')

if __name__ == '__main__':
    run_pipeline()