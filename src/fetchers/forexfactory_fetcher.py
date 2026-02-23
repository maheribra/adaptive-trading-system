import feedparser
import polars as pl
from loguru import logger

FF_RSS_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.xml'

HIGH_IMPACT = [
    'CPI', 'NFP', 'Non-Farm', 'Interest Rate',
    'GDP', 'Unemployment', 'Retail Sales', 'PCE', 'FOMC'
]

def is_high_impact(title: str) -> bool:
    return any(k.lower() in title.lower() for k in HIGH_IMPACT)

def fetch_news_events() -> pl.DataFrame:
    logger.info('Fetching ForexFactory RSS...')
    try:
        feed = feedparser.parse(FF_RSS_URL)
        events = []
        for entry in feed.entries:
            if not is_high_impact(entry.get('title', '')):
                continue
            events.append({
                'event_time': entry.get('published', ''),
                'title':      entry.get('title', ''),
                'currency':   entry.get('ff_currency', ''),
                'impact':     entry.get('ff_impact', ''),
                'actual':     entry.get('ff_actual', ''),
                'forecast':   entry.get('ff_forecast', ''),
                'previous':   entry.get('ff_previous', ''),
            })
        df = pl.DataFrame(events) if events else pl.DataFrame()
        logger.success(f'Found {len(df)} high-impact events this week')
        return df
    except Exception as e:
        logger.error(f'Error fetching news: {e}')
        return pl.DataFrame()

if __name__ == '__main__':
    df = fetch_news_events()
    print(df)