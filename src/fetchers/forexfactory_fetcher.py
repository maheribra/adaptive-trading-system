import feedparser
import polars as pl
import pandas as pd
from loguru import logger
from pathlib import Path

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


def load_cpi_from_csv() -> pl.DataFrame:
    BASE_DIR = Path(__file__).resolve().parents[2]
    CPI_FILE = BASE_DIR / "data" / "raw" / "cpi_events.csv"

    if not CPI_FILE.exists():
        logger.warning(f"cpi_events.csv not found at {CPI_FILE}")
        return pl.DataFrame()

    logger.info("Loading CPI events from cpi_events.csv...")

    try:
        raw = pd.read_csv(CPI_FILE)
        raw.columns = [c.lower() for c in raw.columns]

        # Rename columns to match news_events schema
        cpi_col = [c for c in raw.columns if 'cpi' in c.lower()][0]
        raw = raw.rename(columns={'time': 'event_time', cpi_col: 'actual'})

        raw['title']    = 'CPI'
        raw['currency'] = 'USD'
        raw['impact']   = 'High'
        raw['forecast'] = None
        raw['previous'] = None
        raw['deviation'] = None

        raw = raw[['event_time', 'title', 'currency', 'impact', 'actual', 'forecast', 'previous', 'deviation']]
        raw['actual'] = raw['actual'].astype(str)

        df = pl.from_pandas(raw)
        logger.success(f"Loaded {len(df)} CPI rows from CSV")
        return df

    except Exception as e:
        logger.error(f"Error loading cpi_events.csv: {e}")
        return pl.DataFrame()


if __name__ == '__main__':
    df = load_cpi_from_csv()
    print(df)