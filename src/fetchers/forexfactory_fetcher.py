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

EVENT_FILES = {
    "CPI":          "CPI.csv",
    "NFP":          "NFP.csv",
    "Unemployment": "Unemployment.csv",
    "RetailSales":  "RetailSales.csv",
    "FED_RATE":     "FED_RATE.csv",
}


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


def load_events_from_csv() -> pl.DataFrame:
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data" / "raw"

    all_dfs = []

    for event_name, filename in EVENT_FILES.items():
        filepath = DATA_DIR / filename

        if not filepath.exists():
            logger.warning(f"{filename} not found — skipping {event_name}")
            continue

        try:
            raw = pd.read_csv(filepath)
            raw.columns = [c.lower() for c in raw.columns]

            # Rename time column
            for col in ['time', 'datetime', 'date']:
                if col in raw.columns:
                    raw = raw.rename(columns={col: 'event_time'})
                    break

            # Auto-detect value column
            skip_cols = {'event_time', 'forecast', 'previous', 'deviation', 'currency', 'impact', 'title'}
            value_cols = [c for c in raw.columns if c not in skip_cols]
            if value_cols:
                raw = raw.rename(columns={value_cols[0]: 'actual'})

            raw['title']    = event_name
            raw['currency'] = 'USD'
            raw['impact']   = 'High'

            for col in ['forecast', 'previous', 'deviation']:
                if col not in raw.columns:
                    raw[col] = None

            raw = raw[['event_time', 'title', 'currency', 'impact', 'actual', 'forecast', 'previous', 'deviation']]
            raw['actual'] = raw['actual'].astype(str)

            df = pl.from_pandas(raw)
            logger.success(f"Loaded {len(df)} rows from {filename}")
            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    if not all_dfs:
        logger.warning("No event CSV files loaded")
        return pl.DataFrame()

    combined = pl.concat(all_dfs)
    logger.success(f"Total news events loaded: {len(combined)}")
    return combined


# Keep for backwards compatibility
def load_cpi_from_csv() -> pl.DataFrame:
    return load_events_from_csv()


if __name__ == '__main__':
    df = load_events_from_csv()
    print(df)