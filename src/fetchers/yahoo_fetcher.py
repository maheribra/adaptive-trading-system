import polars as pl
from loguru import logger
import duckdb
import pandas as pd
from pathlib import Path


PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
    'AUDUSD=X', 'USDCAD=X'
]


def load_from_csv(con: duckdb.DuckDBPyConnection):
    """
    Load all CSV files from data/raw into DuckDB prices table.
    Works regardless of where script is executed from.
    """

    # Resolve project root (adaptive-trading-system/)
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data" / "raw"

    logger.info(f"Looking for CSV files in: {DATA_DIR}")

    files = list(DATA_DIR.glob("*.csv"))

    if not files:
        logger.warning(f'No CSV files found in {DATA_DIR}')
        return

    for file in files:
        try:
            # Extract symbol from filename
            symbol = file.stem.replace('_1h', '') + '=X'

            logger.info(f"Loading {file.name} as {symbol}")

            # Read CSV
            raw = pd.read_csv(file)

            # Standardize column names
            raw.columns = [c.lower() for c in raw.columns]
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Rename timestamp column if needed
            for col in ['time', 'datetime', 'date']:
                if col in raw.columns:
                    raw = raw.rename(columns={col: 'timestamp'})
                    break

            # If volume is missing (common for FX), create it
            if 'volume' not in raw.columns:
                raw['volume'] = 0.0

            # Detect timestamp column
            timestamp_found = False
            for col in ['time', 'datetime', 'date', 'timestamp']:
                if col in raw.columns:
                    raw = raw.rename(columns={col: 'timestamp'})
                    timestamp_found = True
                    break

            if not timestamp_found:
                logger.warning(f"No timestamp column found in {file.name}")
                continue

            raw['symbol'] = symbol

            # Keep required columns only
            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            existing_cols = [c for c in required_cols if c in raw.columns]

            raw = raw[existing_cols]

            # Convert to Polars
            df = pl.from_pandas(raw)

            if df.is_empty():
                logger.warning(f"{file.name} produced empty DataFrame")
                continue

            # Insert into DuckDB
            con.register("temp_df", df)

            con.execute("""
                INSERT OR REPLACE INTO prices
                SELECT * FROM temp_df
            """)

            logger.success(f"Loaded {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {file.name}: {e}")


if __name__ == '__main__':
    from src.database import get_connection, create_schema

    logger.info("Starting CSV Loader")

    con = get_connection()
    create_schema(con)

    load_from_csv(con)

    logger.success("CSV loading process finished.")