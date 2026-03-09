import polars as pl
from loguru import logger
import duckdb
import pandas as pd
from pathlib import Path


PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
    'AUDUSD=X', 'USDCAD=X'
]

SKIP_FILES = {'dxy_index.csv', 'cpi_events.csv'}


def load_from_csv(con: duckdb.DuckDBPyConnection):
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data" / "raw"

    logger.info(f"Looking for CSV files in: {DATA_DIR}")

    files = [f for f in DATA_DIR.glob("*.csv") if f.name not in SKIP_FILES]

    if not files:
        logger.warning(f'No CSV files found in {DATA_DIR}')
        return

    for file in files:
        try:
            symbol = file.stem.replace('_1h', '') + '=X'
            logger.info(f"Loading {file.name} as {symbol}")

            raw = pd.read_csv(file)
            raw.columns = [c.lower() for c in raw.columns]

            for col in ['time', 'datetime', 'date']:
                if col in raw.columns:
                    raw = raw.rename(columns={col: 'timestamp'})
                    break

            if 'volume' not in raw.columns:
                raw['volume'] = 0.0

            raw['symbol'] = symbol

            required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            existing_cols = [c for c in required_cols if c in raw.columns]
            raw = raw[existing_cols]

            df = pl.from_pandas(raw)

            if df.is_empty():
                logger.warning(f"{file.name} produced empty DataFrame")
                continue

            con.register("temp_df", df)
            con.execute("INSERT OR REPLACE INTO prices SELECT * FROM temp_df")
            logger.success(f"Loaded {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {file.name}: {e}")


def load_dxy_from_csv(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load DXY from CSV into DuckDB prices table AND return as DataFrame
    so it can be passed directly into feature engineering.
    """
    BASE_DIR = Path(__file__).resolve().parents[2]
    DXY_FILE = BASE_DIR / "data" / "raw" / "dxy_index.csv"

    if not DXY_FILE.exists():
        logger.warning(f"dxy_index.csv not found at {DXY_FILE}")
        return pd.DataFrame()

    logger.info("Loading DXY from dxy_index.csv...")

    try:
        raw = pd.read_csv(DXY_FILE)
        raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={'time': 'timestamp', 'dxy': 'close'})
        raw['timestamp'] = pd.to_datetime(raw['timestamp']).dt.tz_localize(None)

        raw['symbol'] = 'DX=F'
        raw['open']   = raw['close']
        raw['high']   = raw['close']
        raw['low']    = raw['close']
        raw['volume'] = 0.0

        db_df = raw[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
        db_df['timestamp'] = db_df['timestamp'].astype(str)
        pl_df = pl.from_pandas(db_df).with_columns(
            pl.col('timestamp').str.to_datetime()
        )

        if pl_df.is_empty():
            logger.warning("dxy_index.csv produced empty DataFrame")
            return pd.DataFrame()

        con.register("temp_dxy", pl_df)
        con.execute("INSERT OR REPLACE INTO prices SELECT * FROM temp_dxy")
        logger.success(f"Loaded {len(pl_df)} DXY rows from CSV")

        # Return clean DXY DataFrame for feature engineering
        dxy_df = raw[['timestamp', 'close']].copy()
        dxy_df = dxy_df.rename(columns={'close': 'dxy'})
        dxy_df = dxy_df.sort_values('timestamp').reset_index(drop=True)
        return dxy_df

    except Exception as e:
        logger.error(f"Error loading dxy_index.csv: {e}")
        return pd.DataFrame()


if __name__ == '__main__':
    from src.database import get_connection, create_schema

    logger.info("Starting CSV Loader")

    con = get_connection()
    create_schema(con)

    load_from_csv(con)
    dxy_df = load_dxy_from_csv(con)
    logger.info(f"DXY DataFrame shape: {dxy_df.shape}")
    logger.info(f"\n{dxy_df.head()}")

    logger.success("CSV loading process finished.")