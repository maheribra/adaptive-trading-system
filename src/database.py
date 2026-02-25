import duckdb
from loguru import logger
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / 'data' / 'trading_system.duckdb'

def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))

def create_schema(con: duckdb.DuckDBPyConnection):
    con.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol    VARCHAR      NOT NULL,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            volume    DOUBLE,
            PRIMARY KEY (timestamp, symbol)
        )
    ''')
    con.execute('''
        CREATE TABLE IF NOT EXISTS news_events (
            event_time TIMESTAMPTZ NOT NULL,
            title      VARCHAR      NOT NULL,
            currency   VARCHAR,
            impact     VARCHAR,
            actual     VARCHAR,
            forecast   VARCHAR,
            previous   VARCHAR,
            deviation  DOUBLE,
            PRIMARY KEY (event_time, title)
        )
    ''')
    con.execute('''
        CREATE TABLE IF NOT EXISTS dxy_signals (
            timestamp      TIMESTAMPTZ NOT NULL PRIMARY KEY,
            close          DOUBLE,
            dxy_pct_change DOUBLE,
            dxy_ma5        DOUBLE,
            dxy_bullish    BOOLEAN
        )
    ''')
    con.execute('''
        CREATE TABLE IF NOT EXISTS market_regimes (
            timestamp  TIMESTAMPTZ NOT NULL PRIMARY KEY,
            regime     VARCHAR,
            confidence DOUBLE
        )
    ''')
    logger.success('Schema created successfully')

if __name__ == '__main__':
    con = get_connection()
    create_schema(con)
    print('Schema ready:', DB_PATH)