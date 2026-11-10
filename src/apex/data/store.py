"""SQLite candle store. Zero setup today; the interface is small enough to move to Postgres."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import polars as pl

from apex.data.base import candles_from_rows

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol   TEXT    NOT NULL,
    interval TEXT    NOT NULL,
    ts       INTEGER NOT NULL,  -- bar open, epoch milliseconds UTC
    open     REAL    NOT NULL,
    high     REAL    NOT NULL,
    low      REAL    NOT NULL,
    close    REAL    NOT NULL,
    volume   REAL    NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
) WITHOUT ROWID;
"""


class CandleStore:
    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def upsert(self, symbol: str, interval: str, frame: pl.DataFrame, *,
               replace: bool = False) -> int:
        """Write candles. With replace=True the symbol's history is swapped atomically."""
        rows = [
            (symbol, interval, int(ts.timestamp() * 1000), o, h, l, c, v)
            for ts, o, h, l, c, v in frame.select(
                "ts", "open", "high", "low", "close", "volume"
            ).iter_rows()
        ]
        with self._conn:  # one transaction: either everything lands or nothing does
            if replace and rows:
                self._conn.execute(
                    "DELETE FROM candles WHERE symbol = ? AND interval = ?", (symbol, interval)
                )
            self._conn.executemany(
                "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(symbol, interval, ts) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume",
                rows,
            )
        return len(rows)

    def load(self, symbol: str, interval: str, limit: int | None = None) -> pl.DataFrame:
        sql = (
            "SELECT ts, open, high, low, close, volume FROM candles "
            "WHERE symbol = ? AND interval = ? ORDER BY ts DESC"
        )
        params: tuple = (symbol, interval)
        if limit:
            sql += " LIMIT ?"
            params += (limit,)
        with closing(self._conn.execute(sql, params)) as cur:
            rows = cur.fetchall()
        return candles_from_rows(rows[::-1])

    def last_ts_ms(self, symbol: str, interval: str) -> int | None:
        with closing(self._conn.execute(
            "SELECT MAX(ts) FROM candles WHERE symbol = ? AND interval = ?", (symbol, interval)
        )) as cur:
            return cur.fetchone()[0]

    def symbols(self, interval: str) -> list[str]:
        with closing(self._conn.execute(
            "SELECT DISTINCT symbol FROM candles WHERE interval = ? ORDER BY symbol", (interval,)
        )) as cur:
            return [r[0] for r in cur.fetchall()]
