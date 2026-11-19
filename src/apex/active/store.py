"""SQLite storage for the active bot: 5-minute bars, headlines and account settings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from apex.active.news import Item, match_symbols, score

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
  first_seen INTEGER, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS news (
  uid TEXT PRIMARY KEY, source TEXT, title TEXT, published_ms INTEGER, fetched_ms INTEGER,
  url TEXT, score REAL);
CREATE TABLE IF NOT EXISTS news_symbols (uid TEXT, symbol TEXT, PRIMARY KEY (uid, symbol));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class ActiveStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def upsert_bars(self, symbol: str, df: pd.DataFrame, seen_ms: int) -> int:
        """Insert new bars (first_seen = now) and refresh values of known ones. Returns new."""
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO bars VALUES (?,?,?,?,?,?,?,?)",
                [(symbol, int(r.ts), r.open, r.high, r.low, r.close, r.volume, seen_ms)
                 for r in df.itertuples(index=False)])
            new = self.db.total_changes - before
            self.db.executemany(
                "UPDATE bars SET open=?, high=?, low=?, close=?, volume=? WHERE symbol=? AND ts=?",
                [(r.open, r.high, r.low, r.close, r.volume, symbol, int(r.ts))
                 for r in df.itertuples(index=False)])
        return new

    def bars(self, symbol: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume, first_seen FROM bars WHERE symbol=? "
            "ORDER BY ts", self.db, params=(symbol,))

    def bars_between(self, lo_ms: int, hi_ms: int) -> pd.DataFrame:
        """All stocks' bars with lo <= ts < hi, sorted by symbol then time."""
        return pd.read_sql_query(
            "SELECT symbol, ts, open, high, low, close, volume, first_seen FROM bars "
            "WHERE ts >= ? AND ts < ? ORDER BY symbol, ts", self.db, params=(lo_ms, hi_ms))

    def session_days(self) -> list[int]:
        """IST day numbers that have bars, oldest first."""
        cur = self.db.execute("SELECT DISTINCT (ts + 19800000) / 86400000 FROM bars ORDER BY 1")
        return [int(r[0]) for r in cur]

    def latest_ts(self) -> int:
        return int(self.db.execute("SELECT COALESCE(MAX(ts), 0) FROM bars").fetchone()[0])

    def symbols(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT symbol FROM bars")]

    def add_news(self, items: list[Item], symbols: list[str], fetched_ms: int,
                 force_symbol: str | None = None, scorer=None) -> int:
        """Store new headlines, tagged with the stocks they actually name.

        Google News searches return plenty of unrelated stories (the first live day tagged an
        Nvidia CEO salary story to Eicher Motors), so a search result only counts for the
        searched stock when its title names that stock. `force_symbol` is kept for callers
        but no longer tags anything on its own."""
        added = 0
        known = {r[0] for r in self.db.execute(
            "SELECT uid FROM news WHERE uid IN (%s)" % ",".join("?" * len(items)),
            [it.uid for it in items])} if items else set()
        items = [it for it in items if it.uid not in known]
        tones = scorer([it.title for it in items]) if scorer else [score(it.title) for it in items]
        with self.db:
            for it, tone in zip(items, tones):
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?)",
                    (it.uid, it.source, it.title, it.published_ms, fetched_ms, it.url, tone))
                if cur.rowcount == 0:
                    continue
                added += 1
                matched = set(match_symbols(it.title, symbols))
                self.db.executemany("INSERT OR IGNORE INTO news_symbols VALUES (?, ?)",
                                    [(it.uid, s) for s in matched])
        return added

    def news(self, since_ms: int = 0) -> pd.DataFrame:
        """One row per (headline, stock) pair."""
        return pd.read_sql_query(
            "SELECT s.symbol, n.source, n.title, n.published_ms, n.fetched_ms, n.score, n.url "
            "FROM news n JOIN news_symbols s ON s.uid = n.uid WHERE n.fetched_ms >= ? "
            "ORDER BY n.fetched_ms", self.db, params=(since_ms,))

    def latest_news(self, limit: int = 60) -> list[dict]:
        cur = self.db.execute(
            "SELECT n.source, n.title, n.published_ms, n.fetched_ms, n.score, n.url, "
            "GROUP_CONCAT(s.symbol) FROM news n LEFT JOIN news_symbols s ON s.uid = n.uid "
            "GROUP BY n.uid ORDER BY COALESCE(NULLIF(n.published_ms, 0), n.fetched_ms) DESC "
            "LIMIT ?", (limit,))
        keys = ["source", "title", "published_ms", "fetched_ms", "score", "url", "symbols"]
        return [dict(zip(keys, r)) for r in cur]
