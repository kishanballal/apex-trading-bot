"""Free 1-minute crypto candles from Binance's public market-data API (no key needed).

Besides OHLCV we keep two free order-flow fields Binance includes in every candle:
the number of trades and the taker-buy volume (how much of the volume was aggressive buying).
Files are cached as gzip CSV under data/intraday/<SYMBOL>_1m.csv.gz and only the missing
tail is downloaded on later runs.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://data-api.binance.vision"
PAGE = 1000
MINUTE_MS = 60_000
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "trades", "taker_buy"]


def _rows(page: list) -> list[tuple]:
    return [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
             int(k[8]), float(k[9])) for k in page]


def download(symbol: str, start_ms: int, end_ms: int, client: httpx.Client,
             retries: int = 4) -> pd.DataFrame:
    rows: list[tuple] = []
    cursor = start_ms
    while cursor < end_ms:
        for attempt in range(retries):
            try:
                r = client.get(f"{BASE_URL}/api/v3/klines", params={
                    "symbol": symbol, "interval": "1m", "startTime": cursor,
                    "endTime": end_ms - 1, "limit": PAGE})
                r.raise_for_status()
                page = r.json()
                break
            except (httpx.HTTPError, ValueError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if not page:
            break
        rows.extend(_rows(page))
        nxt = int(page[-1][0]) + MINUTE_MS
        if nxt <= cursor:
            break
        cursor = nxt
    return pd.DataFrame(rows, columns=COLUMNS)


def cache_path(root: Path, symbol: str) -> Path:
    return root / f"{symbol}_1m.csv.gz"


def load_or_fetch(symbol: str, days: int, root: Path = Path("data/intraday"),
                  client: httpx.Client | None = None) -> pd.DataFrame:
    """Return 1-minute candles for the last `days` days, using and extending the local cache."""
    root.mkdir(parents=True, exist_ok=True)
    path = cache_path(root, symbol)
    now_ms = int(datetime.now(UTC).timestamp() // 60 * 60 * 1000)
    end_ms = now_ms - MINUTE_MS  # last fully closed minute
    start_ms = end_ms - days * 1440 * MINUTE_MS
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        parts = [old]
        if old.empty or old.ts.min() > start_ms:
            stop = int(old.ts.min()) if not old.empty else end_ms
            parts.append(download(symbol, start_ms, stop, client))
        if not old.empty and old.ts.max() + MINUTE_MS < end_ms:
            parts.append(download(symbol, int(old.ts.max()) + MINUTE_MS, end_ms, client))
    finally:
        if own:
            client.close()
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df[df.ts >= start_ms].reset_index(drop=True)
