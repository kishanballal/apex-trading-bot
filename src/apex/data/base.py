"""Provider contract plus helpers shared by all market data providers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Protocol

import httpx
import pandas as pd

INTERVALS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """A provider could not return usable data. Callers skip the symbol; they never invent data."""


def empty_candles() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDLE_COLUMNS})
    frame["ts"] = pd.to_datetime(frame["ts"], unit="ms", utc=True)
    return frame


def candles_from_rows(rows: list[tuple]) -> pd.DataFrame:
    """rows: (epoch_ms, open, high, low, close, volume)."""
    if not rows:
        return empty_candles()
    frame = pd.DataFrame(rows, columns=CANDLE_COLUMNS)
    for c in CANDLE_COLUMNS[1:]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True)
    return frame


class MarketDataProvider(Protocol):
    name: str
    # True when history is restated (splits, dividends) and must be replaced, not appended.
    full_refresh: bool

    def bar_duration(self, interval: str) -> timedelta:
        """How long after `ts` a bar is final. Unfinished bars must never reach the scanner."""
        ...

    async def fetch_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pd.DataFrame: ...


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    *,
    retries: int = 3,
    backoff: float = 1.0,
):
    """GET with exponential backoff on transport errors, 429 and 5xx. Other 4xx fail at once."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                raise DataError(f"HTTP {response.status_code} from {response.url.host}")
            if response.status_code >= 400:
                raise _Fatal(f"HTTP {response.status_code} from {response.url.host}")
            return response.json()
        except _Fatal as exc:
            raise DataError(str(exc)) from None
        except (httpx.TransportError, DataError, ValueError) as exc:
            last = exc
            if attempt < retries:
                await asyncio.sleep(backoff * 2**attempt)
    raise DataError(f"gave up after {retries + 1} attempts: {last}")


class _Fatal(Exception):
    pass
