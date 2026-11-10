"""Binance public market data. No account or API key is needed for candles."""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import polars as pl

from apex.data.base import INTERVALS, DataError, candles_from_rows, get_json

PAGE_LIMIT = 1000


class BinanceProvider:
    name = "binance"
    full_refresh = False

    def __init__(self, client: httpx.AsyncClient, base_url: str, retries: int = 3,
                 backoff: float = 1.0) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._retries = retries
        self._backoff = backoff

    def bar_duration(self, interval: str) -> timedelta:
        return INTERVALS[interval]

    async def fetch_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        if interval not in INTERVALS:
            raise DataError(f"binance: unsupported interval {interval}")
        step_ms = int(INTERVALS[interval].total_seconds() * 1000)
        cursor, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        rows: list[tuple] = []
        while cursor < end_ms:
            page = await get_json(
                self._client,
                f"{self._base}/api/v3/klines",
                {"symbol": symbol, "interval": interval, "startTime": cursor,
                 "endTime": end_ms, "limit": PAGE_LIMIT},
                retries=self._retries,
                backoff=self._backoff,
            )
            if not isinstance(page, list):
                raise DataError(f"binance: {symbol}: unexpected response {str(page)[:120]}")
            if not page:
                break
            rows.extend(
                (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                for k in page
            )
            next_cursor = int(page[-1][0]) + step_ms
            if len(page) < PAGE_LIMIT or next_cursor <= cursor:
                break
            cursor = next_cursor
        return candles_from_rows(rows)
