"""Yahoo Finance chart API: free, unofficial, fine for end-of-day NSE data in paper trading."""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import quote

import httpx
import polars as pl

from apex.data.base import INTERVALS, DataError, candles_from_rows, get_json

_YAHOO_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
NSE_SESSION = timedelta(hours=6, minutes=15)  # 09:15 to 15:30 IST


def to_yahoo_symbol(symbol: str) -> str:
    if symbol.startswith("^") or "." in symbol or "=" in symbol:
        return symbol
    return f"{symbol}.NS"


class YahooProvider:
    name = "yahoo"
    full_refresh = True  # prices are split/dividend adjusted, so history gets restated

    def __init__(self, client: httpx.AsyncClient, base_url: str, retries: int = 3,
                 backoff: float = 1.0) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._retries = retries
        self._backoff = backoff

    def bar_duration(self, interval: str) -> timedelta:
        # A daily bar opens at 09:15 IST and is final once the session ends.
        return NSE_SESSION if interval == "1d" else INTERVALS[interval]

    async def fetch_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        if interval not in _YAHOO_INTERVAL:
            raise DataError(f"yahoo: unsupported interval {interval}")
        url = f"{self._base}/v8/finance/chart/{quote(to_yahoo_symbol(symbol), safe='^.=-')}"
        params = {
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": _YAHOO_INTERVAL[interval],
            "events": "div,splits",
        }
        payload = await get_json(
            self._client, url, params, retries=self._retries, backoff=self._backoff
        )
        return parse_chart(payload, symbol)


def parse_chart(payload: dict, symbol: str) -> pl.DataFrame:
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise DataError(f"yahoo: {symbol}: {chart['error'].get('description', chart['error'])}")
    results = chart.get("result") or []
    if not results:
        raise DataError(f"yahoo: {symbol}: empty result")
    result = results[0]
    stamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    if not stamps or not quotes.get("close"):
        raise DataError(f"yahoo: {symbol}: no candles")

    rows = []
    for i, stamp in enumerate(stamps):
        o, h, l, c, v = (quotes[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c) or c <= 0:
            continue  # Yahoo emits null bars on holidays and for the in-progress bar
        factor = (adj[i] / c) if adj and adj[i] else 1.0
        rows.append((stamp * 1000, o * factor, h * factor, l * factor, c * factor, v or 0.0))
    return candles_from_rows(rows)
