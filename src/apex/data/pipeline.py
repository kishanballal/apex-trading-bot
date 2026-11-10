"""Fetch, validate, store. One failing symbol never stops the rest."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from apex.config import Settings, UniverseGroup
from apex.data.base import INTERVALS, DataError, MarketDataProvider
from apex.data.binance import BinanceProvider
from apex.data.quality import QualityReport, clean_candles
from apex.data.store import CandleStore
from apex.data.yahoo import YahooProvider

log = logging.getLogger(__name__)
OVERLAP_BARS = 3  # re-fetch a few stored bars so late corrections are picked up
USER_AGENT = "Mozilla/5.0 (compatible; apex-research/0.1)"


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    ok: bool
    rows: int = 0
    error: str = ""
    quality: QualityReport | None = None


def build_providers(client: httpx.AsyncClient, settings: Settings) -> dict[str, MarketDataProvider]:
    return {
        "yahoo": YahooProvider(client, settings.yahoo_base_url, settings.http_retries),
        "binance": BinanceProvider(client, settings.binance_base_url, settings.http_retries),
    }


def make_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=settings.http_timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )


async def update_symbol(
    provider: MarketDataProvider,
    store: CandleStore,
    symbol: str,
    interval: str,
    *,
    history_days: int,
    now: datetime | None = None,
) -> FetchResult:
    now = now or datetime.now(UTC)
    start = now - timedelta(days=history_days)
    if not provider.full_refresh:
        last = store.last_ts_ms(symbol, interval)
        if last is not None:
            resume = datetime.fromtimestamp(last / 1000, UTC) - OVERLAP_BARS * INTERVALS[interval]
            start = max(start, resume)
    try:
        raw = await provider.fetch_candles(symbol, interval, start, now)
        frame, report = clean_candles(
            raw, now=now, bar_duration=provider.bar_duration(interval)
        )
        if frame.empty:
            raise DataError(f"{symbol}: no finished candles returned")
        rows = store.upsert(symbol, interval, frame, replace=provider.full_refresh)
        return FetchResult(symbol, True, rows, quality=report)
    except DataError as exc:
        log.warning("fetch failed for %s: %s", symbol, exc)
        return FetchResult(symbol, False, error=str(exc))


async def update_universe(
    groups: list[UniverseGroup],
    providers: dict[str, MarketDataProvider],
    store: CandleStore,
    *,
    interval: str = "1d",
    history_days: int = 730,
    max_concurrency: int = 5,
    now: datetime | None = None,
) -> list[FetchResult]:
    gate = asyncio.Semaphore(max_concurrency)

    async def one(provider: MarketDataProvider, symbol: str) -> FetchResult:
        async with gate:
            return await update_symbol(
                provider, store, symbol, interval, history_days=history_days, now=now
            )

    tasks = []
    for group in groups:
        provider = providers[group.provider]
        for symbol in dict.fromkeys((group.benchmark, *group.symbols)):
            tasks.append(one(provider, symbol))
    return list(await asyncio.gather(*tasks))
