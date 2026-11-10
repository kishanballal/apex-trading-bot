from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from apex.data.base import candles_from_rows

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def make_candles(
    n: int = 320,
    *,
    drift: float = 0.0015,
    vol: float = 0.010,
    seed: int = 7,
    start_price: float = 100.0,
    volume: float = 2_000_000.0,
    end: datetime = NOW - timedelta(days=1),
    breakout: bool = False,
) -> pl.DataFrame:
    """Synthetic daily candles ending at `end`. With breakout=True the last bar is a
    high-volume close above the prior 20-day high."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    span = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volumes = volume * rng.uniform(0.8, 1.2, n)
    if breakout:
        close[-1] = high[-21:-1].max() * 1.004
        high[-1] = close[-1] * 1.002
        low[-1] = min(low[-1], open_[-1] * 0.999)
        volumes[-1] = volume * 2.0
    rows = [
        (int((end - timedelta(days=n - 1 - i)).timestamp() * 1000),
         open_[i], high[i], low[i], close[i], volumes[i])
        for i in range(n)
    ]
    return candles_from_rows(rows)


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def uptrend() -> pl.DataFrame:
    return make_candles(breakout=True)
