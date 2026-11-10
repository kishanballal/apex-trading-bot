"""Market regime from a benchmark index. Deliberately simple and explainable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from apex.analysis.indicators import add_indicators

MIN_BARS = 220


class Trend(StrEnum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Regime:
    trend: Trend
    high_volatility: bool
    detail: str

    @property
    def allows_longs(self) -> bool:
        return self.trend is Trend.BULL and not self.high_volatility


def percentile_rank(values: pl.Series, lookback: int = 252) -> float | None:
    """Where the latest value sits among the previous `lookback` values, 0 to 1."""
    window = values.drop_nulls().tail(lookback)
    if window.len() < 60:
        return None
    return float((window < window[-1]).sum()) / (window.len() - 1)


def detect_regime(benchmark: pl.DataFrame) -> Regime:
    if benchmark.height < MIN_BARS:
        return Regime(Trend.UNKNOWN, False, f"only {benchmark.height} benchmark bars")
    frame = add_indicators(benchmark)
    last = frame.row(-1, named=True)
    close, ema50, ema200 = last["close"], last["ema50"], last["ema200"]
    if close > ema200 and ema50 > ema200:
        trend = Trend.BULL
    elif close < ema200 and ema50 < ema200:
        trend = Trend.BEAR
    else:
        trend = Trend.SIDEWAYS
    rank = percentile_rank(frame["atr_pct"])
    high_vol = rank is not None and rank >= 0.90
    rank_text = f"{rank:.0%}" if rank is not None else "n/a"
    detail = (
        f"close {close:.2f} vs EMA200 {ema200:.2f}, EMA50 {ema50:.2f}, "
        f"volatility rank {rank_text}"
    )
    return Regime(trend, high_vol, detail)
