"""Market regime from a benchmark index. Deliberately simple and explainable."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from apex.frame import records

from apex.analysis.indicators import prepare

MIN_BARS = 220
HIGH_VOL_RANK = 0.90


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


UNKNOWN = Regime(Trend.UNKNOWN, False, "no benchmark data")


def regime_at(rows: list[dict], i: int) -> Regime:
    """Regime as known at the close of bar `i` of a prepared benchmark. Uses bars 0..i only."""
    if i + 1 < MIN_BARS:
        return Regime(Trend.UNKNOWN, False, f"only {max(i + 1, 0)} benchmark bars")
    last = rows[i]
    close, ema50, ema200 = last["close"], last["ema50"], last["ema200"]
    if close > ema200 and ema50 > ema200:
        trend = Trend.BULL
    elif close < ema200 and ema50 < ema200:
        trend = Trend.BEAR
    else:
        trend = Trend.SIDEWAYS
    rank = last["vol_rank"]
    rank_text = f"{rank:.0%}" if rank is not None else "n/a"
    detail = (
        f"close {close:.2f} vs EMA200 {ema200:.2f}, EMA50 {ema50:.2f}, "
        f"volatility rank {rank_text}"
    )
    return Regime(trend, rank is not None and rank >= HIGH_VOL_RANK, detail)


def detect_regime(benchmark: pd.DataFrame) -> Regime:
    rows = records(prepare(benchmark)) if len(benchmark) else []
    return regime_at(rows, len(rows) - 1)
