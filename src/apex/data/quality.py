"""Candle validation. Bad rows are dropped and counted so problems are visible, not silent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from apex.data.base import CANDLE_SCHEMA

JUMP_WARNING = 0.40  # a one-bar move this large is usually a bad tick or an unadjusted split
_EPS = 1e-9


@dataclass(frozen=True)
class QualityReport:
    rows_in: int
    rows_out: int
    dropped_invalid: int
    dropped_duplicates: int
    dropped_unfinished: int
    suspicious_jumps: int

    @property
    def clean(self) -> bool:
        return self.suspicious_jumps == 0 and self.dropped_invalid == 0


def clean_candles(
    frame: pl.DataFrame, *, now: datetime, bar_duration: timedelta
) -> tuple[pl.DataFrame, QualityReport]:
    """Return sorted, de-duplicated, finished, internally consistent candles."""
    rows_in = frame.height
    frame = frame.select([pl.col(c).cast(t) for c, t in CANDLE_SCHEMA.items()])

    valid = frame.drop_nulls().filter(
        (pl.col("open") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("volume") >= 0)
        & (pl.col("high") * (1 + _EPS) >= pl.max_horizontal("open", "close", "low"))
        & (pl.col("low") * (1 - _EPS) <= pl.min_horizontal("open", "close"))
        & pl.all_horizontal(pl.col("open", "high", "low", "close", "volume").is_finite())
    )
    dropped_invalid = rows_in - valid.height

    unique = valid.unique(subset="ts", keep="last").sort("ts")
    dropped_duplicates = valid.height - unique.height

    # Look-ahead guard: a bar still forming must never be used for a decision.
    finished = unique.filter(pl.col("ts") + bar_duration <= now)
    dropped_unfinished = unique.height - finished.height

    jumps = finished.select(
        ((pl.col("close") / pl.col("close").shift(1) - 1).abs() > JUMP_WARNING).sum()
    ).item() if finished.height > 1 else 0

    return finished, QualityReport(
        rows_in, finished.height, dropped_invalid, dropped_duplicates, dropped_unfinished,
        int(jumps or 0),
    )
