"""Candle validation. Bad rows are dropped and counted so problems are visible, not silent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from apex.data.base import CANDLE_COLUMNS

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
    frame: pd.DataFrame, *, now: datetime, bar_duration: timedelta
) -> tuple[pd.DataFrame, QualityReport]:
    """Return sorted, de-duplicated, finished, internally consistent candles."""
    rows_in = len(frame)
    frame = frame[CANDLE_COLUMNS].copy()
    for c in CANDLE_COLUMNS[1:]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")

    num = frame[CANDLE_COLUMNS[1:]]
    ok = (
        frame["ts"].notna()
        & np.isfinite(num.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
        & (frame["open"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)
        & (frame["volume"] >= 0)
        & (frame["high"] * (1 + _EPS) >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] * (1 - _EPS) <= frame[["open", "close"]].min(axis=1))
    )
    valid = frame[ok]
    dropped_invalid = rows_in - len(valid)

    unique = valid.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    dropped_duplicates = len(valid) - len(unique)

    # Look-ahead guard: a bar still forming must never be used for a decision.
    finished = unique[unique["ts"] + bar_duration <= pd.Timestamp(now)].reset_index(drop=True)
    dropped_unfinished = len(unique) - len(finished)

    jumps = int(((finished["close"] / finished["close"].shift(1) - 1).abs() > JUMP_WARNING).sum()) \
        if len(finished) > 1 else 0

    return finished, QualityReport(
        rows_in, len(finished), dropped_invalid, dropped_duplicates, dropped_unfinished,
        int(jumps or 0),
    )
