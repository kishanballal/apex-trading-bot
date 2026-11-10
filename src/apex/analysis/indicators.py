"""Technical indicators on pandas frames.

Every value on a row uses only that row and earlier rows, so nothing leaks from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_RANK_LOOKBACK = 252
VOL_RANK_MIN_SAMPLES = 60


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder((-delta).clip(lower=0), period)
    out = 100.0 - 100.0 / (1.0 + gain / loss.where(loss != 0))
    out = out.where(loss != 0, np.where(gain == 0, 50.0, 100.0))
    return out.where(gain.notna() & loss.notna())


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return _wilder(true_range(frame), period)


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    tr = _wilder(true_range(frame), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / tr
    minus_di = 100.0 * _wilder(minus_dm, period) / tr
    total = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / total.where(total != 0))
    dx = dx.where(total != 0, 0.0).where(total.notna())
    return _wilder(dx, period)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the indicator columns the scanner uses. Input must be sorted by ts."""
    f = frame.copy()
    close = f["close"]
    f["ema20"] = ema(close, 20)
    f["ema50"] = ema(close, 50)
    f["ema200"] = ema(close, 200)
    f["rsi14"] = rsi(close, 14)
    f["atr14"] = atr(f, 14)
    f["adx14"] = adx(f, 14)
    f["macd_hist"] = macd_hist(close)
    # "prior" windows exclude the current bar, so a breakout is measured against the past.
    f["vol_ratio"] = f["volume"] / f["volume"].shift(1).rolling(20).mean()
    f["prior_high20"] = f["high"].shift(1).rolling(20).max()
    f["turnover20"] = (close * f["volume"]).rolling(20).mean()
    f["atr_pct"] = f["atr14"] / close
    f["ema200_prev"] = f["ema200"].shift(20)
    return f


def rolling_percentile_rank(
    values: pd.Series, lookback: int = VOL_RANK_LOOKBACK, min_samples: int = VOL_RANK_MIN_SAMPLES
) -> pd.Series:
    """For each row, where its value sits among the last `lookback` values up to and
    including itself (0 to 1). Null until `min_samples` values exist. Causal by construction."""
    data = values.to_numpy(dtype=float, na_value=np.nan)
    out: list[float | None] = [None] * len(data)
    valid = np.flatnonzero(~np.isnan(data))
    if len(valid) == 0:
        return pd.Series(out, index=values.index, name=values.name, dtype="float64")
    first = int(valid[0])
    for i in range(first, len(data)):
        if np.isnan(data[i]):
            continue
        window = data[max(first, i - lookback + 1): i + 1]
        window = window[~np.isnan(window)]
        if len(window) >= min_samples:
            out[i] = float((window < window[-1]).sum()) / (len(window) - 1)
    return pd.Series(out, index=values.index, name=values.name, dtype="float64")


def prepare(candles: pd.DataFrame) -> pd.DataFrame:
    """Sorted candles with every column the scanner, regime detector and backtester need."""
    frame = add_indicators(candles.sort_values("ts").reset_index(drop=True))
    frame["vol_rank"] = rolling_percentile_rank(frame["atr_pct"])
    return frame
