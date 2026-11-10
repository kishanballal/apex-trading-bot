"""Technical indicators as Polars expressions.

Every value on a row uses only that row and earlier rows, so nothing leaks from the future.
"""

from __future__ import annotations

import polars as pl


def ema(column: str, span: int) -> pl.Expr:
    return pl.col(column).ewm_mean(span=span, adjust=False, min_samples=span)


def _wilder(expr: pl.Expr, period: int) -> pl.Expr:
    return expr.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)


def rsi(period: int = 14) -> pl.Expr:
    delta = pl.col("close").diff()
    gain = _wilder(delta.clip(lower_bound=0), period)
    loss = _wilder((-delta).clip(lower_bound=0), period)
    return (
        pl.when(loss == 0).then(pl.when(gain == 0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + gain / loss))
    )


def true_range() -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


def atr(period: int = 14) -> pl.Expr:
    return _wilder(true_range(), period)


def adx(period: int = 14) -> pl.Expr:
    up = pl.col("high").diff()
    down = -pl.col("low").diff()
    plus_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    tr = _wilder(true_range(), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / tr
    minus_di = 100.0 * _wilder(minus_dm, period) / tr
    total = plus_di + minus_di
    dx = pl.when(total == 0).then(0.0).otherwise(100.0 * (plus_di - minus_di).abs() / total)
    return _wilder(dx, period)


def macd_hist(fast: int = 12, slow: int = 26, signal: int = 9) -> pl.Expr:
    line = ema("close", fast) - ema("close", slow)
    return line - line.ewm_mean(span=signal, adjust=False, min_samples=signal)


def add_indicators(frame: pl.DataFrame) -> pl.DataFrame:
    """Append the indicator columns the scanner uses. Input must be sorted by ts."""
    return frame.with_columns(
        ema("close", 20).alias("ema20"),
        ema("close", 50).alias("ema50"),
        ema("close", 200).alias("ema200"),
        rsi(14).alias("rsi14"),
        atr(14).alias("atr14"),
        adx(14).alias("adx14"),
        macd_hist().alias("macd_hist"),
        # "prior" windows exclude the current bar, so a breakout is measured against the past.
        (pl.col("volume") / pl.col("volume").shift(1).rolling_mean(20)).alias("vol_ratio"),
        pl.col("high").shift(1).rolling_max(20).alias("prior_high20"),
        (pl.col("close") * pl.col("volume")).rolling_mean(20).alias("turnover20"),
    ).with_columns(
        (pl.col("atr14") / pl.col("close")).alias("atr_pct"),
        pl.col("ema200").shift(20).alias("ema200_prev"),
    )
