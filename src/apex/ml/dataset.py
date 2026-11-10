"""Supervised dataset for trade-outcome models.

One sample is "what if we bought this symbol at the next open after this bar", with the same
2 ATR stop and 2R target the scanner uses. Features use bars up to and including the signal bar.
The label uses bars strictly after it. `label_end` records when the outcome became known, so
training can exclude samples whose outcome was not yet visible at the time (purging).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from apex.analysis.indicators import prepare
from apex.config import CostModel, UniverseGroup
from apex.models import AssetClass
from apex.scanner.scanner import ScannerConfig

HORIZON = 40
FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "ret_120", "ret_250",
    "dist_ema20", "dist_ema50", "dist_ema200", "ema50_vs_200", "ema200_slope",
    "rsi14", "adx14", "macd_rel", "atr_pct", "vol_rank", "vol_ratio", "log_turnover",
    "dist_high20", "dist_high250", "realized_vol20", "up_day", "range_pos",
    "bench_ret_20", "bench_ret_60", "bench_dist_ema200", "bench_vol_rank",
    "rel_strength_60", "rank_ret_60", "rank_ret_120", "is_crypto",
]


@dataclass(frozen=True)
class LabelConfig:
    stop_atr: float = 2.0
    target_r: float = 2.0
    horizon: int = HORIZON


def _feature_frame(candles: pl.DataFrame) -> pl.DataFrame:
    f = prepare(candles)
    c = pl.col("close")
    return f.with_columns(
        *[(c / c.shift(n) - 1).alias(f"ret_{n}") for n in (1, 5, 10, 20, 60, 120, 250)],
        (c / pl.col("ema20") - 1).alias("dist_ema20"),
        (c / pl.col("ema50") - 1).alias("dist_ema50"),
        (c / pl.col("ema200") - 1).alias("dist_ema200"),
        (pl.col("ema50") / pl.col("ema200") - 1).alias("ema50_vs_200"),
        (pl.col("ema200") / pl.col("ema200_prev") - 1).alias("ema200_slope"),
        (pl.col("macd_hist") / c).alias("macd_rel"),
        (pl.col("turnover20").log()).alias("log_turnover"),
        (c / pl.col("prior_high20") - 1).alias("dist_high20"),
        (c / pl.col("high").rolling_max(250) - 1).alias("dist_high250"),
        (c / c.shift(1) - 1).rolling_std(20).alias("realized_vol20"),
        (c > pl.col("open")).cast(pl.Float64).alias("up_day"),
        pl.when(pl.col("high") > pl.col("low"))
        .then((c - pl.col("low")) / (pl.col("high") - pl.col("low"))).otherwise(0.5)
        .alias("range_pos"),
    )


def label_outcome(rows: list[dict], i: int, cfg: LabelConfig, side_cost: float):
    """Outcome of entering at bar i+1's open. Returns (win, net R multiple, exit index) or None
    when the trade could not have been taken."""
    if i + 1 >= len(rows):
        return None
    sig = rows[i]
    atr = sig["atr14"]
    if atr is None or atr <= 0:
        return None
    stop = sig["close"] - cfg.stop_atr * atr
    target = sig["close"] + cfg.target_r * cfg.stop_atr * atr
    entry = rows[i + 1]["open"]
    if not stop < entry < target:
        return None
    exit_price, j, win = None, i + 1, False
    last = min(i + cfg.horizon, len(rows) - 1)
    for j in range(i + 1, last + 1):
        bar = rows[j]
        if bar["low"] <= stop:
            exit_price = min(stop, bar["open"])
            break
        if bar["high"] >= target:
            exit_price, win = max(target, bar["open"]), True
            break
    if exit_price is None:
        if last < i + cfg.horizon:
            return None  # outcome not known yet: never guess a label
        exit_price = rows[last]["close"]
    planned = (entry - stop) + side_cost * (entry + stop)
    net = (exit_price - entry) - side_cost * (entry + exit_price)
    return win, net / planned, j


def build_dataset(
    groups: list[UniverseGroup],
    candles: dict[str, pl.DataFrame],
    *,
    label: LabelConfig | None = None,
    scanner: ScannerConfig | None = None,
    costs: CostModel | None = None,
) -> pl.DataFrame:
    label, scanner, costs = label or LabelConfig(), scanner or ScannerConfig(), costs or CostModel()
    parts: list[pl.DataFrame] = []
    for group in groups:
        bench = candles.get(group.benchmark)
        if bench is None or bench.is_empty():
            continue
        b = prepare(bench)
        bc = pl.col("close")
        bench_feats = b.select(
            pl.col("ts").dt.date().alias("date"),
            (bc / bc.shift(20) - 1).alias("bench_ret_20"),
            (bc / bc.shift(60) - 1).alias("bench_ret_60"),
            (bc / pl.col("ema200") - 1).alias("bench_dist_ema200"),
            pl.col("vol_rank").alias("bench_vol_rank"),
        )
        side_cost = costs.per_side(group.asset_class)
        for symbol in group.symbols:
            raw = candles.get(symbol)
            if raw is None or raw.height < scanner.min_bars + label.horizon:
                continue
            frame = _feature_frame(raw)
            rows = frame.to_dicts()
            wins, rs, ends = [], [], []
            for i in range(len(rows)):
                out = label_outcome(rows, i, label, side_cost) if i + 1 >= scanner.min_bars else None
                wins.append(None if out is None else float(out[0]))
                rs.append(None if out is None else out[1])
                ends.append(None if out is None else rows[out[2]]["ts"])
            frame = frame.with_columns(
                pl.Series("y", wins, dtype=pl.Float64),
                pl.Series("r", rs, dtype=pl.Float64),
                pl.Series("label_end", ends, dtype=frame.schema["ts"]),
                pl.lit(symbol).alias("symbol"),
                pl.lit(group.asset_class.value).alias("asset_class"),
                pl.lit(1.0 if group.asset_class is AssetClass.CRYPTO else 0.0).alias("is_crypto"),
                pl.col("ts").dt.date().alias("date"),
            ).join(bench_feats, on="date", how="left").with_columns(
                (pl.col("ret_60") - pl.col("bench_ret_60")).alias("rel_strength_60"),
            ).filter(pl.col("turnover20") >= scanner.min_turnover[group.asset_class])
            parts.append(frame)
    if not parts:
        return pl.DataFrame()
    data = pl.concat(parts, how="vertical_relaxed")
    # Cross-sectional ranks use only that day's values across symbols of the same asset class.
    data = data.with_columns(
        *[(pl.col(c).rank() / pl.col(c).count()).over("date", "asset_class").alias(f"rank_{c}")
          for c in ("ret_60", "ret_120")]
    )
    keep = ["symbol", "asset_class", "ts", "date", "label_end", "y", "r", *FEATURES]
    return data.select(keep).sort("ts", "symbol")


def to_matrix(data: pl.DataFrame) -> np.ndarray:
    return data.select(FEATURES).fill_null(math.nan).to_numpy().astype(np.float64)
