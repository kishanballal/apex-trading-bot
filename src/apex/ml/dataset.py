"""Supervised dataset for trade-outcome models.

One sample is "what if we bought this symbol at the next open after this bar", with the same
2 ATR stop and 2R target the scanner uses. Features use bars up to and including the signal bar.
The label uses bars strictly after it. `label_end` records when the outcome became known, so
training can exclude samples whose outcome was not yet visible at the time (purging).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apex.analysis.indicators import prepare
from apex.config import CostModel, UniverseGroup
from apex.frame import dates, matrix, records
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


def _feature_frame(candles: pd.DataFrame) -> pd.DataFrame:
    f = prepare(candles)
    c = f["close"]
    for n in (1, 5, 10, 20, 60, 120, 250):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["dist_ema20"] = c / f["ema20"] - 1
    f["dist_ema50"] = c / f["ema50"] - 1
    f["dist_ema200"] = c / f["ema200"] - 1
    f["ema50_vs_200"] = f["ema50"] / f["ema200"] - 1
    f["ema200_slope"] = f["ema200"] / f["ema200_prev"] - 1
    f["macd_rel"] = f["macd_hist"] / c
    f["log_turnover"] = np.log(f["turnover20"])
    f["dist_high20"] = c / f["prior_high20"] - 1
    f["dist_high250"] = c / f["high"].rolling(250).max() - 1
    f["realized_vol20"] = (c / c.shift(1) - 1).rolling(20).std()
    f["up_day"] = (c > f["open"]).astype("float64")
    span = f["high"] - f["low"]
    f["range_pos"] = ((c - f["low"]) / span.where(span > 0)).where(span > 0, 0.5)
    return f


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
    candles: dict[str, pd.DataFrame],
    *,
    label: LabelConfig | None = None,
    scanner: ScannerConfig | None = None,
    costs: CostModel | None = None,
) -> pd.DataFrame:
    label, scanner, costs = label or LabelConfig(), scanner or ScannerConfig(), costs or CostModel()
    parts: list[pd.DataFrame] = []
    for group in groups:
        bench = candles.get(group.benchmark)
        if bench is None or bench.empty:
            continue
        b = prepare(bench)
        bc = b["close"]
        bench_feats = pd.DataFrame({
            "date": dates(b["ts"]),
            "bench_ret_20": bc / bc.shift(20) - 1,
            "bench_ret_60": bc / bc.shift(60) - 1,
            "bench_dist_ema200": bc / b["ema200"] - 1,
            "bench_vol_rank": b["vol_rank"],
        }).drop_duplicates("date", keep="last")
        side_cost = costs.per_side(group.asset_class)
        for symbol in group.symbols:
            raw = candles.get(symbol)
            if raw is None or len(raw) < scanner.min_bars + label.horizon:
                continue
            frame = _feature_frame(raw)
            rows = records(frame[["ts", "open", "high", "low", "close", "atr14"]])
            wins, rs, ends = [], [], []
            for i in range(len(rows)):
                out = label_outcome(rows, i, label, side_cost) if i + 1 >= scanner.min_bars else None
                wins.append(None if out is None else float(out[0]))
                rs.append(None if out is None else out[1])
                ends.append(None if out is None else rows[out[2]]["ts"])
            frame["y"] = pd.Series(wins, index=frame.index, dtype="float64")
            frame["r"] = pd.Series(rs, index=frame.index, dtype="float64")
            frame["label_end"] = pd.to_datetime(pd.Series(ends, index=frame.index), utc=True)
            frame["symbol"] = symbol
            frame["asset_class"] = group.asset_class.value
            frame["is_crypto"] = 1.0 if group.asset_class is AssetClass.CRYPTO else 0.0
            frame["date"] = dates(frame["ts"])
            frame = frame.merge(bench_feats, on="date", how="left")
            frame["rel_strength_60"] = frame["ret_60"] - frame["bench_ret_60"]
            frame = frame[frame["turnover20"] >= scanner.min_turnover[group.asset_class]]
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True)
    # Cross-sectional ranks use only that day's values across symbols of the same asset class.
    grouped = data.groupby(["date", "asset_class"])
    for c in ("ret_60", "ret_120"):
        data[f"rank_{c}"] = grouped[c].rank() / grouped[c].transform("count")
    keep = ["symbol", "asset_class", "ts", "date", "label_end", "y", "r", *FEATURES]
    return data[keep].sort_values(["ts", "symbol"], kind="stable").reset_index(drop=True)


def to_matrix(data: pd.DataFrame) -> np.ndarray:
    return matrix(data, FEATURES)
