"""Minute-bar features. Every value at row t uses only bars up to and including t."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "r1", "r5", "r15", "r60", "r240", "vol15", "vol60", "vol_ratio", "z60", "z240", "rsi14",
    "range_pos_1440", "volu_ratio", "taker_ratio", "taker_ratio15", "trades_ratio",
    "atr_pct", "hour_sin", "hour_cos", "btc_r5", "btc_r60", "rel_r60",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(df: pd.DataFrame, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add indicator and feature columns to a 1-minute candle frame (sorted by ts)."""
    out = df.copy()
    c, h, lo = out.close, out.high, out.low
    lr = np.log(c).diff()
    for n in (1, 5, 15, 60, 240):
        out[f"r{n}"] = np.log(c / c.shift(n))
    out["vol15"] = lr.rolling(15).std()
    out["vol60"] = lr.rolling(60).std()
    out["vol_ratio"] = out.vol15 / out.vol60
    for n in (60, 240):
        m, s = c.rolling(n).mean(), c.rolling(n).std()
        out[f"sma{n}"] = m
        out[f"z{n}"] = (c - m) / s
    out["rsi14"] = _rsi(c)
    hi, lw = h.rolling(1440).max(), lo.rolling(1440).min()
    out["range_pos_1440"] = (c - lw) / (hi - lw)
    out["high60"] = h.rolling(60).max().shift(1)
    vavg = out.volume.rolling(60).mean()
    out["volu_ratio"] = out.volume / vavg
    out["taker_ratio"] = out.taker_buy / out.volume.replace(0, np.nan)
    out["taker_ratio15"] = (out.taker_buy.rolling(15).sum()
                            / out.volume.rolling(15).sum().replace(0, np.nan))
    out["trades_ratio"] = out.trades / out.trades.rolling(60).mean()
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct"] = out.atr / c
    hour = (out.ts // 3_600_000) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if btc is not None:
        b = btc.set_index("ts").close
        bc = out.ts.map(b)
        out["btc_r5"] = np.log(bc / bc.shift(5)).to_numpy()
        out["btc_r60"] = np.log(bc / bc.shift(60)).to_numpy()
    else:
        out["btc_r5"] = out.r5
        out["btc_r60"] = out.r60
    out["rel_r60"] = out.r60 - out.btc_r60
    return out.replace([np.inf, -np.inf], np.nan)
