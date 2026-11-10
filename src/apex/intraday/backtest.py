"""Minute-level backtester for long-only spot trades.

Rules, chosen to be pessimistic rather than flattering:
- A signal on bar t (known at its close) is filled at the OPEN of bar t + 1 + delay, plus slippage.
- Stops and targets are checked on every later bar's high/low. If both are touched in the same
  minute we assume the stop hit first. A gap through the stop fills at the bar's open.
- Positions also close after `max_hold` minutes at that bar's close.
- One position per symbol at a time; signals while in a trade are ignored.
- Fees and slippage are charged on both sides. Indian crypto tax (30% of each profitable
  trade, losses cannot be set off) is applied in `summarize`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee: float = 0.001        # per side, as a fraction (0.1% = Binance spot taker)
    slippage: float = 0.0002  # per side
    tax_on_gains: float = 0.30


@dataclass(frozen=True)
class Exits:
    stop_atr: float = 2.0
    target_atr: float | None = 3.0
    max_hold: int = 60
    exit_on_mean: bool = False  # mean-reversion: also exit when close crosses sma60


def run(df: pd.DataFrame, signal: np.ndarray, exits: Exits, costs: Costs,
        delay: int = 0, symbol: str = "") -> pd.DataFrame:
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy(float)
    sma = df["sma60"].to_numpy(float) if exits.exit_on_mean else None
    ts = df["ts"].to_numpy()
    n = len(df)
    trades = []
    sig_idx = np.flatnonzero(signal)
    busy_until = -1
    for t in sig_idx:
        if t <= busy_until or not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        e = t + 1 + delay
        if e >= n:
            break
        entry = o[e] * (1 + costs.slippage)
        stop = o[e] - exits.stop_atr * atr[t]
        target = o[e] + exits.target_atr * atr[t] if exits.target_atr else np.inf
        last = min(e + exits.max_hold - 1, n - 1)
        exit_px, j, why = c[last], last, "time"
        for j in range(e, last + 1):
            if lo[j] <= stop:
                exit_px, why = (min(o[j], stop) if j > e else stop), "stop"
                break
            if h[j] >= target:
                exit_px, why = (max(o[j], target) if j > e else target), "target"
                break
            if sma is not None and j > e and c[j] >= sma[j]:
                exit_px, why = c[j], "mean"
                break
        else:
            j = last
        exit_net = exit_px * (1 - costs.slippage)
        gross = exit_px / o[e] - 1
        net = (exit_net * (1 - costs.fee)) / (entry * (1 + costs.fee)) - 1
        trades.append((symbol, ts[e], ts[j], j - e + 1, o[e], exit_px, gross, net, why))
        busy_until = j
    return pd.DataFrame(trades, columns=["symbol", "entry_ts", "exit_ts", "minutes", "entry",
                                         "exit", "gross_ret", "net_ret", "reason"])


def summarize(trades: pd.DataFrame, costs: Costs, capital: float = 50_000.0,
              notional: float = 8_000.0, days: float | None = None) -> dict:
    """Portfolio view: each trade uses a fixed notional (default 8k of a 50k account)."""
    if trades.empty:
        return {"trades": 0}
    t = trades.sort_values("exit_ts")
    pnl = t.net_ret.to_numpy() * notional
    gross_pnl = t.gross_ret.to_numpy() * notional
    tax = costs.tax_on_gains * pnl[pnl > 0].sum()
    eq = capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[capital], eq]))[1:]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    span = days or max((t.exit_ts.max() - t.entry_ts.min()) / 86_400_000, 1)
    return {
        "trades": len(t),
        "per_day": round(len(t) / span, 2),
        "win_rate": round(len(wins) / len(t), 3),
        "avg_gross_pct": round(t.gross_ret.mean() * 100, 4),
        "avg_net_pct": round(t.net_ret.mean() * 100, 4),
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else np.inf,
        "gross_pnl": round(gross_pnl.sum()),
        "net_pnl_before_tax": round(pnl.sum()),
        "tax": round(tax),
        "net_pnl_after_tax": round(pnl.sum() - tax),
        "max_dd_pct": round(((peak - eq) / peak).max() * 100, 2),
        "avg_minutes": round(t.minutes.mean(), 1),
    }
