"""Rule-based 1-minute signals. Each takes a feature frame and returns (signal, exits)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from apex.intraday.backtest import Exits


def mean_reversion(f: pd.DataFrame) -> tuple[np.ndarray, Exits]:
    """Buy a sharp stretch below the 60-minute mean, exit back at the mean."""
    sig = (f.z60 < -2.5) & (f.rsi14 < 25) & (f.z240 > -4)
    return sig.fillna(False).to_numpy(), Exits(stop_atr=4.0, target_atr=None, max_hold=60,
                                                exit_on_mean=True)


def breakout(f: pd.DataFrame) -> tuple[np.ndarray, Exits]:
    """Buy a new 60-minute high on a volume burst dominated by aggressive buyers."""
    sig = (f.close > f.high60) & (f.volu_ratio > 3) & (f.taker_ratio > 0.6) & (f.r240 > 0)
    return sig.fillna(False).to_numpy(), Exits(stop_atr=2.0, target_atr=4.0, max_hold=60)


def trend_pullback(f: pd.DataFrame) -> tuple[np.ndarray, Exits]:
    """In a 4-hour uptrend, buy a short dip that is turning back up."""
    sig = (f.r240 > 2 * f.vol60 * np.sqrt(240)) & (f.z60 < -1.0) & (f.r1 > 0) & (f.rsi14 < 40)
    return sig.fillna(False).to_numpy(), Exits(stop_atr=3.0, target_atr=4.5, max_hold=120)


RULES = {"mean_reversion": mean_reversion, "breakout": breakout,
         "trend_pullback": trend_pullback}
