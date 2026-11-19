"""1-minute features for the active bot.

Every minute it looks at the LAST FIVE MINUTES as one move, so a sudden jump is seen within a
minute of happening instead of waiting for a 5-minute candle to close. Every value on a row
uses only bars up to and including that row.

Columns the engine needs: z_move, vol_burst, move, r1, rsi14, vwap_dist, atr, minute_of_day,
and the three signal flags sig_spike / sig_drop / sig_bounce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_MS = 60_000
IST_OFFSET_MS = 19_800_000
WINDOW = 5                 # minutes treated as one "move"
SESSION_BARS = 375         # 09:15 to 15:30
MOVE_Z = 3.0
MOVE_VOLUME = 3.0


def build(df: pd.DataFrame, move_z: float = MOVE_Z, move_volume: float = MOVE_VOLUME
          ) -> pd.DataFrame:
    f = df.reset_index(drop=True).copy()
    local = f.ts + IST_OFFSET_MS
    f["day"] = local // 86_400_000
    f["minute_of_day"] = (local % 86_400_000) // 60_000 - 555   # minutes since 09:15
    c = f.close
    first = f.day != f.day.shift()
    lr = np.log(c).diff().mask(first)          # the overnight gap is not a 1-minute move
    f["r1"] = lr
    # The last five minutes as one move, never reaching back across the open.
    in_day = f.groupby("day").cumcount() >= WINDOW
    f["move"] = lr.rolling(WINDOW).sum().where(in_day)
    sd = lr.rolling(5 * SESSION_BARS, min_periods=2 * SESSION_BARS).std() * np.sqrt(WINDOW)
    f["z_move"] = f.move / sd
    # Volume of the last five minutes against the same clock time on earlier days, so the
    # opening and closing rush do not count as a "burst".
    vol5 = f.volume.rolling(WINDOW).sum().where(in_day)
    grid = pd.DataFrame({"d": f.day, "m": f.minute_of_day, "v": vol5}).pivot_table(
        index="d", columns="m", values="v", aggfunc="last", dropna=False)
    usual_grid = grid.shift().rolling(5, 2).mean()        # mean of up to 5 earlier sessions
    d_pos = grid.index.get_indexer(f.day)
    m_pos = grid.columns.get_indexer(f.minute_of_day)
    usual = pd.Series(usual_grid.to_numpy()[d_pos, m_pos], index=f.index)
    f["vol_burst"] = vol5 / usual.replace(0, np.nan)
    pv = (c * f.volume).groupby(f.day).cumsum()
    vv = f.volume.groupby(f.day).cumsum()
    f["vwap_dist"] = np.log(c / (pv / vv.replace(0, np.nan)))
    # RSI and ATR on a 5-minute scale (smoothing 14 five-minute steps = 70 minutes).
    d = c.diff().mask(first, 0.0)
    up = d.clip(lower=0).ewm(alpha=1 / 70, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 70, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rng5 = f.high.rolling(WINDOW).max() - f.low.rolling(WINDOW).min()
    f["atr"] = rng5.ewm(alpha=1 / 70, adjust=False).mean()

    ok = f.z_move.notna() & f.vol_burst.notna()
    burst = f.vol_burst >= move_volume
    spike = ok & (f.z_move >= move_z) & burst
    drop = ok & (f.z_move <= -move_z) & burst
    # Report a move once, on the minute it first shows up.
    f["sig_spike"] = spike & ~spike.shift(fill_value=False)
    f["sig_drop"] = drop & ~drop.shift(fill_value=False)
    # Bounce: a sharp drop in the last three minutes, and this minute closes above the
    # previous minute's high (buyers are back), while still oversold.
    recent_drop = drop.shift(fill_value=False).rolling(3, 1).max().astype(bool)
    bounce = recent_drop & (c > f.high.shift()) & (f.rsi14 < 40) & ~first
    f["sig_bounce"] = bounce & ~bounce.shift(fill_value=False)
    return f.replace([np.inf, -np.inf], np.nan)
