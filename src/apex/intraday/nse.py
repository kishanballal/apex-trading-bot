"""NSE intraday research on free Yahoo 5-minute candles (the last ~60 days).

Looks for the "sudden move" trades the owner asked for:
- spike: a 5-minute jump far outside normal volatility on a volume burst, bought for continuation
- crash_bounce: a sharp 5-minute fall on a volume burst, bought once a green bar shows buyers
- opening_range: a break above the first 15 minutes' high on rising volume
- ml: a walk-forward model over the same features choosing its own entries

All positions close by the session's last bar (15:25 IST candle). Costs are Indian
equity intraday charges with a discount broker (brokerage 0.03% capped at INR 20, STT 0.025% on
the sell, exchange, SEBI, GST, stamp) plus slippage.

Usage:  python -m apex.intraday.nse
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from apex.intraday import backtest

ROOT = Path("data/intraday/nse")
UA = "Mozilla/5.0 (compatible; apex-research/0.1)"
IST_OFFSET_MS = 19_800_000
BAR_MS = 300_000

# Round trip on INR 8,000: brokerage 2x2.4, STT 2.0, exchange 0.48, GST 0.95, stamp 0.24
# = about INR 8.5 = 0.106%, i.e. 0.053% per side. Slippage 0.02% per side for large caps.
NSE_COSTS = {
    "zero (gross)": backtest.Costs(fee=0.0, slippage=0.0, tax_on_gains=0.0),
    "discount broker": backtest.Costs(fee=0.00053, slippage=0.0002, tax_on_gains=0.0),
    "discount broker, 2x slippage": backtest.Costs(fee=0.00053, slippage=0.0004,
                                                   tax_on_gains=0.0),
}

FEATURES = ["r1", "r3", "r6", "r12", "r_day", "z_move", "vol_burst", "vwap_dist", "or_dist",
            "range_pos_day", "rsi14", "atr_pct", "minute_of_day", "nifty_r1", "nifty_r_day",
            "rel_day", "gap"]


def fetch(symbol: str, client: httpx.Client) -> pd.DataFrame:
    ysym = symbol if symbol.startswith("^") else f"{symbol}.NS"
    r = client.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}",
                   params={"interval": "5m", "range": "60d"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"ts": np.asarray(res["timestamp"], np.int64) * 1000,
                       **{k: q[k] for k in ("open", "high", "low", "close", "volume")}})
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def load_all(symbols: list[str]) -> dict[str, pd.DataFrame]:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = {}
    with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
        for s in symbols:
            path = ROOT / f"{s.replace('^', 'IDX_')}_5m.csv"
            if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
                out[s] = pd.read_csv(path)
                continue
            for attempt in range(4):
                try:
                    df = fetch(s, client)
                    break
                except (httpx.HTTPError, KeyError, TypeError, ValueError):
                    if attempt == 3:
                        df = None
                    time.sleep(3 * (attempt + 1))
            if df is not None and len(df) > 500:
                df.to_csv(path, index=False)
                out[s] = df
            time.sleep(0.7)
    return out


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(df: pd.DataFrame, nifty: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features using only bars up to and including each row."""
    f = df.copy()
    local = f.ts + IST_OFFSET_MS
    f["day"] = local // 86_400_000
    f["minute_of_day"] = (local % 86_400_000) // 60_000 - 555  # minutes since 09:15
    g = f.groupby("day", sort=False)
    c = f.close
    lr = np.log(c).diff()
    first = g.cumcount() == 0
    lr = lr.mask(first)  # overnight gap is not a 5-minute move
    for n in (1, 3, 6, 12):
        f[f"r{n}"] = np.log(c / c.shift(n))
    day_open = g.open.transform("first")
    f["r_day"] = np.log(c / day_open)
    prev_close = c.shift().where(first)
    f["gap"] = np.log(day_open / prev_close.ffill())
    sd = lr.rolling(375, min_periods=300).std()  # about 5 sessions of 5-minute moves
    f["z_move"] = lr / sd
    # Volume burst vs the same slot on previous days, so the opening rush is not a "burst".
    slot = f.minute_of_day
    vol_slot = f.groupby(slot).volume.transform(lambda v: v.shift().rolling(10, 1).mean())
    f["vol_burst"] = f.volume / vol_slot.replace(0, np.nan)
    pv = (c * f.volume).groupby(f.day).cumsum()
    vv = f.volume.groupby(f.day).cumsum()
    f["vwap"] = pv / vv.replace(0, np.nan)
    f["vwap_dist"] = np.log(c / f.vwap)
    or_high = f.high.where(f.minute_of_day < 15).groupby(f.day).transform("max")
    f["or_high"] = or_high.where(f.minute_of_day >= 15)
    f["or_dist"] = np.log(c / f.or_high)
    dh, dl = g.high.cummax(), g.low.cummin()
    f["range_pos_day"] = (c - dl) / (dh - dl).replace(0, np.nan)
    f["rsi14"] = _rsi(c)
    tr = pd.concat([f.high - f.low, (f.high - c.shift()).abs(), (f.low - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = f.atr / c
    f["sma60"] = c.rolling(12).mean()  # 1-hour mean, used by mean-exit only
    if nifty is not None:
        nb = nifty.set_index("ts")
        nc = f.ts.map(nb.close)
        f["nifty_r1"] = np.log(nc / nc.shift()).to_numpy()
        no = f.ts.map(nb.open).groupby(f.day).transform("first")
        f["nifty_r_day"] = np.log(nc / no).to_numpy()
    else:
        f["nifty_r1"], f["nifty_r_day"] = f.r1, f.r_day
    f["rel_day"] = f.r_day - f.nifty_r_day
    last_idx = pd.Series(np.arange(len(f))).groupby(f.day.to_numpy()).transform("max")
    f["last_bar"] = last_idx.to_numpy()
    return f.replace([np.inf, -np.inf], np.nan)


def tradable(f: pd.DataFrame) -> pd.Series:
    # No entries in the first 15 minutes or after 14:45; enough history for the stats.
    return (f.minute_of_day >= 15) & (f.minute_of_day <= 330) & f.z_move.notna()


def spike(f):
    s = tradable(f) & (f.z_move > 3) & (f.vol_burst > 3) & (f.vwap_dist > 0) & (f.rel_day > 0)
    return s.fillna(False).to_numpy(), backtest.Exits(stop_atr=1.5, target_atr=3.0, max_hold=24)


def crash_bounce(f):
    fell = (f.z_move.shift() < -3) & (f.vol_burst.shift() > 3)
    s = tradable(f) & fell & (f.r1 > 0) & (f.rsi14 < 40)
    return s.fillna(False).to_numpy(), backtest.Exits(stop_atr=2.0, target_atr=3.0, max_hold=24)


def opening_range(f):
    cross = (f.close > f.or_high) & (f.close.shift() <= f.or_high.shift())
    s = (tradable(f) & cross & (f.vol_burst > 2) & (f.minute_of_day <= 120)
         & (f.nifty_r_day > -0.003))
    return s.fillna(False).to_numpy(), backtest.Exits(stop_atr=1.5, target_atr=3.0, max_hold=60)


RULES = {"spike": spike, "crash_bounce": crash_bounce, "opening_range": opening_range}


def ml_signals(frames: dict[str, pd.DataFrame], horizon: int = 6, cost: float = 0.0015,
               top_frac: float = 0.01, train_days: int = 20):
    """Walk-forward by trading day: refit every 5 sessions on all earlier sessions."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pooled = []
    for s, f in frames.items():
        x = f.copy()
        x["_sym"], x["_i"] = s, np.arange(len(x))
        x["fwd"] = x.close.shift(-horizon) / x.open.shift(-1) - 1
        x.loc[x.last_bar - x._i < horizon, "fwd"] = np.nan  # label must end the same day
        pooled.append(x)
    p = pd.concat(pooled, ignore_index=True)
    p["y"] = (p.fwd > cost).astype(float).where(p.fwd.notna())
    days = np.sort(p.day.unique())
    sig = {s: np.zeros(len(f), bool) for s, f in frames.items()}
    aucs = []
    ok = tradable(p).to_numpy()
    for k in range(train_days, len(days), 5):
        test_days = days[k:k + 5]
        tr = p[(p.day < test_days[0]) & p.y.notna() & ok]
        te = p[p.day.isin(test_days) & ok]
        if tr.y.nunique() < 2 or te.empty:
            continue
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                           min_samples_leaf=300, l2_regularization=1.0,
                                           random_state=7)
        m.fit(tr[FEATURES].to_numpy(), tr.y.to_numpy())
        thr = np.quantile(m.predict_proba(tr[FEATURES].to_numpy())[:, 1], 1 - top_frac)
        pr = m.predict_proba(te[FEATURES].to_numpy())[:, 1]
        lab = te.y.notna().to_numpy()
        if te.y[lab].nunique() == 2:
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(te.y[lab], pr[lab]))
        pick = pr >= thr
        for s in frames:
            rows = (te._sym == s).to_numpy()
            sig[s][te._i[rows].to_numpy()] = pick[rows]
    exits = backtest.Exits(stop_atr=2.0, target_atr=None, max_hold=horizon)
    first_test = days[train_days] if len(days) > train_days else None
    return sig, exits, aucs, first_test


def main() -> None:
    import tomllib
    cfg = tomllib.loads(Path("config/universe.toml").read_text())
    symbols = cfg["equity"]["symbols"]
    raw = load_all(["^NSEI", *symbols])
    nifty = raw.pop("^NSEI", None)
    frames = {s: build(df, nifty) for s, df in raw.items()}
    n_days = int(np.median([f.day.nunique() for f in frames.values()]))
    print(f"{len(frames)} stocks, about {n_days} sessions of 5-minute bars")
    rows = []
    out = ROOT

    def evaluate(name, sig, ex, days):
        for label, cost in NSE_COSTS.items():
            tr = pd.concat([backtest.run(frames[s], sig[s], ex[s] if isinstance(ex, dict) else ex,
                                         cost, symbol=s,
                                         last_bar=frames[s].last_bar.to_numpy())
                            for s in frames], ignore_index=True)
            if label == "discount broker":
                tr.to_csv(out / f"trades_{name}.csv", index=False)
            rows.append({"strategy": name, "costs": label,
                         **backtest.summarize(tr, cost, days=days)})

    for name, rule in RULES.items():
        sig, ex = {}, {}
        for s, f in frames.items():
            sig[s], ex[s] = rule(f)
        evaluate(name, sig, ex, n_days)
    sig, ex, aucs, first_test = ml_signals(frames)
    test_days = int(np.median([(f.day >= first_test).sum() / 75 for f in frames.values()]))
    evaluate("ml_30min", sig, ex, test_days)
    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print("ML weekly AUC:", [round(a, 3) for a in aucs], "test sessions:", test_days)


if __name__ == "__main__":
    main()
