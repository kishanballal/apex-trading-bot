"""Step-1 intraday research: run every rule and the minute model under several cost levels.

Usage:  python -m apex.intraday.research [--days 180]
Writes a markdown table to stdout and saves all trades to data/intraday/trades_<name>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apex.intraday import backtest, features, model, strategies
from apex.intraday.data import load_or_fetch

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
COSTS = {
    "zero (gross)": backtest.Costs(fee=0.0, slippage=0.0, tax_on_gains=0.0),
    "limit orders 0.02%": backtest.Costs(fee=0.0002, slippage=0.0001),
    "Binance taker 0.1%": backtest.Costs(fee=0.001, slippage=0.0002),
    "Indian app 0.2%": backtest.Costs(fee=0.002, slippage=0.0003),
}


def load(days: int) -> dict[str, pd.DataFrame]:
    raw = {s: load_or_fetch(s, days) for s in SYMBOLS}
    btc = raw["BTCUSDT"]
    return {s: features.build(df, None if s == "BTCUSDT" else btc) for s, df in raw.items()}


def evaluate(frames, signals, exits_by_sym, name: str, days: int, out: Path) -> list[dict]:
    rows = []
    for label, cost in COSTS.items():
        trades = pd.concat([backtest.run(frames[s], signals[s], exits_by_sym[s], cost, symbol=s)
                            for s in frames], ignore_index=True)
        if label == "Binance taker 0.1%":
            trades.to_csv(out / f"trades_{name}.csv", index=False)
        rows.append({"strategy": name, "costs": label,
                     **backtest.summarize(trades, cost, days=days)})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--horizons", default="30,120")
    args = ap.parse_args(argv)
    out = Path("data/intraday")
    frames = load(args.days)
    rows = []
    for name, rule in strategies.RULES.items():
        sig, ex = {}, {}
        for s, f in frames.items():
            sig[s], ex[s] = rule(f)
        rows += evaluate(frames, sig, ex, name, args.days, out)

    folds_all = []
    for h in (int(x) for x in args.horizons.split(",")):
        cost = 2 * (0.001 + 0.0002)
        labelled = {s: model.add_label(f, h, cost) for s, f in frames.items()}
        for kind in ("logistic", "boosting"):
            folds, sig = model.walk_forward(labelled, h, cost, kind)
            folds_all += [{**f.__dict__, "horizon": h} for f in folds]
            ex = {s: backtest.Exits(stop_atr=4.0, target_atr=None, max_hold=h) for s in frames}
            oos_days = sum(1 for _ in folds) * 30
            rows += evaluate(frames, sig, ex, f"ml_{kind}_{h}m", oos_days, out)

    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    pd.DataFrame(folds_all).to_csv(out / "ml_folds.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    print()
    print(pd.DataFrame(folds_all).to_string(index=False))


if __name__ == "__main__":
    main()
