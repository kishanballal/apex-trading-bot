"""Does Kronos (a pretrained candlestick foundation model) predict NSE 5-minute moves?

Runs on your PC (an NVIDIA GPU is used if present). It does not touch either bot.

    .venv\\Scripts\\python.exe tools\\kronos_test.py            full test (GPU: ~10-30 min)
    .venv\\Scripts\\python.exe tools\\kronos_test.py --quick    small sample to check it works

What it does:
1. Downloads Kronos's code (pinned commit, MIT licence) into third_party/, and the
   Kronos-small weights from Hugging Face on first use.
2. Loads the same 60 days of free Yahoo 5-minute bars for the 50 NSE stocks.
3. For every sudden move (3x normal 5-minute move on 3x volume) plus a random sample of
   ordinary bars, gives Kronos ONLY the 400 bars up to that moment and asks for the next
   30 minutes. Kronos was released before this data existed, so it has never seen it.
4. Scores it honestly: direction hit rate, correlation with what really happened, and a
   fake-money trade test with Indian intraday fees, against the plain dip-bounce rule.
Results print at the end and are saved to data/kronos_results.txt.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apex.intraday import backtest, nse  # noqa: E402

KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_DIR = ROOT / "third_party" / f"Kronos-{KRONOS_COMMIT}"
LOOKBACK, HORIZON = 400, 6          # 400 bars of context (~5 sessions), 6 bars = 30 minutes
ENTRY_THRESHOLD = 0.003             # trade when Kronos expects +0.3% or more


def get_kronos():
    if not KRONOS_DIR.exists():
        import httpx
        print("Downloading Kronos code...")
        r = httpx.get(f"https://codeload.github.com/shiyu-coder/Kronos/zip/{KRONOS_COMMIT}",
                      follow_redirects=True, timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(KRONOS_DIR.parent)
    sys.path.insert(0, str(KRONOS_DIR))
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Kronos-small on {device} "
          f"({torch.cuda.get_device_name(0) if device != 'cpu' else 'no GPU found'})...")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tok, device=device, max_context=512), device


def ist(ts_ms: np.ndarray) -> pd.Series:
    return pd.Series(pd.to_datetime(ts_ms + nse.IST_OFFSET_MS, unit="ms"))


def pick_samples(frames: dict[str, pd.DataFrame], n_random: int, rng) -> pd.DataFrame:
    rows = []
    for s, f in frames.items():
        ok = (np.arange(len(f)) >= LOOKBACK) & (f.minute_of_day.between(15, 330)).to_numpy()
        ok &= (f.last_bar.to_numpy() - np.arange(len(f))) >= HORIZON  # 30 min left in session
        move = ok & (f.z_move.abs() >= 3).to_numpy() & (f.vol_burst >= 3).to_numpy()
        for i in np.flatnonzero(move):
            rows.append((s, i, "spike" if f.z_move.iat[i] > 0 else "drop"))
        normal = np.flatnonzero(ok & ~move)
        for i in rng.choice(normal, size=min(n_random, len(normal)), replace=False):
            rows.append((s, i, "normal"))
    return pd.DataFrame(rows, columns=["symbol", "i", "kind"])


def run_predictions(pred, frames, samples: pd.DataFrame, batch: int, sample_count: int):
    out = np.full(len(samples), np.nan)
    t0 = time.time()
    for b in range(0, len(samples), batch):
        chunk = samples.iloc[b:b + batch]
        dfs, xts, yts = [], [], []
        for s, i in zip(chunk.symbol, chunk.i):
            f = frames[s]
            ctx = f.iloc[i - LOOKBACK + 1:i + 1]
            dfs.append(ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
            xts.append(ist(ctx.ts.to_numpy()))
            yts.append(ist(f.ts.iloc[i + 1:i + 1 + HORIZON].to_numpy()))
        preds = pred.predict_batch(dfs, xts, yts, pred_len=HORIZON, T=1.0, top_p=0.9,
                                   sample_count=sample_count, verbose=False)
        for k, (p, (s, i)) in enumerate(zip(preds, zip(chunk.symbol, chunk.i))):
            out[b + k] = p.close.iloc[-1] / frames[s].close.iat[i] - 1
        done = b + len(chunk)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(samples)} forecasts, about {(len(samples) - done) / rate / 60:.1f} "
              "min left", end="\r", flush=True)
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sample, to check it works")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    symbols = tomllib.loads((ROOT / "config/universe.toml").read_text())["equity"]["symbols"]
    import os
    os.chdir(ROOT)
    raw = nse.load_all(["^NSEI", *symbols])
    nifty = raw.pop("^NSEI", None)
    frames = {s: nse.build(df, nifty) for s, df in raw.items()}
    samples = pick_samples(frames, n_random=8 if args.quick else 60, rng=rng)
    if args.quick:
        samples = pd.concat([g.sample(min(len(g), 40), random_state=1)
                             for _, g in samples.groupby("kind")])
    samples = samples.reset_index(drop=True)
    print(f"{len(frames)} stocks loaded; {len(samples)} moments to forecast "
          f"({samples.kind.value_counts().to_dict()})")

    pred, device = get_kronos()
    samples["kronos"] = run_predictions(pred, frames, samples, args.batch,
                                        sample_count=3 if device != "cpu" else 1)
    # What actually happened: buy at the next bar's open, sell 30 minutes later.
    samples["actual"] = [frames[s].close.iat[i + HORIZON] / frames[s].open.iat[i + 1] - 1
                         for s, i in zip(samples.symbol, samples.i)]

    lines = [f"Kronos-small on NSE 5-minute bars, {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             f"device: {device}, forecasts: {len(samples)}", ""]
    lines.append(f"{'group':8} {'n':>5} {'direction hit':>14} {'correlation':>12} "
                 f"{'avg actual if Kronos says up':>30}")
    for kind, g in [("all", samples), *samples.groupby("kind")]:
        g = g.dropna()
        hit = (np.sign(g.kronos) == np.sign(g.actual)).mean()
        corr = g.kronos.corr(g.actual) if len(g) > 2 else np.nan
        up = g[g.kronos > 0].actual.mean() * 100
        lines.append(f"{kind:8} {len(g):5d} {hit:14.1%} {corr:12.3f} {up:29.3f}%")
    lines.append("")
    lines.append("(a coin flip scores about 50% hit and 0.00 correlation)")
    lines.append("")

    # Fake-money trade test on the sudden-move moments only, with Indian intraday fees.
    cost = nse.NSE_COSTS["discount broker"]
    for label, mask_fn in [
        (f"Kronos: buy any sudden move when it forecasts >= +{ENTRY_THRESHOLD:.1%}",
         lambda g: g.kronos >= ENTRY_THRESHOLD),
        ("Kronos: buy sudden drops it forecasts will bounce",
         lambda g: (g.kind == "drop") & (g.kronos >= ENTRY_THRESHOLD)),
        ("Kronos: buy sudden spikes it forecasts will continue",
         lambda g: (g.kind == "spike") & (g.kronos >= ENTRY_THRESHOLD)),
    ]:
        moves = samples[samples.kind != "normal"]
        chosen = moves[mask_fn(moves)]
        trades = []
        for s, grp in chosen.groupby("symbol"):
            f = frames[s]
            sig = np.zeros(len(f), bool)
            sig[grp.i.to_numpy()] = True
            trades.append(backtest.run(f, sig, backtest.Exits(stop_atr=2.0, target_atr=None,
                                                              max_hold=HORIZON),
                                       cost, symbol=s, last_bar=f.last_bar.to_numpy()))
        t = pd.concat(trades) if trades else pd.DataFrame()
        smry = backtest.summarize(t, cost, days=60) if len(t) else {"trades": 0}
        lines.append(label)
        lines.append("   " + ", ".join(f"{k}={v}" for k, v in smry.items()
                                       if k in ("trades", "win_rate", "avg_gross_pct",
                                                "avg_net_pct", "profit_factor",
                                                "net_pnl_before_tax")))
    trades = []
    for s, f in frames.items():
        sig, ex = nse.crash_bounce(f)
        trades.append(backtest.run(f, sig, ex, cost, symbol=s, last_bar=f.last_bar.to_numpy()))
    smry = backtest.summarize(pd.concat(trades), cost, days=60)
    lines.append("Baseline without Kronos: plain dip-bounce rule")
    lines.append("   " + ", ".join(f"{k}={v}" for k, v in smry.items()
                                   if k in ("trades", "win_rate", "avg_gross_pct", "avg_net_pct",
                                            "profit_factor", "net_pnl_before_tax")))
    lines.append("")
    lines.append("Money figures: INR 8,000 per trade on a 50,000 account, after fees, before tax.")
    lines.append("Fewer than about 100 trades is too few to trust either way.")
    text = "\n".join(lines)
    print("\n" + text)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "kronos_results.txt").write_text(text, encoding="utf-8")
    samples.to_csv(ROOT / "data" / "kronos_forecasts.csv", index=False)
    print("\nSaved to data\\kronos_results.txt. Paste it to Claude.")


if __name__ == "__main__":
    main()
