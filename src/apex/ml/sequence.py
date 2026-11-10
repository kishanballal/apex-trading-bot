"""Deep learning comparison: a GRU that reads the last 60 daily bars of each symbol.

Same samples, labels, yearly walk-forward, purging and held-out calibration as the tabular
models in walkforward.py, so the results are directly comparable. Needs PyTorch.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from apex.analysis.indicators import prepare
from apex.config import UniverseGroup
from apex.frame import dates as to_dates, matrix as to_float_matrix
from apex.ml.walkforward import CALIBRATION_FRACTION, FoldScore, _calibrate

WINDOW = 60
SEQ_FEATURES = ["ret", "range", "body", "log_vol_ratio", "dist_ema20", "dist_ema50",
                "rsi", "atr_pct", "bench_ret"]


def _bar_features(candles: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    bs = bench.sort_values("ts")
    b = pd.DataFrame({"date": to_dates(bs["ts"]).to_numpy(),
                      "bench_ret": (bs["close"] / bs["close"].shift(1) - 1).to_numpy()})
    f = prepare(candles)
    f["date"] = to_dates(f["ts"])
    f = f.merge(b.drop_duplicates("date", keep="last"), on="date", how="left")
    c = f["close"]
    return pd.DataFrame({
        "date": f["date"],
        "ret": c / c.shift(1) - 1,
        "range": (f["high"] - f["low"]) / c,
        "body": (c - f["open"]) / c,
        "log_vol_ratio": np.log(f["vol_ratio"].clip(1e-3, 1e3)),
        "dist_ema20": c / f["ema20"] - 1,
        "dist_ema50": c / f["ema50"] - 1,
        "rsi": f["rsi14"] / 100 - 0.5,
        "atr_pct": f["atr_pct"],
        "bench_ret": f["bench_ret"],
    })


def build_sequences(
    groups: list[UniverseGroup], candles: dict[str, pd.DataFrame], data: pd.DataFrame
) -> np.ndarray:
    """One [WINDOW, features] block per row of `data`, ending at that row's bar (inclusive).
    Rows without a full window are NaN."""
    out = np.full((len(data), WINDOW, len(SEQ_FEATURES)), np.nan, dtype=np.float32)
    symbols = data["symbol"].tolist()
    dates = data["date"].tolist()
    where: dict[str, list[int]] = {}
    for n, s in enumerate(symbols):
        where.setdefault(s, []).append(n)
    for group in groups:
        for symbol in group.symbols:
            if symbol not in where or symbol not in candles:
                continue
            feats = _bar_features(candles[symbol], candles[group.benchmark])
            index = {d: i for i, d in enumerate(feats["date"].tolist())}
            matrix = to_float_matrix(feats, SEQ_FEATURES).astype(np.float32)
            for n in where[symbol]:
                i = index.get(dates[n])
                if i is not None and i + 1 >= WINDOW:
                    out[n] = matrix[i + 1 - WINDOW: i + 1]
    return out


class GRUClassifier:
    def __init__(self, hidden: int = 48, epochs: int = 4, lr: float = 1e-3, batch: int = 512,
                 seed: int = 7) -> None:
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((x - self.mean) / self.std, nan=0.0), -5, 5)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GRUClassifier":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        flat = x.reshape(-1, x.shape[-1])
        self.mean, self.std = np.nanmean(flat, 0), np.nanstd(flat, 0) + 1e-8  # training data only
        xs = torch.from_numpy(self._scale(x).astype(np.float32))
        ys = torch.from_numpy(y.astype(np.float32))

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int) -> None:
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

            def forward(self, v):
                _, h = self.gru(v)
                return self.head(h[-1]).squeeze(-1)

        self.net = Net(x.shape[-1], self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(xs), generator=generator)
            for k in range(0, len(xs), self.batch):
                idx = order[k: k + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(xs[idx]), ys[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        xs = torch.from_numpy(self._scale(x).astype(np.float32))
        with torch.no_grad():
            parts = [torch.sigmoid(self.net(xs[k: k + 4096])) for k in range(0, len(xs), 4096)]
        return torch.cat(parts).numpy().astype(np.float64)


def walk_forward_gru(
    data: pd.DataFrame, sequences: np.ndarray, test_years: list[int], *, name: str = "gru",
    factory=GRUClassifier,
) -> tuple[pd.DataFrame, list[FoldScore]]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    data = data.reset_index(drop=True)
    data["_row"] = np.arange(len(data))
    usable = ~np.isnan(sequences).all(axis=(1, 2))
    labelled = data[data["y"].notna().to_numpy() & usable]
    predictions, scores = [], []
    for year in test_years:
        cutoff = datetime(year, 1, 1, tzinfo=UTC)
        train = labelled[labelled["label_end"] < cutoff].sort_values("ts", kind="stable")
        train = train.reset_index(drop=True)
        in_year = ((data["date"] >= date(year, 1, 1)) & (data["date"] <= date(year, 12, 31)))
        test = data[in_year.to_numpy() & usable].reset_index(drop=True)
        if len(train) < 2000 or test.empty:
            continue
        split = int(len(train) * (1 - CALIBRATION_FRACTION))
        cal_start = train["ts"].iloc[split]
        fit = train.head(split)
        fit = fit[fit["label_end"] < cal_start]
        cal = train.tail(len(train) - split)
        model = factory().fit(sequences[fit["_row"].to_numpy()], fit["y"].to_numpy())
        iso = _calibrate(model.predict(sequences[cal["_row"].to_numpy()]), cal["y"].to_numpy())
        p = iso.predict(model.predict(sequences[test["_row"].to_numpy()]))
        fold = test[["symbol", "date", "ts", "y", "r"]].copy()
        fold[name] = p
        predictions.append(fold)
        mask = test["y"].notna().to_numpy()
        if mask.sum() > 50:
            ps, ys = p[mask], test["y"].to_numpy()[mask]
            rs = test["r"].to_numpy()[mask]
            top = ps >= np.quantile(ps, 0.9)
            base = float(fit["y"].mean())
            scores.append(FoldScore(
                name, year, len(fit), int(mask.sum()), float(ys.mean()),
                float(roc_auc_score(ys, ps)), float(brier_score_loss(ys, ps)),
                float(brier_score_loss(ys, np.full_like(ys, base))),
                float(ys[top].mean()), float(rs[top].mean()), float(rs.mean())))
    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), scores
