"""Walk-forward minute model.

Label: does the trade return from the next bar's open to the close `horizon` minutes later
beat the round-trip cost? Models are refit every month on all earlier months only, with a gap
of `horizon` minutes so no training label overlaps the test month. The entry threshold is the
top `top_frac` of the model's scores on its own training data, so it is fixed before the test
month starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apex.intraday.features import FEATURES

MONTH_MS = 30 * 1440 * 60_000


def add_label(f: pd.DataFrame, horizon: int, cost: float) -> pd.DataFrame:
    out = f.copy()
    entry = out.open.shift(-1)
    out["fwd"] = out.close.shift(-horizon) / entry - 1
    out["y"] = (out.fwd > cost).astype(float).where(out.fwd.notna())
    return out


def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=300))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=500, l2_regularization=1.0,
                                          random_state=7)


@dataclass
class Fold:
    model: str
    month: int
    auc: float
    base_rate: float
    top_fwd_pct: float   # mean forward return of the top-scored test rows
    all_fwd_pct: float
    n_signals: int


def walk_forward(frames: dict[str, pd.DataFrame], horizon: int, cost: float, kind: str,
                 top_frac: float = 0.003, min_train_months: int = 2, sample_every: int = 3):
    """Return (folds, {symbol: signal array}) for out-of-sample months only."""
    from sklearn.metrics import roc_auc_score

    pooled = pd.concat([f.assign(_sym=s) for s, f in frames.items()], ignore_index=True)
    t0 = pooled.ts.min()
    pooled["_month"] = ((pooled.ts - t0) // MONTH_MS).astype(int)
    signals = {s: np.zeros(len(f), bool) for s, f in frames.items()}
    folds: list[Fold] = []
    for m in range(min_train_months, pooled._month.max() + 1):
        test_start = t0 + m * MONTH_MS
        train = pooled[(pooled.ts < test_start - horizon * 60_000) & pooled.y.notna()]
        train = train.iloc[::sample_every]
        test = pooled[pooled._month == m]
        if len(test) == 0 or train.y.nunique() < 2:
            continue
        model = make_model(kind)
        model.fit(train[FEATURES].to_numpy(), train.y.to_numpy())
        thr = np.quantile(model.predict_proba(train[FEATURES].to_numpy())[:, 1], 1 - top_frac)
        p = model.predict_proba(test[FEATURES].to_numpy())[:, 1]
        lab = test.y.notna().to_numpy()
        pick = p >= thr
        auc = roc_auc_score(test.y[lab], p[lab]) if test.y[lab].nunique() == 2 else np.nan
        folds.append(Fold(kind, m, round(auc, 4), round(test.y[lab].mean(), 4),
                          round(test.fwd[pick & lab].mean() * 100, 4) if (pick & lab).any()
                          else np.nan,
                          round(test.fwd[lab].mean() * 100, 4), int(pick.sum())))
        for s in frames:
            rows = (test._sym == s).to_numpy()
            idx = test.index[rows] - pooled.index[pooled._sym == s][0]
            signals[s][idx.to_numpy()] = pick[rows]
    return folds, signals
