"""Walk-forward model comparison.

For each test year the models see only samples whose outcome was fully known before that year
began. Probabilities are calibrated on the most recent slice of the training window, which the
model itself was not fitted on. Nothing from the test year touches fitting or calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from apex.ml.dataset import FEATURES, to_matrix

CALIBRATION_FRACTION = 0.2


def make_models(seed: int = 7) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "logistic": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500)),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=200,
                                   max_features=0.4, n_jobs=-1, random_state=seed)),
        "xgboost": lambda: XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
            n_jobs=4, random_state=seed, eval_metric="logloss"),
    }


@dataclass(frozen=True)
class FoldScore:
    model: str
    year: int
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    brier: float
    brier_baseline: float  # always predicting the training base rate
    top_decile_win: float
    top_decile_r: float
    all_r: float


def _calibrate(raw_cal: np.ndarray, y_cal: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y_cal)
    return iso


def walk_forward(
    data: pd.DataFrame, test_years: list[int], *, models: dict | None = None
) -> tuple[pd.DataFrame, list[FoldScore]]:
    """Returns out-of-sample predictions (one column per model) and per-fold scores."""
    from sklearn.metrics import brier_score_loss, roc_auc_score

    models = models or make_models()
    labelled = data[data["y"].notna()]
    predictions: list[pd.DataFrame] = []
    scores: list[FoldScore] = []
    for year in test_years:
        cutoff = datetime(year, 1, 1, tzinfo=UTC)
        train = labelled[labelled["label_end"] < cutoff].sort_values("ts", kind="stable")
        train = train.reset_index(drop=True)
        # Every bar of the test year gets a prediction, including ones with no label yet.
        test = data[(data["date"] >= date(year, 1, 1)) & (data["date"] <= date(year, 12, 31))]
        test = test.reset_index(drop=True)
        if len(train) < 2000 or test.empty:
            continue
        split = int(len(train) * (1 - CALIBRATION_FRACTION))
        # Purge: the fit slice must not contain outcomes that resolve inside the calibration slice.
        cal_start = train["ts"].iloc[split]
        fit = train.head(split)
        fit = fit[fit["label_end"] < cal_start]
        cal = train.tail(len(train) - split)
        x_fit, y_fit = to_matrix(fit), fit["y"].to_numpy()
        x_cal, y_cal = to_matrix(cal), cal["y"].to_numpy()
        x_test = to_matrix(test)
        scored = test[test["y"].notna()]
        mask = test["y"].notna().to_numpy()
        fold = test[["symbol", "date", "ts", "y", "r"]].copy()
        for name, factory in models.items():
            model = factory()
            model.fit(x_fit, y_fit)
            iso = _calibrate(model.predict_proba(x_cal)[:, 1], y_cal)
            p = iso.predict(model.predict_proba(x_test)[:, 1])
            fold[name] = p
            if len(scored) > 50:
                ps, ys, rs = p[mask], scored["y"].to_numpy(), scored["r"].to_numpy()
                top = ps >= np.quantile(ps, 0.9)
                base = float(y_fit.mean())
                scores.append(FoldScore(
                    name, year, len(fit), len(scored), float(ys.mean()),
                    float(roc_auc_score(ys, ps)) if 0 < ys.mean() < 1 else float("nan"),
                    float(brier_score_loss(ys, ps)),
                    float(brier_score_loss(ys, np.full_like(ys, base))),
                    float(ys[top].mean()), float(rs[top].mean()), float(rs.mean()),
                ))
        predictions.append(fold)
    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), scores
