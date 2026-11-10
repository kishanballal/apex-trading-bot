"""The model bundle used for paper and live trading.

Two parts, both trained only on outcomes that are already known:
  * a calibrated classifier for "will this trade reach its target before its stop";
  * ridge regressions for the expected price change 1 to 10 days ahead, which draw the
    forecast line on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from apex.frame import dates

from apex.ml.dataset import FEATURES, to_matrix
from apex.ml.walkforward import CALIBRATION_FRACTION, _calibrate

HORIZONS = (1, 2, 3, 5, 10)


@dataclass
class ModelBundle:
    classifier: object
    calibrator: object
    forecasters: dict[int, object]
    trained_at: datetime
    trained_through: datetime  # latest outcome the bundle has seen
    n_samples: int
    features: list[str]

    def win_probability(self, data: pd.DataFrame) -> np.ndarray:
        raw = self.classifier.predict_proba(to_matrix(data))[:, 1]
        return self.calibrator.predict(raw)

    def forecast_returns(self, data: pd.DataFrame) -> dict[int, np.ndarray]:
        """Expected log return per horizon, in trading days."""
        x = to_matrix(data)
        return {h: m.predict(x) for h, m in self.forecasters.items()}

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        import joblib

        return joblib.load(path)


def add_forward_returns(data: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join log(close[t+h] / close[t]) for each horizon. Null where the future is unknown."""
    parts = []
    for symbol in data["symbol"].unique().tolist():
        frame = candles[symbol].sort_values("ts")
        c = frame["close"].reset_index(drop=True)
        part = pd.DataFrame({"symbol": symbol, "date": dates(frame["ts"]).reset_index(drop=True)})
        for h in HORIZONS:
            part[f"fwd_{h}"] = np.log(c.shift(-h) / c)
        parts.append(part.drop_duplicates("date", keep="last"))
    return data.merge(pd.concat(parts, ignore_index=True), on=["symbol", "date"], how="left")


def train_bundle(data: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> ModelBundle:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labelled = data[data["y"].notna()].sort_values("ts", kind="stable").reset_index(drop=True)
    if len(labelled) < 2000:
        raise ValueError(f"only {len(labelled)} labelled samples; fetch more history first")
    split = int(len(labelled) * (1 - CALIBRATION_FRACTION))
    cal_start = labelled["ts"].iloc[split]
    fit = labelled.head(split)
    fit = fit[fit["label_end"] < cal_start]
    cal = labelled.tail(len(labelled) - split)
    classifier = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.1, max_iter=500))
    classifier.fit(to_matrix(fit), fit["y"].to_numpy())
    calibrator = _calibrate(classifier.predict_proba(to_matrix(cal))[:, 1], cal["y"].to_numpy())

    with_fwd = add_forward_returns(data, candles)
    forecasters = {}
    for h in HORIZONS:
        rows = with_fwd[with_fwd[f"fwd_{h}"].notna()]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
        # Clip extreme moves so a few crypto spikes do not dominate the fit.
        model.fit(to_matrix(rows), np.clip(rows[f"fwd_{h}"].to_numpy(), -0.5, 0.5))
        forecasters[h] = model
    return ModelBundle(classifier, calibrator, forecasters, datetime.now(UTC),
                       labelled["label_end"].max().to_pydatetime(), len(labelled), list(FEATURES))


def forecast_path(last_close: float, returns_by_h: dict[int, float], atr_pct: float,
                  steps: int = 10) -> list[dict]:
    """Daily forecast points with a one-sigma band. Horizons between fitted ones are
    interpolated. The band widens with the square root of time, scaled by recent volatility."""
    hs = sorted(returns_by_h)
    xs, ys = [0, *hs], [0.0, *[returns_by_h[h] for h in hs]]
    sigma_day = (atr_pct or 0.02) / 1.25  # ATR is roughly 1.25 daily standard deviations
    out = []
    for step in range(1, steps + 1):
        mid = float(np.interp(step, xs, ys))
        band = sigma_day * np.sqrt(step)
        out.append({"step": step, "mid": last_close * float(np.exp(mid)),
                    "low": last_close * float(np.exp(mid - band)),
                    "high": last_close * float(np.exp(mid + band))})
    return out
