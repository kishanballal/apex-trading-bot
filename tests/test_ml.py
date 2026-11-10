from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("sklearn")

from apex.analysis.regime import Regime, Trend
from apex.backtest.engine import BacktestConfig, run_backtest
from apex.config import RISK_PROFILES, UniverseGroup, risk_profile
from apex.frame import dates, records
from apex.ml.dataset import FEATURES, LabelConfig, build_dataset, label_outcome, to_matrix
from apex.ml.strategy import make_table_strategy
from apex.ml.walkforward import walk_forward
from apex.models import AssetClass
from apex.scanner import scanner
from conftest import make_candles

END = datetime(2025, 12, 31, 3, 45, tzinfo=UTC)
GROUPS = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", ("A", "B", "C"))]


def universe(n=900):
    c = {s: make_candles(n, seed=i, drift=0.0005, vol=0.012, end=END)
         for i, s in enumerate(("A", "B", "C"))}
    c["BENCH"] = make_candles(n, seed=99, drift=0.0004, vol=0.008, end=END)
    return c


def bar(o, h, l, c, atr=None):
    return {"open": o, "high": h, "low": l, "close": c, "atr14": atr, "ts": None}


def test_label_outcomes():
    cfg = LabelConfig(horizon=3)
    sig = bar(100, 101, 99, 100, atr=2.0)  # stop 96, target 108
    win = [sig, bar(100, 102, 99, 101), bar(101, 109, 100, 108)]
    assert label_outcome(win, 0, cfg, 0.0) == (True, pytest.approx(2.0), 2)
    loss = [sig, bar(100, 101, 95, 96)]
    assert label_outcome(loss, 0, cfg, 0.0) == (False, pytest.approx(-1.0), 1)
    both = [sig, bar(100, 110, 95, 100)]
    assert label_outcome(both, 0, cfg, 0.0)[0] is False  # stop assumed first
    gap = [sig, bar(100, 101, 99, 100), bar(90, 91, 89, 90)]
    assert label_outcome(gap, 0, cfg, 0.0)[1] == pytest.approx(-2.5)  # gap loses more than 1R
    timeout = [sig, *[bar(100, 101, 99, 100)] * 2, bar(100, 102, 99, 102)]
    assert label_outcome(timeout, 0, cfg, 0.0) == (False, pytest.approx(0.5), 3)
    # Costs lower the result: 0.2% per side on about 200 of traded value.
    assert label_outcome(win, 0, cfg, 0.002)[1] < 2.0


def test_label_is_none_when_outcome_unknown_or_untradable():
    cfg = LabelConfig(horizon=5)
    sig = bar(100, 101, 99, 100, atr=2.0)
    assert label_outcome([sig], 0, cfg, 0.0) is None  # no next bar
    assert label_outcome([sig, bar(100, 101, 99, 100)], 0, cfg, 0.0) is None  # still open
    assert label_outcome([sig, bar(95, 96, 94, 95)], 0, cfg, 0.0) is None  # gapped below stop
    assert label_outcome([bar(100, 101, 99, 100), bar(100, 101, 99, 100)], 0, cfg, 0.0) is None


def test_features_do_not_change_when_the_future_is_removed():
    full = universe()
    cut = date(2025, 3, 31)
    truncated = {k: v[dates(v["ts"]) <= cut].reset_index(drop=True) for k, v in full.items()}
    a = build_dataset(GROUPS, full)
    a = a[a["date"] <= cut][["symbol", "date", *FEATURES]].reset_index(drop=True)
    b = build_dataset(GROUPS, truncated)[["symbol", "date", *FEATURES]]
    assert len(a) == len(b) > 1000
    assert a.equals(b)


def test_labels_resolve_strictly_after_the_signal_and_recent_bars_are_unlabelled():
    data = build_dataset(GROUPS, universe())
    labelled = data[data["y"].notna()]
    assert (labelled["label_end"] > labelled["ts"]).all()
    assert set(labelled["y"].unique().tolist()) == {0.0, 1.0}
    last_day = data["date"].max()
    assert data[data["date"] == last_day]["y"].isna().sum() == 3
    assert not np.isinf(to_matrix(data)).any()


def test_walk_forward_never_trains_on_the_test_period(monkeypatch):
    data = build_dataset(GROUPS, universe(1400))
    seen = {}

    class Spy:
        def fit(self, x, y):
            seen.setdefault("n", []).append(len(y))
            self.mean = float(y.mean())
            return self

        def predict_proba(self, x):
            return np.column_stack([np.full(len(x), 1 - self.mean), np.full(len(x), self.mean)])

    year = 2025
    cutoff = datetime(year, 1, 1, tzinfo=UTC)
    eligible = data[data["y"].notna() & (data["label_end"] < cutoff)]
    predictions, scores = walk_forward(data, [year], models={"spy": Spy})
    assert seen["n"][0] < len(eligible) * 0.8 + 1  # calibration slice and purge are held out
    assert predictions["date"].min() >= date(year, 1, 1)
    assert len(predictions) == len(data[data["date"] >= date(year, 1, 1)])
    assert predictions["spy"].between(0, 1).all()
    assert scores and scores[0].year == year and scores[0].n_train == seen["n"][0]


def test_table_strategy_trades_only_above_threshold_and_respects_regime():
    candles = make_candles(330, breakout=True)
    from apex.analysis.indicators import prepare
    rows = records(prepare(candles))
    day = rows[-1]["ts"].date()
    now = rows[-1]["ts"] + timedelta(hours=8)
    bull, bear = Regime(Trend.BULL, False, ""), Regime(Trend.BEAR, False, "")
    strat = make_table_strategy("ml_x", {("T", day): 0.6}, 0.45)
    row = strat("T", AssetClass.EQUITY, rows, len(rows) - 1, bull, now=now)
    assert row.is_opportunity and row.proposal.strategy == "ml_x" and row.score == 60
    assert row.proposal.entry - row.proposal.stop == pytest.approx(2 * rows[-1]["atr14"])
    assert not strat("T", AssetClass.EQUITY, rows, len(rows) - 1, bear, now=now).is_opportunity
    assert not strat("T", AssetClass.EQUITY, rows, len(rows) - 2, bull, now=now).is_opportunity
    low = make_table_strategy("ml_x", {("T", day): 0.3}, 0.45)
    assert not low("T", AssetClass.EQUITY, rows, len(rows) - 1, bull, now=now).is_opportunity


def test_risk_profiles_scale_limits_but_never_leverage():
    assert risk_profile("current").risk_per_trade == 0.0025
    for name in RISK_PROFILES:
        limits = risk_profile(name)
        assert limits.max_leverage == 1.0 and limits.max_deployed_capital <= 1.0
    assert risk_profile("aggressive").max_portfolio_drawdown == 0.20
