import numpy as np
import pandas as pd
import pytest

from apex.intraday import backtest, features, model, strategies


def _bars(closes, spread=0.001):
    c = np.asarray(closes, float)
    n = len(c)
    return pd.DataFrame({
        "ts": np.arange(n) * 60_000, "open": c, "high": c * (1 + spread),
        "low": c * (1 - spread), "close": c, "volume": np.full(n, 10.0),
        "trades": np.full(n, 5), "taker_buy": np.full(n, 5.0)})


def _with_atr(df, atr=1.0):
    df = df.copy()
    df["atr"] = atr
    df["sma60"] = df.close
    return df


def test_entry_is_next_bar_open_and_costs_reduce_return():
    df = _with_atr(_bars([100] * 5 + [100, 110] + [110] * 10, spread=0))
    sig = np.zeros(len(df), bool)
    sig[4] = True
    ex = backtest.Exits(stop_atr=50, target_atr=None, max_hold=3)
    free = backtest.run(df, sig, ex, backtest.Costs(0, 0, 0))
    paid = backtest.run(df, sig, ex, backtest.Costs(fee=0.001, slippage=0.001))
    assert free.entry.iloc[0] == 100  # bar 5 open, not bar 4
    assert free.gross_ret.iloc[0] == pytest.approx(0.10)
    assert paid.net_ret.iloc[0] < free.net_ret.iloc[0]


def test_stop_wins_when_stop_and_target_hit_same_bar():
    df = _with_atr(_bars([100] * 10, spread=0))
    df.loc[3, ["high", "low"]] = [120, 80]
    sig = np.zeros(len(df), bool)
    sig[1] = True
    t = backtest.run(df, sig, backtest.Exits(stop_atr=5, target_atr=5, max_hold=20),
                     backtest.Costs(0, 0, 0))
    assert t.reason.iloc[0] == "stop"
    assert t.exit.iloc[0] == 95


def test_gap_below_stop_fills_at_open():
    df = _with_atr(_bars([100] * 4 + [90] * 4, spread=0))
    sig = np.zeros(len(df), bool)
    sig[0] = True
    t = backtest.run(df, sig, backtest.Exits(stop_atr=5, target_atr=None, max_hold=20),
                     backtest.Costs(0, 0, 0))
    assert t.exit.iloc[0] == 90


def test_no_overlapping_trades_per_symbol():
    df = _with_atr(_bars([100] * 50))
    sig = np.ones(len(df), bool)
    t = backtest.run(df, sig, backtest.Exits(stop_atr=50, target_atr=None, max_hold=10),
                     backtest.Costs(0, 0, 0))
    assert (t.entry_ts.iloc[1:].to_numpy() > t.exit_ts.iloc[:-1].to_numpy()).all()


def test_tax_only_on_winning_trades():
    t = pd.DataFrame({"entry_ts": [0, 1], "exit_ts": [1, 2], "minutes": [1, 1],
                      "gross_ret": [0.01, -0.01], "net_ret": [0.01, -0.01]})
    s = backtest.summarize(t, backtest.Costs(tax_on_gains=0.3), notional=10_000, days=1)
    assert s["net_pnl_before_tax"] == 0
    assert s["tax"] == 30
    assert s["net_pnl_after_tax"] == -30


def test_features_do_not_look_ahead():
    rng = np.random.default_rng(1)
    df = _bars(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 2000))))
    full = features.build(df)
    cut = features.build(df.iloc[:1500])
    cols = features.FEATURES
    pd.testing.assert_frame_equal(full[cols].iloc[:1500], cut[cols], check_exact=False)


def test_rules_return_boolean_signals():
    rng = np.random.default_rng(2)
    f = features.build(_bars(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 3000)))))
    for rule in strategies.RULES.values():
        sig, ex = rule(f)
        assert sig.dtype == bool and len(sig) == len(f)


def test_walk_forward_only_signals_after_training_months():
    rng = np.random.default_rng(3)
    n = 4 * 30 * 1440 // 10  # 4 "months" of 10-minute steps, scaled down below
    df = _bars(100 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    df["ts"] = np.arange(n) * 600_000  # 10-minute spacing keeps the test fast
    f = model.add_label(features.build(df), horizon=3, cost=0.0)
    folds, sig = model.walk_forward({"X": f}, horizon=3, cost=0.0, kind="logistic",
                                    top_frac=0.05)
    first_test = f.ts.min() + 2 * model.MONTH_MS
    assert folds and not sig["X"][(f.ts < first_test).to_numpy()].any()
