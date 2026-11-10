from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from apex.analysis.indicators import prepare, rolling_percentile_rank
from apex.analysis.regime import Regime, Trend, regime_at
from apex.backtest.engine import BacktestConfig, Trade, run_backtest
from apex.backtest.metrics import (
    buy_and_hold, compute_metrics, estimate_tax, financial_year, max_drawdown, yearly_table,
)
from apex.config import CostModel, UniverseGroup
from apex.data.base import candles_from_rows
from apex.models import AssetClass, Side, TradeProposal
from apex.scanner import scanner
from apex.scanner.scanner import ScanRow, scan_bar, scan_bar_pullback, scan_symbol
from conftest import NOW, make_candles

BULL = Regime(Trend.BULL, False, "test")
START = datetime(2025, 1, 1, 3, 45, tzinfo=UTC)
SIDE = 0.002  # equity cost per side


def bars(spec: list[tuple[float, float, float, float]], start: datetime = START) -> pl.DataFrame:
    """spec rows are (open, high, low, close), one per consecutive day."""
    return candles_from_rows([
        (int((start + timedelta(days=i)).timestamp() * 1000), o, h, l, c, 1e7)
        for i, (o, h, l, c) in enumerate(spec)
    ])


FLAT = (100.0, 101.0, 99.0, 100.0)


@pytest.fixture
def fake_strategy(monkeypatch):
    """A strategy that signals on chosen bar numbers with a stop at 96 and a target at 110."""
    fire: dict[str, set[int]] = {}

    def strategy(symbol, asset_class, rows, i, regime, *, config=None, now=None, fx_to_base=1.0):
        if i not in fire.get(symbol, set()):
            return ScanRow(symbol, asset_class, 0, (), (), "NO TRADE: test")
        last = rows[i]
        proposal = TradeProposal(symbol, asset_class, Side.LONG, "fake", last["close"], 96.0,
                                 (110.0,), last["ts"], score=50.0, fx_to_base=fx_to_base)
        return ScanRow(symbol, asset_class, 50, (), (), "CANDIDATE", proposal, last["close"])

    monkeypatch.setitem(scanner.STRATEGIES, "fake", strategy)
    return fire


def run(spec, fire, fake_strategy, **cfg):
    fake_strategy["AAA"] = set(fire)
    groups = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", ("AAA",))]
    return run_backtest(groups, {"AAA": bars(spec)}, BacktestConfig(strategy="fake", **cfg))


def test_live_scan_and_backtest_see_identical_signals():
    """The per-bar strategy on full history must equal the live scan on truncated history."""
    candles = make_candles(330, breakout=True)
    rows = prepare(candles).to_dicts()
    for strategy, fn in (("trend", scan_bar), ("pullback", scan_bar_pullback)):
        candidates = 0
        for i in range(225, 330):
            now = rows[i]["ts"] + timedelta(hours=8)
            a = fn("T", AssetClass.EQUITY, rows, i, BULL, now=now)
            b = scan_symbol("T", AssetClass.EQUITY, candles.head(i + 1), BULL, now=now,
                            strategy=strategy)
            assert a == b, (strategy, i)
            candidates += a.is_opportunity
        assert candidates > 0, strategy  # the comparison covered real signals


def test_rolling_rank_matches_the_naive_definition():
    series = prepare(make_candles(330))["atr_pct"]
    rolled = rolling_percentile_rank(series)
    for i in (10, 59, 80, 200, 329):
        window = series.head(i + 1).drop_nulls().tail(252)
        expected = (None if window.len() < 60
                    else float((window < window[-1]).sum()) / (window.len() - 1))
        assert rolled[i] == (pytest.approx(expected) if expected is not None else None)


def test_regime_at_uses_only_past_bars():
    bench = make_candles(330, drift=0.002, vol=0.005)
    rows = prepare(bench).to_dicts()
    assert regime_at(rows, 250) == regime_at(prepare(bench.head(251)).to_dicts(), 250)
    assert regime_at(rows, 100).trend is Trend.UNKNOWN


def test_entry_is_next_open_and_target_exit(fake_strategy):
    spec = [FLAT, FLAT, (101.0, 103.0, 100.5, 102.0), (102.0, 111.0, 101.0, 110.0), FLAT]
    result = run(spec, {1}, fake_strategy)
    (t,) = result.trades
    assert t.entry == 101.0 and t.entry_time == START + timedelta(days=2)  # not the signal close
    assert t.exit == 110.0 and t.reason == "target" and t.bars_held == 2
    assert t.quantity == 46  # risk budget binds: 250 / (5 + costs 0.394) = 46.3
    assert t.gross_pnl == pytest.approx(46 * 9)
    assert t.costs == pytest.approx(46 * SIDE * (101 + 110))
    # Planned risk is distance to the stop plus both sides of costs.
    assert t.planned_risk == pytest.approx(46 * (5 + SIDE * (101 + 96)))
    assert result.equity_curve[-1][1] == pytest.approx(100_000 + t.pnl)


def test_entry_with_poor_reward_to_risk_is_vetoed_by_the_risk_engine(fake_strategy):
    # Opening at 102 leaves 8 of reward against 6 of risk: below 1.5 after costs.
    spec = [FLAT, FLAT, (102.0, 103.0, 101.0, 102.0), FLAT]
    result = run(spec, {1}, fake_strategy)
    assert result.trades == [] and result.rejections == {"reward_risk": 1}


def test_stop_exit_and_gap_through_stop(fake_strategy):
    stopped = run([FLAT, FLAT, FLAT, (99.0, 100.0, 95.0, 97.0), FLAT], {1}, fake_strategy)
    assert (stopped.trades[0].exit, stopped.trades[0].reason) == (96.0, "stop")
    gapped = run([FLAT, FLAT, FLAT, (90.0, 91.0, 89.0, 90.0), FLAT], {1}, fake_strategy)
    assert (gapped.trades[0].exit, gapped.trades[0].reason) == (90.0, "stop")  # worse than the stop
    assert gapped.trades[0].r_multiple < -1  # a gap can lose more than planned, and we count it


def test_bar_touching_stop_and_target_counts_as_a_stop(fake_strategy):
    result = run([FLAT, FLAT, FLAT, (100.0, 112.0, 95.0, 105.0), FLAT], {1}, fake_strategy)
    assert result.trades[0].reason == "stop"


def test_same_day_stop_after_entry(fake_strategy):
    result = run([FLAT, FLAT, (100.0, 101.0, 95.0, 96.5), FLAT], {1}, fake_strategy)
    assert result.trades[0].reason == "stop" and result.trades[0].bars_held == 1


def test_gap_beyond_stop_or_target_at_entry_is_skipped(fake_strategy):
    result = run([FLAT, FLAT, (95.0, 97.0, 94.0, 96.0), FLAT], {1}, fake_strategy)
    assert result.trades == [] and result.rejections == {"gap_through_stop_or_target": 1}
    assert result.signals == 1


def test_time_stop_and_end_of_data_exit(fake_strategy):
    timed = run([FLAT] * 10, {1}, fake_strategy, max_hold_bars=3)
    assert (timed.trades[0].reason, timed.trades[0].bars_held) == ("time", 3)
    ended = run([FLAT] * 5, {1}, fake_strategy)
    assert ended.trades[0].reason == "end"
    # A flat round trip loses exactly the costs.
    assert ended.trades[0].pnl == pytest.approx(-50 * SIDE * 200)
    assert ended.equity_curve[-1][1] == pytest.approx(100_000 - 20)


def test_signal_on_last_bar_never_trades(fake_strategy):
    assert run([FLAT] * 5, {4}, fake_strategy).trades == []


def test_equity_curve_reconciles_with_trades(fake_strategy):
    spec = [FLAT, FLAT, FLAT, (100.0, 111.0, 99.0, 110.0), FLAT, FLAT, FLAT,
            (99.0, 100.0, 95.0, 96.0), FLAT, FLAT]
    result = run(spec, {1, 5}, fake_strategy)
    assert [t.reason for t in result.trades] == ["target", "stop"]
    assert result.equity_curve[-1][1] == pytest.approx(100_000 + sum(t.pnl for t in result.trades))
    assert len(result.equity_curve) == len(spec) == len(result.exposure)


def test_portfolio_limits_hold_during_a_backtest(fake_strategy):
    symbols = tuple(f"S{i}" for i in range(8))
    candles = {}
    for sym in symbols:
        fake_strategy[sym] = {65}
        candles[sym] = bars([FLAT] * 90)
    groups = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", symbols)]
    result = run_backtest(groups, candles, BacktestConfig(strategy="fake"))
    assert len(result.trades) == 4  # eight signals, four slots
    assert result.rejections.get("max_positions") == 4
    assert sum(t.quantity * t.entry for t in result.trades) <= 20_000 + 1e-6
    assert max(result.exposure) < 0.22
    for t in result.trades:
        assert t.planned_risk <= 250 + 1e-6


def test_kill_switch_stops_new_trades_until_reset(fake_strategy):
    # One catastrophic gap takes the account past the 5% drawdown limit.
    limits_spec = [FLAT, FLAT, FLAT, (1.0, 1.0, 1.0, 1.0)] + [(1.0, 1.01, 0.99, 1.0)] * 6
    fake_strategy["AAA"] = {1, 6}
    groups = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", ("AAA",))]
    from apex.config import RiskLimits
    loose = RiskLimits(max_position_allocation=0.25, max_deployed_capital=1.0)
    result = run_backtest(groups, {"AAA": bars(limits_spec)}, BacktestConfig(strategy="fake"),
                          limits=loose)
    assert len(result.trades) == 1 and len(result.kill_switch_dates) == 1
    assert result.signals == 1  # after the switch trips the scanner is not even consulted


def test_metrics_arithmetic():
    assert max_drawdown([100, 120, 90, 130, 117]) == pytest.approx(0.25)
    assert max_drawdown([]) == 0
    assert financial_year(date(2025, 3, 31)) == 2024 and financial_year(date(2025, 4, 1)) == 2025
    hold = buy_and_hold([(date(2020, 1, 1), 100.0), (date(2021, 1, 1), 50.0),
                         (date(2022, 1, 1), 121.0)])
    assert hold["total_return"] == pytest.approx(0.21) and hold["max_drawdown"] == pytest.approx(0.5)
    assert hold["cagr"] == pytest.approx(0.10, abs=1e-3)


def _trade(pnl, asset=AssetClass.EQUITY, when=datetime(2025, 6, 1, tzinfo=UTC), costs=0.0):
    return Trade("X", asset, "s", when, when, 1, 100, 100 + pnl, 95, 1.0, pnl + costs, costs,
                 10.0, "target" if pnl > 0 else "stop", 5)


def test_tax_estimate():
    equity = [_trade(1000), _trade(-400), _trade(-500, when=datetime(2026, 6, 1, tzinfo=UTC))]
    assert estimate_tax(equity) == pytest.approx(0.20 * 600)  # the later losing year gives no refund
    crypto = [_trade(1000, AssetClass.CRYPTO), _trade(-900, AssetClass.CRYPTO)]
    assert estimate_tax(crypto) == pytest.approx(300)  # losses cannot offset crypto gains


def test_compute_metrics_on_known_trades(fake_strategy):
    spec = [FLAT, FLAT, FLAT, (100.0, 111.0, 99.0, 110.0), FLAT, FLAT, FLAT,
            (99.0, 100.0, 95.0, 96.0), FLAT, FLAT]
    result = run(spec, {1, 5}, fake_strategy)
    m = compute_metrics(result)
    win, loss = result.trades
    assert m.trades == 2 and m.win_rate == 0.5
    assert m.profit_factor == pytest.approx(win.pnl / -loss.pnl)
    assert m.net_profit == pytest.approx(win.pnl + loss.pnl)
    assert m.expectancy == pytest.approx(m.net_profit / 2)
    assert m.total_costs == pytest.approx(win.costs + loss.costs)
    assert m.net_profit_after_tax == pytest.approx(m.net_profit - 0.20 * m.net_profit)
    assert m.exit_reasons == {"target": 1, "stop": 1}
    assert 0 < m.max_drawdown < 0.01
    table = yearly_table(result)
    assert len(table) == 1 and table[0]["trades"] == 2
    assert table[0]["return"] == pytest.approx(m.total_return)


def test_empty_backtest_is_safe():
    groups = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", ("NONE",))]
    result = run_backtest(groups, {}, BacktestConfig())
    m = compute_metrics(result)
    assert result.trades == [] and m.trades == 0 and m.net_profit == 0
