from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from apex.config import CostModel, RiskLimits, Settings
from apex.models import AssetClass, Mode, Position, Side, TradeProposal
from apex.risk.engine import RiskEngine
from apex.risk.state import PortfolioState
from conftest import NOW

CAPITAL = 100_000.0
SIDE_COST = 0.002  # equity: 0.15% fees + 0.05% slippage


def proposal(symbol="AAA", entry=100.0, stop=98.0, targets=(105.0, 107.0), **kw) -> TradeProposal:
    base = dict(symbol=symbol, asset_class=AssetClass.EQUITY, side=Side.LONG, strategy="test",
                entry=entry, stop=stop, targets=targets, as_of=NOW - timedelta(hours=20))
    return TradeProposal(**{**base, **kw})


def position(symbol, notional=5000.0, entry=100.0) -> Position:
    return Position(symbol, AssetClass.EQUITY, Side.LONG, notional / entry, entry, entry * 0.97)


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


@pytest.fixture
def state() -> PortfolioState:
    return PortfolioState.fresh(CAPITAL)


def failed(decision) -> set[str]:
    return {c.name for c in decision.checks if not c.passed}


def test_defaults_match_the_project_brief():
    lim = RiskLimits()
    assert (lim.risk_per_trade, lim.daily_loss_limit, lim.weekly_drawdown_limit) == (0.0025, 0.0075, 0.02)
    assert (lim.max_portfolio_drawdown, lim.max_positions) == (0.05, 4)
    assert (lim.max_position_allocation, lim.max_deployed_capital, lim.max_leverage) == (0.05, 0.20, 1.0)
    assert Settings(_env_file=None).mode is Mode.PAPER and Settings(_env_file=None).capital == CAPITAL


def test_leverage_above_one_cannot_be_configured():
    with pytest.raises(ValidationError):
        RiskLimits(max_leverage=2.0)


def test_live_mode_is_locked_without_authorization(engine, state):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mode=Mode.LIVE)
    live = RiskEngine(mode=Mode.LIVE)
    assert "live_authorization" in failed(live.evaluate(proposal(), state, now=NOW))


def test_approves_and_caps_at_position_allocation(engine, state):
    d = engine.evaluate(proposal(), state, now=NOW)
    assert d.approved, d.rejections
    assert d.quantity == 50  # 5% of 1,00,000 at price 100
    assert d.notional == pytest.approx(5000)
    unit_loss = 2 + SIDE_COST * (100 + 98)
    assert d.max_loss == pytest.approx(50 * unit_loss)
    assert d.max_loss <= CAPITAL * 0.0025
    assert d.expected_return == pytest.approx(50 * (5 - SIDE_COST * (100 + 105)))


def test_wide_stop_is_sized_by_risk_budget(engine, state):
    d = engine.evaluate(proposal(stop=90.0, targets=(125.0,)), state, now=NOW)
    assert d.approved, d.rejections
    assert d.quantity == 24  # 250 / (10 + costs 0.38) = 24.08, whole shares
    assert d.max_loss <= 250.0


def test_no_approved_trade_ever_breaks_a_limit(engine):
    rng = np.random.default_rng(1)
    approved = 0
    for _ in range(500):
        state = PortfolioState.fresh(float(rng.uniform(20_000, 500_000)))
        entry = float(rng.uniform(5, 5000))
        stop = entry * (1 - float(rng.uniform(0.001, 0.2)))
        target = entry * (1 + float(rng.uniform(0.0, 0.5)))
        for i in range(int(rng.integers(0, 4))):
            state.open_position(position(f"H{i}", notional=state.equity * float(rng.uniform(0.01, 0.05))))
        before = state.deployed
        d = engine.evaluate(proposal(entry=entry, stop=stop, targets=(target,)), state, now=NOW,
                            returns={s: list(rng.normal(0, 0.01, 60)) for s in ["AAA", "H0", "H1", "H2"]})
        if not d.approved:
            assert d.quantity == 0 and d.rejections
            continue
        approved += 1
        assert d.max_loss <= state.equity * 0.0025 + 1e-6
        assert d.notional <= state.equity * 0.05 + 1e-6
        assert before + d.notional <= state.equity * 0.20 + 1e-6
        assert d.reward_risk >= 1.5 - 1e-9
        assert d.quantity == int(d.quantity)
    assert approved > 20  # the property was actually exercised


@pytest.mark.parametrize("kwargs, check", [
    (dict(stop=101.0), "stop_loss"),
    (dict(stop=99.95), "stop_distance"),
    (dict(stop=70.0, targets=(200.0,)), "stop_distance"),
    (dict(targets=(99.0,)), "target_side"),
    (dict(targets=(102.5,)), "reward_risk"),
    (dict(as_of=NOW - timedelta(days=10)), "fresh_signal"),
    (dict(as_of=NOW + timedelta(hours=1)), "fresh_signal"),
])
def test_bad_proposals_are_rejected(engine, state, kwargs, check):
    d = engine.evaluate(proposal(**kwargs), state, now=NOW)
    assert not d.approved and check in failed(d)
    assert d.quantity == 0 and d.notional == 0


@pytest.mark.parametrize("kwargs", [
    dict(stop=0.0), dict(stop=float("nan")), dict(entry=-5.0), dict(targets=()),
    dict(entry=float("inf")),
])
def test_missing_or_invalid_stop_and_prices_are_rejected(engine, state, kwargs):
    assert not engine.evaluate(proposal(**kwargs), state, now=NOW).approved


def test_max_positions_and_no_adding_to_a_position(engine, state):
    rets = {s: list(np.random.default_rng(i).normal(0, 0.01, 60))
            for i, s in enumerate(["AAA", "P0", "P1", "P2", "P3"])}
    for i in range(4):
        state.open_position(position(f"P{i}", notional=1000))
    d = engine.evaluate(proposal(), state, now=NOW, returns=rets)
    assert failed(d) == {"max_positions"}
    d = engine.evaluate(proposal(symbol="P0"), state, now=NOW, returns=rets)
    assert "no_duplicate_position" in failed(d)
    with pytest.raises(ValueError):
        state.open_position(position("P0"))


def test_deployed_capital_cap_shrinks_the_trade(engine, state):
    rets = {s: list(np.random.default_rng(i).normal(0, 0.01, 60))
            for i, s in enumerate(["AAA", "P0", "P1", "P2"])}
    for i, notional in enumerate([5000, 5000, 5000]):
        state.open_position(position(f"P{i}", notional=notional))
    d = engine.evaluate(proposal(), state, now=NOW, returns=rets)
    assert d.approved and d.quantity == 50  # 15% deployed, room for exactly 5% more
    state.positions["P0"] = position("P0", notional=9000)
    d = engine.evaluate(proposal(), state, now=NOW, returns=rets)
    assert d.approved and d.notional == pytest.approx(1000)  # only 1% of room left
    state.positions["P1"] = position("P1", notional=6000)
    d = engine.evaluate(proposal(), state, now=NOW, returns=rets)
    assert "position_size" in failed(d)


def test_liquidity_cap(engine, state):
    d = engine.evaluate(proposal(avg_daily_turnover=200_000.0), state, now=NOW)
    assert d.approved and d.quantity == 20  # 1% of turnover = 2,000 = 20 shares


def test_daily_and_weekly_loss_limits_halt_new_trades(engine, state):
    engine.account_checks(state, NOW)  # opens the day and week at 1,00,000
    state.mark(CAPITAL * (1 - 0.0075))
    assert "daily_loss_limit" in failed(engine.evaluate(proposal(), state, now=NOW))
    assert not engine.trading_allowed(state, NOW)

    tomorrow = NOW + timedelta(days=1)
    assert engine.trading_allowed(state, tomorrow)  # a new day resets the daily limit
    state.mark(CAPITAL * (1 - 0.021))
    two_days = NOW + timedelta(days=2)
    assert "weekly_drawdown_limit" in failed(engine.evaluate(proposal(), state, now=two_days))
    next_week = NOW + timedelta(days=7)
    assert "weekly_drawdown_limit" not in failed(engine.evaluate(proposal(), state, now=next_week))


def test_drawdown_kill_switch_latches_until_manual_reset(engine, state):
    engine.account_checks(state, NOW)
    state.mark(CAPITAL * 1.10)
    state.mark(CAPITAL * 1.10 * 0.95)  # exactly 5% below the peak
    later = NOW + timedelta(days=30)
    assert "kill_switch" in failed(engine.evaluate(proposal(), state, now=later))
    assert state.kill_switch and "drawdown" in state.kill_reason

    state.mark(CAPITAL * 1.20)  # recovering does not re-arm trading
    assert "kill_switch" in failed(engine.evaluate(proposal(), state, now=later + timedelta(days=30)))
    state.reset_kill_switch()
    assert engine.evaluate(proposal(as_of=later + timedelta(days=59)), state,
                           now=later + timedelta(days=60)).approved


def test_loss_streak_pauses_for_the_day_and_cooldown_blocks_reentry(engine, state):
    engine.account_checks(state, NOW)
    for i in range(3):
        state.open_position(position(f"L{i}", notional=1000))
        state.close_position(f"L{i}", pnl=-20.0, now=NOW)
    d = engine.evaluate(proposal(), state, now=NOW)
    assert "loss_streak_pause" in failed(d)
    assert "loss_streak_pause" in failed(engine.evaluate(proposal(), state, now=NOW + timedelta(hours=2)))

    tomorrow = NOW + timedelta(days=1)
    assert engine.evaluate(proposal(as_of=NOW + timedelta(hours=10)), state, now=tomorrow).approved
    assert state.consecutive_losses == 0

    d = engine.evaluate(proposal(symbol="L0"), state, now=NOW + timedelta(hours=23))
    assert "symbol_cooldown" in failed(d)
    d = engine.evaluate(proposal(symbol="L0", as_of=NOW + timedelta(hours=10)), state, now=tomorrow + timedelta(hours=1))
    assert "symbol_cooldown" not in failed(d)


def test_a_win_resets_the_loss_streak(state):
    for i, pnl in enumerate([-10.0, -10.0, 30.0]):
        state.open_position(position(f"W{i}"))
        state.close_position(f"W{i}", pnl=pnl, now=NOW)
    assert state.consecutive_losses == 0 and state.equity == pytest.approx(CAPITAL + 10)


def test_correlation_check(engine, state):
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 60)
    state.open_position(position("HELD", notional=1000))
    twin = {"AAA": list(base + rng.normal(0, 0.001, 60)), "HELD": list(base)}
    independent = {"AAA": list(rng.normal(0, 0.01, 60)), "HELD": list(base)}
    assert "correlation" in failed(engine.evaluate(proposal(), state, now=NOW, returns=twin))
    assert engine.evaluate(proposal(), state, now=NOW, returns=independent).approved
    # Missing or thin evidence is a rejection, never a free pass.
    assert "correlation" in failed(engine.evaluate(proposal(), state, now=NOW))
    assert "correlation" in failed(engine.evaluate(proposal(), state, now=NOW, returns={"AAA": list(base)}))
    thin = {"AAA": list(base[:10]), "HELD": list(base[:10])}
    assert "correlation" in failed(engine.evaluate(proposal(), state, now=NOW, returns=thin))


def test_crypto_is_sized_in_base_currency_with_fractions(engine, state):
    p = proposal(symbol="BTCUSDT", asset_class=AssetClass.CRYPTO, entry=100_000.0, stop=96_000.0,
                 targets=(108_000.0,), fx_to_base=88.0)
    d = engine.evaluate(p, state, now=NOW)
    assert d.approved, d.rejections
    assert 0 < d.quantity < 1
    assert d.notional == pytest.approx(d.quantity * 100_000 * 88) and d.notional <= 5000
    assert d.max_loss <= 250


def test_tiny_account_gets_no_trade_instead_of_an_oversized_one(engine):
    state = PortfolioState.fresh(5_000.0)
    d = engine.evaluate(proposal(entry=3000.0, stop=2940.0, targets=(3120.0,)), state, now=NOW)
    assert "position_size" in failed(d) and d.quantity == 0


def test_state_round_trips_through_disk(tmp_path, state):
    state.open_position(Position("AAA", AssetClass.EQUITY, Side.LONG, 10, 100.0, 97.0, 1.0, NOW))
    state.open_position(position("BBB"))
    state.close_position("BBB", -50.0, NOW)
    state.trip_kill_switch("test")
    state.roll(NOW)
    path = tmp_path / "nested" / "state.json"
    state.save(path)
    assert PortfolioState.load(path) == state
