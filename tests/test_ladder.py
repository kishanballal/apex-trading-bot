from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from apex.backtest.ladder import LadderConfig, is_turning_up, run_ladder
from apex.config import UniverseGroup
from apex.data.base import candles_from_rows
from apex.models import AssetClass

START = datetime(2024, 1, 1, 3, 45, tzinfo=UTC)
GROUPS = [UniverseGroup(AssetClass.EQUITY, "x", "BENCH", ("AAA",))]
# 230 quiet bars build the indicators (ATR settles at 2.0), then the scripted bars follow.
WARMUP = [(100.0, 101.0, 99.0, 100.0)] * 230
CFG = LadderConfig(capital=50_000, floor=40_000, require_uptrend=False,
                   symbol_allocation=0.15, dip_atr=3.0)


def run(script, cfg=CFG):
    rows = [(int((START + timedelta(days=i)).timestamp() * 1000), o, h, l, c, 1e7)
            for i, (o, h, l, c) in enumerate(WARMUP + script)]
    return run_ladder(GROUPS, {"AAA": candles_from_rows(rows)}, cfg)


DROP = [(100.0, 100.0, 92.0, 92.0)]  # falls 8 with ATR about 2.4: deep in the buy zone
TURN = [(92.0, 94.0, 91.5, 93.5)]  # up day closing above the prior close


def test_turning_up_rule():
    assert is_turning_up({"open": 92, "close": 93.5}, {"close": 92})
    assert not is_turning_up({"open": 94, "close": 93.5}, {"close": 92})  # red candle
    assert not is_turning_up({"open": 90, "close": 91}, {"close": 92})  # still below yesterday


def test_no_buy_while_still_falling():
    falling = DROP + [(92.0, 92.5, 88.0, 88.5), (88.5, 89.0, 85.0, 85.5)]
    assert run(falling).trades == []


def test_buys_after_the_turn_at_next_open_and_sells_at_profit_threshold():
    script = DROP + TURN + [(94.0, 95.0, 93.0, 94.5), (95.0, 104.0, 94.5, 103.0)]
    (t,) = run(script).trades
    assert t.entry == 94.0 and t.entry_time == START + timedelta(days=232)  # open after the turn
    assert t.reason == "target" and t.exit > t.entry and t.pnl > 0
    # First tranche is 1 / 3.75 of the 15% symbol allocation: 2,000 rupees, 21 whole shares.
    assert t.quantity == 21


def test_adds_a_larger_tranche_after_a_further_dip_and_caps_the_count():
    script = (DROP + TURN + [(94.0, 94.5, 93.0, 93.5)]
              + [(93.0, 93.0, 86.0, 86.5), (86.5, 88.5, 86.0, 88.0), (88.5, 89.0, 87.5, 88.0)]
              + [(88.0, 88.0, 80.0, 80.5), (80.5, 82.5, 80.0, 82.0), (82.5, 83.0, 81.5, 82.0)]
              + [(82.0, 82.0, 76.0, 76.5), (76.5, 78.5, 76.0, 78.0), (78.5, 79.0, 77.5, 78.0)])
    result = run(script, LadderConfig(**{**CFG.__dict__, "stop_atr": 50.0}))
    (t,) = result.trades
    # Three fills at 94, 88.5 and 82.5 with growing sizes; the fourth dip adds nothing.
    assert t.quantity == 21 + 28 + 36
    assert t.entry == pytest.approx((21 * 94 + 28 * 88.5 + 36 * 82.5) / 85)
    assert t.entry < 94  # averaging down lowered the break-even price
    assert max(result.exposure) < 0.16  # never above the 15% symbol allocation at cost


def test_hard_stop_below_lowest_fill():
    script = DROP + TURN + [(94.0, 94.5, 93.0, 93.5), (93.0, 93.0, 70.0, 71.0)]
    (t,) = run(script).trades
    assert t.reason == "stop" and t.pnl < 0
    assert t.exit == pytest.approx(t.stop)


def test_account_floor_liquidates_and_stops_trading_for_good():
    cfg = LadderConfig(**{**CFG.__dict__, "symbol_allocation": 0.95, "tranche_weights": (1.0,),
                          "stop_atr": 500.0, "take_profit_atr": 500.0})
    script = (DROP + TURN + [(94.0, 94.5, 93.0, 93.5), (93.0, 93.0, 75.0, 76.0),
                             (76.0, 77.0, 75.0, 76.0)] + DROP + TURN + [(94.0, 95.0, 93.0, 94.0)] * 3)
    result = run(script, cfg)
    assert len(result.kill_switch_dates) == 1
    assert [t.reason for t in result.trades] == ["kill_switch"]  # sold next open, never re-entered
    assert result.equity_curve[-1][1] == pytest.approx(50_000 + result.trades[0].pnl)


def test_equity_reconciles_with_trades():
    script = DROP + TURN + [(94.0, 95.0, 93.0, 94.5), (95.0, 104.0, 94.5, 103.0)]
    result = run(script)
    assert result.equity_curve[-1][1] == pytest.approx(50_000 + sum(t.pnl for t in result.trades))
