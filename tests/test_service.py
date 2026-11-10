from __future__ import annotations

import json

from apex.config import Settings, UniverseGroup
from apex.data.store import CandleStore
from apex.models import AssetClass
from apex.risk.state import PortfolioState
from apex.service import build_engine, report_to_json, run_scan
from conftest import NOW, make_candles


def test_scan_shortlist_respects_portfolio_limits(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    store = CandleStore(settings.db_path)
    store.upsert("BENCH", "1d", make_candles(drift=0.002, vol=0.005))
    # Six candidates. TWIN copies UP0, so it must fail the correlation check.
    for i in range(5):
        store.upsert(f"UP{i}", "1d", make_candles(seed=7 + 100 * i, breakout=True))
    store.upsert("TWIN", "1d", make_candles(seed=7, breakout=True))
    store.upsert("DOWN", "1d", make_candles(drift=-0.002))
    groups = [UniverseGroup(AssetClass.EQUITY, "fake", "BENCH",
                            ("UP0", "TWIN", "UP1", "UP2", "UP3", "UP4", "DOWN", "MISSING"))]
    state = PortfolioState.fresh(settings.capital)

    report = run_scan(groups, store, build_engine(settings), state, settings, now=NOW)

    candidates = [r for r in report.rows if r.is_opportunity]
    assert len(candidates) >= 5
    assert len(report.approved) <= 4
    assert sum(report.decisions[r.symbol].notional for r in report.approved) <= settings.capital * 0.20
    assert sum(report.decisions[r.symbol].max_loss for r in report.approved) <= settings.capital * 0.01
    both_twins = {"UP0", "TWIN"} <= {r.symbol for r in report.approved}
    assert not both_twins
    by_symbol = {r.symbol: r for r in report.rows}
    assert not by_symbol["DOWN"].is_opportunity
    assert "history" in by_symbol["MISSING"].verdict
    assert state.positions == {}  # a scan never opens positions

    payload = json.loads(report_to_json(report))
    assert payload["regimes"]["equity"]["trend"] == "bull"
    assert len(payload["rows"]) == 8


def test_bear_market_scan_approves_nothing(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    store = CandleStore(settings.db_path)
    store.upsert("BENCH", "1d", make_candles(drift=-0.002, vol=0.005))
    store.upsert("UP0", "1d", make_candles(breakout=True))
    groups = [UniverseGroup(AssetClass.EQUITY, "fake", "BENCH", ("UP0",))]
    report = run_scan(groups, store, build_engine(settings),
                      PortfolioState.fresh(settings.capital), settings, now=NOW)
    assert report.approved == [] and report.decisions == {}
