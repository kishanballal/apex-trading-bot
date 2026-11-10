from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app, future_dates
from apex.backtest.engine import BacktestConfig, run_backtest
from apex.config import Settings
from apex.data.store import CandleStore
from apex.ml.live import HORIZONS, forecast_path
from apex.models import AssetClass
from apex.paper.service import PaperConfig, PaperTrader
from test_ml import END, GROUPS, universe

START = date(2025, 9, 1)


@pytest.fixture(scope="module")
def trader(tmp_path_factory):
    settings = Settings(_env_file=None, data_dir=tmp_path_factory.mktemp("paper"))
    store = CandleStore(settings.db_path)
    for symbol, frame in universe(1400).items():
        store.upsert(symbol, "1d", frame)
    store.close()
    t = PaperTrader(settings, GROUPS)
    t.init(capital=50_000, floor=40_000, start=START)
    return t


def test_floor_sets_the_kill_switch_and_is_validated(trader):
    assert PaperConfig(START).limits().max_portfolio_drawdown == pytest.approx(0.20)
    assert PaperConfig(START, capital=50_000, floor=45_000).limits().max_portfolio_drawdown \
        == pytest.approx(0.10)
    with pytest.raises(ValueError):
        trader.init(capital=50_000, floor=60_000, retrain=False)
    assert trader.store.config().floor == 40_000  # a rejected init changes nothing


def test_cycle_is_idempotent_and_never_rewrites_a_prediction(trader):
    first = trader.snapshot(trader.cycle())
    table = trader.store.predictions()
    assert table and min(d for _, d in table) >= START  # nothing recorded before the start
    assert max(d for _, d in table) == END.date()
    second = trader.snapshot(trader.cycle())
    assert trader.store.get("last_cycle")["new_predictions"] == 0
    for key in ("equity", "cash", "positions", "planned", "trades", "equity_curve"):
        assert first[key] == second[key], key
    # Even a different model cannot change what was already recorded.
    symbol, day = next(iter(table))
    trader.store.add_predictions([(symbol, day.isoformat(), 0.99, "[]", 1.0, 0.01, "later")])
    assert trader.store.predictions()[(symbol, day)] == table[(symbol, day)]


def test_snapshot_accounting(trader):
    result = trader.cycle()
    snap = trader.snapshot(result)
    invested = sum(p["value"] for p in snap["positions"])
    assert snap["equity"] == pytest.approx(snap["cash"] + invested)
    assert snap["room_to_floor"] == pytest.approx(snap["equity"] - 40_000)
    assert snap["days_running"] == len(snap["equity_curve"]) > 50
    assert snap["mode"] == "paper"


def test_replay_keeps_positions_open_only_when_asked(trader):
    candles, cfg = trader.load_candles(), trader.store.config()
    table = {key: 0.99 for key in trader.store.predictions()}  # force signals
    from apex.ml.strategy import make_table_strategy
    from apex.scanner import scanner
    from apex.scanner.scanner import ScannerConfig
    scanner.STRATEGIES["_t"] = make_table_strategy("t", table, 0.5)
    try:
        kw = dict(strategy="_t", capital=50_000, start=START,
                  scanner=ScannerConfig(require_regime=False))
        live = run_backtest(GROUPS, candles, BacktestConfig(**kw, close_at_end=False),
                            limits=cfg.limits())
        closed = run_backtest(GROUPS, candles, BacktestConfig(**kw), limits=cfg.limits())
    finally:
        scanner.STRATEGIES.pop("_t")
    assert live.open_positions and not closed.open_positions
    assert len(closed.trades) == len(live.trades) + len(live.open_positions)
    assert all(t.reason != "end" for t in live.trades)
    held = sum(p["value"] for p in live.open_positions)
    assert live.equity_curve[-1][1] == pytest.approx(live.cash + held)


def test_forecast_path_shape():
    path = forecast_path(100.0, {h: 0.01 * h for h in HORIZONS}, atr_pct=0.02)
    assert [p["step"] for p in path] == list(range(1, 11))
    assert path[0]["mid"] == pytest.approx(100 * 2.718281828 ** 0.01, rel=1e-6)
    assert all(p["low"] < p["mid"] < p["high"] for p in path)
    widths = [p["high"] - p["low"] for p in path]
    assert widths == sorted(widths)  # uncertainty grows with time


def test_future_dates_skip_weekends_for_shares_only():
    friday = END.replace(year=2025, month=11, day=7)
    assert future_dates(friday, 2, AssetClass.EQUITY) == ["2025-11-10", "2025-11-11"]
    assert future_dates(friday, 2, AssetClass.CRYPTO) == ["2025-11-08", "2025-11-09"]


def test_dashboard_api(trader):
    app = create_app(trader.settings, fetch=False, background=False)
    app.state.dashboard.trader = trader
    app.state.dashboard.asset_class = {s: g.asset_class for g in GROUPS for s in g.symbols}
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        assert state["config"]["capital"] == 50_000 and len(state["scanner"]) == 3
        assert {r["status"] for r in state["scanner"]} <= {"held", "planned", "signal", "watch"}
        assert all(0 <= r["p"] <= 1 for r in state["scanner"])
        chart = client.get("/api/chart", params={"symbol": "A"}).json()
        assert len(chart["candles"]) == 180 and len(chart["forecast"]) == 11
        assert chart["forecast"][0]["time"] == chart["candles"][-1]["time"]
        assert chart["forecast"][1]["time"] > chart["candles"][-1]["time"]
        assert chart["predicted"] and chart["predicted"][-1]["time"] <= chart["candles"][-1]["time"]
        assert client.get("/api/chart", params={"symbol": "NOPE"}).status_code == 404
        assert client.post("/api/refresh").status_code == 200
        page = client.get("/")
        assert page.status_code == 200 and "APEX" in page.text
        assert client.get("/static/lightweight-charts.js").status_code == 200
