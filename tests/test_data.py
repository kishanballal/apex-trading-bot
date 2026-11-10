from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from apex.frame import records

from apex.config import UniverseGroup, load_universe, Settings
from apex.data.base import DataError, candles_from_rows, get_json
from apex.data.binance import BinanceProvider
from apex.data.pipeline import update_symbol, update_universe
from apex.data.quality import clean_candles
from apex.data.store import CandleStore
from apex.data.yahoo import YahooProvider, parse_chart, to_yahoo_symbol
from apex.models import AssetClass
from conftest import NOW, make_candles

DAY_MS = 86_400_000


def yahoo_payload(stamps, closes, adj=None):
    return {"chart": {"error": None, "result": [{
        "timestamp": stamps,
        "indicators": {
            "quote": [{"open": closes, "high": [c and c + 1 for c in closes],
                       "low": [c and c - 1 for c in closes], "close": closes,
                       "volume": [1000 for _ in closes]}],
            "adjclose": [{"adjclose": adj or closes}],
        }}]}}


def test_yahoo_symbol_mapping():
    assert to_yahoo_symbol("RELIANCE") == "RELIANCE.NS"
    assert to_yahoo_symbol("^NSEI") == "^NSEI"
    assert to_yahoo_symbol("M&M") == "M&M.NS"


def test_yahoo_parse_skips_null_bars_and_applies_adjustment():
    frame = parse_chart(yahoo_payload([86400, 172800, 259200], [100.0, None, 200.0],
                                      adj=[50.0, None, 200.0]), "X")
    assert len(frame) == 2
    first = records(frame)[0]
    assert first["close"] == pytest.approx(50.0)
    assert first["high"] == pytest.approx(50.5)  # whole bar scaled, not only the close
    assert first["ts"] == datetime(1970, 1, 2, tzinfo=UTC)


def test_yahoo_parse_raises_on_error_and_empty():
    with pytest.raises(DataError):
        parse_chart({"chart": {"error": {"description": "No data found"}, "result": None}}, "X")
    with pytest.raises(DataError):
        parse_chart({"chart": {"error": None, "result": []}}, "X")


async def test_yahoo_provider_encodes_symbol():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.raw_path.decode()
        return httpx.Response(200, json=yahoo_payload([86400], [10.0]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        frame = await YahooProvider(client, "https://y.test").fetch_candles(
            "M&M", "1d", NOW - timedelta(days=5), NOW)
    assert "M%26M.NS" in seen["path"]
    assert len(frame) == 1


async def test_get_json_retries_then_succeeds_and_fails_fast_on_404():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    async with httpx.AsyncClient(transport=httpx.MockTransport(flaky)) as client:
        assert await get_json(client, "https://x.test/a", {}, backoff=0) == {"ok": 1}
    assert calls["n"] == 3

    calls["n"] = 0

    def missing(request):
        calls["n"] += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(missing)) as client:
        with pytest.raises(DataError):
            await get_json(client, "https://x.test/a", {}, backoff=0)
    assert calls["n"] == 1


async def test_binance_paginates(monkeypatch):
    monkeypatch.setattr("apex.data.binance.PAGE_LIMIT", 2)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    base = int(start.timestamp() * 1000)
    all_rows = [[base + i * DAY_MS, "1", "2", "0.5", "1.5", "10"] for i in range(5)]
    requests = []

    def handler(request):
        since = int(request.url.params["startTime"])
        requests.append(since)
        return httpx.Response(200, json=[r for r in all_rows if r[0] >= since][:2])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        frame = await BinanceProvider(client, "https://b.test").fetch_candles(
            "BTCUSDT", "1d", start, start + timedelta(days=30))
    assert len(frame) == 5
    assert len(requests) == 3
    assert frame["close"].tolist() == [1.5] * 5


def test_quality_drops_bad_duplicate_and_unfinished_rows():
    t0 = int((NOW - timedelta(days=3)).timestamp() * 1000)
    rows = [
        (t0, 10, 11, 9, 10.5, 100),
        (t0, 10, 11, 9, 10.6, 100),            # duplicate timestamp, last one wins
        (t0 + DAY_MS, 10, 9, 11, 10, 100),     # high below low
        (t0 + 2 * DAY_MS, 10, 11, 9, -1, 100), # negative close
        (int((NOW - timedelta(hours=2)).timestamp() * 1000), 10, 11, 9, 10, 100),  # still forming
    ]
    frame, report = clean_candles(candles_from_rows(rows), now=NOW, bar_duration=timedelta(days=1))
    assert len(frame) == 1
    assert frame["close"][0] == 10.6
    assert (report.dropped_invalid, report.dropped_duplicates, report.dropped_unfinished) == (2, 1, 1)


def test_quality_flags_suspicious_jump():
    t0 = int((NOW - timedelta(days=5)).timestamp() * 1000)
    rows = [(t0, 10, 11, 9, 10, 1), (t0 + DAY_MS, 20, 21, 19, 20, 1)]
    _, report = clean_candles(candles_from_rows(rows), now=NOW, bar_duration=timedelta(days=1))
    assert report.suspicious_jumps == 1 and not report.clean


def test_store_upsert_is_idempotent_and_replace_swaps_history(tmp_path):
    store = CandleStore(tmp_path / "db.sqlite")
    candles = make_candles(30)
    store.upsert("A", "1d", candles)
    store.upsert("A", "1d", candles)
    assert len(store.load("A", "1d")) == 30
    assert store.load("A", "1d", limit=5)["ts"].tolist() == candles["ts"].tail(5).tolist()
    store.upsert("A", "1d", candles.tail(10), replace=True)
    assert len(store.load("A", "1d")) == 10
    assert store.load("B", "1d").empty
    assert store.symbols("1d") == ["A"]


class FakeProvider:
    name = "fake"

    def __init__(self, full_refresh=False, fail=()):
        self.full_refresh = full_refresh
        self.fail = set(fail)
        self.starts = {}

    def bar_duration(self, interval):
        return timedelta(days=1)

    async def fetch_candles(self, symbol, interval, start, end):
        self.starts.setdefault(symbol, []).append(start)
        if symbol in self.fail:
            raise DataError(f"{symbol}: boom")
        return make_candles(50)


async def test_pipeline_survives_a_failing_symbol(tmp_path):
    store = CandleStore(tmp_path / "db.sqlite")
    provider = FakeProvider(fail={"BAD"})
    groups = [UniverseGroup(AssetClass.CRYPTO, "fake", "BENCH", ("GOOD", "BAD"))]
    results = await update_universe(groups, {"fake": provider}, store, now=NOW)
    by_symbol = {r.symbol: r for r in results}
    assert by_symbol["GOOD"].ok and by_symbol["BENCH"].ok and not by_symbol["BAD"].ok
    assert "boom" in by_symbol["BAD"].error
    assert store.load("BAD", "1d").empty


async def test_pipeline_resumes_incrementally(tmp_path):
    store = CandleStore(tmp_path / "db.sqlite")
    provider = FakeProvider()
    await update_symbol(provider, store, "A", "1d", history_days=730, now=NOW)
    await update_symbol(provider, store, "A", "1d", history_days=730, now=NOW)
    first, second = provider.starts["A"]
    assert first == NOW - timedelta(days=730)
    assert second == NOW - timedelta(days=1) - timedelta(days=3)  # last bar minus overlap
    assert len(store.load("A", "1d")) == 50


def test_universe_file_loads():
    groups = load_universe(Settings().universe_file)
    assert {g.asset_class for g in groups} == {AssetClass.EQUITY, AssetClass.CRYPTO}
    for g in groups:
        assert len(g.symbols) == len(set(g.symbols)) > 0
