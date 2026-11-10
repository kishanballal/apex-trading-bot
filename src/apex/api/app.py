"""Dashboard API. Serves the paper account, scanner and charts, plus the static dashboard."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apex.analysis.regime import detect_regime
from apex.config import Settings
from apex.data.pipeline import build_providers, make_client, update_universe
from apex.data.store import CandleStore
from apex.models import AssetClass
from apex.paper.service import PaperTrader

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
REFRESH_MINUTES = 30


def future_dates(last: datetime, steps: int, asset_class: AssetClass) -> list[str]:
    """Next trading days: every day for crypto, weekdays for shares (holidays are ignored)."""
    out, day = [], last.date()
    while len(out) < steps:
        day += timedelta(days=1)
        if asset_class is AssetClass.CRYPTO or day.weekday() < 5:
            out.append(day.isoformat())
    return out


class Dashboard:
    def __init__(self, settings: Settings, *, fetch: bool = True) -> None:
        self.settings, self.fetch = settings, fetch
        self.trader = PaperTrader(settings)
        self.candles: dict = {}
        self.state: dict | None = None
        self.error: str | None = None
        self.lock = asyncio.Lock()
        self.asset_class = {s: g.asset_class for g in self.trader.groups for s in g.symbols}

    async def refresh(self) -> dict:
        async with self.lock:
            fetch_note = ""
            if self.fetch:
                try:
                    store = CandleStore(self.settings.db_path)
                    async with make_client(self.settings) as client:
                        results = await update_universe(
                            self.trader.groups, build_providers(client, self.settings), store,
                            history_days=self.settings.history_days,
                            max_concurrency=self.settings.max_concurrency)
                    store.close()
                    failed = [r.symbol for r in results if not r.ok]
                    fetch_note = f"{len(failed)} symbols failed to update" if failed else ""
                except Exception as exc:  # keep serving the last good data
                    log.exception("data refresh failed")
                    fetch_note = f"data refresh failed: {exc}"
            try:
                self.candles = await asyncio.to_thread(self.trader.load_candles)
                result = await asyncio.to_thread(self.trader.cycle, self.candles)
                self.state = self._build_state(result, fetch_note)
                self.error = None
            except Exception as exc:
                log.exception("paper cycle failed")
                self.error = str(exc)
            return self.state or {"error": self.error}

    def _build_state(self, result, fetch_note: str) -> dict:
        snap = self.trader.snapshot(result)
        held = {p["symbol"] for p in snap["positions"]}
        planned = {p["symbol"] for p in snap["planned"]}
        threshold = snap["config"]["threshold"]
        scanner = []
        for row in self.trader.store.latest_predictions():
            path = row["forecast"]
            symbol = row["symbol"]
            status = ("held" if symbol in held else "planned" if symbol in planned
                      else "signal" if row["p"] >= threshold else "watch")
            scanner.append({
                "symbol": symbol, "asset_class": self.asset_class[symbol].value,
                "date": row["date"], "close": row["close"], "p": row["p"],
                "forecast_10d": path[-1]["mid"] / row["close"] - 1 if path else 0.0,
                "status": status,
            })
        scanner.sort(key=lambda r: -r["p"])
        regimes = {}
        for g in self.trader.groups:
            regime = detect_regime(self.candles[g.benchmark].tail(600))
            regimes[g.asset_class.value] = {
                "benchmark": g.benchmark, "trend": regime.trend.value,
                "high_volatility": regime.high_volatility, "allows_longs": regime.allows_longs}
        return {**snap, "scanner": scanner, "regimes": regimes, "note": fetch_note,
                "generated_at": datetime.now(UTC).isoformat()}

    def chart(self, symbol: str) -> dict:
        frame = self.candles.get(symbol)
        if frame is None or frame.is_empty():
            raise HTTPException(404, f"no data for {symbol}")
        tail = frame.tail(180)
        candles = [{"time": r["ts"].date().isoformat(), "open": r["open"], "high": r["high"],
                    "low": r["low"], "close": r["close"]} for r in tail.iter_rows(named=True)]
        asset_class = self.asset_class.get(symbol, AssetClass.EQUITY)
        history = self.trader.store.forecast_history(symbol, 60)
        dates = [c["time"] for c in candles]
        # What the model expected for the next day, plotted on the day it was predicting.
        predicted = []
        for h in history:
            if h["date"] in dates and h["forecast"]:
                i = dates.index(h["date"])
                if i + 1 < len(dates):
                    predicted.append({"time": dates[i + 1], "value": h["forecast"][0]["mid"]})
        forecast = []
        if history and history[-1]["date"] == dates[-1]:
            latest = history[-1]
            steps = future_dates(tail["ts"][-1], len(latest["forecast"]), asset_class)
            forecast = [{"time": dates[-1], "mid": latest["close"], "low": latest["close"],
                         "high": latest["close"]}]
            forecast += [{"time": t, **{k: f[k] for k in ("mid", "low", "high")}}
                         for t, f in zip(steps, latest["forecast"])]
        state = self.state or {}
        markers = []
        for t in state.get("trades", []):
            if t["symbol"] == symbol:
                markers.append({"time": t["entry_time"][:10], "kind": "buy", "price": t["entry"]})
                markers.append({"time": t["exit_time"][:10], "kind": "sell", "price": t["exit"],
                                "pnl": t["pnl"]})
        position = next((p for p in state.get("positions", []) if p["symbol"] == symbol), None)
        if position:
            markers.append({"time": position["entry_time"][:10], "kind": "buy",
                            "price": position["entry"]})
        return {"symbol": symbol, "candles": candles, "predicted": predicted,
                "forecast": forecast, "markers": sorted(markers, key=lambda m: m["time"]),
                "position": position, "win_probability": history[-1]["p"] if history else None}


def create_app(settings: Settings | None = None, *, fetch: bool = True,
               background: bool = True) -> FastAPI:
    dash = Dashboard(settings or Settings(), fetch=fetch)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if background:
            async def loop():
                while True:
                    await dash.refresh()
                    await asyncio.sleep(REFRESH_MINUTES * 60)
            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="APEX", lifespan=lifespan)
    app.state.dashboard = dash

    @app.get("/api/state")
    async def state():
        if dash.state is None:
            await dash.refresh()
        if dash.state is None:
            raise HTTPException(503, dash.error or "starting up")
        return dash.state

    @app.post("/api/refresh")
    async def refresh():
        result = await dash.refresh()
        if dash.error:
            raise HTTPException(500, dash.error)
        return result

    @app.get("/api/chart")
    async def chart(symbol: str):
        if dash.state is None:
            await dash.refresh()
        return dash.chart(symbol)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
