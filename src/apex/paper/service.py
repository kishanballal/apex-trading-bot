"""Paper trading.

Design: the paper account is not a pile of mutable state. Each cycle it
  1. downloads the latest finished candles,
  2. asks the frozen model for today's win probabilities and stores them. A stored prediction
     is never overwritten, so hindsight cannot leak into the record,
  3. replays the standard backtester from the paper start date to now using only the stored
     predictions.
The replay IS the account: positions, cash, trades and tomorrow's planned orders come out of
the same tested code as the backtests. Running a cycle twice changes nothing, and a missed day
is caught up on the next run.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from apex.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from apex.backtest.metrics import compute_metrics
from apex.config import RiskLimits, Settings, UniverseGroup, load_universe, risk_profile
from apex.data.store import CandleStore
from apex.ml.dataset import build_dataset
from apex.ml.live import ModelBundle, forecast_path, train_bundle
from apex.ml.strategy import make_table_strategy
from apex.scanner import scanner

STRATEGY_NAME = "ml_logistic"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_predictions (
    symbol TEXT NOT NULL, date TEXT NOT NULL, p REAL NOT NULL, forecast TEXT NOT NULL,
    close REAL NOT NULL, atr_pct REAL, made_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
) WITHOUT ROWID;
"""


@dataclass(frozen=True)
class PaperConfig:
    start: date
    capital: float = 50_000.0
    floor: float = 40_000.0
    profile: str = "aggressive"
    threshold: float = 0.45

    def limits(self) -> RiskLimits:
        base = risk_profile(self.profile).model_dump()
        # The floor defines the kill switch: 50,000 with a 40,000 floor is a 20% drawdown.
        base["max_portfolio_drawdown"] = min(1 - self.floor / self.capital, 0.20)
        return RiskLimits(**base)


class PaperStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # The dashboard runs cycles in a worker thread; callers serialise access with a lock.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str):
        with closing(self._conn.execute("SELECT value FROM paper_meta WHERE key = ?", (key,))) as c:
            row = c.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO paper_meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, default=str)))

    def config(self) -> PaperConfig | None:
        raw = self.get("config")
        return PaperConfig(**{**raw, "start": date.fromisoformat(raw["start"])}) if raw else None

    def save_config(self, cfg: PaperConfig) -> None:
        self.set("config", {**cfg.__dict__, "start": cfg.start.isoformat()})

    def add_predictions(self, rows: list[tuple]) -> int:
        """Insert predictions. Existing (symbol, date) rows are kept as first recorded."""
        with self._conn:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO paper_predictions VALUES (?,?,?,?,?,?,?)", rows)
        return cur.rowcount

    def predictions(self) -> dict[tuple[str, date], float]:
        with closing(self._conn.execute("SELECT symbol, date, p FROM paper_predictions")) as c:
            return {(s, date.fromisoformat(d)): p for s, d, p in c.fetchall()}

    def latest_predictions(self) -> list[dict]:
        sql = ("SELECT p.symbol, p.date, p.p, p.forecast, p.close, p.atr_pct FROM paper_predictions p "
               "JOIN (SELECT symbol, MAX(date) AS d FROM paper_predictions GROUP BY symbol) m "
               "ON p.symbol = m.symbol AND p.date = m.d")
        with closing(self._conn.execute(sql)) as c:
            return [{"symbol": s, "date": d, "p": p, "forecast": json.loads(f), "close": cl,
                     "atr_pct": a} for s, d, p, f, cl, a in c.fetchall()]

    def forecast_history(self, symbol: str, limit: int = 30) -> list[dict]:
        with closing(self._conn.execute(
            "SELECT date, p, forecast, close FROM paper_predictions WHERE symbol = ? "
            "ORDER BY date DESC LIMIT ?", (symbol, limit))) as c:
            return [{"date": d, "p": p, "forecast": json.loads(f), "close": cl}
                    for d, p, f, cl in c.fetchall()][::-1]


class PaperTrader:
    def __init__(self, settings: Settings, groups: list[UniverseGroup] | None = None) -> None:
        self.settings = settings
        self.groups = groups or load_universe(settings.universe_file)
        self.model_path = settings.data_dir / "models" / "paper_model.joblib"
        self.store = PaperStore(settings.db_path)

    # --- setup ---------------------------------------------------------------------------

    def load_candles(self) -> dict[str, pl.DataFrame]:
        candles = CandleStore(self.settings.db_path)
        try:
            return {s: candles.load(s, "1d") for g in self.groups for s in (g.benchmark, *g.symbols)}
        finally:
            candles.close()

    def init(self, *, capital: float = 50_000.0, floor: float = 40_000.0,
             profile: str = "aggressive", threshold: float = 0.45,
             start: date | None = None, retrain: bool = True) -> PaperConfig:
        """Start (or restart) the paper account and freeze the model it will trade with."""
        if not 0 < floor < capital:
            raise ValueError("the floor must be below the starting capital")
        candles = self.load_candles()
        if retrain or not self.model_path.exists():
            train_bundle(build_dataset(self.groups, candles), candles).save(self.model_path)
        cfg = PaperConfig(start or datetime.now(UTC).date(), capital, floor, profile, threshold)
        with self.store._conn:
            self.store._conn.execute("DELETE FROM paper_predictions")
        self.store.save_config(cfg)
        return cfg

    # --- daily cycle ---------------------------------------------------------------------

    def record_predictions(self, candles: dict[str, pl.DataFrame], cfg: PaperConfig) -> int:
        bundle = ModelBundle.load(self.model_path)
        data = build_dataset(self.groups, candles).filter(pl.col("date") >= cfg.start)
        if data.is_empty():
            return 0
        p = bundle.win_probability(data)
        fwd = bundle.forecast_returns(data)
        closes = {(s, r["ts"].date()): r["close"] for s, f in candles.items()
                  for r in f.select("ts", "close").iter_rows(named=True)}
        now = datetime.now(UTC).isoformat()
        rows = []
        for i, (symbol, day, atr_pct) in enumerate(data.select("symbol", "date", "atr_pct").iter_rows()):
            close = closes[(symbol, day)]
            path = forecast_path(close, {h: float(v[i]) for h, v in fwd.items()}, atr_pct)
            rows.append((symbol, day.isoformat(), float(p[i]), json.dumps(path), close, atr_pct, now))
        return self.store.add_predictions(rows)

    def replay(self, candles: dict[str, pl.DataFrame], cfg: PaperConfig) -> BacktestResult:
        table = self.store.predictions()
        scanner.STRATEGIES["_paper"] = make_table_strategy(STRATEGY_NAME, table, cfg.threshold)
        try:
            return run_backtest(
                self.groups, candles,
                BacktestConfig(strategy="_paper", capital=cfg.capital, usd_inr=self.settings.usd_inr,
                               start=cfg.start, close_at_end=False),
                limits=cfg.limits())
        finally:
            scanner.STRATEGIES.pop("_paper", None)

    def cycle(self, candles: dict[str, pl.DataFrame] | None = None) -> BacktestResult:
        cfg = self.store.config()
        if cfg is None:
            raise RuntimeError("paper trading is not set up yet; run `apex paper init` first")
        candles = candles or self.load_candles()
        added = self.record_predictions(candles, cfg)
        result = self.replay(candles, cfg)
        self.store.set("last_cycle", {"at": datetime.now(UTC).isoformat(), "new_predictions": added})
        return result

    # --- read model for the dashboard ----------------------------------------------------

    def snapshot(self, result: BacktestResult) -> dict:
        cfg = self.store.config()
        m = compute_metrics(result)
        equity = result.equity_curve[-1][1] if result.equity_curve else cfg.capital
        peak = max([cfg.capital, *[v for _, v in result.equity_curve]])
        state = result.state
        return {
            "config": {**cfg.__dict__, "start": cfg.start.isoformat()},
            "mode": "paper",
            "equity": equity, "cash": result.cash, "peak": peak,
            "pnl": equity - cfg.capital, "pnl_pct": equity / cfg.capital - 1,
            "drawdown": 1 - equity / peak if peak else 0.0,
            "room_to_floor": equity - cfg.floor,
            "kill_switch": bool(state and state.kill_switch),
            "kill_reason": state.kill_reason if state else "",
            "days_running": len(result.equity_curve),
            "last_bar": result.equity_curve[-1][0].isoformat() if result.equity_curve else None,
            "last_cycle": self.store.get("last_cycle"),
            "stats": {"trades": m.trades, "win_rate": m.win_rate,
                      "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
                      "costs": m.total_costs, "exposure": result.exposure[-1] if result.exposure else 0.0},
            "equity_curve": [{"date": d.isoformat(), "equity": v} for d, v in result.equity_curve],
            "positions": [{**p, "entry_time": p["entry_time"].isoformat()} for p in result.open_positions],
            "planned": [{"symbol": p.symbol, "entry_ref": p.entry, "stop": p.stop,
                         "target": p.targets[0], "score": p.score, "as_of": p.as_of.isoformat()}
                        for p in result.pending],
            "trades": [{"symbol": t.symbol, "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(), "quantity": t.quantity,
                        "entry": t.entry, "exit": t.exit, "pnl": t.pnl, "r": t.r_multiple,
                        "reason": t.reason} for t in result.trades],
            "rejections": result.rejections,
        }
