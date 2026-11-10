"""Glue between data, scanner and risk engine. The CLI and (later) the API both call this."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from apex.analysis.regime import Regime, detect_regime
from apex.config import CostModel, RiskLimits, Settings, UniverseGroup
from apex.data.store import CandleStore
from apex.models import AssetClass, Position, RiskDecision
from apex.risk.engine import RiskEngine
from apex.risk.state import PortfolioState
from apex.scanner.scanner import ScannerConfig, ScanRow, rank, scan_symbol

RETURN_WINDOW = 60


@dataclass(frozen=True)
class ScanReport:
    generated_at: datetime
    regimes: dict[str, Regime]
    rows: list[ScanRow]
    decisions: dict[str, RiskDecision]  # only for scanner candidates

    @property
    def approved(self) -> list[ScanRow]:
        return [r for r in self.rows if r.symbol in self.decisions and self.decisions[r.symbol].approved]


def load_state(settings: Settings) -> PortfolioState:
    path = settings.data_dir / "portfolio_state.json"
    return PortfolioState.load(path) if path.exists() else PortfolioState.fresh(settings.capital)


def build_engine(settings: Settings) -> RiskEngine:
    return RiskEngine(RiskLimits(), CostModel(), mode=settings.mode,
                      live_authorized=settings.live_authorized, timezone=settings.timezone)


def _returns(candles: pd.DataFrame) -> list[float]:
    return candles["close"].pct_change().dropna().tail(RETURN_WINDOW).tolist()


def run_scan(
    groups: list[UniverseGroup],
    store: CandleStore,
    engine: RiskEngine,
    state: PortfolioState,
    settings: Settings,
    *,
    interval: str = "1d",
    now: datetime | None = None,
    scanner_config: ScannerConfig | None = None,
) -> ScanReport:
    now = now or datetime.now(UTC)
    regimes: dict[str, Regime] = {}
    rows: list[ScanRow] = []
    returns: dict[str, list[float]] = {
        s: _returns(store.load(s, interval, 400)) for s in state.positions
    }

    for group in groups:
        regime = detect_regime(store.load(group.benchmark, interval, 600))
        regimes[group.asset_class.value] = regime
        fx = settings.usd_inr if group.asset_class is AssetClass.CRYPTO else 1.0
        for symbol in group.symbols:
            candles = store.load(symbol, interval, 600)
            row = scan_symbol(symbol, group.asset_class, candles, regime,
                              config=scanner_config, now=now, fx_to_base=fx)
            rows.append(row)
            if row.is_opportunity:
                returns[symbol] = _returns(candles)

    rows = rank(rows)
    # Evaluate best-first against a scratch copy of the portfolio, so the shortlist as a whole
    # respects position-count, exposure and correlation limits. Nothing is opened here.
    scratch = copy.deepcopy(state)
    decisions: dict[str, RiskDecision] = {}
    for row in rows:
        if not row.is_opportunity:
            continue
        p = row.proposal
        decision = engine.evaluate(p, scratch, now=now, returns=returns)
        decisions[row.symbol] = decision
        if decision.approved:
            scratch.open_position(Position(p.symbol, p.asset_class, p.side, decision.quantity,
                                           p.entry, p.stop, p.fx_to_base, now))
    return ScanReport(now, regimes, rows, decisions)


def report_to_json(report: ScanReport) -> str:
    def row(r: ScanRow) -> dict:
        d = report.decisions.get(r.symbol)
        out = {"symbol": r.symbol, "asset_class": r.asset_class.value, "score": r.score,
               "close": r.close, "passed": r.passed, "failed": r.failed, "verdict": r.verdict}
        if r.proposal:
            out["proposal"] = {"entry": r.proposal.entry, "stop": r.proposal.stop,
                               "targets": r.proposal.targets, "strategy": r.proposal.strategy}
        if d:
            out["risk"] = {"approved": d.approved, "quantity": d.quantity, "notional": d.notional,
                           "max_loss": d.max_loss, "estimated_costs": d.estimated_costs,
                           "expected_return": d.expected_return, "reward_risk": d.reward_risk,
                           "rejections": d.rejections}
        return out

    return json.dumps({
        "generated_at": report.generated_at.isoformat(),
        "regimes": {k: {"trend": v.trend.value, "high_volatility": v.high_volatility,
                        "detail": v.detail} for k, v in report.regimes.items()},
        "rows": [row(r) for r in report.rows],
    }, indent=2)


def save_report(report: ScanReport, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "last_scan.json"
    path.write_text(report_to_json(report), encoding="utf-8")
    return path
