"""Performance statistics for a backtest. Plain arithmetic, no curve fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from apex.backtest.engine import BacktestResult, Trade
from apex.models import AssetClass

EQUITY_STCG_RATE = 0.20  # India, listed shares held under 12 months
CRYPTO_TAX_RATE = 0.30  # India, flat on gains; losses cannot be set off


@dataclass(frozen=True)
class Metrics:
    start: date | None
    end: date | None
    years: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float  # average net profit per trade, base currency
    expectancy_r: float  # average net profit per trade in units of planned risk
    avg_win: float
    avg_loss: float
    avg_bars_held: float
    net_profit: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    avg_exposure: float
    total_costs: float
    estimated_tax: float
    net_profit_after_tax: float
    exit_reasons: dict[str, int]


def max_drawdown(values: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, 1 - v / peak)
    return worst


def financial_year(day: date) -> int:
    """Indian financial year (April to March), labelled by its starting year."""
    return day.year if day.month >= 4 else day.year - 1


def estimate_tax(trades: list[Trade]) -> float:
    """Rough Indian tax on trading gains. Not tax advice.

    Shares: 20% on each financial year's net gain. Losing years are not carried forward here,
    which overstates tax slightly. Crypto: 30% of every winning trade, losses give no relief.
    """
    by_year: dict[int, float] = {}
    tax = 0.0
    for t in trades:
        if t.asset_class is AssetClass.EQUITY:
            fy = financial_year(t.exit_time.date())
            by_year[fy] = by_year.get(fy, 0.0) + t.pnl
        elif t.gross_pnl > 0:
            tax += CRYPTO_TAX_RATE * t.gross_pnl
    return tax + sum(EQUITY_STCG_RATE * gain for gain in by_year.values() if gain > 0)


def compute_metrics(result: BacktestResult) -> Metrics:
    curve, trades = result.equity_curve, result.trades
    capital = result.config.capital
    values = [v for _, v in curve]
    start = curve[0][0] if curve else None
    end = curve[-1][0] if curve else None
    years = max((end - start).days / 365.25, 1e-9) if curve else 0.0
    final = values[-1] if values else capital
    total_return = final / capital - 1

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    sharpe = sortino = 0.0
    if len(values) > 2:
        rets = np.diff(values) / np.asarray(values[:-1])
        per_year = len(rets) / years
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(per_year))
        downside = math.sqrt(float(np.mean(np.minimum(rets, 0.0) ** 2)))
        if downside > 0:
            sortino = float(rets.mean() / downside * math.sqrt(per_year))

    net = sum(t.pnl for t in trades)
    tax = estimate_tax(trades)
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    return Metrics(
        start=start, end=end, years=years, trades=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win else 0.0),
        expectancy=net / len(trades) if trades else 0.0,
        expectancy_r=float(np.mean([t.r_multiple for t in trades])) if trades else 0.0,
        avg_win=gross_win / len(wins) if wins else 0.0,
        avg_loss=-gross_loss / len(losses) if losses else 0.0,
        avg_bars_held=float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
        net_profit=net, total_return=total_return,
        cagr=(final / capital) ** (1 / years) - 1 if curve and final > 0 else 0.0,
        max_drawdown=max_drawdown(values), sharpe=sharpe, sortino=sortino,
        avg_exposure=float(np.mean(result.exposure)) if result.exposure else 0.0,
        total_costs=sum(t.costs for t in trades), estimated_tax=tax,
        net_profit_after_tax=net - tax, exit_reasons=reasons,
    )


def yearly_table(result: BacktestResult) -> list[dict]:
    """Calendar-year breakdown, so one lucky year cannot hide behind the total."""
    rows = []
    curve = result.equity_curve
    for year in sorted({d.year for d, _ in curve}):
        values = [v for d, v in curve if d.year == year]
        before = [v for d, v in curve if d.year < year]
        base = before[-1] if before else result.config.capital
        year_trades = [t for t in result.trades if t.exit_time.year == year]
        wins = sum(1 for t in year_trades if t.pnl > 0)
        rows.append({
            "year": year,
            "return": values[-1] / base - 1,
            "max_drawdown": max_drawdown([base, *values]),
            "trades": len(year_trades),
            "win_rate": wins / len(year_trades) if year_trades else 0.0,
            "net_profit": sum(t.pnl for t in year_trades),
        })
    return rows


def buy_and_hold(series: list[tuple[date, float]]) -> dict:
    if len(series) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
    values = [v for _, v in series]
    years = max((series[-1][0] - series[0][0]).days / 365.25, 1e-9)
    return {
        "total_return": values[-1] / values[0] - 1,
        "cagr": (values[-1] / values[0]) ** (1 / years) - 1,
        "max_drawdown": max_drawdown(values),
    }
