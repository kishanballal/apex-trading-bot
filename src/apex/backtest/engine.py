"""Portfolio backtester.

Realism rules:
  * A signal is computed on a finished bar and can only be filled at the NEXT bar's open.
  * Every entry goes through the same RiskEngine used live, against the simulated portfolio,
    so position, exposure, loss and kill-switch limits shape the results.
  * Costs (fees plus slippage) are charged on both sides. Stops that gap are filled at the open.
  * If a bar touches both the stop and the target, the stop is assumed to have come first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import polars as pl

from apex.analysis.indicators import prepare
from apex.analysis.regime import UNKNOWN, Regime, regime_at
from apex.config import CostModel, RiskLimits, UniverseGroup
from apex.models import AssetClass, Position, TradeProposal
from apex.risk.engine import RiskEngine
from apex.risk.state import PortfolioState
from apex.scanner.scanner import STRATEGIES, ScannerConfig

RETURN_WINDOW = 60


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "trend"
    capital: float = 100_000.0
    usd_inr: float = 88.0
    max_hold_bars: int = 40  # time stop: capital stuck in a dead trade is also a cost
    start: date | None = None
    end: date | None = None
    # None keeps the real rule: after a 5% drawdown only a human can re-enable trading.
    # A number simulates that human resetting the switch after that many days.
    kill_switch_reset_days: int | None = None
    scanner: ScannerConfig = field(default_factory=ScannerConfig)


@dataclass(frozen=True)
class Trade:
    symbol: str
    asset_class: AssetClass
    strategy: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry: float
    exit: float
    stop: float
    fx: float
    gross_pnl: float  # base currency, before costs
    costs: float
    planned_risk: float  # base currency loss the risk engine budgeted for
    reason: str  # stop, target, time, end
    bars_held: int

    @property
    def pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.planned_risk if self.planned_risk else 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity_curve: list[tuple[date, float]]
    exposure: list[float]  # deployed fraction of equity at each close
    trades: list[Trade]
    signals: int
    rejections: dict[str, int]
    kill_switch_dates: list[date]
    benchmarks: dict[str, list[tuple[date, float]]]


@dataclass
class _Open:
    position: Position
    target: float
    entry_time: datetime
    planned_risk: float
    entry_cost: float
    strategy: str
    bars: int = 0


class _Series:
    def __init__(self, symbol: str, asset_class: AssetClass, candles: pl.DataFrame) -> None:
        self.symbol, self.asset_class = symbol, asset_class
        self.rows = prepare(candles).to_dicts() if candles.height else []
        self.index = {r["ts"].date(): i for i, r in enumerate(self.rows)}
        closes = [r["close"] for r in self.rows]
        self.returns = [0.0] + [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]


def run_backtest(
    groups: list[UniverseGroup],
    candles: dict[str, pl.DataFrame],
    config: BacktestConfig | None = None,
    *,
    limits: RiskLimits | None = None,
    costs: CostModel | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    costs = costs or CostModel()
    engine = RiskEngine(limits, costs)
    signal_fn = STRATEGIES[cfg.strategy]
    empty = pl.DataFrame()

    series: dict[str, _Series] = {}
    bench_of: dict[str, _Series] = {}
    fx_of: dict[str, float] = {}
    for group in groups:
        bench = _Series(group.benchmark, group.asset_class, candles.get(group.benchmark, empty))
        for symbol in group.symbols:
            s = _Series(symbol, group.asset_class, candles.get(symbol, empty))
            if s.rows:
                series[symbol], bench_of[symbol] = s, bench
                fx_of[symbol] = cfg.usd_inr if group.asset_class is AssetClass.CRYPTO else 1.0

    days = sorted({d for s in series.values() for d in s.index
                   if (cfg.start is None or d >= cfg.start) and (cfg.end is None or d <= cfg.end)})
    state = PortfolioState.fresh(cfg.capital)
    cash = cfg.capital
    open_trades: dict[str, _Open] = {}
    pending: dict[str, TradeProposal] = {}
    last_close: dict[str, float] = {}
    trades: list[Trade] = []
    curve: list[tuple[date, float]] = []
    exposure: list[float] = []
    rejections: dict[str, int] = {}
    kill_dates: list[date] = []
    signals = 0

    def market_value() -> float:
        return sum(o.position.quantity * last_close[sym] * o.position.fx_to_base
                   for sym, o in open_trades.items())

    def close_trade(symbol: str, price: float, when: datetime, reason: str) -> None:
        nonlocal cash
        o = open_trades.pop(symbol)
        p = o.position
        proceeds = p.quantity * price * p.fx_to_base
        exit_cost = proceeds * costs.per_side(p.asset_class)
        cash += proceeds - exit_cost
        trade = Trade(symbol, p.asset_class, o.strategy, o.entry_time, when, p.quantity, p.entry,
                      price, p.stop, p.fx_to_base, proceeds - p.notional_base,
                      o.entry_cost + exit_cost, o.planned_risk, reason, o.bars)
        trades.append(trade)
        state.close_position(symbol, trade.pnl, when, equity=cash + market_value())

    for day in days:
        today = {sym: s.index[day] for sym, s in series.items() if day in s.index}

        # 1. Fill yesterday's signals at today's open, best score first.
        for symbol in sorted((s for s in pending if s in today),
                             key=lambda s: -pending[s].score):
            signal = pending.pop(symbol)
            s, i = series[symbol], today[symbol]
            bar = s.rows[i]
            open_price = bar["open"]
            if not signal.stop < open_price < signal.targets[0]:
                rejections["gap_through_stop_or_target"] = rejections.get(
                    "gap_through_stop_or_target", 0) + 1
                continue
            proposal = TradeProposal(
                symbol, signal.asset_class, signal.side, signal.strategy, open_price,
                signal.stop, signal.targets, signal.as_of, signal.score, signal.confirmations,
                signal.avg_daily_turnover, signal.fx_to_base, signal.notes)
            returns = {sym: series[sym].returns[max(0, j - RETURN_WINDOW):j]
                       for sym, j in ((symbol, i), *((h, series[h].index.get(day, len(series[h].rows)))
                                                     for h in open_trades))}
            state.equity = cash + market_value()
            decision = engine.evaluate(proposal, state, now=bar["ts"], returns=returns)
            if not decision.approved:
                for c in decision.checks:
                    if not c.passed:
                        rejections[c.name] = rejections.get(c.name, 0) + 1
                continue
            position = Position(symbol, s.asset_class, proposal.side, decision.quantity,
                                open_price, proposal.stop, proposal.fx_to_base, bar["ts"])
            entry_cost = position.notional_base * costs.per_side(s.asset_class)
            cash -= position.notional_base + entry_cost
            state.open_position(position)
            open_trades[symbol] = _Open(position, proposal.targets[0], bar["ts"],
                                        decision.max_loss, entry_cost, proposal.strategy)
            last_close[symbol] = open_price
        # A signal is good for one bar only. Symbols that did not trade today keep waiting.
        for symbol in [s for s in pending if s in today]:
            pending.pop(symbol)

        # 2. Manage open positions on today's bar.
        for symbol in [s for s in open_trades if s in today]:
            o, bar = open_trades[symbol], series[symbol].rows[today[symbol]]
            o.bars += 1
            if bar["low"] <= o.position.stop:
                close_trade(symbol, min(o.position.stop, bar["open"]), bar["ts"], "stop")
            elif bar["high"] >= o.target:
                close_trade(symbol, max(o.target, bar["open"]), bar["ts"], "target")
            elif o.bars >= cfg.max_hold_bars:
                close_trade(symbol, bar["close"], bar["ts"], "time")

        # 3. Mark to market at the close and let the risk engine see the new equity.
        for symbol, i in today.items():
            last_close[symbol] = series[symbol].rows[i]["close"]
        equity = cash + market_value()
        state.mark(equity)
        stamp = max(series[s].rows[i]["ts"] for s, i in today.items())
        was_killed = state.kill_switch
        engine.account_checks(state, stamp)
        if state.kill_switch and not was_killed:
            kill_dates.append(day)
        if (state.kill_switch and cfg.kill_switch_reset_days is not None and kill_dates
                and day - kill_dates[-1] >= timedelta(days=cfg.kill_switch_reset_days)):
            state.reset_kill_switch()
        curve.append((day, equity))
        exposure.append(state.deployed / equity if equity > 0 else 0.0)

        # 4. Look for new signals on today's finished bars. They can fill tomorrow at the earliest.
        if state.kill_switch:
            continue
        for symbol, i in today.items():
            if symbol in open_trades:
                continue
            s, bench = series[symbol], bench_of[symbol]
            j = bench.index.get(day)
            regime: Regime = regime_at(bench.rows, j) if j is not None else UNKNOWN
            row = signal_fn(symbol, s.asset_class, s.rows, i, regime, config=cfg.scanner,
                            now=s.rows[i]["ts"], fx_to_base=fx_of[symbol])
            if row.proposal:
                pending[symbol] = row.proposal
                signals += 1

    # Close whatever is still open at the last known price so the result is fully realised.
    for symbol in list(open_trades):
        s = series[symbol]
        close_trade(symbol, last_close[symbol], s.rows[-1]["ts"], "end")
    if curve:
        curve[-1] = (curve[-1][0], cash)

    benchmarks = {}
    for group in groups:
        bench = _Series(group.benchmark, group.asset_class, candles.get(group.benchmark, empty))
        benchmarks[group.benchmark] = [(r["ts"].date(), r["close"]) for r in bench.rows
                                       if days and days[0] <= r["ts"].date() <= days[-1]]
    return BacktestResult(cfg, curve, exposure, trades, signals, rejections, kill_dates, benchmarks)
