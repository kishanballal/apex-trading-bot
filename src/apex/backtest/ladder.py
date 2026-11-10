"""Dip-buying ladder: buy a dip once it turns up, add a little more if it dips further,
sell at a profit threshold.

This is deliberate, bounded averaging down, requested by the project owner. It is kept from
becoming a martingale by four hard limits: a fixed number of tranches, a fixed share of equity
per symbol, a hard stop below the lowest fill, and an account floor that liquidates everything.

It runs outside the standard RiskEngine (which forbids adding to positions) and enforces its
own limits here. Fills use the same realism rules as the main backtester: decisions on a
finished bar, orders filled at the next open, costs on both sides, gaps filled at the open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from apex.analysis.regime import UNKNOWN, regime_at
from apex.backtest.engine import BacktestResult, Trade, _Series
from apex.config import CostModel, UniverseGroup
from apex.models import AssetClass
from apex.scanner.scanner import ScannerConfig, _gates


@dataclass(frozen=True)
class LadderConfig:
    capital: float = 50_000.0
    floor: float = 40_000.0  # the account must not fall below this
    kill_buffer: float = 0.05  # liquidate when equity is within 5% of the floor
    usd_inr: float = 88.0
    max_symbols: int = 6
    symbol_allocation: float = 0.15  # share of equity one symbol may take across all tranches
    tranche_weights: tuple[float, ...] = (1.0, 1.25, 1.5)  # "a little more" each time
    dip_atr: float = 3.0  # buy zone: close at least this many ATR below the 20-day high
    add_atr: float = 1.5  # add only after a further drop of this many ATR below the last fill
    take_profit_atr: float = 2.0  # sell threshold above the average entry
    stop_atr: float = 3.0  # hard stop this many ATR below the lowest fill
    max_hold_bars: int = 30
    require_uptrend: bool = True  # only buy dips in stocks above their 200-day average
    require_regime: bool = False  # only buy when the market benchmark allows longs
    gate: dict | None = None  # optional model score per (symbol, date); known at that close
    gate_threshold: float = 0.0
    start: date | None = None
    strategy: str = "dip_ladder"
    scanner: ScannerConfig = field(default_factory=ScannerConfig)

    @property
    def kill_level(self) -> float:
        return self.floor * (1 + self.kill_buffer)


@dataclass
class _Ladder:
    asset_class: AssetClass
    fx: float
    atr0: float
    fills: list[tuple[float, float]] = field(default_factory=list)  # (quantity, price)
    costs: float = 0.0
    opened: datetime | None = None
    bars: int = 0
    pending: str = ""  # "add" or "exit", decided at a close, executed at the next open

    @property
    def quantity(self) -> float:
        return sum(q for q, _ in self.fills)

    @property
    def average(self) -> float:
        return sum(q * p for q, p in self.fills) / self.quantity

    @property
    def lowest(self) -> float:
        return min(p for _, p in self.fills)


def is_turning_up(last: dict, prev: dict) -> bool:
    """The owner's rule: do not catch the falling knife, wait for the first sign of recovery."""
    return last["close"] > last["open"] and last["close"] > prev["close"]


def run_ladder(
    groups: list[UniverseGroup],
    candles: dict[str, pd.DataFrame],
    config: LadderConfig | None = None,
    *,
    costs: CostModel | None = None,
) -> BacktestResult:
    cfg, costs = config or LadderConfig(), costs or CostModel()
    empty = pd.DataFrame()
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
                   if cfg.start is None or d >= cfg.start})
    cash = cfg.capital
    held: dict[str, _Ladder] = {}
    entries: dict[str, float] = {}  # symbol -> ATR at the signal, to buy at the next open
    last_close: dict[str, float] = {}
    trades: list[Trade] = []
    curve: list[tuple[date, float]] = []
    exposure: list[float] = []
    rejections: dict[str, int] = {}
    kill_dates: list[date] = []
    signals = 0
    killed = liquidate = False
    weight_sum = sum(cfg.tranche_weights)

    def note(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    def value() -> float:
        return sum(l.quantity * last_close[s] * l.fx for s, l in held.items())

    def buy(symbol: str, ladder: _Ladder, price: float, equity: float) -> bool:
        nonlocal cash
        weight = cfg.tranche_weights[len(ladder.fills)] / weight_sum
        budget = equity * cfg.symbol_allocation * weight
        quantity = budget / (price * ladder.fx)
        quantity = float(int(quantity)) if ladder.asset_class is AssetClass.EQUITY \
            else int(quantity * 1e6) / 1e6
        notional = quantity * price * ladder.fx
        fee = notional * costs.per_side(ladder.asset_class)
        if quantity <= 0:
            note("share_price_too_high_for_tranche")
            return False
        if notional + fee > cash:
            note("not_enough_cash")
            return False
        cash -= notional + fee
        ladder.fills.append((quantity, price))
        ladder.costs += fee
        return True

    def sell(symbol: str, price: float, when: datetime, reason: str) -> None:
        nonlocal cash
        l = held.pop(symbol)
        proceeds = l.quantity * price * l.fx
        fee = proceeds * costs.per_side(l.asset_class)
        cash += proceeds - fee
        stop = l.lowest - cfg.stop_atr * l.atr0
        planned = sum(q * (p - stop) * l.fx for q, p in l.fills)
        cost_basis = sum(q * p for q, p in l.fills) * l.fx
        trades.append(Trade(symbol, l.asset_class, cfg.strategy, l.opened, when, l.quantity,
                            l.average, price, stop, l.fx, proceeds - cost_basis, l.costs + fee,
                            planned, reason, l.bars))

    for day in days:
        today = {sym: s.index[day] for sym, s in series.items() if day in s.index}
        equity_open = cash + value()

        # 1. Orders decided at the previous close are filled at today's open.
        for symbol in [s for s in held if s in today]:
            l, bar = held[symbol], series[symbol].rows[today[symbol]]
            if liquidate or l.pending == "exit":
                sell(symbol, bar["open"], bar["ts"], "kill_switch" if liquidate else "time")
            elif l.pending == "add":
                l.pending = ""
                buy(symbol, l, bar["open"], equity_open)
        for symbol in [s for s in entries if s in today]:
            atr0 = entries.pop(symbol)
            bar = series[symbol].rows[today[symbol]]
            if killed or len(held) >= cfg.max_symbols:
                note("no_free_slot")
                continue
            ladder = _Ladder(series[symbol].asset_class, fx_of[symbol], atr0, opened=bar["ts"])
            if buy(symbol, ladder, bar["open"], equity_open):
                held[symbol] = ladder
                last_close[symbol] = bar["open"]

        # 2. Intraday exits: hard stop first, then the profit threshold.
        for symbol in [s for s in held if s in today]:
            l, bar = held[symbol], series[symbol].rows[today[symbol]]
            l.bars += 1
            stop = l.lowest - cfg.stop_atr * l.atr0
            target = l.average + cfg.take_profit_atr * l.atr0
            if bar["low"] <= stop:
                sell(symbol, min(stop, bar["open"]), bar["ts"], "stop")
            elif bar["high"] >= target:
                sell(symbol, max(target, bar["open"]), bar["ts"], "target")

        # 3. Mark to market and enforce the account floor.
        for symbol, i in today.items():
            last_close[symbol] = series[symbol].rows[i]["close"]
        equity = cash + value()
        curve.append((day, equity))
        exposure.append(value() / equity if equity > 0 else 0.0)
        if liquidate and not held:
            liquidate = False
        if not killed and equity <= cfg.kill_level:
            killed = liquidate = True
            kill_dates.append(day)
            entries.clear()
        if killed:
            continue

        # 4. Decisions on today's finished bars.
        candidates: list[tuple[float, str, float]] = []
        for symbol, i in today.items():
            s = series[symbol]
            last, prev = s.rows[i], s.rows[i - 1] if i else s.rows[i]
            if symbol in held:
                l = held[symbol]
                if l.bars >= cfg.max_hold_bars:
                    l.pending = "exit"
                elif (len(l.fills) < len(cfg.tranche_weights)
                      and last["close"] <= l.fills[-1][1] - cfg.add_atr * l.atr0
                      and is_turning_up(last, prev)):
                    l.pending = "add"
                continue
            if symbol in entries or _gates(symbol, s.asset_class, s.rows, i, cfg.scanner,
                                           last["ts"]) is not None:
                continue
            depth = (last["prior_high20"] - last["close"]) / last["atr14"]
            if depth < cfg.dip_atr or not is_turning_up(last, prev):
                continue
            if cfg.require_uptrend and not last["close"] > last["ema200"]:
                continue
            if cfg.require_regime:
                j = bench_of[symbol].index.get(day)
                regime = regime_at(bench_of[symbol].rows, j) if j is not None else UNKNOWN
                if not regime.allows_longs:
                    continue
            score = depth
            if cfg.gate is not None:
                score = cfg.gate.get((symbol, day))
                if score is None or score < cfg.gate_threshold:
                    continue
            candidates.append((score, symbol, last["atr14"]))
        signals += len(candidates)
        free = cfg.max_symbols - len(held) - len(entries)
        for _, symbol, atr0 in sorted(candidates, reverse=True)[:max(free, 0)]:
            entries[symbol] = atr0

    for symbol in list(held):
        sell(symbol, last_close[symbol], series[symbol].rows[-1]["ts"], "end")
    if curve:
        curve[-1] = (curve[-1][0], cash)
    benchmarks = {}
    for group in groups:
        bench = _Series(group.benchmark, group.asset_class, candles.get(group.benchmark, empty))
        benchmarks[group.benchmark] = [(r["ts"].date(), r["close"]) for r in bench.rows
                                       if days and days[0] <= r["ts"].date() <= days[-1]]
    return BacktestResult(cfg, curve, exposure, trades, signals, rejections, kill_dates, benchmarks)
