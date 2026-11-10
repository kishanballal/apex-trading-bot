"""Turn a table of per-day scores (model probabilities, momentum ranks) into a strategy the
backtester and scanner can run. The table must only hold values known at that bar's close."""

from __future__ import annotations

from datetime import UTC, date, datetime

from apex.analysis.regime import Regime
from apex.models import AssetClass
from apex.scanner.scanner import (
    BarStrategy, ScannerConfig, ScanRow, _gates, _no_trade, _proposal, _regime_block,
)


def make_table_strategy(
    name: str,
    table: dict[tuple[str, date], float],
    threshold: float,
    *,
    stop_atr: float = 2.0,
) -> BarStrategy:
    def strategy(symbol: str, asset_class: AssetClass, rows: list[dict], i: int, regime: Regime,
                 *, config: ScannerConfig | None = None, now: datetime | None = None,
                 fx_to_base: float = 1.0) -> ScanRow:
        cfg = config or ScannerConfig()
        blocked = _gates(symbol, asset_class, rows, i, cfg, now or datetime.now(UTC))
        if blocked:
            return blocked
        last = rows[i]
        value = table.get((symbol, last["ts"].date()))
        if value is None:
            return _no_trade(symbol, asset_class, "no score for this bar", close=last["close"])
        score = 100.0 * value
        reason = _regime_block(regime, cfg)
        if reason:
            return _no_trade(symbol, asset_class, reason, score=score, close=last["close"])
        if value < threshold:
            return _no_trade(symbol, asset_class, f"score {value:.2f} below {threshold:.2f}",
                             score=score, close=last["close"])
        stop = last["close"] - stop_atr * last["atr14"]
        proposal = _proposal(symbol, asset_class, name, last, stop, cfg, score, (name,),
                             fx_to_base, f"score {value:.2f}")
        return ScanRow(symbol, asset_class, score, (name,), (), f"CANDIDATE: score {value:.2f}",
                       proposal, last["close"])

    return strategy
