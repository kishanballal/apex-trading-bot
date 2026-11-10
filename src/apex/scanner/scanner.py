"""Opportunity scanner and the baseline strategies.

A symbol becomes a trade proposal only when several independent kinds of evidence agree.
The default answer is NO TRADE. Strategies never size a trade: that is the risk engine's job.

Every strategy is a function of prepared bars 0..i, so the live scanner and the backtester run
exactly the same code and neither can see the future.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import polars as pl

from apex.analysis.indicators import prepare
from apex.analysis.regime import Regime
from apex.models import AssetClass, Side, TradeProposal

TREND = "swing_trend_continuation"
PULLBACK = "swing_pullback"
STRATEGY = TREND  # kept for callers that only know the original strategy


@dataclass(frozen=True)
class ScannerConfig:
    min_bars: int = 220
    min_confirmations: int = 4  # out of the five evidence families below
    max_bar_age: timedelta = timedelta(days=5)
    min_turnover: dict[AssetClass, float] = field(
        default_factory=lambda: {
            AssetClass.EQUITY: 5e7,  # INR 5 crore traded per day
            AssetClass.CRYPTO: 5e6,  # USDT 5 million traded per day
        }
    )
    stop_atr_multiple: float = 2.0
    target_r_multiples: tuple[float, ...] = (2.0, 3.0)
    min_adx: float = 20.0
    rsi_band: tuple[float, float] = (50.0, 70.0)
    min_volume_ratio: float = 1.2
    atr_pct_band: tuple[float, float] = (0.005, 0.06)
    max_volatility_rank: float = 0.80
    require_regime: bool = True  # research switch; live scanning always keeps this on
    # pullback strategy
    pullback_rsi_band: tuple[float, float] = (35.0, 55.0)
    pullback_stop_buffer_atr: float = 0.5


@dataclass(frozen=True)
class ScanRow:
    symbol: str
    asset_class: AssetClass
    score: float
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    verdict: str  # human readable reason for the outcome
    proposal: TradeProposal | None = None
    close: float | None = None

    @property
    def is_opportunity(self) -> bool:
        return self.proposal is not None


WEIGHTS = {"trend": 30, "structure": 20, "momentum": 20, "volume": 15, "volatility": 15}
_NEEDED = ("ema20", "ema50", "ema200", "ema200_prev", "rsi14", "atr14", "adx14",
           "macd_hist", "vol_ratio", "prior_high20", "turnover20")

BarStrategy = Callable[..., ScanRow]


def _no_trade(symbol: str, asset_class: AssetClass, reason: str, **kw) -> ScanRow:
    return ScanRow(symbol, asset_class, kw.pop("score", 0.0), kw.pop("passed", ()),
                   kw.pop("failed", ()), f"NO TRADE: {reason}", **kw)


def _gates(symbol, asset_class, rows, i, cfg, now) -> ScanRow | None:
    """Checks shared by all strategies. Returns a NO TRADE row, or None when the bar is usable."""
    if i + 1 < cfg.min_bars:
        return _no_trade(symbol, asset_class,
                         f"only {max(i + 1, 0)} bars of history (need {cfg.min_bars})")
    last = rows[i]
    close = last["close"]
    if now - last["ts"] > cfg.max_bar_age:
        return _no_trade(symbol, asset_class,
                         f"data is stale (last bar {last['ts']:%Y-%m-%d})", close=close)
    if any(last[k] is None for k in _NEEDED):
        return _no_trade(symbol, asset_class, "indicators not ready", close=close)
    if last["turnover20"] < cfg.min_turnover[asset_class]:
        return _no_trade(symbol, asset_class, "not liquid enough to trade safely", close=close)
    return None


def _regime_block(regime: Regime, cfg: ScannerConfig) -> str | None:
    if cfg.require_regime and not regime.allows_longs:
        return (f"market regime is {regime.trend} "
                f"{'with high volatility' if regime.high_volatility else ''}").strip()
    return None


def _volatility_ok(last: dict, cfg: ScannerConfig) -> bool:
    return (
        cfg.atr_pct_band[0] <= last["atr_pct"] <= cfg.atr_pct_band[1]
        and last["vol_rank"] is not None
        and last["vol_rank"] <= cfg.max_volatility_rank
    )


def _proposal(symbol, asset_class, strategy, last, stop, cfg, score, passed, fx, note):
    close = last["close"]
    risk = close - stop
    return TradeProposal(
        symbol=symbol, asset_class=asset_class, side=Side.LONG, strategy=strategy,
        entry=close, stop=stop,
        targets=tuple(close + m * risk for m in cfg.target_r_multiples),
        as_of=last["ts"], score=score, confirmations=passed,
        avg_daily_turnover=last["turnover20"], fx_to_base=fx, notes=(note,),
    )


def scan_bar(
    symbol: str,
    asset_class: AssetClass,
    rows: list[dict],
    i: int,
    regime: Regime,
    *,
    config: ScannerConfig | None = None,
    now: datetime | None = None,
    fx_to_base: float = 1.0,
) -> ScanRow:
    """Trend continuation: buy strength in a confirmed uptrend. Uses prepared bars 0..i only."""
    cfg = config or ScannerConfig()
    now = now or datetime.now(UTC)
    blocked = _gates(symbol, asset_class, rows, i, cfg, now)
    if blocked:
        return blocked
    last, prev = rows[i], rows[i - 1]
    close, atr = last["close"], last["atr14"]

    breakout = close > last["prior_high20"]
    pullback = (
        last["low"] <= last["ema20"] + atr
        and close > last["ema20"]
        and close > prev["high"]
    )
    checks = {
        "trend": (
            close > last["ema50"] > last["ema200"]
            and last["ema200"] > last["ema200_prev"]
            and last["adx14"] >= cfg.min_adx
        ),
        "momentum": cfg.rsi_band[0] <= last["rsi14"] <= cfg.rsi_band[1]
        and last["macd_hist"] > 0,
        "volume": last["vol_ratio"] >= cfg.min_volume_ratio and close > last["open"],
        "structure": breakout or pullback,
        "volatility": _volatility_ok(last, cfg),
    }
    passed = tuple(k for k, ok in checks.items() if ok)
    failed = tuple(k for k, ok in checks.items() if not ok)
    score = float(sum(WEIGHTS[k] for k in passed))
    common = dict(score=score, passed=passed, failed=failed, close=close)

    regime_reason = _regime_block(regime, cfg)
    if regime_reason:
        return _no_trade(symbol, asset_class, regime_reason, **common)
    if not checks["trend"]:
        return _no_trade(symbol, asset_class, "no confirmed uptrend", **common)
    if len(passed) < cfg.min_confirmations:
        return _no_trade(
            symbol, asset_class,
            f"only {len(passed)} of 5 confirmations (missing {', '.join(failed)})", **common
        )
    proposal = _proposal(symbol, asset_class, TREND, last, close - cfg.stop_atr_multiple * atr,
                         cfg, score, passed, fx_to_base, "breakout" if breakout else "pullback")
    return ScanRow(symbol, asset_class, score, passed, failed,
                   f"CANDIDATE: {len(passed)} of 5 confirmations", proposal, close)


def scan_bar_pullback(
    symbol: str,
    asset_class: AssetClass,
    rows: list[dict],
    i: int,
    regime: Regime,
    *,
    config: ScannerConfig | None = None,
    now: datetime | None = None,
    fx_to_base: float = 1.0,
) -> ScanRow:
    """Pullback: buy the first up-day after a dip to the 20-day average inside an uptrend.
    All four conditions are required."""
    cfg = config or ScannerConfig()
    now = now or datetime.now(UTC)
    blocked = _gates(symbol, asset_class, rows, i, cfg, now)
    if blocked:
        return blocked
    last, prev = rows[i], rows[i - 1]
    close, atr = last["close"], last["atr14"]
    checks = {
        "trend": close > last["ema200"] and last["ema50"] > last["ema200"]
        and last["ema200"] > last["ema200_prev"],
        "dip": min(last["low"], prev["low"]) <= last["ema20"]
        and cfg.pullback_rsi_band[0] <= last["rsi14"] <= cfg.pullback_rsi_band[1],
        "reversal": close > last["open"] and close > prev["high"],
        "volatility": _volatility_ok(last, cfg),
    }
    passed = tuple(k for k, ok in checks.items() if ok)
    failed = tuple(k for k, ok in checks.items() if not ok)
    score = 25.0 * len(passed)
    common = dict(score=score, passed=passed, failed=failed, close=close)
    regime_reason = _regime_block(regime, cfg)
    if regime_reason:
        return _no_trade(symbol, asset_class, regime_reason, **common)
    if failed:
        return _no_trade(symbol, asset_class, f"missing {', '.join(failed)}", **common)
    stop = min(last["low"], prev["low"]) - cfg.pullback_stop_buffer_atr * atr
    proposal = _proposal(symbol, asset_class, PULLBACK, last, stop, cfg, score, passed,
                         fx_to_base, "pullback")
    return ScanRow(symbol, asset_class, score, passed, failed, "CANDIDATE: pullback setup",
                   proposal, close)


STRATEGIES: dict[str, BarStrategy] = {"trend": scan_bar, "pullback": scan_bar_pullback}


def scan_symbol(
    symbol: str,
    asset_class: AssetClass,
    candles: pl.DataFrame,
    regime: Regime,
    *,
    config: ScannerConfig | None = None,
    now: datetime | None = None,
    fx_to_base: float = 1.0,
    strategy: str = "trend",
) -> ScanRow:
    """Live entry point: evaluate the most recent finished bar."""
    rows = prepare(candles).to_dicts() if candles.height else []
    return STRATEGIES[strategy](symbol, asset_class, rows, len(rows) - 1, regime,
                                config=config, now=now, fx_to_base=fx_to_base)


def rank(rows: list[ScanRow]) -> list[ScanRow]:
    """Opportunities first, then by score. Stable, so equal rows keep their input order."""
    return sorted(rows, key=lambda r: (not r.is_opportunity, -r.score))
