"""Opportunity scanner.

A symbol becomes a trade proposal only when several independent kinds of evidence agree.
The default answer is NO TRADE. The scanner never sizes a trade: that is the risk engine's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import polars as pl

from apex.analysis.indicators import add_indicators
from apex.analysis.regime import Regime, percentile_rank
from apex.models import AssetClass, Side, TradeProposal

STRATEGY = "swing_trend_continuation"


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


def scan_symbol(
    symbol: str,
    asset_class: AssetClass,
    candles: pl.DataFrame,
    regime: Regime,
    *,
    config: ScannerConfig | None = None,
    now: datetime | None = None,
    fx_to_base: float = 1.0,
) -> ScanRow:
    cfg = config or ScannerConfig()
    now = now or datetime.now(UTC)

    def no_trade(reason: str, **kw) -> ScanRow:
        return ScanRow(symbol, asset_class, kw.pop("score", 0.0), kw.pop("passed", ()),
                       kw.pop("failed", ()), f"NO TRADE: {reason}", **kw)

    if candles.height < cfg.min_bars:
        return no_trade(f"only {candles.height} bars of history (need {cfg.min_bars})")

    frame = add_indicators(candles.sort("ts"))
    last, prev = frame.row(-1, named=True), frame.row(-2, named=True)
    close = last["close"]

    if now - last["ts"] > cfg.max_bar_age:
        return no_trade(f"data is stale (last bar {last['ts']:%Y-%m-%d})", close=close)
    needed = ("ema20", "ema50", "ema200", "ema200_prev", "rsi14", "atr14", "adx14",
              "macd_hist", "vol_ratio", "prior_high20", "turnover20")
    if any(last[k] is None for k in needed):
        return no_trade("indicators not ready", close=close)
    if last["turnover20"] < cfg.min_turnover[asset_class]:
        return no_trade("not liquid enough to trade safely", close=close)

    atr = last["atr14"]
    vol_rank = percentile_rank(frame["atr_pct"])

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
        "volatility": (
            cfg.atr_pct_band[0] <= last["atr_pct"] <= cfg.atr_pct_band[1]
            and vol_rank is not None
            and vol_rank <= cfg.max_volatility_rank
        ),
    }
    passed = tuple(k for k, ok in checks.items() if ok)
    failed = tuple(k for k, ok in checks.items() if not ok)
    score = float(sum(WEIGHTS[k] for k in passed))
    common = dict(score=score, passed=passed, failed=failed, close=close)

    if not regime.allows_longs:
        return no_trade(f"market regime is {regime.trend} "
                        f"{'with high volatility' if regime.high_volatility else ''}".strip(),
                        **common)
    if not checks["trend"]:
        return no_trade("no confirmed uptrend", **common)
    if len(passed) < cfg.min_confirmations:
        return no_trade(
            f"only {len(passed)} of 5 confirmations (missing {', '.join(failed)})", **common
        )

    risk = cfg.stop_atr_multiple * atr
    proposal = TradeProposal(
        symbol=symbol,
        asset_class=asset_class,
        side=Side.LONG,
        strategy=STRATEGY,
        entry=close,
        stop=close - risk,
        targets=tuple(close + m * risk for m in cfg.target_r_multiples),
        as_of=last["ts"],
        score=score,
        confirmations=passed,
        avg_daily_turnover=last["turnover20"],
        fx_to_base=fx_to_base,
        notes=("breakout" if breakout else "pullback",),
    )
    return ScanRow(symbol, asset_class, score, passed, failed,
                   f"CANDIDATE: {len(passed)} of 5 confirmations", proposal, close)


def rank(rows: list[ScanRow]) -> list[ScanRow]:
    """Opportunities first, then by score. Stable, so equal rows keep their input order."""
    return sorted(rows, key=lambda r: (not r.is_opportunity, -r.score))
