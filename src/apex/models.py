"""Shared domain types. No business logic lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class Mode(StrEnum):
    ALERTS = "alerts"
    PAPER = "paper"
    LIVE = "live"


# Canonical candle columns. `ts` is the bar's opening time, timezone-aware UTC.
CANDLE_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class TradeProposal:
    """What a strategy asks for. It carries no size: only the risk engine sizes trades."""

    symbol: str
    asset_class: AssetClass
    side: Side
    strategy: str
    entry: float
    stop: float
    targets: tuple[float, ...]
    as_of: datetime
    score: float = 0.0
    confirmations: tuple[str, ...] = ()
    avg_daily_turnover: float | None = None  # in the quote currency
    fx_to_base: float = 1.0  # quote currency -> base currency
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Position:
    symbol: str
    asset_class: AssetClass
    side: Side
    quantity: float
    entry: float
    stop: float
    fx_to_base: float = 1.0
    opened_at: datetime | None = None

    @property
    def notional_base(self) -> float:
        return self.quantity * self.entry * self.fx_to_base


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class RiskDecision:
    """The risk engine's final word on a proposal. All money values are in base currency."""

    approved: bool
    symbol: str
    quantity: float = 0.0
    notional: float = 0.0
    max_loss: float = 0.0  # loss if stopped out, including costs
    estimated_costs: float = 0.0  # round trip
    expected_return: float = 0.0  # net profit at the first target
    reward_risk: float = 0.0
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def rejections(self) -> tuple[str, ...]:
        return tuple(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)
