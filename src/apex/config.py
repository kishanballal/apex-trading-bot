"""Central configuration. Every value can be overridden with an APEX_* environment variable."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from apex.models import AssetClass, Mode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RiskLimits(BaseSettings):
    """Hard limits enforced by the independent risk engine. Fractions, not percent."""

    model_config = SettingsConfigDict(env_prefix="APEX_RISK_", extra="ignore")

    risk_per_trade: float = Field(0.0025, gt=0, le=0.01)
    daily_loss_limit: float = Field(0.0075, gt=0, le=0.05)
    weekly_drawdown_limit: float = Field(0.02, gt=0, le=0.10)
    max_portfolio_drawdown: float = Field(0.05, gt=0, le=0.20)
    max_positions: int = Field(4, ge=1, le=20)
    max_position_allocation: float = Field(0.05, gt=0, le=0.25)
    max_deployed_capital: float = Field(0.20, gt=0, le=1.0)
    max_leverage: float = Field(1.0, gt=0, le=1.0)

    min_reward_risk: float = Field(1.5, ge=1.0)
    min_stop_distance: float = Field(0.002, gt=0)  # stops tighter than 0.2% are noise
    max_stop_distance: float = Field(0.15, gt=0)
    max_correlation: float = Field(0.80, gt=0, le=1.0)
    min_correlation_samples: int = 30
    max_participation: float = Field(0.01, gt=0, le=0.1)  # share of average daily turnover
    min_notional: float = Field(500.0, ge=0)  # base currency; smaller trades are eaten by costs
    max_signal_age_hours: float = Field(96.0, gt=0)
    symbol_cooldown_hours: float = Field(24.0, ge=0)  # after a loss, no re-entry in that symbol
    max_consecutive_losses: int = Field(3, ge=1)  # then no new trades until the next day


class CostModel(BaseSettings):
    """Per-side cost assumptions as fractions of traded value. Deliberately conservative."""

    model_config = SettingsConfigDict(env_prefix="APEX_COST_", extra="ignore")

    equity_fees: float = 0.0015  # STT, exchange, stamp, GST on delivery trades (approx.)
    crypto_fees: float = 0.0010
    slippage: float = 0.0005

    def per_side(self, asset_class: AssetClass) -> float:
        fees = self.equity_fees if asset_class is AssetClass.EQUITY else self.crypto_fees
        return fees + self.slippage


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APEX_", env_file=".env", extra="ignore")

    mode: Mode = Mode.PAPER
    capital: float = Field(100_000.0, gt=0)
    base_currency: str = "INR"
    usd_inr: float = Field(88.0, gt=0)
    data_dir: Path = Path("data")
    universe_file: Path = PROJECT_ROOT / "config" / "universe.toml"
    timezone: str = "Asia/Kolkata"

    yahoo_base_url: str = "https://query1.finance.yahoo.com"
    binance_base_url: str = "https://data-api.binance.vision"
    http_timeout: float = 20.0
    http_retries: int = 3
    max_concurrency: int = 5
    history_days: int = 730

    # Live trading stays locked until the later, explicitly authorised stage.
    live_authorized: bool = False

    @model_validator(mode="after")
    def _live_requires_authorization(self) -> "Settings":
        if self.mode is Mode.LIVE and not self.live_authorized:
            raise ValueError("Live mode requires explicit authorization (APEX_LIVE_AUTHORIZED).")
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / "apex.sqlite"


@dataclass(frozen=True)
class UniverseGroup:
    asset_class: AssetClass
    provider: str
    benchmark: str
    symbols: tuple[str, ...]


def load_universe(path: Path) -> list[UniverseGroup]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    groups: list[UniverseGroup] = []
    for key, body in raw.items():
        asset_class = AssetClass(key)
        symbols = tuple(dict.fromkeys(s.strip().upper() for s in body["symbols"] if s.strip()))
        groups.append(
            UniverseGroup(asset_class, body["provider"], body["benchmark"], symbols)
        )
    return groups
