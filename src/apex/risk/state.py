"""Portfolio state the risk engine reasons about. Plain data plus bookkeeping, JSON-persistable."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from apex.models import AssetClass, Position, Side


@dataclass
class PortfolioState:
    equity: float  # mark-to-market account value, base currency
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    day_key: str = ""  # local calendar day, e.g. 2026-11-10
    week_key: str = ""  # ISO week, e.g. 2026-W46
    positions: dict[str, Position] = field(default_factory=dict)
    kill_switch: bool = False
    kill_reason: str = ""
    consecutive_losses: int = 0
    loss_pause_day: str = ""  # day on which the consecutive-loss pause applies
    last_loss_at: dict[str, datetime] = field(default_factory=dict)

    @classmethod
    def fresh(cls, capital: float) -> "PortfolioState":
        return cls(capital, capital, capital, capital)

    @property
    def deployed(self) -> float:
        return sum(p.notional_base for p in self.positions.values())

    def roll(self, now: datetime, timezone: str = "Asia/Kolkata") -> None:
        """Start a new day or week when the local calendar turns over."""
        local = now.astimezone(ZoneInfo(timezone))
        day = local.strftime("%Y-%m-%d")
        iso = local.isocalendar()
        week = f"{iso.year}-W{iso.week:02d}"
        if day != self.day_key:
            self.day_key, self.day_start_equity = day, self.equity
        if week != self.week_key:
            self.week_key, self.week_start_equity = week, self.equity

    def mark(self, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def open_position(self, position: Position) -> None:
        if position.symbol in self.positions:
            raise ValueError(f"{position.symbol} is already open; adding to positions is not allowed")
        self.positions[position.symbol] = position

    def close_position(
        self, symbol: str, pnl: float, now: datetime, *, equity: float | None = None
    ) -> None:
        """Record a closed trade. `pnl` is net of costs, in base currency.

        Pass `equity` when the caller tracks mark-to-market equity itself: the open profit or
        loss is then already inside `self.equity`, and adding `pnl` again would count it twice.
        """
        self.positions.pop(symbol)
        self.mark(self.equity + pnl if equity is None else equity)
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_at[symbol] = now
        else:
            self.consecutive_losses = 0

    def trip_kill_switch(self, reason: str) -> None:
        if not self.kill_switch:
            self.kill_switch, self.kill_reason = True, reason

    def reset_kill_switch(self) -> None:
        """Manual action only. Nothing in the automated path calls this."""
        self.kill_switch, self.kill_reason = False, ""
        self.peak_equity = self.equity

    # --- persistence -----------------------------------------------------------------

    def save(self, path: Path) -> None:
        data = asdict(self)
        data["last_loss_at"] = {k: v.isoformat() for k, v in self.last_loss_at.items()}
        for pos in data["positions"].values():
            pos["opened_at"] = pos["opened_at"].isoformat() if pos["opened_at"] else None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic, so a crash cannot leave a half-written state file

    @classmethod
    def load(cls, path: Path) -> "PortfolioState":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["last_loss_at"] = {
            k: datetime.fromisoformat(v) for k, v in data["last_loss_at"].items()
        }
        data["positions"] = {
            symbol: Position(
                **{
                    **pos,
                    "asset_class": AssetClass(pos["asset_class"]),
                    "side": Side(pos["side"]),
                    "opened_at": datetime.fromisoformat(pos["opened_at"])
                    if pos["opened_at"] else None,
                }
            )
            for symbol, pos in data["positions"].items()
        }
        return cls(**data)
