"""The independent risk engine. It has the final say on every trade.

Design rules:
  * It imports nothing from the scanner, strategies or ML code, so none of them can bypass it.
  * It sizes every trade itself. A proposal never carries a quantity.
  * Every check runs and is recorded, so a rejection lists all of its reasons.
  * Kill switches latch. Only a human can reset the portfolio-drawdown switch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from apex.config import CostModel, RiskLimits
from apex.models import AssetClass, CheckResult, Mode, RiskDecision, Side, TradeProposal
from apex.risk.state import PortfolioState

_TOL = 1e-9


class RiskEngine:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        costs: CostModel | None = None,
        *,
        mode: Mode = Mode.PAPER,
        live_authorized: bool = False,
        timezone: str = "Asia/Kolkata",
    ) -> None:
        self.limits = limits or RiskLimits()
        self.costs = costs or CostModel()
        self.mode = mode
        self.live_authorized = live_authorized
        self.timezone = timezone

    # --- account-level guards --------------------------------------------------------

    def account_checks(self, state: PortfolioState, now: datetime) -> list[CheckResult]:
        """Loss limits and kill switches. Trips the latched switch as a side effect."""
        lim = self.limits
        state.roll(now, self.timezone)
        state.peak_equity = max(state.peak_equity, state.equity)

        drawdown = 1 - state.equity / state.peak_equity
        if drawdown >= lim.max_portfolio_drawdown - _TOL:
            state.trip_kill_switch(
                f"portfolio drawdown {drawdown:.2%} reached the {lim.max_portfolio_drawdown:.2%} limit"
            )
        day_loss = 1 - state.equity / state.day_start_equity
        week_loss = 1 - state.equity / state.week_start_equity
        paused = (
            state.consecutive_losses >= lim.max_consecutive_losses
            and state.loss_pause_day in ("", state.day_key)
        )
        if paused:
            state.loss_pause_day = state.day_key
        elif state.consecutive_losses >= lim.max_consecutive_losses:
            # A new day has started: the pause is served, the streak counter starts over.
            state.consecutive_losses, state.loss_pause_day = 0, ""

        return [
            CheckResult("kill_switch", not state.kill_switch, state.kill_reason or "not tripped"),
            CheckResult(
                "live_authorization",
                self.mode is not Mode.LIVE or self.live_authorized,
                "live trading is not authorized",
            ),
            CheckResult(
                "daily_loss_limit",
                day_loss < lim.daily_loss_limit - _TOL,
                f"down {max(day_loss, 0):.2%} today, limit {lim.daily_loss_limit:.2%}",
            ),
            CheckResult(
                "weekly_drawdown_limit",
                week_loss < lim.weekly_drawdown_limit - _TOL,
                f"down {max(week_loss, 0):.2%} this week, limit {lim.weekly_drawdown_limit:.2%}",
            ),
            CheckResult(
                "loss_streak_pause",
                not paused,
                f"{state.consecutive_losses} losses in a row, no new trades today",
            ),
        ]

    def trading_allowed(self, state: PortfolioState, now: datetime | None = None) -> bool:
        return all(c.passed for c in self.account_checks(state, now or datetime.now(UTC)))

    # --- per-trade evaluation --------------------------------------------------------

    def evaluate(
        self,
        proposal: TradeProposal,
        state: PortfolioState,
        *,
        now: datetime | None = None,
        returns: Mapping[str, Sequence[float]] | None = None,
    ) -> RiskDecision:
        """Approve and size a proposal, or reject it with every reason.

        `returns` maps symbols to aligned recent return series, used for the correlation check.
        """
        now = now or datetime.now(UTC)
        lim = self.limits
        checks = self.account_checks(state, now)

        p = proposal
        long = p.side is Side.LONG
        numbers_ok = all(
            isinstance(x, (int, float)) and math.isfinite(x) and x > 0
            for x in (p.entry, p.stop, p.fx_to_base, *p.targets)
        )
        checks.append(CheckResult("valid_prices", numbers_ok, "entry, stop and targets must be positive numbers"))
        if not numbers_ok or not p.targets:
            checks.append(CheckResult("has_target", bool(p.targets), "a trade needs at least one target"))
            return RiskDecision(False, p.symbol, checks=tuple(checks))

        stop_side_ok = p.stop < p.entry if long else p.stop > p.entry
        target = p.targets[0]
        target_side_ok = target > p.entry if long else target < p.entry
        stop_distance = abs(p.entry - p.stop) / p.entry
        checks += [
            CheckResult("stop_loss", stop_side_ok, "a stop-loss on the losing side of entry is mandatory"),
            CheckResult("target_side", target_side_ok, "the first target must be on the winning side of entry"),
            CheckResult(
                "stop_distance",
                lim.min_stop_distance <= stop_distance <= lim.max_stop_distance,
                f"stop is {stop_distance:.2%} from entry, allowed "
                f"{lim.min_stop_distance:.2%} to {lim.max_stop_distance:.2%}",
            ),
        ]
        age = now - p.as_of
        checks.append(CheckResult(
            "fresh_signal",
            timedelta(0) <= age <= timedelta(hours=lim.max_signal_age_hours),
            f"signal is {age.total_seconds() / 3600:.0f}h old",
        ))
        checks.append(CheckResult(
            "no_duplicate_position",
            p.symbol not in state.positions,
            "already holding this symbol; adding to a position is not allowed",
        ))
        checks.append(CheckResult(
            "max_positions",
            len(state.positions) < lim.max_positions,
            f"{len(state.positions)} positions open, limit {lim.max_positions}",
        ))
        last_loss = state.last_loss_at.get(p.symbol)
        cooling = last_loss is not None and now - last_loss < timedelta(hours=lim.symbol_cooldown_hours)
        checks.append(CheckResult(
            "symbol_cooldown", not cooling, "recent loss in this symbol, no re-entry yet"
        ))
        checks.append(self._correlation_check(p.symbol, state, returns))

        # --- sizing: the smallest of four independent caps ---------------------------
        side_cost = self.costs.per_side(p.asset_class)
        unit_entry = p.entry * p.fx_to_base
        unit_stop = p.stop * p.fx_to_base
        unit_target = target * p.fx_to_base
        unit_loss = abs(unit_entry - unit_stop) + side_cost * (unit_entry + unit_stop)
        unit_gain = abs(unit_target - unit_entry) - side_cost * (unit_entry + unit_target)

        equity = max(state.equity, 0.0)
        caps = {
            "risk per trade": equity * lim.risk_per_trade / unit_loss,
            "position allocation": equity * lim.max_position_allocation / unit_entry,
            "deployed capital": max(
                equity * min(lim.max_deployed_capital, lim.max_leverage) - state.deployed, 0.0
            ) / unit_entry,
        }
        if p.avg_daily_turnover is not None:
            caps["liquidity"] = p.avg_daily_turnover * lim.max_participation / p.entry
        binding = min(caps, key=caps.get)
        quantity = caps[binding]
        if p.asset_class is AssetClass.EQUITY:
            quantity = float(math.floor(quantity + _TOL))  # whole shares only
        else:
            quantity = math.floor(quantity * 1e6) / 1e6

        notional = quantity * unit_entry
        max_loss = quantity * unit_loss
        reward_risk = unit_gain / unit_loss if unit_loss > 0 else 0.0
        checks += [
            CheckResult(
                "position_size",
                quantity > 0 and notional >= lim.min_notional,
                f"size {quantity:g} (limited by {binding}) is too small to be worth its costs",
            ),
            CheckResult(
                "reward_risk",
                reward_risk >= lim.min_reward_risk - _TOL,
                f"reward:risk after costs is {reward_risk:.2f}, minimum {lim.min_reward_risk:.2f}",
            ),
        ]

        approved = all(c.passed for c in checks)
        return RiskDecision(
            approved=approved,
            symbol=p.symbol,
            quantity=quantity if approved else 0.0,
            notional=notional if approved else 0.0,
            max_loss=max_loss if approved else 0.0,
            estimated_costs=quantity * side_cost * (unit_entry + unit_target) if approved else 0.0,
            expected_return=quantity * unit_gain if approved else 0.0,
            reward_risk=reward_risk,
            checks=tuple(checks),
        )

    def _correlation_check(
        self,
        symbol: str,
        state: PortfolioState,
        returns: Mapping[str, Sequence[float]] | None,
    ) -> CheckResult:
        name = "correlation"
        held = [s for s in state.positions if s != symbol]
        if not held:
            return CheckResult(name, True, "no open positions to compare with")
        mine = (returns or {}).get(symbol)
        if mine is None:
            # Missing evidence is not permission: without data we cannot show diversification.
            return CheckResult(name, False, "no return history to check correlation with open positions")
        worst, worst_symbol = 0.0, ""
        for other in held:
            theirs = (returns or {}).get(other)
            if theirs is None:
                return CheckResult(name, False, f"no return history for open position {other}")
            n = min(len(mine), len(theirs))
            if n < self.limits.min_correlation_samples:
                return CheckResult(name, False, f"only {n} overlapping returns with {other}")
            a, b = np.asarray(mine[-n:], float), np.asarray(theirs[-n:], float)
            if a.std() == 0 or b.std() == 0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if corr > worst:
                worst, worst_symbol = corr, other
        ok = worst <= self.limits.max_correlation
        detail = (
            f"correlation {worst:.2f} with {worst_symbol}, limit {self.limits.max_correlation:.2f}"
            if worst_symbol else "uncorrelated with open positions"
        )
        return CheckResult(name, ok, detail)
