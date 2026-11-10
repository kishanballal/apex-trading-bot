"""Command line entry point: `apex fetch`, `apex scan`, `apex status`."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from apex.config import Settings, load_universe
from apex.data.pipeline import build_providers, make_client, update_universe
from apex.data.store import CandleStore
from apex.service import build_engine, load_state, run_scan, save_report

app = typer.Typer(add_completion=False, help="APEX: selective, risk-first trading system.")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def fetch(interval: str = "1d") -> None:
    """Download and validate market data for the whole watchlist."""
    settings = Settings()
    groups = load_universe(settings.universe_file)
    store = CandleStore(settings.db_path)

    async def run():
        async with make_client(settings) as client:
            return await update_universe(
                groups, build_providers(client, settings), store, interval=interval,
                history_days=settings.history_days, max_concurrency=settings.max_concurrency,
            )

    results = asyncio.run(run())
    store.close()
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    warned = [r for r in ok if r.quality and r.quality.suspicious_jumps]
    console.print(f"[green]Updated {len(ok)} symbols[/green], {len(failed)} failed.")
    for r in warned:
        console.print(f"[yellow]Check {r.symbol}: {r.quality.suspicious_jumps} suspicious price jumps[/yellow]")
    for r in failed:
        console.print(f"[red]Skipped {r.symbol}: {r.error}[/red]")
    if not ok:
        raise typer.Exit(1)


@app.command()
def scan(interval: str = "1d", show_all: bool = typer.Option(False, "--all")) -> None:
    """Scan the watchlist and show what the risk engine would allow. Places no trades."""
    settings = Settings()
    store = CandleStore(settings.db_path)
    state = load_state(settings)
    report = run_scan(load_universe(settings.universe_file), store, build_engine(settings),
                      state, settings, interval=interval, now=datetime.now(UTC))
    store.close()

    for name, regime in report.regimes.items():
        console.print(f"[bold]{name}[/bold] regime: {regime.trend.value}"
                      f"{' (high volatility)' if regime.high_volatility else ''}. {regime.detail}")

    table = Table(title="Opportunities" if not show_all else "Full scan")
    for col in ("Symbol", "Score", "Entry", "Stop", "Target", "Qty", "Max loss", "Exp. profit", "Verdict"):
        table.add_column(col)
    shown = 0
    for row in report.rows:
        decision = report.decisions.get(row.symbol)
        if not row.is_opportunity and not show_all:
            continue
        shown += 1
        if row.proposal and decision:
            verdict = ("[green]APPROVED[/green]" if decision.approved
                       else "[red]REJECTED[/red] " + "; ".join(decision.rejections))
            p = row.proposal
            table.add_row(row.symbol, f"{row.score:.0f}", f"{p.entry:.2f}", f"{p.stop:.2f}",
                          f"{p.targets[0]:.2f}", f"{decision.quantity:g}",
                          f"{decision.max_loss:.0f}", f"{decision.expected_return:.0f}", verdict)
        else:
            table.add_row(row.symbol, f"{row.score:.0f}", f"{row.close:.2f}" if row.close else "-",
                          "-", "-", "-", "-", "-", row.verdict)
    if shown:
        console.print(table)
    console.print(f"{len(report.approved)} approved out of {len(report.rows)} scanned. "
                  f"Mode: {settings.mode.value}. Money values in {settings.base_currency}.")
    if not report.approved:
        console.print("No trade today. That is a valid outcome, not an error.")
    console.print(f"Saved to {save_report(report, settings.data_dir)}")


@app.command()
def status() -> None:
    """Show the risk engine's limits and the account's current standing."""
    settings = Settings()
    state = load_state(settings)
    engine = build_engine(settings)
    checks = engine.account_checks(state, datetime.now(UTC))
    console.print(f"Mode: {settings.mode.value}. Equity: {state.equity:,.0f} {settings.base_currency}. "
                  f"Open positions: {len(state.positions)}. Deployed: {state.deployed:,.0f}.")
    for c in checks:
        console.print(f"  {'[green]OK[/green]  ' if c.passed else '[red]STOP[/red]'} {c.name}"
                      + ("" if c.passed else f": {c.detail}"))
    lim = engine.limits
    console.print(f"Limits: {lim.risk_per_trade:.2%} risk per trade, {lim.daily_loss_limit:.2%} daily, "
                  f"{lim.weekly_drawdown_limit:.2%} weekly, {lim.max_portfolio_drawdown:.2%} max drawdown, "
                  f"{lim.max_positions} positions, {lim.max_position_allocation:.0%} per position, "
                  f"{lim.max_deployed_capital:.0%} deployed.")


if __name__ == "__main__":
    app()
