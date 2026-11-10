"""Command line entry point: `apex fetch`, `apex scan`, `apex status`."""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import asdict
from datetime import UTC, datetime

import typer
from rich.console import Console
from rich.table import Table

from apex.backtest.engine import BacktestConfig, run_backtest
from apex.backtest.metrics import buy_and_hold, compute_metrics, yearly_table
from apex.config import Settings, load_universe
from apex.data.pipeline import build_providers, make_client, update_universe
from apex.data.store import CandleStore
from apex.scanner.scanner import STRATEGIES, ScannerConfig
from apex.service import build_engine, load_state, run_scan, save_report

app = typer.Typer(add_completion=False, help="APEX: selective, risk-first trading system.")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def fetch(interval: str = "1d",
          days: int = typer.Option(0, help="Days of history. Use 3300 before a backtest.")) -> None:
    """Download and validate market data for the whole watchlist."""
    settings = Settings()
    if days:
        settings.history_days = days
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


@app.command()
def backtest(
    strategy: str = typer.Option("trend", help=f"One of: {', '.join(STRATEGIES)}"),
    reset_days: int = typer.Option(
        -1, help="Simulate a human resetting the drawdown kill switch after this many days. "
                 "Default: never, which is the real rule."),
    no_regime: bool = typer.Option(False, help="Research only: ignore the market regime gate."),
    interval: str = "1d",
) -> None:
    """Replay history through the strategy and the risk engine, with costs and delays."""
    if strategy not in STRATEGIES:
        raise typer.BadParameter(f"unknown strategy {strategy}")
    settings = Settings()
    groups = load_universe(settings.universe_file)
    store = CandleStore(settings.db_path)
    candles = {s: store.load(s, interval) for g in groups for s in (g.benchmark, *g.symbols)}
    store.close()
    config = BacktestConfig(
        strategy=strategy, capital=settings.capital, usd_inr=settings.usd_inr,
        kill_switch_reset_days=reset_days if reset_days >= 0 else None,
        scanner=ScannerConfig(require_regime=not no_regime),
    )
    result = run_backtest(groups, candles, config)
    m = compute_metrics(result)
    if not result.equity_curve:
        console.print("[red]No data. Run `apex fetch --days 3300` first.[/red]")
        raise typer.Exit(1)

    cur = settings.base_currency
    console.print(f"[bold]{strategy}[/bold] from {m.start} to {m.end} ({m.years:.1f} years), "
                  f"starting capital {settings.capital:,.0f} {cur}")
    summary = Table(show_header=False)
    for label, value in [
        ("Trades", f"{m.trades}"),
        ("Win rate", f"{m.win_rate:.1%}"),
        ("Profit factor", f"{m.profit_factor:.2f}"),
        ("Expectancy per trade", f"{m.expectancy:,.0f} {cur} ({m.expectancy_r:+.2f} R)"),
        ("Net profit after costs", f"{m.net_profit:,.0f} {cur}"),
        ("Estimated tax", f"{m.estimated_tax:,.0f} {cur}"),
        ("Net profit after tax", f"{m.net_profit_after_tax:,.0f} {cur}"),
        ("Return per year (CAGR)", f"{m.cagr:.2%}"),
        ("Maximum drawdown", f"{m.max_drawdown:.2%}"),
        ("Sharpe / Sortino", f"{m.sharpe:.2f} / {m.sortino:.2f}"),
        ("Average capital deployed", f"{m.avg_exposure:.1%}"),
        ("Costs paid", f"{m.total_costs:,.0f} {cur}"),
        ("Exits", ", ".join(f"{k} {v}" for k, v in m.exit_reasons.items())),
        ("Kill switch tripped", ", ".join(map(str, result.kill_switch_dates)) or "never"),
    ]:
        summary.add_row(label, value)
    console.print(summary)

    years = Table(title="By calendar year")
    for col in ("Year", "Return", "Max drawdown", "Trades", "Win rate"):
        years.add_column(col)
    table = yearly_table(result)
    for y in table:
        years.add_row(str(y["year"]), f"{y['return']:+.2%}", f"{y['max_drawdown']:.2%}",
                      str(y["trades"]), f"{y['win_rate']:.0%}")
    console.print(years)
    holds = {name: buy_and_hold(series) for name, series in result.benchmarks.items()}
    for name, h in holds.items():
        console.print(f"Buy and hold {name}: {h['cagr']:.2%} per year, "
                      f"maximum drawdown {h['max_drawdown']:.2%}")
    console.print("Past results do not predict future results. The watchlist is today's large "
                  "companies, which flatters any backtest (survivorship bias).")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    out = settings.data_dir / f"backtest_{strategy}.json"
    out.write_text(json.dumps({
        "metrics": asdict(m), "yearly": table, "buy_and_hold": holds,
        "kill_switch_dates": result.kill_switch_dates, "rejections": result.rejections,
        "equity_curve": result.equity_curve,
        "trades": [{**asdict(t), "pnl": t.pnl, "r_multiple": t.r_multiple} for t in result.trades],
    }, indent=1, default=str), encoding="utf-8")
    console.print(f"Saved to {out}")


if __name__ == "__main__":
    app()
