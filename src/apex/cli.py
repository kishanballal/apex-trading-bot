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
from apex.config import RISK_PROFILES, Settings, load_universe, risk_profile
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


def _report(result, settings: Settings, title: str) -> None:
    m = compute_metrics(result)
    if not result.equity_curve:
        console.print("[red]No data. Run `apex fetch --days 3300` first.[/red]")
        raise typer.Exit(1)

    cur = settings.base_currency
    console.print(f"[bold]{title}[/bold] from {m.start} to {m.end} ({m.years:.1f} years), "
                  f"starting capital {result.config.capital:,.0f} {cur}")
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
    out = settings.data_dir / f"backtest_{title}.json"
    out.write_text(json.dumps({
        "metrics": asdict(m), "yearly": table, "buy_and_hold": holds,
        "kill_switch_dates": result.kill_switch_dates, "rejections": result.rejections,
        "equity_curve": result.equity_curve,
        "trades": [{**asdict(t), "pnl": t.pnl, "r_multiple": t.r_multiple} for t in result.trades],
    }, indent=1, default=str), encoding="utf-8")
    console.print(f"Saved to {out}")




@app.command()
def backtest(
    strategy: str = typer.Option("trend", help=f"One of: {', '.join(STRATEGIES)}"),
    reset_days: int = typer.Option(
        -1, help="Simulate a human resetting the drawdown kill switch after this many days. "
                 "Default: never, which is the real rule."),
    no_regime: bool = typer.Option(False, help="Research only: ignore the market regime gate."),
    profile: str = typer.Option("current", help=f"Risk limits: {', '.join(RISK_PROFILES)}"),
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
    _report(run_backtest(groups, candles, config, limits=risk_profile(profile)), settings, strategy)


@app.command()
def ladder(
    capital: float = typer.Option(50_000.0, help="Starting capital."),
    floor: float = typer.Option(40_000.0, help="The account is liquidated before it gets here."),
    no_uptrend: bool = typer.Option(False, help="Also buy dips in stocks below their 200-day average."),
    interval: str = "1d",
) -> None:
    """Backtest the dip-buying ladder: buy a dip once it turns up, add on further dips,
    sell at a profit threshold."""
    from apex.backtest.ladder import LadderConfig, run_ladder

    settings = Settings()
    groups = load_universe(settings.universe_file)
    store = CandleStore(settings.db_path)
    candles = {s: store.load(s, interval) for g in groups for s in (g.benchmark, *g.symbols)}
    store.close()
    config = LadderConfig(capital=capital, floor=floor, usd_inr=settings.usd_inr,
                          require_uptrend=not no_uptrend)
    _report(run_ladder(groups, candles, config), settings, "dip_ladder")


@app.command()
def research(
    first_test_year: int = 2021,
    threshold: float = typer.Option(0.45, help="Minimum predicted win probability to trade."),
    profile: str = typer.Option("current", help=f"Risk limits: {', '.join(RISK_PROFILES)}"),
    interval: str = "1d",
) -> None:
    """Walk-forward test of the ML models, then backtest each one on years it never saw.
    Needs the extras: pip install -e ".[ml]"."""
    from datetime import date

    from apex.ml.dataset import build_dataset
    from apex.ml.strategy import make_table_strategy
    from apex.ml.walkforward import walk_forward

    settings = Settings()
    groups = load_universe(settings.universe_file)
    store = CandleStore(settings.db_path)
    candles = {s: store.load(s, interval) for g in groups for s in (g.benchmark, *g.symbols)}
    store.close()
    data = build_dataset(groups, candles)
    if data.empty:
        console.print("[red]No data. Run `apex fetch --days 3300` first.[/red]")
        raise typer.Exit(1)
    last_year = data["date"].max().year
    console.print(f"{len(data):,} samples. Training and testing year by year, please wait.")
    predictions, scores = walk_forward(data, list(range(first_test_year, last_year + 1)))

    folds = Table(title="Prediction quality on unseen years (AUC 0.50 is a coin flip)")
    for col in ("Model", "Year", "AUC", "Brier", "Brier of base rate", "Top 10% win", "All win"):
        folds.add_column(col)
    for sc in scores:
        folds.add_row(sc.model, str(sc.year), f"{sc.auc:.3f}", f"{sc.brier:.4f}",
                      f"{sc.brier_baseline:.4f}", f"{sc.top_decile_win:.1%}", f"{sc.base_rate:.1%}")
    console.print(folds)

    limits = risk_profile(profile)
    start = date(first_test_year, 1, 1)
    results = Table(title=f"Backtest on unseen years from {first_test_year}, {profile} limits")
    for col in ("Strategy", "Trades", "Win rate", "Profit factor", "Avg R", "Net profit",
                "After tax", "Per year", "Max drawdown"):
        results.add_column(col)

    def run(name: str, strategy: str) -> None:
        cfg = BacktestConfig(strategy=strategy, capital=settings.capital, usd_inr=settings.usd_inr,
                             start=start, kill_switch_reset_days=90)
        m = compute_metrics(run_backtest(groups, candles, cfg, limits=limits))
        results.add_row(name, str(m.trades), f"{m.win_rate:.0%}", f"{m.profit_factor:.2f}",
                        f"{m.expectancy_r:+.2f}", f"{m.net_profit:,.0f}",
                        f"{m.net_profit_after_tax:,.0f}", f"{m.cagr:.2%}", f"{m.max_drawdown:.2%}")

    run("trend (no ML)", "trend")
    run("pullback (no ML)", "pullback")
    for model in [c for c in predictions.columns if c not in ("symbol", "date", "ts", "y", "r")]:
        table = {(a, b): c for a, b, c in predictions[["symbol", "date", model]].itertuples(index=False, name=None)}
        STRATEGIES["_ml"] = make_table_strategy(f"ml_{model}", table, threshold)
        run(f"{model} p >= {threshold:.2f}", "_ml")
    STRATEGIES.pop("_ml", None)
    console.print(results)
    console.print("The kill switch is reset after 90 days in these runs so all rows cover the "
                  "same years. Unseen-year results are evidence, not proof.")


paper_app = typer.Typer(help="Paper trading with fake money.")
app.add_typer(paper_app, name="paper")


@paper_app.command("init")
def paper_init(
    capital: float = typer.Option(50_000.0, help="Fake starting money."),
    floor: float = typer.Option(40_000.0, help="Trading stops for good if the account gets here."),
    profile: str = typer.Option("aggressive", help=f"Risk limits: {', '.join(RISK_PROFILES)}"),
    backfill_days: int = typer.Option(0, help="Demo only: pretend the account started this many "
                                              "days ago so the dashboard has history to show."),
) -> None:
    """Start a fresh paper account and train the model it will trade with."""
    from datetime import timedelta

    from apex.paper.service import PaperTrader

    trader = PaperTrader(Settings())
    start = datetime.now(UTC).date() - timedelta(days=backfill_days)
    console.print("Training the model on all finished trades in the database, please wait.")
    cfg = trader.init(capital=capital, floor=floor, profile=profile, start=start)
    console.print(f"Paper account ready: {cfg.capital:,.0f} starting {cfg.start}, floor "
                  f"{cfg.floor:,.0f}, {cfg.profile} limits. Now run `apex serve`.")


@paper_app.command("run")
def paper_run(no_fetch: bool = typer.Option(False, help="Skip the data download.")) -> None:
    """Run one paper trading cycle: update data, record predictions, update the account."""
    from apex.paper.service import PaperTrader

    settings = Settings()
    if not no_fetch:
        fetch()
    trader = PaperTrader(settings)
    snap = trader.snapshot(trader.cycle())
    console.print(f"Equity {snap['equity']:,.0f} ({snap['pnl_pct']:+.2%}), "
                  f"{len(snap['positions'])} open, {len(snap['planned'])} planned for next open, "
                  f"{snap['stats']['trades']} closed trades. "
                  f"Room above the floor: {snap['room_to_floor']:,.0f}.")
    for p in snap["planned"]:
        console.print(f"  planned buy {p['symbol']}: stop {p['stop']:.2f}, target {p['target']:.2f}")
    if snap["kill_switch"]:
        console.print(f"[red]Kill switch active: {snap['kill_reason']}[/red]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000,
          no_fetch: bool = typer.Option(False, help="Do not download data (offline demo).")) -> None:
    """Start the dashboard and keep the paper account updated every 30 minutes."""
    import uvicorn

    from apex.api.app import create_app

    console.print(f"Dashboard: http://{host}:{port}  (leave this window open, Ctrl+C to stop)")
    uvicorn.run(create_app(Settings(), fetch=not no_fetch), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
