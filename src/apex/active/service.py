"""The active bot's loop and its dashboard page (http://127.0.0.1:8001).

    python -m apex.active            run the bot and the page (what run_active.bat does)
    python -m apex.active --once     one cycle, print a summary, exit

Every 5 minutes during NSE hours it downloads the latest 5-minute bars for the watchlist,
looks for sudden moves, pulls headlines for stocks that moved, and replays the fake-money
account. Market news and Reddit are read every 10 minutes (every 30 outside market hours).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from apex.active import engine
from apex.active.news import NewsClient, now_ms
from apex.active.store import ActiveStore
from apex.intraday import nse

log = logging.getLogger("apex.active")
DATA = Path("data")
STATIC = Path(__file__).parent / "static"
CYCLE_S = 300
IST = nse.IST_OFFSET_MS


def ist_parts(ms: int) -> tuple[int, int]:
    """(weekday, minutes since midnight) in IST."""
    local = ms + IST
    return datetime.fromtimestamp(local / 1000, UTC).weekday(), (local % 86_400_000) // 60_000


def market_open(ms: int) -> bool:
    wd, minute = ist_parts(ms)
    return wd < 5 and 9 * 60 + 15 <= minute <= 15 * 60 + 35


def fetch_bars(client: httpx.Client, symbol: str, rng: str) -> pd.DataFrame | None:
    ysym = symbol if symbol.startswith("^") else f"{symbol}.NS"
    for attempt in range(3):
        try:
            r = client.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}",
                           params={"interval": "5m", "range": rng})
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"ts": np.asarray(res.get("timestamp") or [], np.int64) * 1000,
                               **{k: q.get(k) or [] for k in
                                  ("open", "high", "low", "close", "volume")}})
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df.volume.fillna(0)
            # Keep only finished bars: the last one may still be forming.
            return df[df.ts + nse.BAR_MS <= now_ms()].reset_index(drop=True)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
            time.sleep(2 * (attempt + 1))
    return None


class ActiveBot:
    def __init__(self, root: Path = DATA, cfg: engine.ActiveConfig = engine.ActiveConfig()):
        self.cfg = cfg
        self.store = ActiveStore(root / "active.db")
        self.state_path = root / "active_state.json"
        universe = tomllib.loads(Path("config/universe.toml").read_text())
        self.symbols: list[str] = universe["equity"]["symbols"]
        self.http = httpx.Client(timeout=30, headers={"User-Agent": nse.UA})
        self.news = NewsClient()
        self.state: dict = {}
        self.last_market_news = 0
        self.news_checked: dict[str, int] = {}
        self.data_errors: list[str] = []
        if self.store.get("start_ms") is None:
            self.store.set("start_ms", str(now_ms()))
            log.info("opened a new active fake-money account")

    # ---- data -----------------------------------------------------------------------------
    def update_bars(self) -> int:
        seen = now_ms()
        first_time = not self.store.symbols()
        rng = "60d" if first_time else "5d"
        new, failed = 0, []
        for s in ["^NSEI", *self.symbols]:
            df = fetch_bars(self.http, s, rng)
            if df is None or df.empty:
                failed.append(s)
                continue
            new += self.store.upsert_bars(s, df, seen)
            time.sleep(0.4)
        self.data_errors = failed
        return new

    def update_news(self, movers: list[str]) -> int:
        t = now_ms()
        added = 0
        gap = 10 * 60_000 if market_open(t) else 30 * 60_000
        if t - self.last_market_news >= gap:
            added += self.store.add_news(self.news.market(), self.symbols, t)
            self.last_market_news = t
        for s in movers[:8]:
            if t - self.news_checked.get(s, 0) < 30 * 60_000:
                continue
            self.news_checked[s] = t
            added += self.store.add_news(self.news.for_symbol(s), self.symbols, t,
                                         force_symbol=s)
            time.sleep(0.5)
        return added

    # ---- account --------------------------------------------------------------------------
    def frames(self) -> dict[str, pd.DataFrame]:
        nifty = self.store.bars("^NSEI")
        out = {}
        for s in self.symbols:
            b = self.store.bars(s)
            if len(b) < 100:
                continue
            f = nse.build(b.drop(columns="first_seen"), nifty if len(nifty) else None)
            out[s] = engine.prepare(f, b.set_index("ts").first_seen)
        return out

    def movers(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        """Stocks whose latest finished bar was a sudden move."""
        hot = []
        for s, f in frames.items():
            if f.empty:
                continue
            r = f.iloc[-1]
            if now_ms() - (r.ts + nse.BAR_MS) > 15 * 60_000:
                continue
            if abs(r.z_move) >= self.cfg.move_z and r.vol_burst >= self.cfg.move_volume:
                hot.append((abs(r.z_move), s))
        return [s for _, s in sorted(hot, reverse=True)]

    def cycle(self) -> dict:
        t0 = time.time()
        new = self.update_bars()
        frames = self.frames()
        added = self.update_news(self.movers(frames))
        start = int(self.store.get("start_ms"))
        news = self.store.news(since_ms=start - self.cfg.news_window_ms)
        res = engine.replay(frames, news, start, self.cfg)
        self.state = self.snapshot(res, frames, new, added, time.time() - t0)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, default=float))
        tmp.replace(self.state_path)
        return self.state

    def snapshot(self, res: engine.Result, frames, new_bars: int, new_news: int,
                 secs: float) -> dict:
        trades = res.trades
        wins = [t for t in trades if t["pnl"] > 0]
        start = int(self.store.get("start_ms"))
        latest = max((int(f.ts.iloc[-1]) for f in frames.values() if len(f)), default=0)
        return {
            "generated_ms": now_ms(), "market_open": market_open(now_ms()),
            "start_ms": start, "latest_bar_ms": latest + nse.BAR_MS if latest else 0,
            "capital": self.cfg.capital, "equity": res.equity_now, "cash": res.cash,
            "pnl": round(res.equity_now - self.cfg.capital, 2),
            "halted": res.halted, "today": res.today,
            "stats": {"trades": len(trades), "wins": len(wins),
                      "win_rate": round(len(wins) / len(trades), 3) if trades else None,
                      "net": round(sum(t["pnl"] for t in trades), 2)},
            "positions": res.positions, "trades": trades[-100:][::-1],
            "events": res.events[-80:][::-1], "equity_curve": res.equity[-600:],
            "news": self.store.latest_news(40),
            "feeds": {"errors": self.news.errors, "data_failed": self.data_errors},
            "cycle": {"new_bars": new_bars, "new_headlines": new_news, "seconds": round(secs, 1)},
            "rules": {"positions": self.cfg.max_positions, "per_trade": self.cfg.notional,
                      "daily_loss_limit": self.cfg.daily_loss_limit,
                      "floor": self.cfg.equity_floor},
        }

    def run_forever(self) -> None:
        while True:
            try:
                s = self.cycle()
                log.info("equity %.0f, %d trades, %d open, %d new bars, %d new headlines",
                         s["equity"], s["stats"]["trades"], len(s["positions"]),
                         s["cycle"]["new_bars"], s["cycle"]["new_headlines"])
            except Exception:
                log.exception("cycle failed; trying again next cycle")
            time.sleep(self.sleep_seconds())

    def sleep_seconds(self) -> float:
        t = now_ms()
        if market_open(t):
            # wake 25 s after the next 5-minute bar closes
            return max(30.0, (CYCLE_S * 1000 - t % (CYCLE_S * 1000)) / 1000 + 25)
        return 15 * 60


def serve(bot: ActiveBot, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse

    from fastapi.staticfiles import StaticFiles

    api = FastAPI(title="APEX active")
    api.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "api" / "static"),
              name="static")

    @api.get("/")
    def page():
        return FileResponse(STATIC / "active.html")

    @api.get("/api/state")
    def state():
        if bot.state:
            return JSONResponse(json.loads(json.dumps(bot.state, default=float)))
        if bot.state_path.exists():
            return JSONResponse(json.loads(bot.state_path.read_text()))
        return JSONResponse({"loading": True})

    uvicorn.run(api, host="127.0.0.1", port=port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m apex.active")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    bot = ActiveBot()
    if args.once:
        s = bot.cycle()
        print(json.dumps({k: s[k] for k in ("equity", "stats", "today", "halted", "cycle",
                                             "feeds")}, indent=1, default=float))
        print(f"{len(s['events'])} recent sudden moves, {len(s['news'])} recent headlines")
        return
    threading.Thread(target=bot.run_forever, daemon=True).start()
    print(f"APEX active bot running. Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    serve(bot, args.port)
