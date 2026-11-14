"""The active bot's loop and its dashboard page (http://127.0.0.1:8001).

    python -m apex.active            run the bot and the page (what run_active.bat does)
    python -m apex.active --once     one cycle, print a summary, exit

Every minute during NSE hours it downloads the latest 1-minute bars for the watchlist,
looks for sudden moves, pulls headlines for stocks that moved, and replays the fake-money
account. Market news is read every 10 minutes (every 30 outside market hours).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from apex.active import bars1m, engine
from apex.active.news import NewsClient, Scorer, load_aliases, now_ms
from apex.active.store import ActiveStore
from apex.intraday import nse

log = logging.getLogger("apex.active")
DATA = Path("data")
STATIC = Path(__file__).parent / "static"
BAR_MS = bars1m.BAR_MS
WAKE_AFTER_S = 12      # seconds after each minute closes
FEATURE_DAYS = 8       # sessions of history the features need (5) plus a margin
LIVE = engine.ActiveConfig(live_grace_ms=3 * 60_000, max_fill_wait_ms=3 * 60_000)
IST = nse.IST_OFFSET_MS


def ist_parts(ms: int) -> tuple[int, int]:
    """(weekday, minutes since midnight) in IST."""
    local = ms + IST
    return datetime.fromtimestamp(local / 1000, UTC).weekday(), (local % 86_400_000) // 60_000


def market_open(ms: int) -> bool:
    wd, minute = ist_parts(ms)
    return wd < 5 and 9 * 60 + 15 <= minute <= 15 * 60 + 35


def fetch_bars(client: httpx.Client, symbol: str, params: dict) -> pd.DataFrame | None:
    """Finished 1-minute bars inside NSE hours, or None if Yahoo did not answer."""
    ysym = symbol if symbol.startswith("^") else f"{symbol}.NS"
    for attempt in range(3):
        try:
            r = client.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}",
                           params={"interval": "1m", **params})
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"ts": np.asarray(res.get("timestamp") or [], np.int64) * 1000,
                               **{k: q.get(k) or [] for k in
                                  ("open", "high", "low", "close", "volume")}})
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df.volume.fillna(0)
            minute = ((df.ts + IST) % 86_400_000) // 60_000 - 555
            # Keep only finished bars: the last one may still be forming.
            keep = (df.ts + BAR_MS <= now_ms()) & (minute >= 0) & (minute < 375)
            return df[keep].drop_duplicates("ts").reset_index(drop=True)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
            time.sleep(1 + attempt)
    return None


class ActiveBot:
    def __init__(self, root: Path = DATA, cfg: engine.ActiveConfig = LIVE):
        self.cfg = cfg
        # Version 2 (1-minute bars) keeps its own file; the 5-minute account stays in active.db.
        self.store = ActiveStore(root / "active1m.db")
        self._past: dict[str, pd.DataFrame] = {}   # prepared frames for finished days
        self._past_day = -1
        self.state_path = root / "active_state.json"
        universe = tomllib.loads(Path("config/universe.toml").read_text())
        extra = load_aliases(Path("config/active_universe.toml"))
        self.symbols: list[str] = list(dict.fromkeys([*universe["equity"]["symbols"], *extra]))
        self.scorer = Scorer()
        log.info("watching %d stocks; headline reader: %s", len(self.symbols), self.scorer.name)
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
        stale = seen - self.store.latest_ts() > 20 * 3600 * 1000
        if stale:   # first start, or back after a day off: Yahoo serves 7 days per request
            t = seen // 1000
            windows = [{"period1": t - 14 * 86400, "period2": t - 7 * 86400},
                       {"period1": t - 7 * 86400, "period2": t}]
        else:
            windows = [{"range": "1d"}]

        def get(sym: str):
            parts = [fetch_bars(self.http, sym, w) for w in windows]
            parts = [p for p in parts if p is not None and not p.empty]
            return sym, (pd.concat(parts).drop_duplicates("ts") if parts else None)

        new, failed = 0, []
        with ThreadPoolExecutor(6) as pool:
            for sym, df in pool.map(get, ["^NSEI", *self.symbols]):
                if df is None:
                    failed.append(sym)
                else:
                    new += self.store.upsert_bars(sym, df, seen)
        self.data_errors = failed
        return new

    def update_news(self, movers: list[str]) -> int:
        t = now_ms()
        added = 0
        gap = 10 * 60_000 if market_open(t) else 30 * 60_000
        if t - self.last_market_news >= gap:
            added += self.store.add_news(self.news.market(), self.symbols, t,
                                         scorer=self.scorer)
            self.last_market_news = t
        todo = [s for s in movers if t - self.news_checked.get(s, 0) >= 30 * 60_000][:6]
        if todo:
            with ThreadPoolExecutor(len(todo)) as pool:
                found = list(pool.map(self.news.for_symbol, todo))
            for s, items in zip(todo, found):
                self.news_checked[s] = t
                added += self.store.add_news(items, self.symbols, t, force_symbol=s,
                                             scorer=self.scorer)
        return added

    # ---- account --------------------------------------------------------------------------
    def _build(self, bars: pd.DataFrame, keep_from_day: int | None, keep_to_day: int | None,
               finished: bool) -> dict[str, pd.DataFrame]:
        out = {}
        for s, b in bars.groupby("symbol", sort=False):
            if s == "^NSEI" or len(b) < 2 * bars1m.SESSION_BARS:
                continue
            f = bars1m.build(b.drop(columns=["symbol", "first_seen"]), self.cfg.move_z,
                             self.cfg.move_volume)
            f = engine.prepare(f, b.set_index("ts").first_seen, bar_ms=BAR_MS, cfg=self.cfg)
            if finished:   # a finished day's final bar is its last, even if 15:29 is missing
                f.loc[f.day != f.day.shift(-1), "is_last_"] = True
            if keep_from_day is not None:
                f = f[f.day >= keep_from_day]
            if keep_to_day is not None:
                f = f[f.day < keep_to_day]
            if len(f):
                out[s] = f
        return out

    def frames(self) -> dict[str, pd.DataFrame]:
        """Feature frames from the account's start to now.

        Features only ever look back 5 sessions, so finished days are built once and kept;
        each minute only today's rows are rebuilt from the last FEATURE_DAYS sessions.
        """
        days = self.store.session_days()
        if not days:
            return {}
        day_ms = 86_400_000
        today = (now_ms() + IST) // day_ms
        start_day = (int(self.store.get("start_ms")) + IST) // day_ms
        if self._past_day != today:
            past_days = [d for d in days if start_day <= d < today]
            self._past = {}
            for d in past_days:   # one day at a time keeps each build small and identical
                lo = [x for x in days if x < d][-(FEATURE_DAYS - 1):]
                first = (lo[0] if lo else d) * day_ms - IST
                part = self._build(self.store.bars_between(first, (d + 1) * day_ms - IST),
                                   d, d + 1, finished=True)
                for s, f in part.items():
                    self._past[s] = pd.concat([self._past[s], f]) if s in self._past else f
            self._past_day = today
        lo = [x for x in days if x < today][-(FEATURE_DAYS - 1):]
        first = (lo[0] if lo else today) * day_ms - IST
        now_frames = self._build(self.store.bars_between(first, (today + 1) * day_ms - IST),
                                 today, None, finished=False)
        out = dict(self._past)
        for s, f in now_frames.items():
            out[s] = pd.concat([out[s], f], ignore_index=True) if s in out else f
        return out

    def movers(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        """Stocks with a fresh sudden move in the last three minutes, biggest first."""
        hot, cutoff = [], now_ms() - 4 * 60_000
        for s, f in frames.items():
            tail = f.iloc[-3:]
            tail = tail[(tail.ts >= cutoff) & (tail.sig_spike | tail.sig_drop | tail.sig_bounce)]
            if len(tail):
                hot.append((float(tail.z_move.abs().max()), s))
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
            "start_ms": start, "latest_bar_ms": latest + BAR_MS if latest else 0,
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
            "watching": len(self.symbols), "scorer": self.scorer.name,
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
        if market_open(t):   # wake shortly after the next minute closes
            wait = (60_000 - t % 60_000) / 1000 + WAKE_AFTER_S
            return wait
        return 5 * 60


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
