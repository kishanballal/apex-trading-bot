"""Active fake-money account: a deterministic replay over stored 1-minute bars and news.

Every cycle rebuilds the whole account from the stored bars and headlines, the same way the
daily paper bot does. That makes it restart-safe, and a crash or a PC restart never loses or
duplicates a trade. Trades only happen on bars the bot saw live (within a few minutes of the
bar closing), and a buy fills at the first price AFTER the bot saw the signal, so when the PC is off nothing is traded, as in real life.

Two entry rules, both buying (long only, no leverage, every position closed the same day):
- news_momentum: a sudden 5-minute jump (3x normal size, 3x normal volume) above VWAP, in a
  stock with positive headlines in the last 6 hours. The idea: moves with news continue.
- dip_bounce: after a sudden drop on heavy volume, the first green bar, unless the news on
  that stock is negative (a falling knife). The best rule in the 60-day price test.

Safety rules, several borrowed from freqtrade's "protections":
- max 4 positions, one per stock, INR 8,000 each, 2-ATR stop on every trade
- no entries before 09:30 or after 14:45, everything closed on the 15:25 bar
- 30-minute cooldown on a stock after an exit
- pause 60 minutes after 3 stop-outs in a day (StoplossGuard)
- stop trading for the day after INR 1,000 of losses (2% of the account)
- kill switch: no new trades at all if equity falls below INR 45,000
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from apex.intraday.nse import IST_OFFSET_MS

MIN = 60_000


@dataclass(frozen=True)
class ActiveConfig:
    capital: float = 50_000.0
    notional: float = 8_000.0
    max_positions: int = 4
    fee: float = 0.00053        # per side: brokerage, STT, exchange, GST, stamp
    slippage: float = 0.0002    # per side
    move_z: float = 3.0
    move_volume: float = 3.0
    news_window_ms: int = 6 * 3600 * 1000
    news_positive: float = 0.2
    news_negative: float = -0.2
    first_entry_minute: int = 15     # 09:30
    last_entry_minute: int = 330     # 14:45
    live_grace_ms: int = 10 * MIN
    cooldown_ms: int = 30 * MIN
    stoploss_guard: int = 3
    stoploss_pause_ms: int = 60 * MIN
    daily_loss_limit: float = 1_000.0
    equity_floor: float = 45_000.0
    max_hold_ms: int = 120 * MIN
    max_fill_wait_ms: int = 5 * MIN   # give up on an entry if no price arrives in time
    target_atr: float = 3.0


RULE_STOPS = {"news_momentum": 1.5, "dip_bounce": 2.0}


@dataclass
class Position:
    symbol: str
    rule: str
    entry_ts: int
    entry: float          # price paid incl. slippage
    qty: float
    stop: float
    target: float
    news: str = ""


@dataclass
class Result:
    trades: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    equity: list[dict] = field(default_factory=list)
    cash: float = 0.0
    equity_now: float = 0.0
    halted: str = ""
    today: dict = field(default_factory=dict)


def news_context(news: pd.DataFrame, symbol: str, t_close: int, cfg: ActiveConfig) -> dict:
    """Headlines about `symbol` that the bot had already fetched by the bar's close."""
    if news.empty:
        return {"count": 0, "tone": 0.0, "top": ""}
    pub = news.published_ms.where(news.published_ms > 0, news.fetched_ms)
    m = (news.symbol == symbol) & (news.fetched_ms <= t_close + cfg.live_grace_ms) & (
        pub >= t_close - cfg.news_window_ms) & (pub <= t_close + cfg.live_grace_ms)
    rel = news[m]
    if rel.empty:
        return {"count": 0, "tone": 0.0, "top": ""}
    toned = rel[rel.score != 0]
    tone = float(toned.score.mean()) if len(toned) else 0.0
    top = rel.iloc[int(np.argmax(np.abs(rel.score.to_numpy())))].title
    return {"count": int(len(rel)), "tone": round(tone, 3), "top": str(top)[:160]}


def replay(frames: dict[str, pd.DataFrame], news: pd.DataFrame, start_ms: int,
           cfg: ActiveConfig = ActiveConfig()) -> Result:
    """frames: symbol -> a prepare()d frame. Only the minutes that matter are visited
    (signals, pending orders, open positions), so a month of 1-minute bars replays quickly.

    Timing is honest about what the bot could know: a signal on a bar is acted on at the
    moment the bot first saw that bar (`first_seen`), and the buy fills at the OPEN of the
    first bar that starts after that moment.
    """
    res = Result(cash=cfg.capital)
    A: dict[str, dict[str, np.ndarray]] = {}
    signals: dict[int, list[tuple[str, int]]] = {}
    all_ts: set[int] = set()
    cols = ["ts", "open", "high", "low", "close", "z_move", "vol_burst", "move", "r1",
            "vwap_dist", "atr", "minute_of_day", "first_seen", "bar_ms"]
    flags = ["sig_spike", "sig_drop", "sig_bounce", "is_last_"]
    for s, f in frames.items():
        if f.empty:
            continue
        a = {c: f[c].to_numpy(float) for c in cols if c != "ts"}
        a["ts"] = f.ts.to_numpy(np.int64)
        for c in flags:
            a[c] = f[c].to_numpy(bool)
        A[s] = a
        live = a["ts"] >= start_ms
        all_ts.update(a["ts"][live].tolist())
        for i in np.flatnonzero(live & (a["sig_spike"] | a["sig_drop"] | a["sig_bounce"])):
            signals.setdefault(int(a["ts"][i]), []).append((s, int(i)))
    if not all_ts:
        res.equity_now = cfg.capital
        return res

    open_pos: dict[str, Position] = {}
    pending: dict[str, dict] = {}   # symbol -> entry waiting for the next bar to open
    last_close: dict[str, float] = {}
    cooldown_until: dict[str, int] = {}
    day_stats: dict[int, dict] = {}
    halted = ""

    def day_of(ts: int) -> int:
        return (ts + IST_OFFSET_MS) // 86_400_000

    def row(s: str, ts: int) -> int | None:
        t = A[s]["ts"]
        i = int(np.searchsorted(t, ts))
        return i if i < len(t) and t[i] == ts else None

    def mark() -> float:
        return res.cash + sum(pp.qty * last_close.get(pp.symbol, pp.entry)
                              for pp in open_pos.values())

    def close(pos: Position, ts: int, px: float, why: str) -> None:
        proceeds = pos.qty * px * (1 - cfg.slippage) * (1 - cfg.fee)
        cost = pos.qty * pos.entry * (1 + cfg.fee)
        pnl = proceeds - cost
        res.cash += proceeds
        d = day_stats.setdefault(day_of(ts), {"pnl": 0.0, "stops": 0, "pause_until": 0,
                                              "trades": 0})
        d["pnl"] += pnl
        d["trades"] += 1
        if why == "stop":
            d["stops"] += 1
            if d["stops"] >= cfg.stoploss_guard:
                d["pause_until"] = ts + cfg.stoploss_pause_ms
        cooldown_until[pos.symbol] = ts + cfg.cooldown_ms
        res.trades.append({"symbol": pos.symbol, "rule": pos.rule, "entry_ts": pos.entry_ts,
                           "exit_ts": ts, "entry": round(pos.entry, 2), "exit": round(px, 2),
                           "qty": round(pos.qty, 4), "pnl": round(pnl, 2),
                           "ret_pct": round(pnl / cost * 100, 3), "reason": why,
                           "news": pos.news})
        del open_pos[pos.symbol]

    for ts in sorted(all_ts):
        day = day_of(ts)
        d = day_stats.setdefault(day, {"pnl": 0.0, "stops": 0, "pause_until": 0, "trades": 0})
        # 1) fill entries whose decision time has passed, at this bar's open
        for s in sorted(pending):
            order = pending[s]
            i = row(s, ts)
            if ts < order["fill_after"] or (i is None and ts <= order["expires"]):
                continue
            del pending[s]
            why_not = ""
            if i is None or ts > order["expires"] or day_of(ts) != order["day"]:
                why_not = "no fresh price in time"
            elif halted:
                why_not = "kill switch"
            elif s in open_pos:
                why_not = "already holding"
            elif len(open_pos) >= cfg.max_positions:
                why_not = f"{cfg.max_positions} positions already open"
            elif d["pnl"] <= -cfg.daily_loss_limit:
                why_not = "daily loss limit hit"
            elif ts < d["pause_until"]:
                why_not = "paused after 3 stop-outs"
            elif ts < cooldown_until.get(s, 0):
                why_not = "cooldown after last trade"
            spend = min(cfg.notional, res.cash)
            if not why_not and spend < cfg.notional * 0.5:
                why_not = "not enough cash"
            order["event"]["action"] = "skipped: " + why_not if why_not else "bought"
            if not why_not:
                o = float(A[s]["open"][i])
                price = o * (1 + cfg.slippage)
                qty = spend / (price * (1 + cfg.fee))
                res.cash -= qty * price * (1 + cfg.fee)
                open_pos[s] = Position(
                    s, order["rule"], ts, price, qty,
                    stop=o - RULE_STOPS[order["rule"]] * order["atr"],
                    target=o + cfg.target_atr * order["atr"], news=order["news"])
        # 2) exits on every bar a position is open (stop checked first, pessimistic)
        for s in sorted(open_pos):
            i = row(s, ts)
            if i is None:
                continue
            a, pos = A[s], open_pos[s]
            o, h, lo, c = (float(a[k][i]) for k in ("open", "high", "low", "close"))
            last_close[s] = c
            first = pos.entry_ts == ts
            if lo <= pos.stop:
                close(pos, ts, pos.stop if first else min(o, pos.stop), "stop")
            elif h >= pos.target:
                close(pos, ts, pos.target if first else max(o, pos.target), "target")
            elif ts - pos.entry_ts >= cfg.max_hold_ms:
                close(pos, ts, c, "time")
            elif a["is_last_"][i]:
                close(pos, ts, c, "session end")
        # 3) signals on this bar's close
        for s, i in signals.get(ts, ()):
            a = A[s]
            spike, bounce = bool(a["sig_spike"][i]), bool(a["sig_bounce"][i])
            bar_ms = int(a["bar_ms"][i])
            t_close = ts + bar_ms
            ctx = news_context(news, s, t_close, cfg)
            kind = "spike" if spike else "bounce" if bounce else "drop"
            ev = {"ts": t_close, "symbol": s, "kind": kind,
                  "move_pct": round(float(a["move"][i]) * 100, 2),
                  "volume_x": round(float(a["vol_burst"][i]), 1), "news_count": ctx["count"],
                  "news_tone": ctx["tone"], "headline": ctx["top"], "action": "watch only"}
            res.events.append(ev)
            rule = None
            if spike and a["vwap_dist"][i] > 0:
                if ctx["tone"] >= cfg.news_positive:
                    rule = "news_momentum"
                else:
                    ev["action"] = "no positive news, not chasing"
            elif bounce:
                if ctx["tone"] <= cfg.news_negative:
                    ev["action"] = "bad news, not catching the knife"
                else:
                    rule = "dip_bounce"
            if rule is None:
                continue
            seen = a["first_seen"][i]
            live = np.isfinite(seen) and seen <= t_close + cfg.live_grace_ms
            minute = a["minute_of_day"][i]
            if not live:
                ev["action"] = "bot was not running"
            elif not cfg.first_entry_minute <= minute <= cfg.last_entry_minute:
                ev["action"] = "outside 09:30-14:45 entry window"
            elif s in open_pos or s in pending:
                ev["action"] = "already holding"
            elif not np.isfinite(a["atr"][i]) or a["atr"][i] <= 0:
                ev["action"] = "no volatility estimate"
            else:
                decided = max(t_close, int(seen))
                ev["action"] = "buying at the next price"
                pending[s] = {"fill_after": decided, "expires": decided + cfg.max_fill_wait_ms,
                              "day": day, "rule": rule, "atr": float(a["atr"][i]),
                              "news": ctx["top"], "event": ev}
        eq = mark()
        if not halted and eq < cfg.equity_floor:
            halted = f"equity {eq:,.0f} fell below the {cfg.equity_floor:,.0f} floor"
        res.equity.append({"ts": ts, "equity": round(eq, 2)})

    res.positions = [{**asdict(pp), "last": last_close.get(pp.symbol, pp.entry),
                      "unrealized": round(pp.qty * (last_close.get(pp.symbol, pp.entry)
                                                    - pp.entry), 2)}
                     for pp in open_pos.values()]
    res.equity_now = round(mark(), 2)
    res.cash = round(res.cash, 2)
    res.halted = halted
    if day_stats:
        last_day = max(day_stats)
        res.today = {k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in day_stats[last_day].items()}
    return res


def prepare(f: pd.DataFrame, first_seen: pd.Series | None, bar_ms: int = 300_000,
            cfg: ActiveConfig = ActiveConfig()) -> pd.DataFrame:
    """Add the replay's helper columns. Works for 5-minute frames (nse.build, where the
    signal flags are derived here) and 1-minute frames (bars1m.build, which sets them)."""
    out = f.copy()
    out["first_seen"] = (out.ts.map(first_seen) if first_seen is not None
                         else np.nan).astype(float)
    out["bar_ms"] = bar_ms
    if "move" not in out:
        out["move"] = out.r1
    if "sig_spike" not in out:
        ok = out.z_move.notna() & out.vol_burst.notna()
        burst = out.vol_burst >= cfg.move_volume
        out["sig_spike"] = ok & (out.z_move >= cfg.move_z) & burst
        out["sig_drop"] = ok & (out.z_move <= -cfg.move_z) & burst
        out["sig_bounce"] = (ok & out.sig_drop.shift(fill_value=False) & (out.r1 > 0)
                             & (out.rsi14 < 40))
    # The session's final bar: the last candle before 15:30, or the last bar before a new
    # day starts. (Today's latest bar is not "last" until the session is over.)
    nxt_day = ((out.ts.shift(-1) + IST_OFFSET_MS) // 86_400_000)
    day = (out.ts + IST_OFFSET_MS) // 86_400_000
    final_minute = 375 - bar_ms // MIN
    out["is_last_"] = ((out.minute_of_day >= final_minute)
                       | (nxt_day.notna() & (nxt_day != day))).to_numpy()
    return out
