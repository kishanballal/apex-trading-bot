"""Active fake-money account: a deterministic replay over stored 5-minute bars and news.

Every cycle rebuilds the whole account from the stored bars and headlines, the same way the
daily paper bot does. That makes it restart-safe, and a crash or a PC restart never loses or
duplicates a trade. Trades only happen on bars the bot saw live (within 10 minutes of the
bar closing), so when the PC is off nothing is traded, as in real life.

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

from apex.intraday.nse import BAR_MS, IST_OFFSET_MS

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
    max_hold_bars: int = 24          # 2 hours
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
    bars: int = 0
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
    """frames: symbol -> nse.build() output with a `first_seen` column (ms, NaN if never)."""
    res = Result(cash=cfg.capital)
    rows = []
    for s, f in frames.items():
        sel = f[f.ts >= start_ms]
        if sel.empty:
            continue
        x = sel.assign(sym_=s, next_open_=f.open.shift(-1).loc[sel.index],
                       next_ts_=f.ts.shift(-1).loc[sel.index],
                       prev_z_=f.z_move.shift().loc[sel.index],
                       prev_vb_=f.vol_burst.shift().loc[sel.index])
        rows.append(x)
    if not rows:
        res.equity_now = cfg.capital
        return res
    p = pd.concat(rows, ignore_index=True).sort_values(["ts", "sym_"], kind="stable")
    open_pos: dict[str, Position] = {}
    pending: dict[str, dict] = {}   # symbol -> entry decided on the previous bar
    last_close: dict[str, float] = {}
    cooldown_until: dict[str, int] = {}
    day_stats: dict[int, dict] = {}
    halted = ""

    def day_of(ts: int) -> int:
        return (ts + IST_OFFSET_MS) // 86_400_000

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

    for ts, bars in p.groupby("ts", sort=True):
        ts = int(ts)
        day = day_of(ts)
        d = day_stats.setdefault(day, {"pnl": 0.0, "stops": 0, "pause_until": 0, "trades": 0})
        for r in bars.itertuples(index=False):
            s = r.sym_
            # 1) fill an entry decided on the previous bar, at this bar's open
            order = pending.pop(s, None)
            if order is not None and order["fill_ts"] == ts and s not in open_pos:
                why_not = ""
                if halted:
                    why_not = "kill switch"
                elif len(open_pos) >= cfg.max_positions:
                    why_not = "4 positions already open"
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
                    price = r.open * (1 + cfg.slippage)
                    qty = spend / (price * (1 + cfg.fee))
                    res.cash -= qty * price * (1 + cfg.fee)
                    open_pos[s] = Position(
                        s, order["rule"], ts, price, qty,
                        stop=r.open - RULE_STOPS[order["rule"]] * order["atr"],
                        target=r.open + cfg.target_atr * order["atr"], news=order["news"])
            # 2) exits, checked on every bar a position is open (stop first, pessimistic)
            pos = open_pos.get(s)
            if pos is not None:
                pos.bars += 1
                first = pos.entry_ts == ts
                if r.low <= pos.stop:
                    close(pos, ts, pos.stop if first else min(r.open, pos.stop), "stop")
                elif r.high >= pos.target:
                    close(pos, ts, pos.target if first else max(r.open, pos.target), "target")
                elif pos.bars >= cfg.max_hold_bars:
                    close(pos, ts, r.close, "time")
            last_close[s] = r.close
            # 3) signals on this bar's close
            if not np.isfinite(r.z_move) or not np.isfinite(r.vol_burst):
                continue
            spike = r.z_move >= cfg.move_z and r.vol_burst >= cfg.move_volume
            drop = r.z_move <= -cfg.move_z and r.vol_burst >= cfg.move_volume
            bounce = (np.isfinite(r.prev_z_) and r.prev_z_ <= -cfg.move_z
                      and r.prev_vb_ >= cfg.move_volume and r.r1 > 0 and r.rsi14 < 40)
            if not (spike or drop or bounce):
                continue
            t_close = ts + BAR_MS
            ctx = news_context(news, s, t_close, cfg)
            kind = "spike" if spike else "drop" if drop else "bounce"
            ev = {"ts": t_close, "symbol": s, "kind": kind, "move_pct": round(r.r1 * 100, 2),
                  "volume_x": round(float(r.vol_burst), 1), "news_count": ctx["count"],
                  "news_tone": ctx["tone"], "headline": ctx["top"], "action": "watch only"}
            res.events.append(ev)
            rule = None
            if spike and r.vwap_dist > 0:
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
            live = np.isfinite(r.first_seen) and r.first_seen <= t_close + cfg.live_grace_ms
            in_window = cfg.first_entry_minute <= r.minute_of_day <= cfg.last_entry_minute
            same_day = np.isfinite(r.next_ts_) and day_of(int(r.next_ts_)) == day
            if not live:
                ev["action"] = "bot was not running"
            elif not in_window:
                ev["action"] = "outside 09:30-14:45 entry window"
            elif s in open_pos:
                ev["action"] = "already holding"
            elif not np.isfinite(r.atr) or r.atr <= 0:
                ev["action"] = "no volatility estimate"
            elif not np.isfinite(r.next_ts_):
                ev["action"] = "buy at next bar"   # filled on the next cycle
                pending[s] = {"fill_ts": int(ts + BAR_MS), "rule": rule, "atr": float(r.atr),
                              "news": ctx["top"], "event": ev}
            elif same_day:
                pending[s] = {"fill_ts": int(r.next_ts_), "rule": rule, "atr": float(r.atr),
                              "news": ctx["top"], "event": ev}
        # session end: close anything open on its stock's last bar of the day
        for r in bars.itertuples(index=False):
            pos = open_pos.get(r.sym_)
            if pos is not None and r.is_last_:
                close(pos, ts, r.close, "session end")
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


def prepare(f: pd.DataFrame, first_seen: pd.Series | None) -> pd.DataFrame:
    """Add the replay's helper columns to an nse.build() frame."""
    out = f.copy()
    out["first_seen"] = (out.ts.map(first_seen) if first_seen is not None
                         else np.nan).astype(float)
    # The session's final bar: the 15:25 candle, or the last bar before a new day starts.
    # (Today's latest bar is not "last" until the session is over.)
    nxt_day = ((out.ts.shift(-1) + IST_OFFSET_MS) // 86_400_000)
    day = (out.ts + IST_OFFSET_MS) // 86_400_000
    out["is_last_"] = ((out.minute_of_day >= 370) | (nxt_day.notna() & (nxt_day != day))
                       ).to_numpy()
    return out
