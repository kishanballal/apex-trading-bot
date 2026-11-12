import numpy as np
import pandas as pd

from apex.active import engine, news
from apex.intraday.nse import BAR_MS

DAY0 = 1_790_000_000_000 - (1_790_000_000_000 + 19_800_000) % 86_400_000 - 19_800_000


def test_score_and_matching():
    assert news.score("Tata Steel shares surge after record profit") > 0
    assert news.score("Adani Ports plunges on SEBI probe") < 0
    assert news.score("Board meeting on Friday") == 0
    syms = ["RELIANCE", "SBIN", "SBILIFE", "LT", "TCS"]
    assert news.match_symbols("Reliance Industries jumps 3%", syms) == ["RELIANCE"]
    assert news.match_symbols("SBI Life shares rise", syms) == ["SBILIFE"]
    assert set(news.match_symbols("SBI and TCS lead gains", syms)) == {"SBIN", "TCS"}
    assert news.match_symbols("salt prices", syms) == []  # no lowercase "lt" match


def test_parse_rss():
    xml = """<rss><channel><item><title>Infosys wins big order</title>
    <pubDate>Wed, 11 Nov 2026 09:30:00 GMT</pubDate><link>https://x/1</link></item>
    <item><title></title></item></channel></rss>"""
    items = news.parse_rss(xml, "test")
    assert len(items) == 1 and items[0].title == "Infosys wins big order"
    assert items[0].published_ms > 0


def _frame(kind: str, n: int = 30, live: bool = True) -> pd.DataFrame:
    """One session of flat 5-minute bars with a single move on bar 10."""
    ts = DAY0 + 13_500_000 + np.arange(n) * BAR_MS  # 09:15 IST onwards
    c = np.full(n, 100.0)
    f = pd.DataFrame({"ts": ts, "open": c, "high": c + 0.05, "low": c - 0.05, "close": c,
                      "z_move": 0.0, "vol_burst": 1.0, "r1": 0.0, "rsi14": 50.0,
                      "vwap_dist": 0.0, "atr": 0.5, "minute_of_day": np.arange(n) * 5})
    if kind == "spike":
        f.loc[10, ["z_move", "vol_burst", "r1", "vwap_dist"]] = [4, 5, 0.01, 0.01]
    else:  # sharp drop then a green bar
        f.loc[10, ["z_move", "vol_burst", "r1"]] = [-4, 5, -0.01]
        f.loc[11, ["r1", "rsi14"]] = [0.002, 30]
    f["first_seen"] = (ts + BAR_MS + 60_000) if live else np.nan
    f["is_last_"] = np.arange(n) == n - 1
    return f


def _news(symbol, tone, t):
    title = "X shares surge on record order" if tone > 0 else "X plunges on fraud probe"
    return pd.DataFrame([{"symbol": symbol, "published_ms": t, "fetched_ms": t,
                          "score": news.score(title), "title": title}])


EMPTY = pd.DataFrame(columns=["symbol", "published_ms", "fetched_ms", "score", "title"])


def test_spike_needs_positive_news():
    f = _frame("spike")
    t = int(f.ts[10])
    assert engine.replay({"A": f}, EMPTY, DAY0).trades == []
    r = engine.replay({"A": f}, _news("A", 1, t - 600_000), DAY0)
    assert len(r.trades) == 1 and r.trades[0]["rule"] == "news_momentum"
    assert r.trades[0]["entry_ts"] == int(f.ts[11])  # next bar's open, never the signal bar


def test_dip_bounce_skips_bad_news_and_needs_live_bars():
    f = _frame("drop")
    t = int(f.ts[11])
    assert len(engine.replay({"A": f}, EMPTY, DAY0).trades) == 1
    bad = engine.replay({"A": f}, _news("A", -1, t - 600_000), DAY0)
    assert bad.trades == [] and any("knife" in e["action"] for e in bad.events)
    off = engine.replay({"A": _frame("drop", live=False)}, EMPTY, DAY0)
    assert off.trades == [] and any("not running" in e["action"] for e in off.events)


def test_news_fetched_after_the_bar_is_not_used():
    f = _frame("spike")
    late = int(f.ts[10]) + BAR_MS + 30 * 60_000
    assert engine.replay({"A": f}, _news("A", 1, late), DAY0).trades == []


def test_everything_closes_by_session_end_and_fees_are_charged():
    f = _frame("drop", n=20)
    r = engine.replay({"A": f}, EMPTY, DAY0, engine.ActiveConfig(max_hold_bars=100))
    assert r.positions == [] and r.trades[0]["reason"] == "session end"
    assert r.trades[0]["pnl"] < 0  # flat prices: fees and slippage only
    assert r.equity_now == r.cash < 50_000


def test_max_positions_and_kill_switch():
    frames = {s: _frame("drop") for s in "ABCDEF"}
    r = engine.replay(frames, EMPTY, DAY0, engine.ActiveConfig(max_hold_bars=100))
    assert len(r.trades) == 4
    assert sum("4 positions" in e["action"] for e in r.events) == 2
    halted = engine.replay(frames, EMPTY, DAY0, engine.ActiveConfig(equity_floor=60_000))
    assert halted.halted and halted.trades == []


def test_search_results_only_tag_stocks_they_name(tmp_path):
    from apex.active.store import ActiveStore
    st = ActiveStore(tmp_path / "a.db")
    items = [news.Item("1", "google-news", "Jensen Huang salary revealed", 0, ""),
             news.Item("2", "google-news", "Eicher Motors shares jump on record sales", 0, "")]
    st.add_news(items, ["EICHERMOT", "TATASTEEL"], 1, force_symbol="EICHERMOT")
    tagged = st.news()
    assert list(tagged.title) == ["Eicher Motors shares jump on record sales"]
