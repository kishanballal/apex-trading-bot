"""Free news and social feeds, matched to stocks and scored for tone.

Sources (no keys, no accounts):
- Google News RSS search per stock, fetched when that stock makes a sudden move
- Economic Times markets and Moneycontrol RSS, polled for general market news
- (Reddit is off: it blocks clients without an API key)

Scoring uses a small built-in finance word list (in the spirit of the Loughran-McDonald
lexicon that many open-source bots use), so it needs no model download. Headlines are data:
they are only matched and counted, never executed or followed.
"""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx

UA = "Mozilla/5.0 (compatible; apex-research/0.1)"

MARKET_FEEDS = {
    "economictimes": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
}
# Reddit answers "403 Blocked" to unauthenticated clients (seen on the first live day), so it
# is off until a free Reddit API app is registered. parse_reddit stays for that.
REDDIT_FEEDS: dict[str, str] = {}

# Names people actually use in headlines. The ticker itself is matched too.
ALIASES: dict[str, list[str]] = {
    "RELIANCE": ["Reliance Industries", "RIL", "Reliance Jio", "Reliance Retail"],
    "TCS": ["Tata Consultancy"], "HDFCBANK": ["HDFC Bank"], "ICICIBANK": ["ICICI Bank"],
    "INFY": ["Infosys"], "BHARTIARTL": ["Bharti Airtel", "Airtel"], "SBIN": ["SBI", "State Bank"],
    "ITC": ["ITC Ltd", "ITC Hotels"], "LT": ["Larsen", "L&T"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"], "KOTAKBANK": ["Kotak Mahindra", "Kotak Bank"],
    "AXISBANK": ["Axis Bank"], "BAJFINANCE": ["Bajaj Finance"], "MARUTI": ["Maruti Suzuki"],
    "SUNPHARMA": ["Sun Pharma"], "HCLTECH": ["HCL Tech", "HCLTech"], "TITAN": ["Titan Company"],
    "ASIANPAINT": ["Asian Paints"], "ULTRACEMCO": ["UltraTech"], "NTPC": ["NTPC"],
    "POWERGRID": ["Power Grid"], "ONGC": ["ONGC"], "M&M": ["Mahindra & Mahindra", "M&M"],
    "WIPRO": ["Wipro"], "ADANIPORTS": ["Adani Ports"], "ADANIENT": ["Adani Enterprises"],
    "JSWSTEEL": ["JSW Steel"], "TATASTEEL": ["Tata Steel"], "COALINDIA": ["Coal India"],
    "NESTLEIND": ["Nestle India"], "BAJAJFINSV": ["Bajaj Finserv"],
    "BAJAJ-AUTO": ["Bajaj Auto"], "TECHM": ["Tech Mahindra"], "GRASIM": ["Grasim"],
    "HINDALCO": ["Hindalco"], "DRREDDY": ["Dr Reddy", "Dr. Reddy"], "CIPLA": ["Cipla"],
    "EICHERMOT": ["Eicher Motors", "Royal Enfield"], "HEROMOTOCO": ["Hero MotoCorp"],
    "BRITANNIA": ["Britannia"], "APOLLOHOSP": ["Apollo Hospitals"], "DIVISLAB": ["Divi's Lab",
                                                                                 "Divis Lab"],
    "TATACONSUM": ["Tata Consumer"], "SBILIFE": ["SBI Life"], "HDFCLIFE": ["HDFC Life"],
    "INDUSINDBK": ["IndusInd"], "BPCL": ["BPCL", "Bharat Petroleum"],
    "SHRIRAMFIN": ["Shriram Finance"], "TRENT": ["Trent", "Zudio"],
    "BEL": ["Bharat Electronics"],
}

POSITIVE = """beat beats beating surge surges surged soar soars soared jump jumps jumped rally
rallies rallied gain gains gained rise rises rose upgrade upgraded upgrades outperform record
high highs strong stronger robust boost boosts boosted win wins won bags bagged order orders
contract approval approved approves buyback dividend bonus profit profits growth grows expands
expansion raises raised hike target upside bullish breakout acquire acquires acquisition
partnership launch launches milestone recovery rebound rebounds""".split()
NEGATIVE = """miss misses missed fall falls fell plunge plunges plunged slump slumps slumped drop
drops dropped crash crashes crashed tumble tumbles tumbled sink sinks sank decline declines
declined downgrade downgraded downgrades underperform loss losses weak weaker slowdown cut cuts
lower lowers probe raid fraud penalty fine fined ban banned recall resigns resignation default
defaults debt warning warns bearish selloff sell-off concern concerns lawsuit notice scam
investigation pledge pledged halt halted delay delayed""".split()
_POS, _NEG = set(POSITIVE), set(NEGATIVE)
_WORD = re.compile(r"[a-z][a-z\-']+")


def load_aliases(path) -> list[str]:
    """Merge extra stocks and their names from config/active_universe.toml, if present."""
    import tomllib
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    for sym, names in (cfg.get("aliases") or {}).items():
        ALIASES.setdefault(sym, [])
        ALIASES[sym] = list(dict.fromkeys([*ALIASES[sym], *names]))
    return list((cfg.get("active") or {}).get("extra_symbols") or [])


class Scorer:
    """Headline tone in [-1, 1]. Uses FinBERT (a free model trained on financial news) when
    the `transformers` package is installed, on the GPU if there is one; otherwise the word
    list below. `name` says which one is active."""

    def __init__(self, use_finbert: bool = True) -> None:
        self.name = "word list"
        self._clf = None
        if use_finbert:
            try:
                import torch
                from transformers import pipeline
                self._clf = pipeline("text-classification", model="ProsusAI/finbert",
                                     top_k=None, device=0 if torch.cuda.is_available() else -1)
                self.name = "FinBERT" + (" (GPU)" if torch.cuda.is_available() else " (CPU)")
            except Exception as exc:  # missing package, blocked DLL, no download: fall back
                self.error = f"{type(exc).__name__}: {str(exc)[:120]}"
                self._clf = None

    def __call__(self, titles: list[str]) -> list[float]:
        if not titles:
            return []
        if self._clf is not None:
            try:
                out = []
                for res in self._clf(titles, truncation=True, batch_size=32):
                    d = {r["label"].lower(): r["score"] for r in res}
                    out.append(round(d.get("positive", 0.0) - d.get("negative", 0.0), 3))
                return out
            except Exception:
                pass
        return [score(t) for t in titles]


def score(text: str) -> float:
    """Tone from -1 (all negative words) to +1 (all positive). 0 when no finance words."""
    words = _WORD.findall(text.lower())
    p = sum(w in _POS for w in words)
    n = sum(w in _NEG for w in words)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def match_symbols(text: str, symbols: list[str]) -> list[str]:
    """Stocks named in `text`. Longer names are matched first and blanked out, so
    "SBI Life" counts for SBILIFE and not also for SBI (SBIN)."""
    names = sorted(((n, s) for s in symbols for n in [s, *ALIASES.get(s, [])]),
                   key=lambda x: -len(x[0]))
    found: list[str] = []
    for name, sym in names:
        # Whole-word match. Tickers and one-word names ("SBI", "Eternal", "Trent") must match
        # with their capitals, so ordinary words in a headline are not mistaken for a stock.
        one_word = " " not in name.strip()
        pat = re.compile(r"(?<![A-Za-z])" + re.escape(name.strip()) + r"(?![A-Za-z])",
                         0 if one_word or name.isupper() else re.IGNORECASE)
        if pat.search(text):
            if sym not in found:
                found.append(sym)
            text = pat.sub(" ", text)
    return found


@dataclass(frozen=True)
class Item:
    uid: str
    source: str
    title: str
    published_ms: int
    url: str


def _uid(source: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{title.strip().lower()}".encode()).hexdigest()[:16]


def parse_rss(xml_text: str, source: str) -> list[Item]:
    out = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        if not title:
            continue
        pub = it.findtext("pubDate")
        try:
            ms = int(parsedate_to_datetime(pub).timestamp() * 1000) if pub else 0
        except (TypeError, ValueError):
            ms = 0
        out.append(Item(_uid(source, title), source, title, ms, it.findtext("link") or ""))
    return out


def parse_reddit(payload: dict, source: str) -> list[Item]:
    out = []
    for child in (payload.get("data") or {}).get("children") or []:
        d = child.get("data") or {}
        title = (d.get("title") or "").strip()
        if title:
            out.append(Item(_uid(source, title), source, title,
                            int(float(d.get("created_utc") or 0) * 1000),
                            "https://www.reddit.com" + (d.get("permalink") or "")))
    return out


def google_news_url(symbol: str) -> str:
    name = (ALIASES.get(symbol) or [symbol])[0]
    return ("https://news.google.com/rss/search?q=" + quote_plus(f'"{name}" share')
            + "+when:1d&hl=en-IN&gl=IN&ceid=IN:en")


class NewsClient:
    """Fetches feeds. Every failure is swallowed and reported: news is optional input."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.http = httpx.Client(timeout=timeout, headers={"User-Agent": UA},
                                 follow_redirects=True)
        self.errors: dict[str, str] = {}

    def close(self) -> None:
        self.http.close()

    def _get(self, key: str, url: str) -> httpx.Response | None:
        try:
            r = self.http.get(url)
            r.raise_for_status()
            self.errors.pop(key, None)
            return r
        except httpx.HTTPError as exc:
            self.errors[key] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return None

    def market(self) -> list[Item]:
        items = []
        for src, url in MARKET_FEEDS.items():
            r = self._get(src, url)
            if r is not None:
                try:
                    items += parse_rss(r.text, src)
                except ET.ParseError:
                    self.errors[src] = "bad RSS"
        for src, url in REDDIT_FEEDS.items():
            r = self._get(src, url)
            if r is not None:
                try:
                    items += parse_reddit(r.json(), src)
                except ValueError:
                    self.errors[src] = "bad JSON"
        return items

    def for_symbol(self, symbol: str) -> list[Item]:
        r = self._get("google-news", google_news_url(symbol))
        if r is None:
            return []
        try:
            return parse_rss(r.text, "google-news")
        except ET.ParseError:
            self.errors["google-news"] = "bad RSS"
            return []


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)
