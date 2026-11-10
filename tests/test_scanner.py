from __future__ import annotations

from datetime import timedelta

import pytest

from apex.analysis.indicators import add_indicators
from apex.analysis.regime import Regime, Trend, detect_regime
from apex.models import AssetClass, Side
from apex.scanner.scanner import ScannerConfig, rank, scan_symbol
from conftest import NOW, make_candles

BULL = Regime(Trend.BULL, False, "test")
INDICATORS = ["ema20", "ema50", "ema200", "rsi14", "atr14", "adx14", "macd_hist",
              "vol_ratio", "prior_high20", "turnover20"]


def test_indicators_have_no_lookahead():
    """Values for a bar must not change when later bars are appended."""
    full = make_candles(320)
    cut = 280
    a = add_indicators(full).head(cut)[INDICATORS]
    b = add_indicators(full.head(cut))[INDICATORS]
    for col in INDICATORS:
        assert a[col].isna().tolist() == b[col].isna().tolist(), col
        assert a[col].dropna().size > 0, col
        assert a[col].dropna().tolist() == pytest.approx(
            b[col].dropna().tolist(), rel=1e-12), col


def test_rsi_extremes_and_bounds():
    rising = make_candles(100, drift=0.01, vol=0.0)
    falling = make_candles(100, drift=-0.01, vol=0.0)
    assert add_indicators(rising)["rsi14"].iloc[-1] == pytest.approx(100.0)
    assert add_indicators(falling)["rsi14"].iloc[-1] == pytest.approx(0.0)
    noisy = add_indicators(make_candles(300))["rsi14"].dropna()
    assert noisy.min() >= 0 and noisy.max() <= 100


def test_atr_of_constant_range_bars():
    n = 60
    frame = make_candles(n).assign(open=100.0, close=100.0, high=102.0, low=98.0)
    assert add_indicators(frame)["atr14"].iloc[-1] == pytest.approx(4.0)


def test_breakout_in_uptrend_becomes_a_candidate(uptrend):
    row = scan_symbol("TEST", AssetClass.EQUITY, uptrend, BULL, now=NOW)
    assert row.is_opportunity, row.verdict
    p = row.proposal
    assert len(row.passed) >= 4 and "trend" in row.passed
    assert p.side is Side.LONG and p.stop < p.entry < p.targets[0] < p.targets[1]
    atr = add_indicators(uptrend)["atr14"].iloc[-1]
    assert p.entry - p.stop == pytest.approx(2 * atr)
    assert p.targets[0] - p.entry == pytest.approx(2 * (p.entry - p.stop))
    assert p.as_of == uptrend["ts"].iloc[-1]


def test_no_trade_cases(uptrend):
    bear = Regime(Trend.BEAR, False, "test")
    choppy = Regime(Trend.BULL, True, "test")
    cases = {
        "regime": scan_symbol("T", AssetClass.EQUITY, uptrend, bear, now=NOW),
        "high volatility": scan_symbol("T", AssetClass.EQUITY, uptrend, choppy, now=NOW),
        "history": scan_symbol("T", AssetClass.EQUITY, uptrend.tail(100), BULL, now=NOW),
        "stale": scan_symbol("T", AssetClass.EQUITY, uptrend, BULL, now=NOW + timedelta(days=30)),
        "liquid": scan_symbol("T", AssetClass.EQUITY, make_candles(volume=1000, breakout=True),
                              BULL, now=NOW),
        "uptrend": scan_symbol("T", AssetClass.EQUITY, make_candles(drift=-0.002), BULL, now=NOW),
    }
    for expected, row in cases.items():
        assert not row.is_opportunity
        assert row.verdict.startswith("NO TRADE") and expected in row.verdict, row.verdict


def test_requires_enough_confirmations(uptrend):
    strict = ScannerConfig(min_confirmations=6)  # impossible: there are five families
    assert not scan_symbol("T", AssetClass.EQUITY, uptrend, BULL, config=strict, now=NOW).is_opportunity


def test_rank_puts_opportunities_first(uptrend):
    good = scan_symbol("GOOD", AssetClass.EQUITY, uptrend, BULL, now=NOW)
    bad = scan_symbol("BAD", AssetClass.EQUITY, make_candles(drift=-0.002), BULL, now=NOW)
    assert [r.symbol for r in rank([bad, good])] == ["GOOD", "BAD"]


def test_regime_detection():
    assert detect_regime(make_candles(drift=0.002, vol=0.005)).trend is Trend.BULL
    assert detect_regime(make_candles(drift=-0.002, vol=0.005)).trend is Trend.BEAR
    assert detect_regime(make_candles(50)).trend is Trend.UNKNOWN
    assert not detect_regime(make_candles(50)).allows_longs
