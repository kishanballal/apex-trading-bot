# APEX

A selective, risk-first trading system. Capital preservation comes first, and
**NO TRADE is the default answer**. It runs in paper mode. Live trading is locked.

This is stage 1 of 4: the core engine.

| Part | What it does | Where |
| --- | --- | --- |
| Data pipeline | Downloads daily candles for NSE stocks (Yahoo Finance) and crypto (Binance public data), validates them, stores them in a local SQLite file | `src/apex/data/` |
| Scanner | Scores every symbol on five independent kinds of evidence and proposes a trade only when at least four agree and the market regime is healthy | `src/apex/scanner/`, `src/apex/analysis/` |
| Risk engine | Sizes every trade and can veto it. Nothing else in the system can set a position size | `src/apex/risk/` |

## Setup (Windows)

1. Install Python 3.13 from python.org and tick "Add Python to PATH".
2. Open PowerShell in this folder and run:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

On macOS or Linux use `python3.13 -m venv .venv` and `source .venv/bin/activate`.

## Use

```powershell
pytest              # run the 51 automated tests
apex fetch          # download and validate market data (about a minute)
apex scan           # show trades the risk engine would approve
apex scan --all     # show every symbol and why it was or was not picked
apex status         # show limits and whether any kill switch is active
```

No API keys or accounts are needed at this stage. `apex scan` places no trades.
It prints a shortlist and saves it to `data/last_scan.json`.

## How a trade gets proposed

The scanner looks for swing trades (days to weeks) on daily candles, long only.

Hard gates, any failure means NO TRADE:
- At least 220 days of history, and data no older than 5 days.
- Liquid: average daily turnover above INR 5 crore (stocks) or USDT 5 million (crypto).
- The benchmark (Nifty 50 or Bitcoin) is in an uptrend and not in a volatility spike.
- The symbol itself is in a confirmed uptrend.

Then at least four of these five must agree:

| Evidence | Passes when |
| --- | --- |
| Trend | Close above EMA50 above EMA200, EMA200 rising, ADX at least 20 |
| Momentum | RSI between 50 and 70 and MACD histogram positive |
| Volume | Volume at least 1.2 times its 20-day average on an up day |
| Structure | Close above the prior 20-day high, or a bounce off the EMA20 |
| Volatility | ATR between 0.5% and 6% of price and not in its top 20% for the year |

Entry is the last close, the stop is 2 ATR below, targets are 2R and 3R.
Only finished candles are used. A test proves indicators never see future data.

## Risk rules (enforced in code, covered by tests)

| Rule | Value |
| --- | --- |
| Capital (simulated) | INR 1,00,000 |
| Risk per trade, including costs | 0.25% |
| Daily loss limit | 0.75%, no new trades for the rest of the day |
| Weekly drawdown limit | 2%, no new trades for the rest of the week |
| Maximum portfolio drawdown | 5%, kill switch latches until a human resets it |
| Open positions | 4 at most |
| One position | 5% of equity at most |
| Total deployed | 20% of equity at most |
| Leverage | 1x, cannot be configured higher |
| Stop-loss | Mandatory, between 0.2% and 15% from entry |
| Reward to risk after costs | At least 1.5 |
| Correlation with open positions | At most 0.80. Missing data counts as a rejection |
| Liquidity | At most 1% of average daily turnover |
| Adding to a position | Never allowed, so no averaging down or martingale |
| After a loss | 24 hour cooldown on that symbol. Three losses in a row pauses the day |

Costs assumed per side: 0.15% fees plus 0.05% slippage for stocks, 0.10% plus
0.05% for crypto. These are conservative estimates, not broker quotes.

## Settings

Copy `.env.example` to `.env` to change capital, the USD to INR rate or the data
folder. Edit `config/universe.toml` to change the watchlist.

## Known limits of stage 1

- Yahoo Finance is a free, unofficial source. It is fine for daily candles in
  paper trading and not good enough for live or intraday trading.
- One strategy only (swing). Intraday, long-term and ML models come later, and
  each must beat this baseline in backtests to be kept.
- No backtester yet, so there is **no evidence yet that these rules make money**.
  That is stage 2.
- Storage is a local SQLite file. PostgreSQL, Redis and Docker arrive with the
  dashboard stage, when several services need to share data.

## Roadmap

1. Core engine: data, scanner, risk engine (this stage).
2. Backtester with costs, taxes and slippage, plus baseline strategies.
3. Paper trading loop, Telegram alerts, React dashboard.
4. ML models (XGBoost, LSTM, Transformer, Kronos), then authorised live trading.
