# APEX

A selective, risk-first trading system. Capital preservation comes first, and
**NO TRADE is the default answer**. It runs in paper mode. Live trading is locked.

Built so far: the core engine, the backtester and the first ML research.

- [Backtest results](docs/backtest-results.md): the two rule-based strategies have no edge worth trading.
- [ML research results](docs/ml-research-results.md): a simple ML filter shows a promising edge on unseen years. Not proven yet.
- [Higher risk, dip-buying and deep learning](docs/higher-risk-and-deep-learning.md): dip-buying with averaging down loses money. The ML filter sized for a 20% drawdown budget beat the index on unseen years. A GRU matches the simple model.

| Part | What it does | Where |
| --- | --- | --- |
| Data pipeline | Downloads daily candles for NSE stocks (Yahoo Finance) and crypto (Binance public data), validates them, stores them in a local SQLite file | `src/apex/data/` |
| Scanner | Scores every symbol on five independent kinds of evidence and proposes a trade only when at least four agree and the market regime is healthy | `src/apex/scanner/`, `src/apex/analysis/` |
| Risk engine | Sizes every trade and can veto it. Nothing else in the system can set a position size | `src/apex/risk/` |
| ML research | Builds a leak-free dataset, compares logistic regression, random forest and XGBoost year by year, and backtests each on years it never saw | `src/apex/ml/` |
| Backtester | Replays history through the same strategy code and the same risk engine, with costs, next-day fills, gaps and a tax estimate | `src/apex/backtest/` |

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
pytest              # run the 85 automated tests
apex fetch          # download and validate market data (about a minute)
apex scan           # show trades the risk engine would approve
apex scan --all     # show every symbol and why it was or was not picked
apex status         # show limits and whether any kill switch is active
apex fetch --days 3300                # long history, needed once before backtesting
apex backtest --strategy trend        # or --strategy pullback
apex backtest --profile moderate      # try looser risk limits (research only)
pip install -e ".[ml]"                 # once, for the ML tools
apex research                         # walk-forward ML test, takes a few minutes
apex ladder                           # dip-buying ladder backtest (50k capital, 40k floor)
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

## Known limits

- Yahoo Finance is a free, unofficial source. It is fine for daily candles in
  paper trading and not good enough for live or intraday trading.
- Two swing strategies only (trend continuation and pullback), both long only.
  Intraday needs better data than the free sources give.
- The rule-based strategies show no edge. The ML filter looks promising on unseen
  years and still has to prove itself in paper trading.
- A GRU deep learning model was tested (`src/apex/ml/sequence.py`, needs `pip install -e ".[dl]"`).
  Transformer and Kronos models have not been tested yet.
- Storage is a local SQLite file. PostgreSQL, Redis and Docker arrive with the
  dashboard stage, when several services need to share data.

## Roadmap

1. Core engine: data, scanner, risk engine. Done.
2. Backtester with costs, taxes and slippage, plus baseline strategies. Done.
3. Paper trading loop, Telegram alerts, React dashboard.
4. ML models (XGBoost, LSTM, Transformer, Kronos), then authorised live trading.
