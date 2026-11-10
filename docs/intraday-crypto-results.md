# Intraday step 1: 1-minute crypto research (2026-11-10)

**Result: nothing tested makes money after costs and Indian crypto tax. Do not paper or live
trade any of these yet.**

## Setup

- Data: 180 days of free Binance 1-minute candles (May to Nov 2026) for BTC, ETH, SOL, BNB, XRP
  and DOGE against USDT. About 1.5 million candles. Each candle also carries two free
  order-flow factors: number of trades and taker-buy volume.
- Factors (22, `src/apex/intraday/features.py`): returns over 1/5/15/60/240 minutes,
  short and long volatility, distance from the 60- and 240-minute mean, RSI, position in the
  24-hour range, volume burst, buy pressure (taker-buy share), trade-count burst, ATR, hour of
  day, BTC's own move and each coin's move relative to BTC.
- Backtester (`src/apex/intraday/backtest.py`): signal at a candle's close, fill at the next
  candle's open, long only, stop checked before target inside a candle, gaps fill at the open,
  one position per coin, fixed 8,000 per trade on a 50,000 account. Tax: 30% of every winning
  trade with no loss set-off (Indian VDA rules). The 1% TDS cash drag is not modelled.
- Cost levels per side: zero, 0.02% fee + 0.01% slippage (limit orders), 0.1% + 0.02%
  (Binance taker), 0.2% + 0.03% (typical Indian app).
- ML minute model (`src/apex/intraday/model.py`): logistic regression and gradient boosting
  predict whether the next 30 or 120 minutes beats the round-trip cost. Refit every month on
  earlier months only, with a gap so no label overlaps the test month. Trades only the top 0.3%
  of scores, threshold fixed from training data. Out-of-sample period: the last 4 months.

Run it again with `python -m apex.intraday.research` (writes `data/intraday/results.csv`).

## Results (net profit on the 50,000 account)

| Strategy | Trades/day | Gross edge per trade | Before costs | Binance 0.1% fees, before tax | After tax |
|---|---|---|---|---|---|
| Mean reversion (rules) | 29 | +0.010% | +4,027 | -96,528 | -1,05,053 |
| Breakout + volume + buy pressure | 30 | +0.001% | +515 | -1,02,984 | -1,08,437 |
| Trend pullback | 1.2 | +0.013% | +217 | -3,906 | -4,173 |
| ML logistic, 30 min | 2.6 | +0.19% | +4,754 | -1,164 | -4,987 |
| ML boosting, 30 min | 2.9 | +0.13% | +3,568 | -3,153 | -6,410 |
| ML logistic, 2 h | 1.4 | -0.07% | -906 | -4,087 | -6,547 |
| ML boosting, 2 h | 1.8 | +0.27% | +4,701 | +509 | -2,718 |

The three rule periods cover 180 days; the ML rows cover the 4 out-of-sample months.
With cheap limit orders (0.02%) the best ML rows make +3,273 to +3,652 before tax, but tax on
winners still turns them into -54 to -1,281. At 0.2% (Indian app fees) every row loses.

## What this means

1. Simple 1-minute rules have almost no edge (about 0.01% per trade) while a round trip costs
   0.24% or more. Trading more often only multiplies the fees.
2. The ML model does find something real: AUC 0.59 to 0.70 every month, and its top picks gain
   0.13% to 0.27% per trade before costs. That is about the size of the fees, so it is not
   enough. Much of the AUC comes from predicting volatility (big moves), not direction.
3. India's crypto tax is the killer for frequent trading. Taxing every winning trade at 30%
   with no loss set-off means a strategy needs a large gross edge, not a thin one.
4. Four months of out-of-sample data is short. The one positive pre-tax row (boosting, 2 h,
   profit factor 1.05) is within noise.

## Next options

- NSE intraday instead of crypto: Indian stock intraday costs are roughly 0.06% to 0.1% per
  round trip with a discount broker, and losses can be offset against gains. That is 3 to 4
  times cheaper than crypto here. Free minute data comes from a broker API (for example Angel
  One SmartAPI) or a short Yahoo history (5-minute bars for 60 days).
- Slower crypto (15-minute to 4-hour holds) where the ML edge per trade is larger than fees.
