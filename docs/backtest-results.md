# Backtest results, 10 November 2026

**Verdict: neither baseline strategy has an edge worth trading.** Both are slightly
profitable before tax, roughly break-even after tax, and far behind simply holding
the Nifty. The risk engine did its job: the worst drawdown stayed near 5%.

## Setup

- Period: October 2017 to November 2026 (9 years). Trading starts in 2019 because
  indicators need 220 days of history first.
- Universe: 50 large NSE stocks and 10 crypto pairs, daily candles.
- Capital INR 1,00,000 with the full risk rules (0.25% risk per trade, 4 positions,
  5% per position, 20% deployed, kill switches).
- Signals are computed on a finished day and filled at the next day's open.
- Costs per side: 0.20% for stocks and 0.15% for crypto, slippage included.
- Tax estimate: 20% on yearly net share gains, 30% on every winning crypto trade.

Reproduce with `apex fetch --days 3300` then `apex backtest --strategy trend` or
`--strategy pullback`.

## Headline numbers

| | Trend | Pullback | Hold Nifty | Hold Bitcoin |
| --- | --- | --- | --- | --- |
| Trades | 240 | 430 | | |
| Win rate | 40% | 39% | | |
| Profit factor | 1.16 | 1.14 | | |
| Expectancy per trade | +0.08 R | +0.07 R | | |
| Net profit after costs | INR 4,985 | INR 7,192 | | |
| Net profit after tax | INR 1,522 | INR 449 | | |
| Return per year | 0.54% | 0.77% | 10.3% | 31.9% |
| Maximum drawdown | 5.3% | 4.9% | 38.4% | 83.2% |
| Sharpe ratio | 0.32 | 0.38 | | |
| Average capital deployed | 5.9% | 8.3% | 100% | 100% |
| Costs paid | INR 3,739 | INR 7,048 | | |

The trend strategy hit the 5% drawdown kill switch on 25 October 2023 and, by the
real rule, never traded again. With a simulated human reset after 90 days it trades
on to 2026 and does worse: 376 trades, profit factor 1.07, INR 3,676 net and a
loss after tax.

## By year (return on the whole account)

| Year | Trend, with 90-day reset | Pullback |
| --- | --- | --- |
| 2019 | +0.83% | +1.42% |
| 2020 | +5.22% | +0.67% |
| 2021 | +2.43% | +4.94% |
| 2022 | -0.36% | -1.25% |
| 2023 | -2.86% | +1.87% |
| 2024 | +0.66% | +1.76% |
| 2025 | -0.06% | +0.95% |
| 2026 | -1.84% | -3.19% |

Most of the profit came from 2020 and 2021. Since 2022 the trend strategy has lost
money every year but one.

## What the experiments show

| Variant | Trades | Profit factor | Net profit | After tax |
| --- | --- | --- | --- | --- |
| Trend, regime gate off | 456 | 1.03 | INR 1,924 | INR -3,195 |
| Trend, stocks only | 338 | 1.13 | INR 5,031 | INR 3,505 |
| Trend, crypto only | 171 | 1.31 | INR 8,195 | INR -2,351 |
| Pullback, regime gate off | 517 | 1.10 | INR 5,929 | INR 594 |
| Pullback, stocks only | 342 | 1.09 | INR 3,221 | INR 1,321 |
| Pullback, crypto only | 133 | 0.90 | INR -2,111 | INR -7,908 |

(All with the 90-day reset so the runs cover the same years.)

1. **The regime gate helps.** Turning it off adds trades and lowers the profit factor.
2. **Costs eat about half the gross edge.** The pullback strategy paid INR 7,048 in
   costs to keep INR 7,192.
3. **Crypto tax is punishing.** Trend-following crypto had the best raw edge (1.31)
   and still loses after the 30% tax on winners with no relief for losers.
4. **The risk limits cap the upside as well as the downside.** With about 6% to 8% of
   capital deployed on average, even a good strategy could only move the account a
   few percent a year. The idle 80% or more would sit in cash, which this backtest
   counts as earning nothing.

## Caveats

- **Survivorship bias.** The watchlist is today's large companies. Stocks that fell
  out of the index are missing, so real results would likely be worse.
- The rules were chosen once, before testing, and not tuned to this data. That makes
  the result honest and is also why it is unimpressive.
- USD to INR is fixed at 88. Daily bars cannot show the order of intraday moves, so
  a bar touching both stop and target is counted as a stop.
- The tax model is an estimate, not tax advice.
