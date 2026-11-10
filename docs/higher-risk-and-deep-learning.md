# Higher risk, dip-buying and deep learning, 10 November 2026

New brief from the owner: capital INR 50,000, the account must not fall below
INR 40,000 (a 20% drawdown budget), more risk is acceptable, and a dip-buying logic
should be tested along with a deep learning model.

**Verdict.** The dip-buying ladder loses money on its own. The ML filter, sized for
the 20% budget, turned INR 50,000 into about INR 1,12,000 over six unseen years
without going below INR 47,974. The deep learning model matches the simple model and
does not beat it. Returns are lumpy, not continuous.

Reproduce: `apex ladder`, `APEX_CAPITAL=50000 apex research --profile aggressive`.

## 1. Dip-buying ladder

Rules as requested, with hard limits so it cannot become a martingale:

- Watch every stock and coin. A symbol enters the buy zone when it closes at least
  3 ATR below its 20-day high.
- Do not buy while it is still falling. Buy at the next open after the first up-day
  (green candle that closes above the previous close).
- If it then falls another 1.5 ATR and turns up again, buy a little more. At most
  three buys per symbol, sized 1 : 1.25 : 1.5, together at most 15% of the account.
- Sell everything at average cost plus 2 ATR. Hard stop 3 ATR below the lowest buy.
  Give up after 30 days. At most 6 symbols at once.
- If the account reaches INR 42,000, sell everything and stop for good.

Results on INR 50,000, 2017 to 2026 unless noted:

| Variant | Trades | Win rate | Profit factor | Net profit | Max drawdown | Floor reached |
| --- | --- | --- | --- | --- | --- | --- |
| Dips in uptrending stocks only | 144 | 54% | 0.53 | INR -8,043 | 19.8% | Yes, Sept 2020 |
| Any dip | 768 | 62% | 0.96 | INR -3,660 | 28.7% | No (low INR 43,998) |
| Only when market is healthy | 96 | 54% | 0.39 | INR -8,485 | 19.7% | Yes, Sept 2020 |
| No buying more, single buy | 79 | 52% | 0.65 | INR -8,109 | 19.6% | Yes, Aug 2019 |
| Stocks only | 520 | 60% | 0.89 | INR -3,121 | 9.9% | No |
| Crypto only | 27 | 30% | 0.23 | INR -8,153 | 19.3% | Yes, Sept 2020 |
| Any dip, 2021 onward | 430 | 58% | 0.82 | INR -7,703 | 31.4% | No |
| + random forest filter 40%, 2021 on | 230 | 64% | 1.45 | INR +7,816 | 5.7% | No |
| + logistic filter 40%, 2021 on | 202 | 63% | 0.99 | INR -138 | 8.6% | No |

Why it loses: it wins often (55 to 62% of trades) but the average loss is about twice
the average win (for example INR 286 lost against INR 164 won). Small frequent wins,
occasional large losses. That is the usual shape of buying dips and averaging down, and
it is why a high win rate alone means little. One ML filter rescued it and the other
did not, and nearly all of that profit came from 2021, so this is not reliable.

Also, with INR 50,000 a first buy is about INR 2,000, which cannot buy even one share
of 43 otherwise valid signals (stocks such as Maruti or Bajaj Auto).

## 2. The ML filter with a 20% drawdown budget

Same model and rules as in the previous research note, only sized larger
(aggressive profile: 1% risk per trade, 20% per position, kill switch at 20%).
Out of sample 2021 to 2026, capital INR 50,000, real kill-switch rule:

| Model | Trades | Win rate | Profit factor | Final value | Per year | Max drawdown | Lowest value | After tax profit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 334 | 49% | 1.73 | INR 1,11,647 | 14.7% | 13.3% | INR 47,974 | INR 40,808 |
| Random forest | 349 | 47% | 1.51 | INR 95,961 | 11.8% | 16.4% | INR 49,489 | INR 22,172 |
| GRU (deep learning) | 261 | 50% | 1.65 | INR 93,830 | 11.4% | 13.0% | not recorded | INR 24,590 |
| Nifty buy and hold | | | | | 10.4% | 17.2% | | |

By year, logistic regression: 2021 +36.6%, 2022 +4.3%, 2023 0.0%, 2024 +36.2%,
2025 +17.1%, 2026 -1.7%.

This is the first configuration that beats holding the index with less drawdown. It
is not a steady income: two years did almost nothing and 2026 is slightly down.
Position sizing is also touchy at this capital. The same model with the moderate
profile hit its 10% kill switch in August 2024 and stopped.

## 3. Deep learning (GRU)

A GRU network read the last 60 daily bars (9 inputs per bar) for each stock and coin,
tested with the same year-by-year method as the other models.

| | Logistic regression | GRU |
| --- | --- | --- |
| AUC on unseen years (0.50 is a coin flip) | 0.51 to 0.57 | 0.49 to 0.56 |
| Trades, current limits, 45% threshold | 176 | 133 |
| Win rate | 50% | 53% |
| Profit factor | 1.81 | 1.88 |
| Return per year, current limits | 2.3% | 2.0% |
| Return per year, aggressive limits, INR 50,000 | 14.7% | 11.4% |

The GRU is about as good as the logistic model and not better. Two very different
models reaching the same conclusion is mild extra evidence that the effect is real.
Both earn most of their profit in the same years, so they are not independent proof.

## 4. Honest limits

- "Continuous returns" is not what any tested strategy delivers. Profits arrive in
  bursts when the market trends, and the models mostly sit out otherwise.
- Six years of unseen data hold only a few distinct market phases. Survivorship bias
  still flatters all numbers. The INR 40,000 floor can be breached by an overnight gap.
- Nothing has been tuned on the test years so far. Further tweaking on the same data
  would weaken the evidence. Live paper trading is the next real test.
