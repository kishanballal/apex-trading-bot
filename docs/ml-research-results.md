# Looser limits and ML research, 10 November 2026

**Verdict: a simple ML filter is the first thing that shows a real edge on unseen
data. It is promising, not proven.** Loosening the risk limits alone does not fix
weak strategies. It only makes their wins and losses bigger.

Reproduce with `pip install -e ".[ml]"`, `apex fetch --days 3300`, then
`apex research` and `apex research --profile moderate`.

## 1. Looser risk limits on the old strategies (2017 to 2026)

| Profile | Risk per trade | Per position | Deployed | Kill switch |
| --- | --- | --- | --- | --- |
| current | 0.25% | 5% | 20% | 5% drawdown |
| moderate | 0.5% | 10% | 50% | 10% drawdown |
| aggressive | 1% | 20% | 100% | 20% drawdown |

| Strategy and profile | Trades | Profit factor | Per year | Max drawdown | After tax |
| --- | --- | --- | --- | --- | --- |
| Trend, current | 240 | 1.16 | 0.54% | 5.3% | INR 1,522 |
| Trend, moderate | 556 | 1.24 | 3.93% | 10.1% | INR 16,914 |
| Trend, aggressive | 358 | 1.15 | 2.81% | 20.1% | INR 12,461 |
| Pullback, current | 430 | 1.14 | 0.77% | 4.9% | INR 449 |
| Pullback, moderate | 91 | 0.91 | -0.24% | 10.5% | INR -3,483 |
| Pullback, aggressive | 105 | 0.90 | -0.60% | 21.6% | INR -7,908 |

(Real kill-switch rule: once tripped, trading stops for good. Pullback tripped it
in February 2020 under both looser profiles.)

Looser limits multiply whatever edge exists. The edge of these strategies is thin,
so the best case is about 4% a year with a 10% drawdown, still well behind the
Nifty's 10% a year. Bigger limits also hit the kill switch sooner.

## 2. ML models

**What the model predicts.** For every stock and coin on every day: if we bought at
tomorrow's open with a 2 ATR stop and a 2R target, would the target be hit first
within 40 days? 138,706 samples, 32 features (momentum, trend, volatility, volume,
market state, relative strength).

**How it was tested.** Walk-forward by year from 2021 to 2026. The model for a year
is trained only on trades whose outcome was already known before 1 January of that
year. Probabilities are calibrated on a held-out slice. A test confirms that features
do not change when future data is deleted.

**Prediction quality is weak, as expected for markets.** AUC on unseen years runs
from 0.49 to 0.60, where 0.50 is a coin flip. The models are slightly better than
chance, not crystal balls.

**Trading only the most confident predictions is what works.** Backtest on the
unseen years 2021 to 2026, trading only when predicted win probability is at least
45%, with the market regime gate on:

| Strategy | Trades | Win rate | Profit factor | Avg per trade | Per year | Max drawdown | After tax |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trend, no ML | 316 | 38% | 1.04 | 0.00 R | 0.28% | 4.3% | INR -2,673 |
| Pullback, no ML | 327 | 39% | 1.11 | +0.05 R | 0.67% | 4.8% | INR -906 |
| Buy every signal bar, no ML | 327 | 39% | 1.03 | +0.04 R | 0.22% | 8.8% | INR -4,741 |
| Momentum top 10%, no ML | 304 | 41% | 1.17 | +0.11 R | 1.17% | 6.2% | INR 1,051 |
| **Logistic regression** | 176 | 50% | **1.81** | +0.35 R | 2.33% | 3.8% | INR 9,785 |
| Random forest | 190 | 49% | 1.60 | +0.28 R | 2.04% | 3.2% | INR 6,491 |
| XGBoost | 192 | 41% | 1.20 | +0.11 R | 0.77% | 3.7% | INR 212 |

With the moderate limits over the same years:

| Strategy | Trades | Profit factor | Per year | Max drawdown | After tax |
| --- | --- | --- | --- | --- | --- |
| Trend, no ML | 422 | 1.05 | 0.91% | 10.5% | INR -5,121 |
| Logistic regression | 255 | 1.70 | 6.27% | 7.5% | INR 28,572 |
| Random forest | 291 | 1.55 | 5.90% | 8.2% | INR 20,320 |
| XGBoost | 250 | 1.35 | 3.36% | 8.4% | INR 8,629 |

For scale, holding the Nifty over 2021 to 2026 returned about 10% a year with a
drawdown near 17%.

The thresholds 40%, 45% and 50% were all profitable for logistic regression (profit
factor 1.47, 1.81, 2.49) and random forest (1.51, 1.60, 1.74), so the result does not
hinge on one lucky setting.

## 3. Checks against fooling ourselves

| Check | Result |
| --- | --- |
| Future leak: features recomputed with data after June 2023 deleted | Identical |
| Placebo: same pipeline trained on shuffled outcomes | Profit factor 1.01, no edge |
| Luck: resampling whole months of trades 4,000 times (logistic) | Average per trade +0.15 R to +0.55 R (90% range), below zero in 0.2% of resamples |
| Same test for random forest | +0.06 R to +0.51 R, below zero in 1.8% |
| Same test for XGBoost | -0.08 R to +0.30 R, below zero in 19%: not reliable |

## 4. Reasons for caution

- **The profit is concentrated.** Logistic regression made its money in 2021, 2024
  and 2025. It took 4 trades in 2022, none in 2023 and 5 in 2026. The model mostly
  works as a market-timing switch, and six years contain only a handful of
  independent market phases.
- **The simplest model won.** XGBoost, the most flexible model, was the worst. That
  suggests there is little complex pattern to find, and that deep learning (LSTM,
  Transformer, Kronos) is unlikely to help much. They are still worth one fair test.
- **Survivorship bias** still applies: the watchlist is today's large companies.
- Even the best case is a few percent a year under the current limits, because only
  about 7% of capital is ever deployed.
- About 176 trades is a small sample. Paper trading on live data is the real test.

## 5. Where this leaves the plan

The logistic-regression filter clears the bar set in stage 2 (beat profit factor
1.15 and +0.07 R on unseen data). It is the strategy worth paper trading.
