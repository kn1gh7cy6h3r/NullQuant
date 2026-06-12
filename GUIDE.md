# Understanding NullQuant — a plain-English guide

This guide assumes **zero finance background**. By the end you'll understand what
this project is, what every part does, what happens when you run it, and how to
read the results. No jargon goes unexplained.

---

## 1. The one-paragraph version

NullQuant is a **research project that tests a trading idea honestly**. The idea:
"buy crypto coins that are going up, sell short the ones going down, and you'll
make money." We built a careful simulator to check whether that idea actually
works once you account for real-world frictions (trading fees) and once you stop
fooling yourself (proper testing). **The honest answer it produced is: no — this
particular idea does not beat simply buying and holding Bitcoin.** The point of
the project is not the idea; it's the *machinery and discipline* used to test it,
which is exactly what professional quant firms care about.

---

## 2. The big idea, by analogy

Imagine you think you've found a system to win at a casino. Before betting real
money, a smart person would:

1. **Replay history** — "if I'd used this system every day for the last 6 years,
   would I have made money?" (This is a **backtest**.)
2. **Subtract the costs** — the casino takes a cut on every bet. Does the system
   still win *after* the house's cut? (These are **transaction costs**.)
3. **Avoid fooling themselves** — it's easy to invent a system that "would have
   won" on past data by accident. The real test is whether it works on data you
   *didn't* use to build it. (This is **out-of-sample testing**.)
4. **Compare to doing nothing clever** — if just leaving your money in the bank
   beats your fancy system, the system is worthless. (This is the **benchmark**.)

NullQuant does all four, rigorously, for a crypto trading idea. Most amateur
trading projects skip steps 2–4 and "discover" edges that don't exist.

---

## 3. Every concept, explained

### Prices and returns
A **price** is what one coin costs today (e.g. Bitcoin = \$61,000). A **return**
is the *percentage change* in price over a day. Returns matter more than prices
because they're comparable across coins of different prices.

### Moving average (SMA) and "crossover"
A **moving average** is just the average price over the last N days — it smooths
out daily noise so you can see the trend. We use:
- **SMA50** = average of the last 50 days (fast to react)
- **SMA200** = average of the last 200 days (slow, the long-term trend)

When the fast line crosses *above* the slow line, it's a **"golden cross"** —
often read as "the trend just turned up, consider buying." The opposite (fast
crosses below slow) is a **"death cross"** — "trend turned down." These crossovers
are our **entry signals**: the moments the strategy decides to act.

### Long, short, and "long/short"
- Going **long** = buying something, betting it goes **up**.
- Going **short** = borrowing and selling something, betting it goes **down** (you
  profit if the price falls).
- A **long/short** strategy does both at once: long the strong coins, short the
  weak ones. The hope is to make money whether the overall market rises or falls.

### Cross-sectional ranking
"Cross-sectional" just means **comparing the coins to each other at the same
moment**. Each week we rank the 8 coins from strongest to weakest trend, go long
the top few, short the bottom few. (Contrast with "time-series," which compares a
coin to *its own* past.)

### Volatility and "vol targeting"
**Volatility** = how wildly a price swings (risk). A calm coin and a wild coin are
not equally risky to hold the same dollar amount of. **Vol targeting** means we
size positions so the *whole portfolio* aims for a steady, chosen level of risk
(here, 20% annualized) — putting less money in wild coins, more in calm ones, and
scaling the whole book up or down so risk stays roughly constant.

### Position sizing and leverage
**Sizing** = deciding how much to put into each bet. **Leverage** = betting more
than you actually have (borrowed). We cap leverage at 2× and any single coin at
40% of the book, so one coin can't blow everything up.

### Transaction costs (fees, spread, slippage)
Every time you trade, you lose a little:
- **Fee** — the exchange's commission (we assume 0.10% per trade).
- **Spread** — the gap between the buy price and sell price.
- **Slippage** — big orders move the price against you.
We charge all three on every trade. A backtest **without** costs is fantasy; ours
includes them and even **stress-tests** them at 2× and 4×.

### Backtest
A **backtest** replays the strategy day-by-day over history to produce a
hypothetical track record. The cardinal rule: at any day, you may only use
information that was *actually known by then* — no peeking at the future.

### Look-ahead bias and "repainting"
**Look-ahead bias** = accidentally using future information (e.g. today's final
price before the day is over). **Repainting** = when a chart silently rewrites the
past as new data arrives, making a strategy look better than it was. NullQuant
forbids both: all calculations use only **completed** daily bars, and the live
price is shown for display only — it never edits history.

### Benchmark / "buy-and-hold" / "HODL"
A **benchmark** is the do-nothing-clever comparison. Here it's **buy-and-hold**
(a.k.a. **HODL** in crypto slang): just buy and never trade. If our clever
strategy can't beat buy-and-hold, the cleverness isn't worth it.

### Sharpe ratio
The **Sharpe ratio** measures **return per unit of risk** — how much profit you
got for how much stomach-churning you endured. Higher is better. ~1.0 is decent;
near 0 means "no better than noise." It lets you compare a calm 8%/year strategy
to a wild 80%/year one fairly.

### Drawdown
A **drawdown** is how far you fell from your peak — the worst losing streak.
"−60% max drawdown" means at some point you'd have lost 60% of your money from a
high. Big drawdowns are what make people panic-sell.

### Overfitting — the central villain
**Overfitting** = building a system that fits the past *too* perfectly, capturing
random flukes instead of a real pattern — like memorizing the answers to last
year's exam instead of learning the subject. It always looks brilliant on old
data and fails on new data. Almost every fake trading "edge" is overfitting. The
next few tools exist specifically to catch it.

### Out-of-sample testing, train/test, walk-forward
The fix for overfitting: **only trust results on data the system never saw while
being built.**
- **Train/test split** — build on older data, grade on newer data.
- **Walk-forward** — train on a few years, test on the next 6 months, roll
  forward, repeat. This mimics how you'd really deploy it over time.

### Purged cross-validation (for the ML)
A subtle finance trap: a single "bet" plays out over many days, so naively
splitting data lets a bet's outcome **leak** between the training and testing
sets. **Purged cross-validation with an embargo** deletes any training example
whose outcome window overlaps the test period (and adds a buffer). It's the
finance-correct way to grade a machine-learning model.

### Deflated Sharpe ratio
If you try **100 strategies** and pick the best, it'll look great *by luck alone*.
The **Deflated Sharpe ratio** mathematically discounts your headline number by how
many things you tried, telling you the probability the result is *real* rather
than the luckiest of many guesses. It's the project's built-in lie detector.

### The four machine-learning models (what they actually do)
Instead of trying to *predict price* (nearly impossible), these four let ML do
what it's better at — learning **order, regimes, relationships, and certainty**:
- **Learning-to-Rank** — doesn't predict how much each coin moves, just the
  **order** of which will do best vs worst this week; we then buy the top and
  short the bottom. (A much easier question than "what's the price tomorrow?")
- **Regime-switching (HMM)** — discovers **hidden market "moods"** (e.g. calm
  trending vs choppy) and **switches which strategy runs** in each mood.
- **Lead–lag network** — learns whether some coins **move first and others
  follow**, then trades the followers after a leader moves. It also blends in a
  **structural funding-rate tilt** (see below) so the book isn't pure price-action.
- **Conformal gate** — the clever one. It doesn't predict direction; it produces
  a **statistically honest confidence level** and tells the book **how much to
  bet**: size up toward the full risk target when the model is confident (its
  prediction interval is tight), scale down continuously when it isn't. Its
  confidence claims are *calibrated* — when it says "90% sure," it's right ~90% of
  the time (we check this: it scored **90.7%**).

**A new structural input — funding rates.** Perpetual-swap "funding" is a small
fee the crowded side of a leveraged bet pays the other side. Persistently **high
positive** funding means everyone is piling into longs — a classic setup for a
sharp reversal — so we read it as a **bearish** cross-sectional tilt; **negative**
funding is a tailwind. We pull it from Binance (cached locally; the system runs
fine without it) and feed it to the ranking models as an *exogenous* signal that
price charts alone don't contain.

### Ablation (the "with vs without" test)
The decisive experiment: run the strategy **with** each model and **without** it,
on the same data, and see if it **actually improves the money made** out-of-sample.
A model that doesn't help gets **shelved**, no matter how fancy. The honest result
(on the last 6 years, 2020–2026): the three "predicting" models (rank, regime,
lead–lag) all got **shelved** — they made things worse. The **conformal confidence
gate genuinely helped**, but modestly: it lifts the baseline Sharpe **0.10 → 0.12**
with lower drawdown and risk, while now staying **~94% invested** (a continuous
sizer) rather than parking in cash. Crucially, an earlier *binary* version of the
gate looked far more impressive (Sharpe 0.05 → 0.33) — but most of that came from
**sitting in cash through bad stretches**, i.e. market-timing, not smarter sizing.
Forced to stay invested, the honest lift is real but small. (And even so, none beat
just holding Bitcoin — beta is hard.)

---

## 4. What happens when you run it (step by step)

When you run the pipeline, NullQuant:

1. **Downloads daily prices** for 8 big cryptocurrencies (Bitcoin, Ethereum, etc.)
   and caches them locally.
2. **Computes indicators** (moving averages, volatility, momentum) using only
   past data.
3. **Generates signals** — each week, ranks the coins and picks which to go long
   and which to short.
4. **Sizes the positions** to hit the 20% risk target, applying leverage/position
   caps.
5. **Simulates trading** day-by-day, subtracting realistic costs, producing an
   equity curve (your hypothetical account value over time).
6. **Runs the three ML models** and, via ablation, checks whether each one
   improves results out-of-sample.
7. **Stress-tests costs** (0× to 4×) to see how fragile any edge is.
8. **Scores everything** with Sharpe, drawdown, and the Deflated Sharpe lie
   detector, and **compares against buy-and-hold**.
9. **Writes the results** (tables + charts) into `research/results/`.

---

## 5. How to run it

```bash
# one-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# the three things you can do:
./run.sh             # run the full research pipeline (prints results, saves charts)
./run.sh tests       # run the automated checks (71 of them)
./run.sh dashboard   # open the interactive visual dashboard in your browser
```

The dashboard opens at `http://localhost:8050`. Use the left sidebar to switch
between Overview, Equity (the headline chart), Positions, Signals, Costs, and ML
Intel.

---

## 6. How to read the results

After `./run.sh`, look at the printed table and `research/results/`:

- **`variants.csv` / the Overview table.** Each row is a version of the strategy.
  Compare the `sharpe` column across rows. The `[bench] btc_hold` and
  `[bench] equal_weight` rows are buy-and-hold benchmarks. **If a strategy row's
  Sharpe is below the benchmark rows, the strategy lost.**
- **`equity_curves.png`.** Hypothetical account value over time (log scale). The
  benchmark lines climbing far above the strategy lines = the strategy
  underperformed.
- **`cost_sweep.png`.** Sharpe as trading costs rise. A real edge stays positive
  as costs grow; a fragile one collapses. The plain baseline collapses by ~1×
  costs; the **conformal-gated** baseline stays positive out to ~2×.
- **ML diagnostics (`summary.json` / ML Intel panel).** The truth-tellers: rank IC
  ≈ −0.006 (no ranking skill — but no longer *inverse*, which the refit fixed),
  lead-lag hit-rate ≈ 0.51 (coin-flip), regime trend-dominated (~87%) — all
  shelved; but conformal coverage ≈ 0.91 (calibration works) and it lifts the
  baseline Sharpe 0.10 → 0.12 while staying nearly fully invested.

**What a full "win" would have looked like:** a strategy Sharpe clearly *above*
the benchmarks, staying positive across the cost sweep, with a Deflated Sharpe
near 1.0. We got a genuine *partial* win (conformal helps, validated) but not a
beat-the-benchmark win — and we report both plainly.

---

## 7. Why a "negative" result is the whole point

It would have been easy to tweak knobs until a chart looked amazing — and
completely fake. Instead, NullQuant was built so it **cannot lie to itself**, and
the honest verdict is: *this simple idea doesn't beat just holding Bitcoin once
you account for costs and test it properly.*

To a serious quant employer that is a **strong** result, because it demonstrates:
- you can build a correct, realistic simulator,
- you understand the traps (overfitting, look-ahead, costs, multiple-testing),
- and you have the **integrity to report a negative finding** instead of selling
  a mirage. That judgment is the actual job.

---

## 8. Project map (what each folder does, plainly)

```
config/config.yaml      All the settings (which coins, risk level, costs) in one place.
nullquant/
  data/        Downloads & organizes price history + perp funding rates (no future-peeking).
  features/    Turns raw prices into indicators (averages, volatility, cross-sectional scores) and labels.
  signals/     Decides which coins to long/short each week.
  portfolio/   The cost model + the trading simulator (the backtest).
  metrics/     Scores performance (Sharpe, drawdown, the lie-detector Sharpe).
  validation/  The honest-testing machinery (walk-forward, purged CV).
  ml/          The four reframed models (rank, regime, lead-lag, conformal) + funding tilt.
  ablation.py  The "does the ML actually help?" experiment.
  pipeline.py  Runs everything end-to-end and saves results.
dashboard.py   The interactive visual app.
tests/         71 automated checks that the math is correct and nothing cheats.
research/report.md   The formal write-up of methodology and findings.
```

---

## 9. Quick glossary

| Term | Plain meaning |
|---|---|
| Long | Bet it goes up (buy) |
| Short | Bet it goes down |
| SMA | Average price over N days (a trend line) |
| Golden/death cross | Fast trend line crossing above/below the slow one |
| Volatility | How wildly the price swings (risk) |
| Vol targeting | Sizing bets to keep total risk steady |
| Leverage | Betting more than you have (borrowed) |
| Backtest | Replaying a strategy over history |
| Transaction cost | The money lost to fees/spread/slippage when trading |
| Benchmark / HODL | The "just buy and hold" comparison |
| Sharpe ratio | Return per unit of risk (higher = better) |
| Drawdown | Worst drop from a peak |
| Overfitting | Fitting past flukes, not a real pattern |
| Look-ahead / repainting | Cheating by using future info |
| Out-of-sample | Testing on data you didn't build on |
| Walk-forward | Train on the past, test on the next chunk, roll forward |
| Purged CV | Finance-correct ML testing that prevents leakage |
| Deflated Sharpe | Sharpe discounted for how many things you tried (lie detector) |
| Ablation | "With vs without" test of whether a component helps |
| Learning-to-Rank | ML that predicts the *order* of winners/losers (not prices) |
| HMM regime-switching | ML that finds hidden market "moods" and switches strategy |
| Lead–lag network | ML that learns which coins move first and which follow |
| Conformal prediction | Calibrated confidence; size bets by how certain the model is |
| Funding rate | Fee the crowded side of a perpetual-swap bet pays; high = crowded longs (bearish tilt) |

For the rigorous version of all this, see [`research/report.md`](research/report.md).
