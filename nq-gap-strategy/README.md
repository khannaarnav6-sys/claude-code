# NQ New York open gap strategy — backtest harness

Tests one idea on Nasdaq 100 futures: **wait for the New York open, take the
first unmitigated gap that forms, and trade it with a defined stop.**

The point of this project is not the equity curve. It is the set of checks
around it — control permutations, a parameter sweep, a bar-resolution test and
a slippage ladder — that decide whether the equity curve means anything. On
this sample the answer is "probably something, but the sample is too small to
bet on."

## The rules

1. **Wait for 09:30 ET.** Nothing before the New York open counts, including
   overnight imbalances.
2. **Find the first gap.** A gap is a three-bar fair value gap on the 5-minute
   chart: bar 3's low above bar 1's high (bullish), or bar 3's high below bar
   1's low (bearish). The untouched range between them is the gap zone. It is
   unmitigated by construction the moment it forms.
3. **Take the first one only,** if it is at least 5 points wide and forms
   by 11:00.
4. **Trade with the displacement.** Bullish gap → long, bearish gap → short.
5. **Enter on a limit back inside the gap,** at the 50% level by default
   (the proximal edge and far edge are both selectable).
6. **Stop beyond the far edge of the gap** plus a 4-tick buffer: if the gap
   fills completely, the reason for the trade is gone.
7. **Target 2R.** Flat by 15:55 either way. Unfilled orders cancel at 12:00.
   One trade per session.

## Results

49 full sessions, 2026-06-29 to 2026-09-04, front-month NQ.

| | |
|---|---|
| Setups found | 41 of 49 sessions |
| Orders filled | 29 (71% of setups) |
| Win rate | 55% (16W / 13L) |
| Expectancy | **+0.65R per trade**, 95% CI [+0.12, +1.17] |
| Total | +18.8R, $6,845 on one contract |
| Max drawdown | 3.1R ($984) |
| Profit factor | 2.41 |

Context that matters: NQ went **nowhere** over this stretch (−0.2%, only 43% of
sessions closed above their open), so this is not a long-bias artifact.

### What the checks say

**Direction carries information.** Re-running with a coin-flip direction 500
times — same gaps, same levels, same risk, orders mirrored to the correct side
of the market — averages +0.22R. The real rule beat 99% of those runs
(p = 0.012).

**"First" gap is weakly special.** Trading a randomly chosen gap from the same
session averages +0.32R; the first gap beat 94% of those runs (p = 0.062).
Suggestive, not significant.

**The shape of the edge is consistent.** 18 of 25 sweep variants are positive,
median +0.36R. Every limit-back-into-the-gap variant makes money; every
stop-entry breakout variant and the fade loses. A single lucky cell would not
look like that — but the sweep also means the *best* cell (+0.84R at a 10:30
cutoff) is not to be trusted.

**Costs are not the problem.** Stops average ~15 points, so quadrupling
slippage to 4 ticks a side moves expectancy from +0.66R to +0.62R.

**Five-minute bars understate the result.** On the 20 sessions where 1-minute
data is available, the same trades score +0.07R simulated on 5-minute bars and
+0.48R on 1-minute bars. Whenever a bar contains both the stop and the target,
the simulator takes the stop; on 5-minute bars that happened to a third of all
trades. The headline +0.65R is therefore a *pessimistic* bound, not an
optimistic one.

### Why not to trade this yet

- **29 trades.** The confidence interval's lower edge is +0.12R, barely above
  zero. One extra losing streak flips the conclusion.
- **68 calendar days of a single regime.** No high-volatility stretch, no rate
  shock, no earnings-season cluster.
- **The sweep is a multiple-comparisons machine.** 25 variants on one sample.
- **Shorts are much weaker than longs** (+0.39R vs +0.93R) in a flat market,
  which the strategy's logic does not explain.
- **Yahoo bars, not tick data.** No bid/ask, no volume profile, continuous
  front-month with roll artifacts.

What would actually settle it: several years of proper 1-minute data — the
sample here is capped by what a free feed will serve.

## Running it

```bash
pip install -r requirements.txt
python3 run_backtest.py              # full report
python3 run_backtest.py --refresh    # re-download bars first
python3 run_backtest.py --quick      # skip the 500-draw control permutations
python3 run_backtest.py --target-r 3 --entry-style proximal
python3 -m pytest tests -q           # 36 tests
```

Writes `results/trades_baseline.csv` (the ledger), `results/sweep.csv` and
`results/summary.json`.

## Layout

| File | What's in it |
|---|---|
| `gapstrat/data.py` | Bar download, caching, resampling, session slicing |
| `gapstrat/gaps.py` | Fair value gap detection and mitigation tracking |
| `gapstrat/strategy.py` | Rules: which gap, which side, what levels |
| `gapstrat/backtest.py` | Bar-by-bar fill simulation |
| `gapstrat/metrics.py` | Statistics, bootstrap intervals |
| `gapstrat/controls.py` | Permutation nulls the strategy has to beat |
| `run_backtest.py` | The report |

## Execution assumptions

Every ambiguity in an OHLC bar is resolved against the strategy:

- A limit fills only once price trades **through** it by a tick, never on a
  touch.
- A profit target is a resting limit: it fills at its own price and never
  better, however far past it the market runs.
- On the bar that fills the entry, that bar's open is unusable — it happened
  before the fill — so exits on that bar can only be the exact stop or target,
  and the bar contributes no excursion statistics.
- One bar containing both stop and target is scored as a loss.
- Stop entries, stop exits and time exits pay slippage; limit fills do not.
- $4.50 commission round turn, $20/point, 0.25 tick.

Two of these were bugs found by testing, not design decisions made up front.
Filling exits at the entry bar's open inflated expectancy from +0.65R to
+1.31R, and letting the profit target fill at a better-than-limit price
produced average wins of 3.2R against a 2R target. Both are the kind of error
that makes a backtest look like an edge.

## Data

Yahoo Finance's chart endpoint, the only free intraday futures source reachable
here. It serves ~60 trading days of 5-minute bars and ~30 days of 1-minute bars
(in ≤8-day windows). Cached CSVs under `data/` are gitignored — rerun with
`--refresh` to rebuild them. The window moves with the calendar, so numbers
from a fresh pull will not match the ones above exactly.

**Not investment advice.** A backtest is a hypothesis, not a forecast.
