# NQ New York open gap strategy — backtest harness

Tests one idea on Nasdaq 100 futures: **wait for the New York open, take the
first unmitigated gap that forms, and trade it with a defined stop.**

It also tests two ways of sharpening that: filtering gaps by the session's
**bias**, and a separate **sweep-and-reclaim** setup that anchors the entry and
the stop to the same reversal.

The point of this project is not the equity curve. It is the set of checks
around it — control permutations, a parameter grid, a bar-resolution test and a
slippage ladder — that decide whether the equity curve means anything. On this
sample the answer is "probably something in the base rules, nothing yet in the
setup that looks best."

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

**The shape of the edge is consistent.** 18 of 25 variants in the parameter
grid are positive, median +0.36R. Every limit-back-into-the-gap variant makes
money; every stop-entry breakout variant and the fade loses. A single lucky cell
would not look like that — but a 25-cell grid also means the *best* cell (+0.84R
at a 10:30 cutoff) is not to be trusted.

**Costs are not the problem.** Stops average ~15 points, so quadrupling
slippage to 4 ticks a side moves expectancy from +0.66R to +0.62R.

**Five-minute bars understate the result.** On the 20 sessions where 1-minute
data is available, the same trades score +0.07R simulated on 5-minute bars and
+0.48R on 1-minute bars. Whenever a bar contains both the stop and the target,
the simulator takes the stop; on 5-minute bars that happened to a third of all
trades. The headline +0.65R is therefore a *pessimistic* bound, not an
optimistic one.

## Session bias

The base rules take the first gap and trade whichever way it points. Adding a
bias filter asks the question a discretionary trader asks first — *which side am
I looking for today?* — and skips gaps that argue against it. Every method is
evaluated only from bars that had already printed when the order would go in.

| Bias | Trades | Win rate | Expectancy | 95% CI |
|---|---|---|---|---|
| Opening-range break | 21 | 62% | **+0.86R** | [+0.15, +1.44] |
| Prior day's close | 18 | 56% | +0.66R | [−0.01, +1.33] |
| None (base rules) | 29 | 55% | +0.65R | [+0.12, +1.17] |
| Overnight-range break | 15 | 47% | +0.39R | [−0.41, +1.20] |
| Sweep-and-reclaim | 21 | 38% | +0.19R | [−0.44, +0.82] |

The most recent break of the first 15 minutes' range is the only filter that
clearly beats taking every gap, and it costs 8 of 29 trades to get there. That
is the trade-off every filter makes: a better average on a thinner sample is not
automatically a better strategy, and the interval barely narrows.

### The structural stop makes things worse

Placing the stop beyond the recent swing low instead of the gap's far edge —
the placement in the chart this was modelled on — loses money in **nine of ten**
bias/stop combinations, taking the base rules from +0.65R to −0.32R.

The reason is geometry, not stop placement as such. A gap entry fires *after* a
displacement, so the entry sits high in the move while the structural stop stays
at the pre-move low: average risk goes from 15 to 72 points while the room left
above is unchanged. A 2R target then needs a 144-point move after the move has
already happened. Capping risk at 40–55 points does not rescue it.

### It is already an AM-session strategy

Moving the flat-by time from 15:55 to 13:00, 12:00 or 11:30 changes nothing at
all: **no trade has ever reached a time exit**. Every one resolves at its stop
or its target before 11:30, because the stop is a fraction of a session's range
and the target is 2R away. The afternoon exit is a backstop that has not once
been needed.

Tightening the order-expiry from 12:00 to 11:30 does matter, since it cancels
orders that had not yet filled: 29 trades become 25, and expectancy slips from
+0.65R to +0.55R. With the opening-range bias, an 11:30 expiry gives 18 trades
at +0.83R.

## Sweep and reclaim

Anchoring entry and stop to the *same* event fixes that geometry, and is the
setup the chart actually shows: price runs a known low, fails to hold under it,
and reclaims it; the entry is a limit back at the reclaimed level with the stop
just below the wick that swept it. Both sit at the origin of the move, so risk
is small relative to the room above.

The best cell — sweeping the overnight low, entering at the level, requiring an
imbalance on the reclaim — returns **+1.14R over 7 trades**, 71% win rate,
26-point average risk. That is the most attractive number in this repository and
it should be ignored, for two reasons:

- **It fails its own control.** Randomising the direction on the same setups
  averages +0.97R (p = 0.37). With seven trades and a 2R target, a coin flip
  produces the same result.
- **10 of 18 variants are positive**, which is what a coin flip looks like.

The setup fires on 12 of 49 sessions and fills 7. Whether it works is simply not
answerable on two months of data.

### Why not to trade any of this yet

- **29 trades.** The confidence interval's lower edge is +0.12R, barely above
  zero. One extra losing streak flips the conclusion.
- **68 calendar days of a single regime.** No high-volatility stretch, no rate
  shock, no earnings-season cluster.
- **The parameter grid is a multiple-comparisons machine.** 25 variants on one sample.
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
python3 -m pytest tests -q           # 64 tests
```

Writes `results/trades_baseline.csv` and `results/trades_sweep.csv` (the trade
ledgers), `results/sweep.csv` (the parameter grid) and `results/summary.json`.

## Layout

| File | What's in it |
|---|---|
| `gapstrat/data.py` | Bar download, caching, resampling, session slicing |
| `gapstrat/gaps.py` | Fair value gap detection and mitigation tracking |
| `gapstrat/strategy.py` | Rules: which gap, which side, what levels |
| `gapstrat/bias.py` | Session bias: which side to look for today |
| `gapstrat/sweep.py` | The stop-run-and-reverse setup |
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
