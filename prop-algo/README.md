# prop-algo — Apex eval pass-probability optimizer

An algo + backtester built around one question: **what maximizes the probability
of passing an Apex $50K evaluation ($3,000 target before a $2,500 real-time
trailing drawdown)?**

There are no guaranteed wins. What this tool does instead is the defensible
version of "eval passing demon": find positive-expectancy intraday setups on
index futures, then size them to maximize **P(pass before bust)** — and tell
you the expected number of attempts and total cost in eval fees, so blowing
some evals is a priced-in part of the strategy, not a surprise.

## What's inside

- **Data** (`src/propalgo/data/`)
  - `dukascopy.py` — free, no-API-key, multi-year 15m history for index CFDs
    that track the futures nearly tick-for-tick (`E_NQ-100`, `E_SandP-500`,
    `E_D&J-Ind`, `USSC2000.IDX`). This is the backtest dataset.
  - `yahoo.py` — recent true-futures 15m bars (`NQ=F`, `ES=F`, ~60 days,
    ~10 min delayed). Powers the live loop and the parity check.
  - Both share a parquet cache (`data_cache/`) with incremental append; every
    `fetch` extends the history.
- **Apex rules** (`src/propalgo/rules/apex.py`) — the trailing drawdown trails
  the *real-time* equity high-water mark, **including unrealized peaks of open
  trades**. Most naive backtests get this wrong and overstate pass rates.
  Intra-bar ordering is conservative (the bar's low is checked against the
  threshold before its high can raise the high-water mark).
- **Strategies** (`src/propalgo/strategies/`) — high-conviction, one signal per
  session each:
  - `orb.py` — 30-minute opening range breakout, ATR-capped stop, 2R target.
    Longs only; breakdowns aren't traded but *veto* other entries while the
    phantom short would be open (see Results).
  - `momentum.py` — overnight gap (≥0.5 ATR) holding one side of VWAP →
    trend-day continuation.
  - Signals carry a grade (`A` / `A+`); A+ setups get sized up.
- **Backtest** (`src/propalgo/backtest/`) — bar-by-bar engine with conservative
  fills (stop before target when both hit in one bar, 1-tick slippage/side,
  $4 RT commission). Trades store per-contract, per-bar equity excursions so
  the same history can be replayed at any contract size. The eval replay runs
  back-to-back $50K attempts over the whole history; the Monte Carlo
  bootstraps trade-days into 10,000 synthetic eval months.
- **Live** (`src/propalgo/live/`) — polls each 15m bar close, re-runs the
  strategies on today's session, posts Discord alerts (instrument, side,
  entry/stop/target, contracts). **Alerts only — you click the trades in
  Tradovate yourself.** Apex prohibits fully automated trading; this design
  keeps you compliant.

## Sizing model

Fixed-contract sizing does not survive contact with this account: index
futures stops are ATR-scale ($2,000-6,000 per mini) against a $2,500 trailing
drawdown, so a fixed 4-mini position busts almost every attempt (verified:
900 attempts, 2 passes). Instead, every trade risks a **fraction of the
trailing drawdown**, converted into position size through the signal's stop
distance, in **micro granularity** (Apex 50K allows 10 minis = 100 micros).
Tight stops get big size, wide stops get small size, and the `montecarlo`
sweep finds the risk fraction that maximizes P(pass).

## Quick start

```bash
pip install -r requirements.txt

# 1. pull data (Dukascopy multi-year + Yahoo recent)
python scripts/cli.py fetch --symbols NQ ES --start 2023-01-01

# 2. trade stats + sequential eval replay over the full history
python scripts/cli.py backtest --symbols NQ ES --start 2023-01-01 --risk 0.20

# 3. the headline numbers: P(pass), attempts, cost across risk levels
python scripts/cli.py montecarlo --symbols NQ ES --start 2023-01-01

# promo accounts often waive the 7-day minimum:
python scripts/cli.py montecarlo --start 2023-01-01 --min-days 1

# 4. sanity-check proxy data against true futures over the overlap window
python scripts/cli.py parity --symbols NQ ES

# 5. go online (signal alerts via Discord webhook)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
python scripts/cli.py live --risk 0.20            # or --dry-run / --once
```

Docker: `docker build -t propalgo . && docker run -e DISCORD_WEBHOOK_URL=... propalgo`

## Results (NQ+ES 15m, Jan 2023 - Jun 2026, 557 trades)

After the optimization pass below: **P(pass) 36.9% per 30-trading-day month
at 25% risk, expected cost ~$95 (≈2.7 attempts) per funded account** — up
from the 29.2% / $120 baseline.

```
Monte Carlo (bootstrapped 30-trading-day months; size in micros):
  risk/trade  avg size  P(pass)  P(bust)  med days  E[attempts]   E[cost]
        10%       2.1     3.5%     3.2%        25         28.3      $992
        15%       3.0    21.5%    26.5%        21          4.7      $163
        20%       3.9    33.2%    51.2%        17          3.0      $105
        25%       4.8    36.9%    59.3%        14          2.7       $95
        30%       5.6    35.0%    63.0%        12          2.9      $100
        50%       7.5    26.7%    71.1%        15          3.7      $131
  -> best: risk 25% of DD per trade (P(pass)=36.9%, expected cost $95)
```

The sequential replay over the same history (attempts not capped at 30 days)
passed 14 of 37 back-to-back evals (38%, median 11 trading days per pass).
Proxy-vs-futures signal parity over the recent overlap window: **91%**.

**What moved the needle (kept):**
- **ORB short veto** (+4.5pp): breakdown trades lose money on long-biased
  indices (PF 0.94), and entries taken *during* a morning breakdown window
  average -0.01R — so breakdowns aren't traded AND they block other entries
  until the phantom short would have exited.
- **Momentum gap filter 0.3 → 0.5 ATR** (+2pp): bigger gaps mark real trend
  days. The 0.35-0.60 scan is a smooth plateau (34-37%), not a fitted spike.
- **Risk taper near the target** (+1pp): never risk much more than the
  remaining distance to the target — busting while almost done is the most
  expensive way to fail.

**What was tried and made things worse (rejected):** breakeven stops at
0.8/1.0/1.5R (-3 to -6pp — they cut the EOD runners that carry the edge),
time stops (-9pp), adaptive timid/bold sizing (-3 to -6pp), one-loss-per-day
(neutral to -2pp), wider 45/60-min opening ranges (-9pp), 3R/4R/no targets
(-1 to -2pp), adding YM/RTY (neutral — one position at a time means they just
compete for the same slot).

**Regime honesty** (chosen config, per-period P(pass) at 20-25% risk):
2023: 47.5% · 2024: 38.5% · 2025: **13.6%** · 2026 ytd: 42.4%. In a chop
year like 2025 the expected cost per pass rises to ~$260 (~7 attempts). The
edge is regime-dependent; the headline 36.9% averages over good and bad
regimes.

## Honest caveats

- **No guarantees.** Backtests overfit; the bootstrap assumes the future
  resembles the sampled history. Treat P(pass) as an estimate with error bars,
  not a promise.
- **Proxy data.** Backtests run on Dukascopy index CFDs, not CME futures —
  same shape, small basis/session differences. PnL is converted with real
  futures multipliers (NQ $20/pt, ES $50/pt). Run `parity` to measure signal
  agreement on the overlap window; Databento's free signup credits can buy a
  one-off true-CME cross-check.
- **Passing ≠ keeping.** Apex's 30% consistency rule applies at *payout* on
  funded accounts. The aggressive sizing that passes evals fast will violate
  it if you keep trading the same way after funding. That is a different
  problem this tool does not solve.
- **Compliance.** Apex bans fully automated/unattended trading. This system
  only sends alerts; you place every order manually.
- Yahoo live data is ~10 min delayed — fine for 15m-close signals, not for
  scalping.
