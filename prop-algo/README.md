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
  - `momentum.py` — overnight gap holding one side of VWAP → trend-day
    continuation.
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

## Results (NQ+ES 15m, Jan 2023 - Jun 2026, 1,026 trades)

Strategy edge per 1 mini contract: momentum PF 1.35 (n=155), ORB PF 1.02
(n=871), combined +$46k total but with a losing 2025 (-$30k) — regime risk
is real and the eval numbers below include it.

```
Monte Carlo (bootstrapped 30-trading-day months; size in micros):
  risk/trade  avg size  P(pass)  P(bust)  med days  E[attempts]   E[cost]
        10%       2.3     5.4%    12.4%        24         18.7      $653
        15%       3.3    22.4%    48.9%        18          4.5      $156
        20%       4.3    29.2%    66.7%        12          3.4      $120
        25%       5.4    28.7%    71.2%         9          3.5      $122
        30%       6.3    26.1%    73.9%         7          3.8      $134
        50%       8.8    19.8%    80.2%         7          5.1      $177
  -> best: risk 20% of DD per trade (P(pass)=29.2%, expected cost $120)
```

Read it honestly: this is a gambler's-ruin curve, and 20% risk per trade is
its peak — ~29% of eval months pass, so budget **~3-4 attempts (~$120 in
promo fees) per funded account**. Risking more passes faster but busts more;
risking less starves the target. The sequential replay over the same history
(attempts not capped at 30 days) passed 11 of 29 back-to-back evals (38%).
Proxy-vs-futures signal parity over the recent overlap window: **91%**.

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
