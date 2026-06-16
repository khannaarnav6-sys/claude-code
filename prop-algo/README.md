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

## Standalone 3-sigma Telegram notifier (`lambda_sd_notifier.py`)

A separate, self-contained one-file tool — **zero pip dependencies, standard
library only** — that alerts via Telegram when MES / MNQ price action
approaches the 3-sigma band (|z-score| >= 2.8 over a rolling 15m window). It's
built to drop straight into AWS Lambda on an EventBridge schedule (no layers,
no packaging). All config is via environment variables; see the docstring at
the top of the file for deploy steps and the tunables (`SYMBOLS`, `LOOKBACK`,
`SD_THRESHOLD`, ...). Test locally with:

```bash
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python lambda_sd_notifier.py
# or DRY_RUN=1 to print alerts instead of sending
```

Data source failover order is `SOURCES` (default `yahoo,tradier,dukascopy`):
Yahoo gives the real future and the 18:00 session VWAP; Tradier (set
`TRADIER_TOKEN`) is a real-volume SPY/QQQ failover (RTH-anchored); Dukascopy is
a last-resort cash-index proxy. Yahoo gets throttled from AWS IPs, so hosting on
a non-AWS box is more reliable — see below.

### Host on an Oracle Cloud "Always Free" VM (free, dodges AWS throttling)

The script is one dependency-free file, so the VM only needs Python 3.9+. It
self-schedules with `--loop` (no cron). This keeps the accurate Yahoo
session-anchored bands while avoiding the AWS-IP rate limiting that hits Lambda.

1. **Create the instance.** Oracle Cloud console -> Compute -> Instances ->
   Create. Pick an **Always Free-eligible** shape (`VM.Standard.E2.1.Micro`,
   AMD, is simplest/most available; the Ampere `A1.Flex` also works), image
   **Ubuntu 22.04+**. Download the SSH key it offers. No inbound ports are
   needed (the notifier only makes outbound calls), so leave the default
   security list alone.
2. **SSH in:** `ssh -i your_key ubuntu@<public-ip>` and confirm Python:
   `python3 --version` (3.10 ships on 22.04 — fine; `zoneinfo`/tzdata are
   already present).
3. **Install the script:**
   ```bash
   sudo mkdir -p /opt/sdnotifier
   sudo curl -fsSL -o /opt/sdnotifier/lambda_sd_notifier.py \
     https://raw.githubusercontent.com/khannaarnav6-sys/claude-code/claude/session-6tv9tq/prop-algo/lambda_sd_notifier.py
   ```
   (or `scp` it up / paste it in).
4. **Secrets in a root-only env file** `/etc/sdnotifier.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=123456:abc...
   TELEGRAM_CHAT_ID=987654321
   # optional: enables the Tradier failover
   TRADIER_TOKEN=
   # keep dedup/cache across restarts (matches StateDirectory below)
   STATE_DIR=/var/lib/sdnotifier
   ```
   `sudo chmod 600 /etc/sdnotifier.env`
5. **systemd service** `/etc/systemd/system/sdnotifier.service`:
   ```ini
   [Unit]
   Description=MES/MNQ 3-SD VWAP Telegram notifier
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   EnvironmentFile=/etc/sdnotifier.env
   Environment=PYTHONUNBUFFERED=1
   ExecStart=/usr/bin/python3 /opt/sdnotifier/lambda_sd_notifier.py --loop 60
   Restart=always
   RestartSec=10
   DynamicUser=yes
   StateDirectory=sdnotifier

   [Install]
   WantedBy=multi-user.target
   ```
6. **Start it and watch logs:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now sdnotifier
   journalctl -u sdnotifier -f
   ```
   You'll see one `[loop ...] N new alert(s)` line per minute. `Restart=always`
   + `enable` means it survives crashes and reboots. To update the script,
   re-download and `sudo systemctl restart sdnotifier`.

(GCP's free `e2-micro` works identically — same steps from #2.)

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
python scripts/cli.py backtest --symbols NQ ES --start 2023-01-01 --profile grind

# 3. the headline numbers: P(pass), E[attempts], months, cost per risk level
python scripts/cli.py montecarlo --symbols NQ ES --start 2023-01-01

# promo accounts often waive the 7-day minimum:
python scripts/cli.py montecarlo --start 2023-01-01 --min-days 1

# 4. sanity-check proxy data against true futures over the overlap window
python scripts/cli.py parity --symbols NQ ES

# 5. go online (signal alerts via Discord webhook)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
python scripts/cli.py live --profile grind        # or --dry-run / --once
```

Docker: `docker build -t propalgo . && docker run -e DISCORD_WEBHOOK_URL=... propalgo`

## Results (NQ+ES 15m, Jan 2023 - Jun 2026, 557 trades)

**Attempts and speed trade off directly** — Apex evals have no time limit
(fees accrue monthly), so risking less per trade busts less but grinds
longer. Pick a profile (`--profile`, default `grind`):

```
Monte Carlo (attempts capped at 250 trading days, fees accrue monthly):
  profile    risk/trade  P(pass)  P(bust)  med days  mo/att  E[attempts]  E[cost]
  grind            8.5%    77.2%    19.2%        98    5.77         1.30     $262
  balanced        12.0%    62.0%    38.0%        54    3.22         1.61     $182
  sprint          25.0%    40.3%    59.7%        14    1.14         2.48      $99
```

- **grind (default): ~1.3 attempts on average** — three out of four evals
  pass, but a pass takes ~5 months of trading and ~$260 in monthly fees.
- **sprint: ~2.5 attempts**, median pass in ~3 weeks, cheapest in fees —
  you just eat more blown evals along the way.
- There is no setting that passes fast AND rarely busts; that point sits
  outside what this edge (PF 1.14) can buy. Anyone promising both is selling
  something.

The sequential replay over the actual 2023-2026 sequence at grind risk went
**3 for 3** (median 80 trading days per pass, surviving 2025); at sprint
risk it passed 14 of 37 (38%, median 11 days). Proxy-vs-futures signal
parity over the recent overlap window: **91%**.

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

**Also tried for the attempts goal and rejected:** a midday range-breakout
strategy to raise trade frequency (PF 0.65, loses money every full year) and
ORB re-entries after stop-outs (slightly positive alone, but neutral-to-worse
through the eval math — extra trades add bust risk faster than progress).

**Regime honesty** (grind profile, per-year E[attempts]): 2023: 1.12 ·
2024: 1.60 · 2025: **5.46** · 2026 ytd: 1.20. The 1.3 average holds across
mixed regimes, but a pure chop year like 2025 alone is ~5 attempts even at
grind risk. The edge is regime-dependent; no sizing setting fixes that.

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
