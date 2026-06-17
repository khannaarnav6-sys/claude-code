#!/usr/bin/env python3
"""Telegram notifier: alerts when MES / MNQ price approaches the 3rd VWAP SD band.

Single file, ZERO third-party dependencies (standard library only) so it drops
straight into AWS Lambda with no layers or packaging step. It reproduces
TradingView's *VWAP with standard-deviation bands* indicator: a session-anchored,
volume-weighted average price plus bands at +/- N volume-weighted standard
deviations, and sends a Telegram message when the latest price gets close to the
3rd SD band in either direction.

"Approaching the 3 SD band" => alert fires once |z| >= SD_THRESHOLD (default
2.8), where z = (price - VWAP) / sigma_vwap.

Band math (matches TradingView), anchored to the current session, typical price
hlc3 = (high+low+close)/3:
    VWAP    = sum(volume * hlc3) / sum(volume)
    sigma   = sqrt( sum(volume * hlc3^2) / sum(volume) - VWAP^2 )
    band_k  = VWAP +/- k * sigma          (k = BAND_SD, default 3)

------------------------------------------------------------------------------
DATA SOURCES (failover)
------------------------------------------------------------------------------
Tried in the order given by SOURCES (default "yahoo,tradier,dukascopy"):

  * yahoo     -- real CME futures (MES=F/MNQ=F) WITH real volume, so the VWAP
                 and SD bands match your trading platform, INCLUDING the 18:00
                 ET session anchor. Primary. Downside: Yahoo throttles AWS IP
                 ranges, so it can return 429/999 from Lambda; the function
                 retries, then fails over.
  * tradier   -- first failover. Free real-time equities with a Tradier
                 brokerage account (set TRADIER_TOKEN); cloud-friendly REST,
                 not throttled like Yahoo. No futures, so it queries the
                 tracking ETF (MES/ES->SPY, MNQ/NQ->QQQ) via markets/timesales
                 with REAL consolidated volume. ETFs are RTH-only, so it forces
                 the RTH VWAP anchor (it cannot reproduce the 18:00 session
                 anchor). z-score tracks the future intraday; printed levels are
                 the ETF. Skipped automatically if TRADIER_TOKEN is unset.
  * dukascopy -- last-resort backup, free, no key, cloud-friendly. Pulls
                 the underlying CASH index via the public tick feed and decodes
                 it with stdlib lzma/struct. IMPORTANT: its index "volume" is
                 flat, so the SD bands degrade to an UNWEIGHTED standard
                 deviation, and prices are the cash index (offset from the
                 futures by the basis, ~tens of points). The z-score / "near 3
                 SD" trigger is still valid (price and VWAP share the same
                 series, so the offset cancels); only the printed absolute
                 levels differ. Used only as a keep-alive backup when Yahoo
                 fails -- alerts from it are clearly labeled BACKUP.

So normal operation is accurate (Yahoo); during a Yahoo/AWS block the notifier
stays alive on Dukascopy with honestly-labeled approximate bands.

------------------------------------------------------------------------------
DEPLOY ON AWS LAMBDA
------------------------------------------------------------------------------
  * Runtime: Python 3.11+, handler = lambda_sd_notifier.lambda_handler
  * No dependencies, so just zip this one file (or paste it into the console).
  * Trigger: EventBridge (CloudWatch Events) schedule -> rate(1 minute).
    Polling every minute catches both the 5m and 15m bar closes within a
    minute; the 5m/15m bars are far less noisy than 1m. The per-bar dedup
    (keyed on symbol+interval) means a once-a-minute wake never double-alerts
    on the same bar. Cost on the AWS always-free tier is negligible:
      ~43,800 runs/month x ~0.13 GB-s (128 MB, ~1s) = ~1.5% of 400k GB-s,
      and ~4.4% of the 1M free invocations. Yahoo load is 4 requests/run =
      240 req/hour, comfortably under its informal throttle.
  * Set these environment variables:
      TELEGRAM_BOT_TOKEN   (required)  from @BotFather
      TELEGRAM_CHAT_ID     (required)  your chat/channel id (talk to @userinfobot)
      SYMBOLS              default "MES=F,MNQ=F"
      INTERVALS            default "15m,5m"  (checked independently; one alert
                             per symbol per timeframe per bar)
      SOURCES              default "yahoo,tradier,dukascopy"  (failover order)
      TRADIER_TOKEN        production access token (enables the tradier source)
      TRADIER_BASE         default "https://api.tradier.com" (sandbox = delayed)
      RANGE                default "5d"    (Yahoo history window to pull)
      VWAP_ANCHOR          default "session"  where the VWAP resets each day:
                             "session" = 18:00 ET (CME Globex open, TV default
                                          for futures)
                             "rth"     = 09:30 ET (regular session open)
                             "day"     = 00:00 ET
      BAND_SD              default "3.0"  the band level you care about (sigmas)
      SD_THRESHOLD         default "2.8"  |z| at/above this fires the alert
      MIN_BARS             default "3"    bars needed since the anchor to bother
      WARMUP_MIN           default "0"    suppress alerts for N minutes after the
                             anchor (use ~30 with VWAP_ANCHOR=rth so the opening
                             bars' tiny-sample bands don't fire false 3-SD signals)
      MAX_BAR_AGE_MIN      default "0"    skip if the latest bar is older than this
                             many minutes -- guards against stale data (e.g. a
                             real-time source going quiet, or an RTH-only source
                             like tradier being polled overnight). NOTE Yahoo's
                             futures feed is ~10 min delayed, so keep this >~20
                             if Yahoo is in your SOURCES.
      USE_LAST_CLOSED_BAR  default "1"    ignore the still-forming current bar
  * Give the function ~128MB and a 30s timeout; outbound internet (default
    Lambda networking, or a NAT gateway if you place it in a VPC).

Run locally to test:  TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
                      python lambda_sd_notifier.py
Add DRY_RUN=1 to print alerts instead of sending them.

------------------------------------------------------------------------------
RUN ON AN ALWAYS-ON VM (Oracle Cloud "Always Free", GCP free e2-micro, ...)
------------------------------------------------------------------------------
Instead of Lambda you can self-host (sidesteps Yahoo's AWS-IP throttling, keeps
the real-futures session-anchored bands). No dependencies -> just Python 3.9+.
Run the built-in scheduler (no cron needed):

    python3 lambda_sd_notifier.py --loop          # every 60s
    python3 lambda_sd_notifier.py --loop 30        # every 30s

Best run as a systemd service with the secrets in a root-only EnvironmentFile;
see the repo README for the exact unit file and Oracle setup steps.
"""
from __future__ import annotations

import json
import lzma
import math
import os
import struct
import time as wallclock  # 'time' name is taken by datetime.time below
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
DUKAS_FEED = "https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0"}

# session anchor name -> local ET time of day the VWAP resets at
ANCHORS = {
    "session": time(18, 0),   # CME Globex open (prior calendar day)
    "rth": time(9, 30),       # regular trading hours open
    "day": time(0, 0),        # midnight ET
}

# Dukascopy backup: futures symbol -> (cash-index instrument, integer scale)
DUKAS_MAP = {
    "MES=F": ("USA500IDXUSD", 1000.0), "ES=F": ("USA500IDXUSD", 1000.0),
    "MNQ=F": ("USATECHIDXUSD", 1000.0), "NQ=F": ("USATECHIDXUSD", 1000.0),
}

# Tradier backup: futures symbol -> tracking ETF (equities only; RTH only)
TRADIER_MAP = {
    "MES=F": "SPY", "ES=F": "SPY", "MNQ=F": "QQQ", "NQ=F": "QQQ",
}
TRADIER_INTERVALS = {"1m": "1min", "5m": "5min", "15m": "15min"}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_interval_seconds(interval: str) -> int:
    interval = interval.strip().lower()
    unit = interval[-1]
    n = int(interval[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400}[unit]


# Every fetcher returns (bars, meta). meta keys:
#   real_volume    bands are properly volume-weighted
#   proxy          instrument actually queried, if not the configured symbol
#   anchor_override force this VWAP anchor regardless of VWAP_ANCHOR
#   approx         bands are a rough estimate -> alert is labeled BACKUP
def _meta(real_volume=True, proxy=None, anchor_override=None, approx=False):
    return {"real_volume": real_volume, "proxy": proxy,
            "anchor_override": anchor_override, "approx": approx}


# --------------------------------------------------------------------------- #
# Source 1: Yahoo (primary, real futures + real volume, 24h session anchor)
# --------------------------------------------------------------------------- #
def fetch_yahoo(symbol: str, interval: str, rng: str, retries: int = 2):
    url = YAHOO_CHART.format(symbol=urllib.parse.quote(symbol))
    url += "?" + urllib.parse.urlencode({"interval": interval, "range": rng})
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as exc:
            last_err = exc
            note = " (rate-limited / bot-blocked)" if exc.code in (429, 999) else ""
            print(f"[{symbol} {interval}] yahoo HTTP {exc.code}{note}, "
                  f"attempt {attempt + 1}/{retries + 1}")
        except Exception as exc:  # timeouts, transient network
            last_err = exc
            print(f"[{symbol} {interval}] yahoo {type(exc).__name__}, "
                  f"attempt {attempt + 1}/{retries + 1}")
    else:
        raise last_err

    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    bars = []
    for t, h, l, c, v in zip(stamps, q["high"], q["low"], q["close"], q["volume"]):
        if None in (h, l, c, v):
            continue
        bars.append((int(t), float(h), float(l), float(c), float(v)))
    return bars, _meta(real_volume=True)


# --------------------------------------------------------------------------- #
# Source 2: Tradier (backup, SPY/QQQ ETF proxy, real volume, RTH only)
# --------------------------------------------------------------------------- #
def _parse_tradier(payload: dict) -> list[tuple[int, float, float, float, float]]:
    series = payload.get("series") or {}
    data = series.get("data")
    if not data:
        return []
    if isinstance(data, dict):
        data = [data]
    bars = []
    for row in data:
        try:
            bars.append((int(row["timestamp"]), float(row["high"]), float(row["low"]),
                         float(row["close"]), float(row["volume"])))
        except (KeyError, TypeError, ValueError):
            continue
    bars.sort()
    return bars


def fetch_tradier(symbol: str, interval: str, rng: str):
    token = _env("TRADIER_TOKEN", "")
    if not token:
        raise RuntimeError("TRADIER_TOKEN not set")
    etf = TRADIER_MAP.get(symbol, symbol)  # pass through if already an equity
    tv_interval = TRADIER_INTERVALS.get(interval)
    if tv_interval is None:
        raise ValueError(f"Tradier has no interval for {interval}")
    base = _env("TRADIER_BASE", "https://api.tradier.com")
    now_et = datetime.now(timezone.utc).astimezone(ET)
    start_et = now_et - timedelta(days=4)  # covers the latest RTH session + slack
    params = urllib.parse.urlencode({
        "symbol": etf, "interval": tv_interval,
        "start": start_et.strftime("%Y-%m-%d %H:%M"),
        "end": now_et.strftime("%Y-%m-%d %H:%M"),
        "session_filter": "open",  # RTH only -> clean regular-session VWAP
    })
    url = f"{base}/v1/markets/timesales?{params}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        payload = json.load(r)
    bars = _parse_tradier(payload)
    # ETFs are RTH-only, so force the RTH anchor; volume is real consolidated vol
    return bars, _meta(real_volume=True, proxy=etf, anchor_override="rth")


# --------------------------------------------------------------------------- #
# Source 3: Dukascopy (last-resort backup, cash index, tick-count weighted)
# --------------------------------------------------------------------------- #
def _state_dir() -> str:
    """Where dedup + Dukascopy hour cache live. Override with STATE_DIR (e.g. a
    systemd StateDirectory) so they survive restarts; defaults to /tmp."""
    return os.environ.get("STATE_DIR", "/tmp")


def _dukas_hour_bytes(inst: str, hour_dt: datetime, now: datetime) -> bytes:
    """Raw .bi5 for one UTC hour. Completed hours are cached (immutable)."""
    url = DUKAS_FEED.format(inst=inst, y=hour_dt.year, m=hour_dt.month - 1,
                            d=hour_dt.day, h=hour_dt.hour)
    completed = hour_dt + timedelta(hours=1) <= now
    cache = os.path.join(_state_dir(), f"dukas_{inst}_{hour_dt:%Y%m%d%H}.bi5")
    if completed:
        try:
            with open(cache, "rb") as fh:
                return fh.read()
        except OSError:
            pass
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raw = b""  # no data this hour (weekend / maintenance break)
        else:
            raise
    if completed and raw:
        try:
            with open(cache, "wb") as fh:
                fh.write(raw)
        except OSError:
            pass
    return raw


def fetch_dukascopy(symbol: str, interval: str, rng: str, hours_back: int = 26):
    inst, scale = DUKAS_MAP[symbol]  # KeyError -> caller logs + fails over
    interval_sec = parse_interval_seconds(interval)
    now = datetime.now(timezone.utc)
    start_hour = (now - timedelta(hours=hours_back)).replace(minute=0, second=0, microsecond=0)

    ohlc: dict[int, list[float]] = {}  # bar_epoch -> [high, low, close, tick_count]
    hour_dt = start_hour
    while hour_dt <= now:
        hour_epoch = int(hour_dt.timestamp())
        raw = _dukas_hour_bytes(inst, hour_dt, now)
        if raw:
            data = lzma.decompress(raw)
            for i in range(0, len(data) - (len(data) % 20), 20):
                ms, ask, bid, _av, _bv = struct.unpack(">IIIff", data[i:i + 20])
                price = ((ask + bid) / 2.0) / scale
                epoch = hour_epoch + ms // 1000
                key = (epoch // interval_sec) * interval_sec
                b = ohlc.get(key)
                if b is None:
                    ohlc[key] = [price, price, price, 1.0]
                else:
                    if price > b[0]:
                        b[0] = price
                    if price < b[1]:
                        b[1] = price
                    b[2] = price
                    b[3] += 1.0
        hour_dt += timedelta(hours=1)
    # No real exchange volume on the index feed -> weight by TICK COUNT, which
    # tracks activity and pulls the SD closer to the real volume-weighted bands.
    bars = [(k, b[0], b[1], b[2], b[3]) for k, b in sorted(ohlc.items())]
    return bars, _meta(real_volume=False, proxy=inst, approx=True)


SOURCE_FUNCS = {"yahoo": fetch_yahoo, "tradier": fetch_tradier, "dukascopy": fetch_dukascopy}


def fetch_bars(symbol: str, interval: str, rng: str, sources: list[str]):
    """Try each source in order; return (source_name, meta, bars)."""
    last_err = None
    for src in sources:
        fn = SOURCE_FUNCS.get(src)
        if fn is None:
            print(f"unknown source '{src}', skipping")
            continue
        try:
            bars, meta = fn(symbol, interval, rng)
            if bars:
                return src, meta, bars
            print(f"[{symbol} {interval}] source {src} returned no bars")
        except Exception as exc:
            last_err = exc
            print(f"[{symbol} {interval}] source {src} failed: "
                  f"{type(exc).__name__}: {exc}")
    if last_err:
        raise last_err
    raise RuntimeError("no data from any source")


# --------------------------------------------------------------------------- #
# VWAP + SD band math
# --------------------------------------------------------------------------- #
def anchor_epoch(latest_epoch: int, anchor: str) -> int:
    tod = ANCHORS.get(anchor, ANCHORS["session"])
    latest_et = datetime.fromtimestamp(latest_epoch, tz=timezone.utc).astimezone(ET)
    candidate = latest_et.replace(hour=tod.hour, minute=tod.minute,
                                  second=0, microsecond=0)
    if latest_et < candidate:
        candidate -= timedelta(days=1)
    return int(candidate.timestamp())


def compute_vwap_bands(bars: list[tuple], anchor: str, min_bars: int) -> dict | None:
    if not bars:
        return None
    start = anchor_epoch(bars[-1][0], anchor)
    session = [b for b in bars if b[0] >= start]
    if len(session) < min_bars:
        return None

    cum_v = cum_pv = cum_pv2 = 0.0
    for _t, h, l, c, v in session:
        tp = (h + l + c) / 3.0
        cum_v += v
        cum_pv += v * tp
        cum_pv2 += v * tp * tp
    if cum_v <= 0:
        return None
    vwap = cum_pv / cum_v
    variance = cum_pv2 / cum_v - vwap * vwap
    sigma = math.sqrt(variance) if variance > 0 else 0.0
    if sigma == 0:
        return None

    price = session[-1][3]
    return {
        "price": price, "vwap": vwap, "sigma": sigma, "z": (price - vwap) / sigma,
        "session_start": start, "session_bars": len(session),
    }


def format_alert(symbol: str, stat: dict, bar_time: int, band_sd: float,
                 anchor: str, interval: str, source: str, meta: dict) -> str:
    z = stat["z"]
    direction = "ABOVE" if z > 0 else "BELOW"
    band = stat["vwap"] + math.copysign(band_sd * stat["sigma"], z)
    dist = abs(band - stat["price"])
    when = datetime.fromtimestamp(bar_time, tz=timezone.utc).astimezone(ET).strftime(
        "%Y-%m-%d %H:%M ET")
    sess = datetime.fromtimestamp(stat["session_start"], tz=timezone.utc).astimezone(
        ET).strftime("%m-%d %H:%M ET")
    arrow = "\U0001F4C8" if z > 0 else "\U0001F4C9"
    proxy = meta.get("proxy")
    title_sym = f"{symbol} (via {proxy})" if proxy and proxy != symbol else symbol
    lines = [
        f"{arrow} *{title_sym}* approaching {band_sd:g} SD from VWAP ({direction})",
        f"z-score: *{z:+.2f}* sigma   ({interval}, {anchor} VWAP, src={source})",
        f"price: `{stat['price']:.2f}`   VWAP: `{stat['vwap']:.2f}`   "
        f"1 sigma: `{stat['sigma']:.2f}`",
        f"{band_sd:g} SD band ({direction.lower()}): `{band:.2f}`   "
        f"distance: `{dist:.2f}` pts",
        f"anchored {sess} ({stat['session_bars']} bars)",
        f"bar: {when}",
    ]
    if source == "tradier":
        lines.append(f"ℹ️ via {proxy} ETF (real volume), RTH-anchored — z-score "
                     f"tracks {symbol} intraday; printed levels are {proxy}, not "
                     "the future.")
    elif meta.get("approx"):
        lines.append("⚠️ BACKUP (" + source + ", cash-index proxy, tick-weighted "
                     "SD): APPROXIMATE — levels offset from the futures and the "
                     "band is a rough estimate; a heads-up to check your chart, "
                     f"not an exact {band_sd:g} SD touch.")
    return "\n".join(lines)


def send_telegram(token: str, chat_id: str, text: str) -> None:
    data = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text, "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(TELEGRAM_API.format(token=token), data=data)
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.load(resp)
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body}")


# --- best-effort, per-bar dedup so warm restarts/retries don't double-alert ----
def _dedup_path() -> str:
    return os.path.join(_state_dir(), "sd_notifier_seen.json")


def _load_seen() -> dict:
    try:
        with open(_dedup_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_seen(seen: dict) -> None:
    try:
        with open(_dedup_path(), "w") as fh:
            json.dump(seen, fh)
    except OSError:
        pass


def check_symbols() -> list[dict]:
    """Evaluate every configured symbol x timeframe; return the alerts emitted."""
    symbols = [s.strip() for s in _env("SYMBOLS", "MES=F,MNQ=F").split(",") if s.strip()]
    intervals = [i.strip() for i in _env("INTERVALS", _env("INTERVAL", "15m,5m")).split(",")
                 if i.strip()]
    sources = [s.strip().lower() for s in _env("SOURCES", "yahoo,tradier,dukascopy").split(",")
               if s.strip()]
    rng = _env("RANGE", "5d")
    anchor = _env("VWAP_ANCHOR", "session").lower()
    band_sd = float(_env("BAND_SD", "3.0"))
    threshold = float(_env("SD_THRESHOLD", "2.8"))
    min_bars = int(_env("MIN_BARS", "3"))
    warmup_min = int(_env("WARMUP_MIN", "0"))  # skip alerts for N min after anchor
    max_bar_age = int(_env("MAX_BAR_AGE_MIN", "0"))  # skip if latest bar older than this
    use_closed = _env("USE_LAST_CLOSED_BAR", "1") == "1"
    dry_run = _env("DRY_RUN", "0") == "1"

    if anchor not in ANCHORS:
        print(f"unknown VWAP_ANCHOR '{anchor}', falling back to 'session'")
        anchor = "session"

    token = _env("TELEGRAM_BOT_TOKEN", "")
    chat_id = _env("TELEGRAM_CHAT_ID", "")
    if not dry_run and (not token or not chat_id):
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set "
                           "(or use DRY_RUN=1 to print).")

    seen = _load_seen()
    alerts = []
    for symbol in symbols:
        for interval in intervals:
            tag = f"{symbol} {interval}"
            try:
                source, meta, bars = fetch_bars(symbol, interval, rng, sources)
            except Exception as exc:
                print(f"[{tag}] all sources failed: {type(exc).__name__}: {exc}")
                continue

            if use_closed and len(bars) > 1:
                bars = bars[:-1]  # drop the still-forming bar
            eff_anchor = meta.get("anchor_override") or anchor
            stat = compute_vwap_bands(bars, eff_anchor, min_bars)
            if stat is None:
                print(f"[{tag}] not enough session data / flat VWAP; skipping")
                continue

            bar_time = bars[-1][0]
            bar_age_min = (datetime.now(timezone.utc).timestamp() - bar_time) / 60.0
            if max_bar_age and bar_age_min > max_bar_age:
                print(f"[{tag}] src={source} latest bar is {bar_age_min:.0f} min old "
                      f"(> {max_bar_age}); stale, skip")
                continue
            elapsed_min = (bar_time - stat["session_start"]) / 60.0
            if elapsed_min < warmup_min:
                print(f"[{tag}] src={source} warming up "
                      f"({elapsed_min:.0f}/{warmup_min} min since anchor); skip")
                continue
            triggered = abs(stat["z"]) >= threshold
            print(f"[{tag}] src={source} anchor={eff_anchor} z={stat['z']:+.2f} "
                  f"price={stat['price']:.2f} vwap={stat['vwap']:.2f} "
                  f"sigma={stat['sigma']:.2f} ({stat['session_bars']} bars) "
                  f"{'ALERT' if triggered else 'ok'}")
            if not triggered:
                continue

            dedup_key = f"{symbol}:{interval}"
            if seen.get(dedup_key) == bar_time:
                print(f"[{tag}] already alerted for bar {bar_time}; skipping")
                continue

            text = format_alert(symbol, stat, bar_time, band_sd, eff_anchor, interval,
                                source, meta)
            if dry_run:
                print("--- ALERT (dry-run) ---\n" + text + "\n-----------------------")
            else:
                send_telegram(token, chat_id, text)
            seen[dedup_key] = bar_time
            alerts.append({"symbol": symbol, "interval": interval, "source": source,
                           "proxy": meta.get("proxy"), "z": round(stat["z"], 3),
                           "bar_time": bar_time})

    _save_seen(seen)
    return alerts


def lambda_handler(event, context):  # noqa: ARG001 - AWS signature
    alerts = check_symbols()
    return {"statusCode": 200,
            "body": json.dumps({"alerts_sent": len(alerts), "alerts": alerts})}


def run_loop(interval_sec: int = 60) -> None:
    """Self-scheduling loop for always-on hosts (Oracle/GCP free VM, etc.) so no
    cron is needed. Checks every `interval_sec`, aligned to the wall clock + a
    few seconds so the latest 5m/15m bar has closed. One cycle's failure never
    kills the loop, and the /tmp dedup file persists across cycles + restarts."""
    print(f"[loop] starting; checking every {interval_sec}s "
          f"(Ctrl-C / systemd stop to exit)")
    while True:
        try:
            n = len(check_symbols())
            print(f"[loop {datetime.now(timezone.utc):%H:%M}Z] {n} new alert(s)")
        except Exception as exc:
            print(f"[loop] cycle error: {type(exc).__name__}: {exc}")
        now = wallclock.time()
        next_t = (int(now // interval_sec) + 1) * interval_sec + 5  # +5s past close
        wallclock.sleep(max(1, next_t - now))


if __name__ == "__main__":
    import sys
    if "--loop" in sys.argv:
        i = sys.argv.index("--loop")
        secs = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 60
        run_loop(secs)
    else:
        result = check_symbols()
        print(f"\nDone. {len(result)} alert(s) sent.")
