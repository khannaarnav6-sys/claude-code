#!/usr/bin/env python3
"""Telegram notifier: alerts when MES / MNQ price approaches the 3rd VWAP SD band.

Single file, ZERO third-party dependencies (standard library only) so it drops
straight into AWS Lambda with no layers or packaging step. It pulls free 15m
bars from Yahoo Finance's chart API and reproduces TradingView's *VWAP with
standard-deviation bands* indicator: a session-anchored, volume-weighted
average price (VWAP) plus bands at +/- N volume-weighted standard deviations.
It sends a Telegram message when the latest price gets close to the 3rd SD band
in either direction.

"Approaching the 3 SD band" => alert fires once |z| >= SD_THRESHOLD (default
2.8), where z = (price - VWAP) / sigma_vwap, i.e. just before price reaches
+/-3 sigma.

How the bands are computed (matches TradingView's VWAP), anchored to the
current session and using the typical price hlc3 = (high+low+close)/3:
    VWAP    = sum(volume * hlc3) / sum(volume)
    sigma   = sqrt( sum(volume * hlc3^2) / sum(volume) - VWAP^2 )
    band_k  = VWAP +/- k * sigma          (k = BAND_SD, default 3)

------------------------------------------------------------------------------
DEPLOY ON AWS LAMBDA
------------------------------------------------------------------------------
  * Runtime: Python 3.11+, handler = lambda_sd_notifier.lambda_handler
  * No dependencies, so just zip this one file (or paste it into the console).
  * Trigger: EventBridge (CloudWatch Events) schedule. To check on every 15m
    bar close during US futures RTH, e.g. rate(15 minutes), or a cron like
    cron(2/15 13-20 ? * MON-FRI *) to run a couple minutes after each close.
  * Set these environment variables:
      TELEGRAM_BOT_TOKEN   (required)  from @BotFather
      TELEGRAM_CHAT_ID     (required)  your chat/channel id (talk to @userinfobot)
      SYMBOLS              default "MES=F,MNQ=F"
      INTERVAL             default "15m"
      RANGE                default "5d"    (Yahoo history window to pull)
      VWAP_ANCHOR          default "session"  where the VWAP resets each day:
                             "session" = 18:00 ET (CME Globex open, TV default
                                          for futures)
                             "rth"     = 09:30 ET (regular session open)
                             "day"     = 00:00 ET
      BAND_SD              default "3.0"  the band level you care about (sigmas)
      SD_THRESHOLD         default "2.8"  |z| at/above this fires the alert
      MIN_BARS             default "3"    bars needed since the anchor to bother
      USE_LAST_CLOSED_BAR  default "1"    ignore the still-forming current bar
  * Give the function ~128MB and a 30s timeout; outbound internet (default
    Lambda networking, or a NAT gateway if you place it in a VPC).

Run locally to test:  TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
                      python lambda_sd_notifier.py
Add DRY_RUN=1 to print alerts instead of sending them.
"""
from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
ET = ZoneInfo("America/New_York")

# session anchor name -> local ET time of day the VWAP resets at
ANCHORS = {
    "session": time(18, 0),   # CME Globex open (prior calendar day)
    "rth": time(9, 30),       # regular trading hours open
    "day": time(0, 0),        # midnight ET
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def fetch_bars(symbol: str, interval: str, rng: str) -> list[tuple[int, float, float, float, float]]:
    """Return complete bars as (epoch, high, low, close, volume), oldest first.

    Bars missing any of high/low/close/volume (gaps) are dropped so VWAP and its
    variance only accumulate real prints.
    """
    url = YAHOO_CHART.format(symbol=urllib.parse.quote(symbol))
    url += "?" + urllib.parse.urlencode({"interval": interval, "range": rng})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)

    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    bars = []
    for t, h, l, c, v in zip(stamps, q["high"], q["low"], q["close"], q["volume"]):
        if None in (h, l, c, v):
            continue
        bars.append((int(t), float(h), float(l), float(c), float(v)))
    return bars


def anchor_epoch(latest_epoch: int, anchor: str) -> int:
    """Epoch (UTC seconds) of the session anchor active for `latest_epoch`."""
    tod = ANCHORS.get(anchor, ANCHORS["session"])
    latest_et = datetime.fromtimestamp(latest_epoch, tz=timezone.utc).astimezone(ET)
    candidate = latest_et.replace(hour=tod.hour, minute=tod.minute,
                                  second=0, microsecond=0)
    if latest_et < candidate:           # haven't reached today's reset yet
        candidate -= timedelta(days=1)  # so we're still in yesterday's session
    return int(candidate.timestamp())


def compute_vwap_bands(bars: list[tuple], anchor: str, min_bars: int) -> dict | None:
    """Session-anchored VWAP + volume-weighted SD bands over the latest session.

    Returns None if there aren't enough bars since the anchor or volume/variance
    is degenerate.
    """
    if not bars:
        return None
    start = anchor_epoch(bars[-1][0], anchor)
    session = [b for b in bars if b[0] >= start]
    if len(session) < min_bars:
        return None

    cum_v = cum_pv = cum_pv2 = 0.0
    for _t, h, l, c, v in session:
        tp = (h + l + c) / 3.0          # typical price (hlc3), TradingView source
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

    price = session[-1][3]              # latest close
    z = (price - vwap) / sigma
    return {
        "price": price,
        "vwap": vwap,
        "sigma": sigma,
        "z": z,
        "session_start": start,
        "session_bars": len(session),
    }


def format_alert(symbol: str, stat: dict, bar_time: int, band_sd: float,
                 anchor: str, interval: str) -> str:
    z = stat["z"]
    direction = "ABOVE" if z > 0 else "BELOW"
    band = stat["vwap"] + math.copysign(band_sd * stat["sigma"], z)
    dist = abs(band - stat["price"])
    when = datetime.fromtimestamp(bar_time, tz=timezone.utc).astimezone(ET).strftime(
        "%Y-%m-%d %H:%M ET")
    sess = datetime.fromtimestamp(stat["session_start"], tz=timezone.utc).astimezone(
        ET).strftime("%m-%d %H:%M ET")
    arrow = "\U0001F4C8" if z > 0 else "\U0001F4C9"  # chart up / chart down
    return (
        f"{arrow} *{symbol}* approaching {band_sd:g} SD from VWAP ({direction})\n"
        f"z-score: *{z:+.2f}* sigma   ({interval}, {anchor} VWAP)\n"
        f"price: `{stat['price']:.2f}`   VWAP: `{stat['vwap']:.2f}`   "
        f"1 sigma: `{stat['sigma']:.2f}`\n"
        f"{band_sd:g} SD band ({direction.lower()}): `{band:.2f}`   "
        f"distance: `{dist:.2f}` pts\n"
        f"session anchored {sess} ({stat['session_bars']} bars)\n"
        f"bar: {when}"
    )


def send_telegram(token: str, chat_id: str, text: str) -> None:
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(TELEGRAM_API.format(token=token), data=data)
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.load(resp)
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body}")


# --- best-effort, per-bar dedup so warm Lambda retries don't double-alert -----
_DEDUP_PATH = "/tmp/sd_notifier_seen.json"


def _load_seen() -> dict:
    try:
        with open(_DEDUP_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_seen(seen: dict) -> None:
    try:
        with open(_DEDUP_PATH, "w") as fh:
            json.dump(seen, fh)
    except OSError:
        pass  # /tmp not writable -> just skip dedup, never fatal


def check_symbols() -> list[dict]:
    """Evaluate every configured symbol; return the list of alerts emitted."""
    symbols = [s.strip() for s in _env("SYMBOLS", "MES=F,MNQ=F").split(",") if s.strip()]
    interval = _env("INTERVAL", "15m")
    rng = _env("RANGE", "5d")
    anchor = _env("VWAP_ANCHOR", "session").lower()
    band_sd = float(_env("BAND_SD", "3.0"))
    threshold = float(_env("SD_THRESHOLD", "2.8"))
    min_bars = int(_env("MIN_BARS", "3"))
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
        try:
            bars = fetch_bars(symbol, interval, rng)
        except Exception as exc:  # one bad symbol shouldn't sink the others
            print(f"[{symbol}] fetch failed: {type(exc).__name__}: {exc}")
            continue

        if use_closed and len(bars) > 1:
            bars = bars[:-1]  # drop the still-forming bar
        stat = compute_vwap_bands(bars, anchor, min_bars)
        if stat is None:
            print(f"[{symbol}] not enough session data / flat VWAP; skipping")
            continue

        bar_time = bars[-1][0]
        triggered = abs(stat["z"]) >= threshold
        print(f"[{symbol}] z={stat['z']:+.2f} price={stat['price']:.2f} "
              f"vwap={stat['vwap']:.2f} sigma={stat['sigma']:.2f} "
              f"({stat['session_bars']} bars) {'ALERT' if triggered else 'ok'}")
        if not triggered:
            continue

        if seen.get(symbol) == bar_time:
            print(f"[{symbol}] already alerted for bar {bar_time}; skipping")
            continue

        text = format_alert(symbol, stat, bar_time, band_sd, anchor, interval)
        if dry_run:
            print("--- ALERT (dry-run) ---\n" + text + "\n-----------------------")
        else:
            send_telegram(token, chat_id, text)
        seen[symbol] = bar_time
        alerts.append({"symbol": symbol, "z": round(stat["z"], 3),
                       "bar_time": bar_time})

    _save_seen(seen)
    return alerts


def lambda_handler(event, context):  # noqa: ARG001 - AWS signature
    alerts = check_symbols()
    return {"statusCode": 200,
            "body": json.dumps({"alerts_sent": len(alerts), "alerts": alerts})}


if __name__ == "__main__":
    result = check_symbols()
    print(f"\nDone. {len(result)} alert(s) sent.")
