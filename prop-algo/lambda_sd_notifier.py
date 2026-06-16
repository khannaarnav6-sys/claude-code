#!/usr/bin/env python3
"""Telegram notifier: alerts when MES / MNQ price action approaches the 3-sigma band.

Single file, ZERO third-party dependencies (standard library only) so it drops
straight into AWS Lambda with no layers or packaging step. It pulls free 15m
bars from Yahoo Finance's chart API, computes a Bollinger-style z-score of the
latest close over a rolling window, and sends a Telegram message when the price
gets close to +/-3 standard deviations from its rolling mean.

"Approaching 3 SD" => alert fires once |z| >= SD_THRESHOLD (default 2.8), i.e.
just before price reaches the 3-sigma band, in either direction.

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
      RANGE                default "5d"   (Yahoo history window to pull)
      LOOKBACK             default "20"   (bars in the rolling mean/std window)
      SD_THRESHOLD         default "2.8"  (|z| at/above this fires the alert)
      USE_LAST_CLOSED_BAR  default "1"    (ignore the still-forming current bar)
  * Give the function ~128MB and a 30s timeout; outbound internet (default
    Lambda networking, or a NAT gateway if you place it in a VPC).

Run locally to test:  TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
                      python lambda_sd_notifier.py
Add DRY_RUN=1 to print alerts instead of sending them.
"""
from __future__ import annotations

import json
import os
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def fetch_bars(symbol: str, interval: str, rng: str) -> tuple[list[int], list[float]]:
    """Return (timestamps, closes) of completed+forming bars from Yahoo.

    Bars with a null close (gaps / not-yet-traded) are dropped so the rolling
    statistics only see real prints.
    """
    url = YAHOO_CHART.format(symbol=urllib.parse.quote(symbol))
    url += "?" + urllib.parse.urlencode({"interval": interval, "range": rng})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)

    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    ts, px = [], []
    for t, c in zip(stamps, closes):
        if c is not None:
            ts.append(int(t))
            px.append(float(c))
    return ts, px


def compute_zscore(closes: list[float], lookback: int) -> dict | None:
    """Bollinger-style z-score of the last close over the rolling window.

    Returns None if there isn't enough data or the window is flat (std == 0).
    """
    if len(closes) <= lookback:
        return None
    window = closes[-lookback:]
    last = window[-1]
    mean = statistics.fmean(window)
    # population std matches the usual Bollinger Band convention
    std = statistics.pstdev(window)
    if std == 0:
        return None
    z = (last - mean) / std
    return {
        "price": last,
        "mean": mean,
        "std": std,
        "z": z,
        "upper_3sd": mean + 3 * std,
        "lower_3sd": mean - 3 * std,
    }


def format_alert(symbol: str, stat: dict, bar_time: int, lookback: int,
                 interval: str) -> str:
    z = stat["z"]
    direction = "ABOVE" if z > 0 else "BELOW"
    band = stat["upper_3sd"] if z > 0 else stat["lower_3sd"]
    dist = abs(band - stat["price"])
    when = datetime.fromtimestamp(bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    arrow = "\U0001F4C8" if z > 0 else "\U0001F4C9"  # chart up / chart down
    return (
        f"{arrow} *{symbol}* approaching 3 SD ({direction})\n"
        f"z-score: *{z:+.2f}* sigma ({interval}, {lookback}-bar window)\n"
        f"price: `{stat['price']:.2f}`   mean: `{stat['mean']:.2f}`   "
        f"1 sigma: `{stat['std']:.2f}`\n"
        f"3 sigma band: `{stat['lower_3sd']:.2f}` / `{stat['upper_3sd']:.2f}`\n"
        f"distance to {direction.lower()} band: `{dist:.2f}` pts\n"
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
    lookback = int(_env("LOOKBACK", "20"))
    threshold = float(_env("SD_THRESHOLD", "2.8"))
    use_closed = _env("USE_LAST_CLOSED_BAR", "1") == "1"
    dry_run = _env("DRY_RUN", "0") == "1"

    token = _env("TELEGRAM_BOT_TOKEN", "")
    chat_id = _env("TELEGRAM_CHAT_ID", "")
    if not dry_run and (not token or not chat_id):
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set "
                           "(or use DRY_RUN=1 to print).")

    seen = _load_seen()
    alerts = []
    for symbol in symbols:
        try:
            stamps, closes = fetch_bars(symbol, interval, rng)
        except Exception as exc:  # one bad symbol shouldn't sink the others
            print(f"[{symbol}] fetch failed: {type(exc).__name__}: {exc}")
            continue

        if use_closed and len(closes) > 1:
            stamps, closes = stamps[:-1], closes[:-1]  # drop the forming bar
        stat = compute_zscore(closes, lookback)
        if stat is None:
            print(f"[{symbol}] not enough data / flat window; skipping")
            continue

        bar_time = stamps[-1]
        triggered = abs(stat["z"]) >= threshold
        print(f"[{symbol}] z={stat['z']:+.2f} price={stat['price']:.2f} "
              f"band=({stat['lower_3sd']:.2f}/{stat['upper_3sd']:.2f}) "
              f"{'ALERT' if triggered else 'ok'}")
        if not triggered:
            continue

        dedup_key = f"{symbol}:{bar_time}"
        if seen.get(symbol) == bar_time:
            print(f"[{symbol}] already alerted for bar {bar_time}; skipping")
            continue

        text = format_alert(symbol, stat, bar_time, lookback, interval)
        if dry_run:
            print("--- ALERT (dry-run) ---\n" + text + "\n-----------------------")
        else:
            send_telegram(token, chat_id, text)
        seen[symbol] = bar_time
        alerts.append({"symbol": symbol, "z": stat["z"], "bar_time": bar_time,
                       "key": dedup_key})

    _save_seen(seen)
    return alerts


def lambda_handler(event, context):  # noqa: ARG001 - AWS signature
    alerts = check_symbols()
    return {"statusCode": 200,
            "body": json.dumps({"alerts_sent": len(alerts), "alerts": alerts})}


if __name__ == "__main__":
    result = check_symbols()
    print(f"\nDone. {len(result)} alert(s) sent.")
