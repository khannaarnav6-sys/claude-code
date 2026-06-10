"""Discord webhook alerts. Set DISCORD_WEBHOOK_URL; --dry-run prints instead."""
from __future__ import annotations

import os

import requests

from ..strategies.base import Signal


def format_signal(sig: Signal, micros: int, risk_per_micro: float | None = None) -> str:
    side = "LONG" if sig.side > 0 else "SHORT"
    minis, rem = divmod(micros, 10)
    size = f"{micros} micros" if minis == 0 else (
        f"{minis} minis" if rem == 0 else f"{minis} minis + {rem} micros")
    lines = [
        f"**{side} {sig.symbol} x {size}**  [{sig.strategy} / {sig.grade}]",
        f"entry ~ `{sig.entry_ref:.2f}`  stop `{sig.stop:.2f}`  target `{sig.target:.2f}`",
        sig.note,
    ]
    if risk_per_micro:
        total = micros * risk_per_micro
        lines.append(
            f"risk ≈ ${total:,.0f} (${risk_per_micro:,.0f}/micro) — taper: if you're "
            f"less than ${total:,.0f} from the target, take ~remaining/"
            f"${risk_per_micro:,.0f} micros instead.")
    lines.append("_Manual execution only — place the order yourself in Tradovate._")
    return "\n".join(lines)


def send_alert(message: str, webhook_env: str = "DISCORD_WEBHOOK_URL",
               dry_run: bool = False) -> bool:
    if dry_run:
        print("--- ALERT (dry-run) ---")
        print(message)
        print("-----------------------")
        return True
    url = os.environ.get(webhook_env, "")
    if not url:
        print(f"WARNING: {webhook_env} not set; printing alert instead:\n{message}")
        return False
    resp = requests.post(url, json={"content": message}, timeout=10)
    resp.raise_for_status()
    return True
