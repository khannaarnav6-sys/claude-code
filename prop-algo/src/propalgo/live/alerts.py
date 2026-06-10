"""Discord webhook alerts. Set DISCORD_WEBHOOK_URL; --dry-run prints instead."""
from __future__ import annotations

import os

import requests

from ..strategies.base import Signal


def format_signal(sig: Signal, contracts: int) -> str:
    side = "LONG" if sig.side > 0 else "SHORT"
    return (
        f"**{side} {sig.symbol} x{contracts}**  [{sig.strategy} / {sig.grade}]\n"
        f"entry ~ `{sig.entry_ref:.2f}`  stop `{sig.stop:.2f}`  target `{sig.target:.2f}`\n"
        f"{sig.note}\n"
        f"_Manual execution only — place the order yourself in Tradovate._"
    )


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
