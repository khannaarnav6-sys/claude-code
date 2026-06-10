"""Online signal loop.

Wakes shortly after each 15m bar close, pulls the latest Yahoo futures bars,
re-runs the strategies on today's session so far, and alerts on new signals.
State (which signals already fired today) persists to JSON so restarts are
safe. Yahoo futures data is ~10 min delayed — signals are evaluated on bar
closes, so the alert lands within minutes of the bar completing.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..backtest.engine import (ET, daily_atr_series, rth_sessions, simulate_trade,
                               size_micros)
from ..config import Config, PROJECT_ROOT
from ..data.yahoo import YahooSource
from ..strategies import build_strategies
from .alerts import format_signal, send_alert


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"fired": []}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def _signal_key(day, sig) -> str:
    return f"{day}:{sig.symbol}:{sig.strategy}"


def run_cycle(cfg: Config, risk_frac: float, dry_run: bool = False,
              state_path: Path | None = None) -> int:
    """One polling cycle; returns number of new alerts sent."""
    state_path = state_path or PROJECT_ROOT / cfg.live.get("state_file", "live_state.json")
    state = _load_state(state_path)
    symbols = cfg.live.get("instruments", ["NQ", "ES"])
    source = YahooSource(cfg.instruments)
    strategies = build_strategies(cfg.strategies)
    aplus_mult = float(cfg.sizing.get("aplus_multiplier", 1.5))

    sent = 0
    start = datetime.now(timezone.utc) - timedelta(days=40)
    # gather today's signals across all symbols, then walk them in time order
    # with the same one-position/veto logic as the backtest engine
    candidates = []   # (ts, bar_idx, sig, session, today)
    for sym in symbols:
        try:
            bars = source.fetch(sym, start)
        except Exception as exc:
            print(f"[live] data fetch failed for {sym}: {exc}")
            continue
        sessions = rth_sessions(bars)
        if not sessions:
            continue
        atrs = daily_atr_series(sessions)
        today = sorted(sessions)[-1]
        session = sessions[today]
        atr = atrs.get(today) or 0.0
        for strat in strategies:
            for bar_idx, sig in strat.on_session(sym, session, atr):
                candidates.append((session.index[bar_idx], bar_idx, sig, session, today))

    candidates.sort(key=lambda c: c[0])
    busy_until = None
    for ts, bar_idx, sig, session, today in candidates:
        if busy_until is not None and ts < busy_until:
            continue
        instr = cfg.instruments[sig.symbol]
        if bar_idx < len(session) - 2:
            # earlier today: replay it (trade or phantom veto) to know how
            # long it occupies the account
            trade = simulate_trade(session, bar_idx + 1, sig, instr, cfg.costs)
            busy_until = trade.exit_time
            continue
        if sig.action == "veto":
            # live breakdown just fired: hostile window for the rest of the
            # day (next cycles will re-derive the phantom exit as bars arrive)
            busy_until = session.index[-1] + timedelta(hours=8)
            continue
        # signal on the most recent closed bar -> actionable now
        key = _signal_key(today, sig)
        if key in state["fired"]:
            continue
        budget = risk_frac * cfg.account.trailing_drawdown
        if sig.grade == "A+":
            budget *= aplus_mult
        risk_per_mini = sig.risk_points * instr.point_value
        micros = size_micros(budget, risk_per_mini, cfg.account.max_contracts)
        if micros == 0:
            continue
        send_alert(format_signal(sig, micros, risk_per_mini / 10.0),
                   cfg.live.get("webhook_env", "DISCORD_WEBHOOK_URL"), dry_run)
        state["fired"].append(key)
        sent += 1
        busy_until = session.index[-1] + timedelta(hours=8)  # one position at a time
    state["fired"] = state["fired"][-200:]
    state["last_cycle"] = datetime.now(timezone.utc).isoformat()
    _save_state(state_path, state)
    return sent


def run_forever(cfg: Config, risk_frac: float, dry_run: bool = False) -> None:
    interval = int(cfg.live.get("poll_interval_min", 15)) * 60
    print(f"[live] polling every {interval // 60}m for {cfg.live.get('instruments')}; "
          f"{'DRY RUN' if dry_run else 'alerts live'}")
    while True:
        now = time.time()
        next_close = (int(now // interval) + 1) * interval + 30  # 30s after bar close
        time.sleep(max(1, next_close - now))
        try:
            n = run_cycle(cfg, risk_frac, dry_run)
            stamp = datetime.now(timezone.utc).strftime("%H:%M")
            print(f"[live {stamp}Z] cycle done, {n} new alert(s)")
        except Exception as exc:
            print(f"[live] cycle error: {exc}")
