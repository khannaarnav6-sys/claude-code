"""Bar-by-bar backtest over 15m bars + sequential Apex eval simulation.

Two layers:
1. `generate_trades` — runs strategies over RTH sessions and produces
   TradeRecords with *per-contract* PnL and per-bar equity excursions, so the
   same trade history can be replayed at any contract size.
2. `run_eval_sequence` — replays day-ordered trades through EvalTracker,
   starting a fresh $50K eval after every pass/bust, tallying attempts.

Conservative assumptions: entries fill at next bar open + slippage; if a bar
contains both stop and target, the stop fills first; intra-bar adverse
excursion is checked against the trailing threshold before the favorable
excursion can raise the high-water mark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..config import AccountRules, Costs, Instrument
from ..rules.apex import EvalStatus, EvalTracker
from ..strategies.base import Signal, Strategy

RTH_START = "09:30"
RTH_END = "16:00"
ET = "America/New_York"


def rth_sessions(bars: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split UTC 15m bars into ET regular-trading-hours sessions, attaching
    the prior session's close (for gap logic) via DataFrame.attrs."""
    et_index = bars.index.tz_convert(ET)
    rth = bars[(et_index.time >= pd.Timestamp(RTH_START).time())
               & (et_index.time < pd.Timestamp(RTH_END).time())]
    sessions: dict[date, pd.DataFrame] = {}
    prev_close = None
    for day, sess in rth.groupby(rth.index.tz_convert(ET).date):
        sess = sess.copy()
        sess.attrs["prev_rth_close"] = prev_close
        if len(sess) >= 8:  # skip half days / data gaps
            sessions[day] = sess
        prev_close = sess["close"].iloc[-1]
    return sessions


def daily_atr_series(sessions: dict[date, pd.DataFrame], period: int = 14) -> dict[date, float]:
    """ATR of daily RTH ranges, shifted so each day only sees prior days."""
    days = sorted(sessions)
    rows = []
    for d in days:
        s = sessions[d]
        rows.append((d, s["high"].max(), s["low"].min(), s["close"].iloc[-1]))
    df = pd.DataFrame(rows, columns=["day", "high", "low", "close"]).set_index("day")
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=5).mean().shift(1)
    return atr.to_dict()


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    grade: str
    side: int
    day: date
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    exit_reason: str
    pnl_per_contract: float            # $ after commission+slippage, 1 contract
    risk_per_contract: float = 0.0     # |entry - stop| in $ for 1 (mini) contract
    bar_low_pnl: list[float] = field(default_factory=list)   # per-bar worst unrealized $/contract
    bar_high_pnl: list[float] = field(default_factory=list)  # per-bar best unrealized $/contract


def size_micros(risk_budget: float, risk_per_contract: float, max_contracts: int) -> int:
    """Contracts come in 0.1-mini steps because Apex allows 10 micros per mini.

    Returns the position size in micro units (10 micros == 1 mini)."""
    if risk_per_contract <= 0:
        return 0
    micros = int(risk_budget / (risk_per_contract / 10.0))
    return max(0, min(micros, max_contracts * 10))


def simulate_trade(session: pd.DataFrame, entry_idx: int, sig: Signal,
                   instr: Instrument, costs: Costs,
                   mgmt: dict | None = None) -> TradeRecord:
    """`mgmt` options (all optional):
      breakeven_at_r: once price has moved this many R in favor, stop moves to
        entry (applied at bar close — a stop can't be saved by the same bar
        that would have moved it).
      time_stop_bars / time_stop_min_r: exit at close after N bars unless the
        trade has reached min_r R.
    """
    mgmt = mgmt or {}
    be_at_r = mgmt.get("breakeven_at_r")
    ts_bars = mgmt.get("time_stop_bars")
    ts_min_r = float(mgmt.get("time_stop_min_r", 0.5))

    slip = costs.slippage_ticks * instr.tick_size
    entry = float(session["open"].iloc[entry_idx]) + sig.side * slip
    pv = instr.point_value
    risk_pts = abs(entry - sig.stop)
    low_pnl, high_pnl = [], []
    exit_price, exit_reason, exit_i = None, "eod", len(session) - 1
    stop = sig.stop

    for i in range(entry_idx, len(session)):
        bar = session.iloc[i]
        if sig.side > 0:
            worst, best = (bar["low"] - entry) * pv, (bar["high"] - entry) * pv
            hit_stop, hit_target = bar["low"] <= stop, bar["high"] >= sig.target
        else:
            worst, best = (entry - bar["high"]) * pv, (entry - bar["low"]) * pv
            hit_stop, hit_target = bar["high"] >= stop, bar["low"] <= sig.target
        if hit_stop:  # stop assumed first when both hit in one bar
            exit_price = stop - sig.side * slip
            worst = min(worst, (exit_price - entry) * pv * sig.side)
            low_pnl.append(worst)
            high_pnl.append(min(best, 0.0) if hit_stop and not hit_target else best)
            exit_reason = "breakeven" if stop != sig.stop else "stop"
            exit_i = i
            break
        low_pnl.append(worst)
        high_pnl.append(best)
        if hit_target:
            exit_price = sig.target - sig.side * slip
            exit_reason, exit_i = "target", i
            break
        bars_held = i - entry_idx + 1
        unrealized_r = (bar["close"] - entry) * sig.side / risk_pts if risk_pts > 0 else 0.0
        if ts_bars and bars_held >= ts_bars and unrealized_r < ts_min_r:
            exit_price = float(bar["close"]) - sig.side * slip
            exit_reason, exit_i = "time", i
            break
        if be_at_r and risk_pts > 0 and unrealized_r >= 0 and \
                best >= be_at_r * risk_pts * pv:
            stop = max(stop, entry) if sig.side > 0 else min(stop, entry)

    if exit_price is None:
        exit_price = float(session["close"].iloc[-1]) - sig.side * slip
    pnl = (exit_price - entry) * sig.side * pv - costs.commission_rt
    # fold commission into excursions so the eval tracker sees net equity
    low_pnl = [p - costs.commission_rt for p in low_pnl]
    high_pnl = [p - costs.commission_rt for p in high_pnl]
    return TradeRecord(
        symbol=sig.symbol, strategy=sig.strategy, grade=sig.grade, side=sig.side,
        day=session.index[0].tz_convert(ET).date(),
        entry_time=session.index[entry_idx], exit_time=session.index[exit_i],
        entry=entry, exit=exit_price, exit_reason=exit_reason,
        pnl_per_contract=pnl, risk_per_contract=abs(entry - sig.stop) * pv,
        bar_low_pnl=low_pnl, bar_high_pnl=high_pnl,
    )


def generate_trades(bars_by_symbol: dict[str, pd.DataFrame],
                    strategies: list[Strategy],
                    instruments: dict[str, Instrument],
                    costs: Costs,
                    mgmt: dict | None = None) -> list[TradeRecord]:
    """One position at a time, account-wide (high-conviction sizing needs the
    full drawdown budget behind each trade). Signals are taken in time order;
    overlapping ones are skipped."""
    sessions_by_symbol = {s: rth_sessions(b) for s, b in bars_by_symbol.items()}
    atr_by_symbol = {s: daily_atr_series(sess) for s, sess in sessions_by_symbol.items()}
    all_days = sorted({d for sess in sessions_by_symbol.values() for d in sess})

    trades: list[TradeRecord] = []
    for day in all_days:
        candidates: list[tuple[pd.Timestamp, int, Signal, pd.DataFrame]] = []
        for sym, sessions in sessions_by_symbol.items():
            if day not in sessions:
                continue
            session = sessions[day]
            atr = atr_by_symbol[sym].get(day) or 0.0
            if not np.isfinite(atr):
                atr = 0.0
            for strat in strategies:
                for bar_idx, sig in strat.on_session(sym, session, atr):
                    candidates.append((session.index[bar_idx], bar_idx, sig, session))
        candidates.sort(key=lambda c: c[0])
        busy_until = None
        for ts, bar_idx, sig, session in candidates:
            if busy_until is not None and ts < busy_until:
                continue
            trade = simulate_trade(session, bar_idx + 1, sig, instruments[sig.symbol], costs, mgmt)
            busy_until = trade.exit_time
            if sig.action != "veto":  # veto signals block the window untraded
                trades.append(trade)
    return trades


@dataclass
class EvalAttempt:
    passed: bool
    trading_days: int
    n_trades: int
    final_balance: float
    start_day: date
    end_day: date


def run_eval_sequence(trades: list[TradeRecord], rules: AccountRules,
                      risk_frac: float, aplus_multiplier: float = 1.5,
                      taper: bool = True) -> list[EvalAttempt]:
    """Replay day-ordered trades through back-to-back eval attempts.

    Sizing: each trade risks `risk_frac` of the trailing drawdown (A+ signals
    risk `risk_frac * aplus_multiplier`), converted to micro-granular size via
    the trade's stop distance. With `taper`, the budget never much exceeds the
    remaining distance to the target — no point risking $625 when $200 of
    profit finishes the eval. Once the target is hit but min trading days
    aren't met, later signals drop to 1 micro to log days without risking the
    pass.
    """
    attempts: list[EvalAttempt] = []
    tracker = EvalTracker(rules)
    n_trades, start_day = 0, None

    for tr in sorted(trades, key=lambda t: t.entry_time):
        if start_day is None:
            start_day = tr.day
        budget = risk_frac * rules.trailing_drawdown
        if tr.grade == "A+":
            budget *= aplus_multiplier
        if taper:
            remaining = max(0.0, tracker.target_balance - tracker.balance)
            budget = min(budget, max(remaining, 0.04 * rules.trailing_drawdown))
        micros = size_micros(budget, tr.risk_per_contract, rules.max_contracts)
        if tracker.target_pending():
            micros = min(micros, 1)
        if micros == 0:
            continue  # stop too wide even for 1 micro at this budget
        scale = micros / 10.0
        tracker.mark_trading_day(tr.day)
        n_trades += 1
        for lo, hi in zip(tr.bar_low_pnl, tr.bar_high_pnl):
            if tracker.on_equity_extremes(tracker.balance + scale * lo,
                                          tracker.balance + scale * hi) is EvalStatus.BUSTED:
                break
        if tracker.status is EvalStatus.ACTIVE:
            tracker.on_trade_closed(scale * tr.pnl_per_contract)
        if tracker.status is not EvalStatus.ACTIVE:
            attempts.append(EvalAttempt(
                passed=tracker.status is EvalStatus.PASSED,
                trading_days=len(tracker.trading_days), n_trades=n_trades,
                final_balance=tracker.balance, start_day=start_day, end_day=tr.day,
            ))
            tracker = EvalTracker(rules)
            n_trades, start_day = 0, None
    return attempts
