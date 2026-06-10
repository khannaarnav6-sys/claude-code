"""Text reports: trade stats, eval attempt tally, Monte Carlo sweep table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import EvalAttempt, TradeRecord
from .montecarlo import MonteCarloResult


def trades_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "entry_time": t.entry_time, "exit_time": t.exit_time, "day": t.day,
        "symbol": t.symbol, "strategy": t.strategy, "grade": t.grade,
        "side": "long" if t.side > 0 else "short", "entry": t.entry, "exit": t.exit,
        "exit_reason": t.exit_reason, "pnl_per_contract": t.pnl_per_contract,
    } for t in sorted(trades, key=lambda t: t.entry_time)])


def trade_stats(trades: list[TradeRecord]) -> str:
    df = trades_frame(trades)
    if df.empty:
        return "No trades generated."
    lines = [f"Trades: {len(df)}  ({df['day'].min()} -> {df['day'].max()})"]
    for key, grp in [("ALL", df)] + list(df.groupby("strategy")):
        pnl = grp["pnl_per_contract"]
        wins = pnl > 0
        pf_denom = -pnl[~wins].sum()
        pf = pnl[wins].sum() / pf_denom if pf_denom > 0 else float("inf")
        lines.append(
            f"  {key:<10} n={len(grp):<5} win%={100 * wins.mean():5.1f}  "
            f"avg=${pnl.mean():8.2f}  total=${pnl.sum():10.2f}  PF={pf:5.2f}  (per contract)"
        )
    by_year = df.assign(year=pd.to_datetime(df["day"].astype(str)).dt.year) \
                .groupby("year")["pnl_per_contract"].agg(["count", "sum"])
    lines.append("  Per-year PnL (1 contract): " + "  ".join(
        f"{y}: ${row['sum']:,.0f}/{int(row['count'])}t" for y, row in by_year.iterrows()))
    return "\n".join(lines)


def eval_report(attempts: list[EvalAttempt], base_contracts: int) -> str:
    if not attempts:
        return "No eval attempt completed within the data window."
    n_pass = sum(a.passed for a in attempts)
    lines = [
        f"Sequential eval replay @ {base_contracts} contracts: "
        f"{len(attempts)} attempts, {n_pass} passed ({100 * n_pass / len(attempts):.0f}%)"
    ]
    for a in attempts:
        lines.append(
            f"  {'PASS' if a.passed else 'BUST'}  {a.start_day} -> {a.end_day}  "
            f"days={a.trading_days:<3} trades={a.n_trades:<4} final=${a.final_balance:,.0f}"
        )
    return "\n".join(lines)


def montecarlo_table(results: list[MonteCarloResult], horizon_days: int) -> str:
    lines = [
        f"Monte Carlo (10k bootstrapped {horizon_days}-trading-day months):",
        f"  {'size':>4} {'P(pass)':>8} {'P(bust)':>8} {'med days':>9} {'E[attempts]':>12} {'E[cost]':>9}",
    ]
    for r in results:
        att = f"{r.expected_attempts:.1f}" if r.expected_attempts else "inf"
        cost = f"${r.expected_cost:,.0f}" if r.expected_cost else "-"
        days = f"{r.median_days_to_pass:.0f}" if r.median_days_to_pass else "-"
        lines.append(f"  {r.contracts:>4} {100 * r.p_pass:>7.1f}% {100 * r.p_bust:>7.1f}% "
                     f"{days:>9} {att:>12} {cost:>9}")
    best = max(results, key=lambda r: r.p_pass)
    lines.append(f"  -> best size: {best.contracts} contracts "
                 f"(P(pass)={100 * best.p_pass:.1f}%, expected cost ${best.expected_cost:,.0f})")
    return "\n".join(lines)


def equity_curve(trades: list[TradeRecord], contracts: int = 1) -> pd.Series:
    df = trades_frame(trades)
    if df.empty:
        return pd.Series(dtype=float)
    return (df.set_index("exit_time")["pnl_per_contract"] * contracts).cumsum()
