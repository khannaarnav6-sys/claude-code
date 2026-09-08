#!/usr/bin/env python3
"""Run the New York open gap strategy and everything needed to judge it.

    python3 run_backtest.py                 # full report
    python3 run_backtest.py --refresh       # re-download bars first
    python3 run_backtest.py --quick         # skip the control permutations

Writes the trade ledger, summary statistics and sweep table into results/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gapstrat import bias as bias_module
from gapstrat import data
from gapstrat.backtest import ExecutionConfig, plans_for, r_multiples, rth_sessions, run, run_plans
from gapstrat.controls import ControlResult, random_direction, run_controls
from gapstrat.data import NQ
from gapstrat.metrics import bootstrap_ci, equity_curve, format_stats, summarize
from gapstrat.strategy import StrategyConfig, describe
from gapstrat.sweep import SweepConfig, plan_sweep_session, run_sweep
from gapstrat.sweep import describe as describe_sweep

RESULTS = Path(__file__).resolve().parent / "results"
SYMBOL = "NQ=F"


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def sweep(signal_bars, exec_bars, days, base: StrategyConfig, execution) -> pd.DataFrame:
    """Same data, many rule variations -- a flat table is the honest way to see
    whether the baseline is a real effect or the luckiest cell in a grid."""
    rows = []
    grid = []
    for entry_style in ("proximal", "mid"):
        for target_r in (1.0, 1.5, 2.0, 3.0):
            grid.append(replace(base, entry_style=entry_style, target_r=target_r))
    for target_r in (1.0, 2.0, 3.0):
        # A distal entry sits on the far edge, so its stop has to come from the
        # pattern; against the gap edge the risk would be the buffer alone.
        grid.append(replace(base, entry_style="distal", stop_style="pattern", target_r=target_r))
    for target_r in (1.0, 2.0, 3.0):
        grid.append(replace(base, entry_type="stop", target_r=target_r))
    for stop_style in ("pattern",):
        for target_r in (1.0, 2.0, 3.0):
            grid.append(replace(base, stop_style=stop_style, target_r=target_r))
    for cutoff in ("10:00", "10:30", "12:00"):
        grid.append(replace(base, signal_cutoff=cutoff))
    for min_gap in (0.0, 10.0, 20.0):
        grid.append(replace(base, min_gap_points=min_gap))
    grid.append(replace(base, displacement="fade"))
    grid.append(replace(base, breakeven_at_r=1.0))

    for cfg in grid:
        result = run(signal_bars, exec_bars, cfg, NQ, execution, days=days)
        stats = summarize(result)
        rows.append(
            {
                "variant": describe(cfg),
                "entry_style": cfg.entry_style,
                "cutoff": cfg.signal_cutoff,
                "min_gap": cfg.min_gap_points,
                "trades": stats.trades,
                "win_rate": stats.win_rate,
                "expectancy_r": stats.expectancy_r,
                "ci_low": stats.expectancy_r_ci[0],
                "ci_high": stats.expectancy_r_ci[1],
                "total_r": stats.total_r,
                "net_dollars": stats.net_dollars,
                "max_dd_r": stats.max_drawdown_r,
                "profit_factor": stats.profit_factor,
            }
        )
    return pd.DataFrame(rows).sort_values("expectancy_r", ascending=False)


def bias_comparison(signal_bars, exec_bars, days, base, execution) -> pd.DataFrame:
    """Does knowing the session's lean improve the gap trade, or just thin it out?

    Filtering always removes trades, and removing trades always widens the
    interval, so a higher expectancy on fewer trades is not automatically an
    improvement. The trade count sits next to the expectancy for that reason.
    """
    rows = []
    for method in bias_module.METHODS:
        for stop_style in ("gap_far", "swing"):
            cfg = replace(base, bias_method=method, stop_style=stop_style)
            result = run(signal_bars, exec_bars, cfg, NQ, execution, days=days)
            stats = summarize(result)
            risks = [t.risk_points for t in result.filled]
            rows.append(
                {
                    "bias": method,
                    "stop": stop_style,
                    "setups": stats.setups,
                    "trades": stats.trades,
                    "win_rate": stats.win_rate,
                    "expectancy_r": stats.expectancy_r,
                    "ci_low": stats.expectancy_r_ci[0],
                    "ci_high": stats.expectancy_r_ci[1],
                    "total_r": stats.total_r,
                    "avg_risk_pts": round(sum(risks) / len(risks), 1) if risks else 0.0,
                    "net_dollars": stats.net_dollars,
                }
            )
    return pd.DataFrame(rows)


def sweep_setup_study(signal_bars, exec_bars, days, execution) -> pd.DataFrame:
    """The stop-run-and-reverse setup across its own parameter grid."""
    rows = []
    for reference in ("overnight", "prior_day", "opening_range"):
        for anchor in ("level", "sweep_mid", "sweep_close"):
            for displacement in (True, False):
                cfg = SweepConfig(
                    reference=reference, entry_anchor=anchor, require_displacement=displacement
                )
                result = run_sweep(signal_bars, exec_bars, cfg, NQ, execution, days=days)
                stats = summarize(result)
                risks = [t.risk_points for t in result.filled]
                rows.append(
                    {
                        "reference": reference,
                        "entry_anchor": anchor,
                        "needs_displacement": displacement,
                        "setups": stats.setups,
                        "trades": stats.trades,
                        "win_rate": stats.win_rate,
                        "expectancy_r": stats.expectancy_r,
                        "ci_low": stats.expectancy_r_ci[0],
                        "ci_high": stats.expectancy_r_ci[1],
                        "total_r": stats.total_r,
                        "avg_risk_pts": round(sum(risks) / len(risks), 1) if risks else 0.0,
                        "net_dollars": stats.net_dollars,
                    }
                )
    return pd.DataFrame(rows).sort_values("expectancy_r", ascending=False)


def market_context(bars: pd.DataFrame, days: list) -> str:
    """What the market itself did over the sample.

    A continuation strategy tested through a one-way trend will look good for
    reasons that have nothing to do with its rules, so this belongs next to
    every result.
    """
    rth = [data.session_slice(bars, day, "09:30", "15:55") for day in days]
    rth = [frame for frame in rth if not frame.empty]
    opens = np.array([float(f["open"].iloc[0]) for f in rth])
    closes = np.array([float(f["close"].iloc[-1]) for f in rth])
    ranges = np.array([float(f["high"].max() - f["low"].min()) for f in rth])
    net = closes[-1] - opens[0]
    up_days = int((closes > opens).sum())
    return (
        f"NQ over the sample: {opens[0]:,.0f} -> {closes[-1]:,.0f}  "
        f"({net:+,.0f} pts, {net / opens[0]:+.1%})\n"
        f"{up_days} of {len(rth)} sessions closed above their 09:30 open  "
        f"({up_days / len(rth):.0%})\n"
        f"average RTH range {ranges.mean():.0f} pts   median {np.median(ranges):.0f} pts"
    )


def slippage_sensitivity(signal_bars, exec_bars, days, base, ticks=(0, 1, 2, 4)) -> pd.DataFrame:
    """Stops here are ~15 points wide, so execution cost is not a rounding error."""
    rows = []
    for t in ticks:
        stats = summarize(run(signal_bars, exec_bars, base, NQ, ExecutionConfig(slippage_ticks=t), days=days))
        rows.append(
            {
                "slippage_ticks": t,
                "dollars_per_side": round(t * NQ.tick_value, 2),
                "trades": stats.trades,
                "expectancy_r": stats.expectancy_r,
                "total_r": stats.total_r,
                "net_dollars": stats.net_dollars,
            }
        )
    return pd.DataFrame(rows)


def split_half(result) -> str:
    """Does the first half of the sample look like the second half?"""
    trades = result.filled
    if len(trades) < 6:
        return "not enough trades to split"
    mid = len(trades) // 2
    first = np.array([t.r_multiple for t in trades[:mid]])
    second = np.array([t.r_multiple for t in trades[mid:]])
    return (
        f"first half  n={first.size:2d}  expectancy {first.mean():+.3f}R  "
        f"({trades[0].day} .. {trades[mid - 1].day})\n"
        f"second half n={second.size:2d}  expectancy {second.mean():+.3f}R  "
        f"({trades[mid].day} .. {trades[-1].day})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download bars")
    parser.add_argument("--quick", action="store_true", help="skip control permutations")
    parser.add_argument("--draws", type=int, default=500, help="control permutations")
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--entry-style", default="mid", choices=["proximal", "mid", "distal"])
    parser.add_argument("--slippage-ticks", type=int, default=1)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)

    bars_5m = data.load(SYMBOL, "5m", refresh=args.refresh)
    bars_1m = data.load(SYMBOL, "1m", refresh=args.refresh)
    days_5m = rth_sessions(bars_5m)
    days_1m = rth_sessions(bars_1m)

    execution = ExecutionConfig(slippage_ticks=args.slippage_ticks)
    base = StrategyConfig(target_r=args.target_r, entry_style=args.entry_style)

    heading("DATA")
    print(f"5m bars: {len(bars_5m):,}  {bars_5m.index[0]:%Y-%m-%d} .. {bars_5m.index[-1]:%Y-%m-%d}  "
          f"({len(days_5m)} full RTH sessions)")
    print(f"1m bars: {len(bars_1m):,}  {bars_1m.index[0]:%Y-%m-%d} .. {bars_1m.index[-1]:%Y-%m-%d}  "
          f"({len(days_1m)} full RTH sessions)")
    print(f"contract: NQ, ${NQ.point_value}/pt, {NQ.tick_size} tick, "
          f"${NQ.commission_round_turn} commission round turn, {args.slippage_ticks} tick slippage")
    print()
    print(market_context(bars_5m, days_5m))

    heading(f"BASELINE  ({describe(base)}, gaps on 5m, fills on 5m)")
    result = run(bars_5m, bars_5m, base, NQ, execution, days=days_5m)
    stats = summarize(result)
    print(format_stats(stats))
    print("\n" + split_half(result))

    ledger = result.frame()
    ledger.to_csv(RESULTS / "trades_baseline.csv", index=False)

    heading("BAR-RESOLUTION CHECK  (same sessions, 5m fills vs 1m fills)")
    overlap = sorted(set(days_5m) & set(days_1m))
    signal_from_1m = data.resample(bars_1m, "5min")  # identical gaps in both runs
    fidelity = {}
    for label, exec_bars in (("5m fills", bars_5m), ("1m fills", bars_1m)):
        sub = run(signal_from_1m, exec_bars, base, NQ, execution, days=overlap)
        sub_stats = summarize(sub)
        fidelity[label] = sub_stats.to_dict()
        print(f"\n{label} ({len(overlap)} sessions)")
        print(format_stats(sub_stats))

    heading("SESSION BIAS  (does knowing the day's lean help?)")
    bias_table = bias_comparison(bars_5m, bars_5m, days_5m, base, execution)
    print(bias_table.to_string(index=False))
    print("\nFewer trades is the cost of every filter here: a higher expectancy on")
    print("half the sample is not automatically a better strategy.")

    heading("SWEEP AND RECLAIM  (stop-run, reversal, limit back at the level)")
    sweep_table = sweep_setup_study(bars_5m, bars_5m, days_5m, execution)
    print(sweep_table.to_string(index=False))
    best = sweep_table.iloc[0]
    sweep_best = SweepConfig(
        reference=best["reference"],
        entry_anchor=best["entry_anchor"],
        require_displacement=bool(best["needs_displacement"]),
    )
    sweep_result = run_sweep(bars_5m, bars_5m, sweep_best, NQ, execution, days=days_5m)
    print(f"\nbest cell: {describe_sweep(sweep_best)}")
    print(format_stats(summarize(sweep_result)))
    sweep_result.frame().to_csv(RESULTS / "trades_sweep.csv", index=False)

    sweep_controls = []
    if not args.quick:
        sweep_plans = [
            p for p in (plan_sweep_session(bars_5m, d, sweep_best, NQ) for d in days_5m)
            if p is not None
        ]
        observed = float(r_multiples(sweep_result).mean()) if sweep_result.filled else 0.0
        sweep_controls = [
            ControlResult(
                "random direction",
                args.draws,
                random_direction(
                    sweep_plans, bars_5m, NQ, execution,
                    StrategyConfig(target_r=sweep_best.target_r), args.draws,
                ),
                observed,
            )
        ]
        for control in sweep_controls:
            print(control.summary())
    positive_sweep = int((sweep_table["expectancy_r"] > 0).sum())
    print(f"\n{positive_sweep} of {len(sweep_table)} sweep variants positive; "
          f"best cell rests on {int(best['trades'])} trades")

    heading("SLIPPAGE SENSITIVITY")
    slip_table = slippage_sensitivity(bars_5m, bars_5m, days_5m, base)
    print(slip_table.to_string(index=False))

    heading("SWEEP  (every variation on the same data)")
    table = sweep(bars_5m, bars_5m, days_5m, base, execution)
    table.to_csv(RESULTS / "sweep.csv", index=False)
    print(table.to_string(index=False))
    positive = int((table["expectancy_r"] > 0).sum())
    print(f"\n{positive} of {len(table)} variants positive; "
          f"median expectancy {table['expectancy_r'].median():+.3f}R")

    controls = []
    if not args.quick:
        heading(f"CONTROLS  ({args.draws} permutations each)")
        plans = plans_for(bars_5m, base, NQ, days=days_5m)
        controls = run_controls(
            result, plans, bars_5m, bars_5m, days_5m, base, NQ, execution, draws=args.draws
        )
        for control in controls:
            print(control.summary())

    r = r_multiples(result)
    payload = {
        "generated_for_sessions": [str(d) for d in days_5m],
        "config": {
            "target_r": base.target_r,
            "entry_style": base.entry_style,
            "entry_type": base.entry_type,
            "stop_style": base.stop_style,
            "signal_cutoff": base.signal_cutoff,
            "entry_expiry": base.entry_expiry,
            "exit_time": base.exit_time,
            "min_gap_points": base.min_gap_points,
            "slippage_ticks": args.slippage_ticks,
            "commission_round_turn": NQ.commission_round_turn,
        },
        "market_context": market_context(bars_5m, days_5m),
        "baseline": stats.to_dict(),
        "bias_comparison": bias_table.to_dict(orient="records"),
        "sweep_setup": {
            "grid": sweep_table.to_dict(orient="records"),
            "best": {
                "reference": sweep_best.reference,
                "entry_anchor": sweep_best.entry_anchor,
                "require_displacement": sweep_best.require_displacement,
                "target_r": sweep_best.target_r,
            },
            "best_stats": summarize(sweep_result).to_dict(),
            "best_equity": equity_curve(sweep_result),
            "controls": [
                {
                    "name": c.name,
                    "draws": c.draws,
                    "control_mean_r": round(c.mean, 4),
                    "actual_r": round(c.actual, 4),
                    "percentile": round(c.percentile_of_actual, 1),
                    "p_value": round(c.p_value, 4),
                    "distribution": [round(float(v), 4) for v in c.expectancies],
                }
                for c in sweep_controls
            ],
        },
        "split_half": split_half(result),
        "slippage_sensitivity": slip_table.to_dict(orient="records"),
        "equity_curve": equity_curve(result),
        "bar_resolution_check": {"sessions": len(overlap), **fidelity},
        "controls": [
            {
                "name": c.name,
                "draws": c.draws,
                "control_mean_r": round(c.mean, 4),
                "actual_r": round(c.actual, 4),
                "percentile": round(c.percentile_of_actual, 1),
                "p_value": round(c.p_value, 4),
                "distribution": [round(float(v), 4) for v in c.expectancies],
            }
            for c in controls
        ],
        "expectancy_bootstrap_ci": list(bootstrap_ci(r)) if r.size else None,
        "sweep": table.to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2, default=str))

    heading("FILES")
    for path in sorted(RESULTS.glob("*")):
        print(f"  {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
