#!/usr/bin/env python3
"""Experiment harness: slice the trade history and compare Monte Carlo P(pass)
across filters, trade-management variants, and sizing policies — all from the
local parquet cache (no network).

Usage: python scripts/experiment.py [slices|variants|all]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propalgo.backtest.engine import generate_trades
from propalgo.backtest.montecarlo import run_montecarlo, sweep_risk
from propalgo.config import load_config
from propalgo.data.dukascopy import DukascopySource
from propalgo.strategies import build_strategies

START = datetime(2023, 1, 1, tzinfo=timezone.utc)
SIMS = 4000


def load_bars(cfg, symbols):
    src = DukascopySource(cfg.instruments)
    bars = {}
    for sym in symbols:
        df = src.load_cached(sym)
        df = df[df.index >= START]
        if len(df):
            bars[sym] = df
    return bars


def mc_line(label, trades, cfg, fracs=(0.10, 0.15, 0.20, 0.25, 0.30), policy=None):
    if len({t.day for t in trades}) < 30:
        print(f"  {label:<34} (too few trade days: {len(trades)} trades)")
        return None
    best = None
    cells = []
    for f in fracs:
        r = run_montecarlo(trades, cfg.account, f, n_sims=SIMS, policy=policy)
        cells.append(f"{f:.0%}:{100 * r.p_pass:4.1f}%")
        if best is None or r.p_pass > best.p_pass:
            best = r
    cost = f"${best.expected_cost:,.0f}" if best.expected_cost else "-"
    print(f"  {label:<34} n={len(trades):<5} best P(pass)={100 * best.p_pass:4.1f}% "
          f"@ {best.risk_frac:.0%} (E[cost] {cost})   [{'  '.join(cells)}]")
    return best


def run_slices(cfg, trades):
    print(f"\n=== SLICES ({len(trades)} baseline trades, {SIMS} sims/cell) ===")
    mc_line("baseline (all)", trades, cfg)
    for strat in sorted({t.strategy for t in trades}):
        mc_line(f"strategy={strat}", [t for t in trades if t.strategy == strat], cfg)
    for sym in sorted({t.symbol for t in trades}):
        mc_line(f"symbol={sym}", [t for t in trades if t.symbol == sym], cfg)
    mc_line("grade=A+ only", [t for t in trades if t.grade == "A+"], cfg)
    mc_line("longs only", [t for t in trades if t.side > 0], cfg)
    mc_line("shorts only", [t for t in trades if t.side < 0], cfg)
    mc_line("momentum + ORB-A+ only",
            [t for t in trades if t.strategy == "momentum" or t.grade == "A+"], cfg)
    mc_line("drop ORB shorts",
            [t for t in trades if not (t.strategy == "orb" and t.side < 0)], cfg)
    mc_line("momentum + NQ-ORB",
            [t for t in trades if t.strategy == "momentum" or t.symbol == "NQ"], cfg)


def pnl_stats(label, trades):
    import numpy as np
    pnl = np.array([t.pnl_per_contract for t in trades])
    if not len(pnl):
        return
    r = np.array([t.pnl_per_contract / t.risk_per_contract for t in trades
                  if t.risk_per_contract > 0])
    wins = pnl > 0
    pf = pnl[wins].sum() / -pnl[~wins].sum() if (~wins).any() and pnl[~wins].sum() < 0 else float("inf")
    print(f"  {label:<34} n={len(pnl):<5} win%={100 * wins.mean():5.1f} "
          f"avgR={r.mean():+.3f}  PF={pf:5.2f}")


def run_diagnostics(cfg, trades):
    print("\n=== DIAGNOSTICS (per-contract economics, R = pnl/initial risk) ===")
    for strat in sorted({t.strategy for t in trades}):
        sub = [t for t in trades if t.strategy == strat]
        pnl_stats(f"{strat}", sub)
        for reason in ("target", "stop", "eod"):
            pnl_stats(f"  {strat}/{reason}", [t for t in sub if t.exit_reason == reason])
        for grade in ("A", "A+"):
            pnl_stats(f"  {strat}/grade {grade}", [t for t in sub if t.grade == grade])
        for side, nm in ((1, "long"), (-1, "short")):
            pnl_stats(f"  {strat}/{nm}", [t for t in sub if t.side == side])
    print("\n  by entry hour (ET):")
    for h in range(9, 16):
        pnl_stats(f"  hour {h:02d}", [t for t in trades
                                      if t.entry_time.tz_convert('America/New_York').hour == h])


def drop_orb_shorts(trades):
    return [t for t in trades if not (t.strategy == "orb" and t.side < 0)]


def run_variants(cfg, bars):
    strategies = build_strategies(cfg.strategies)
    print(f"\n=== VARIANTS ({SIMS} sims/cell; all applied on top of 'drop ORB shorts') ===")
    base = drop_orb_shorts(generate_trades(bars, strategies, cfg.instruments, cfg.costs))
    mc_line("base (drop ORB shorts)", base, cfg)

    for be in (0.8, 1.0, 1.5):
        t = drop_orb_shorts(generate_trades(bars, strategies, cfg.instruments,
                                            cfg.costs, mgmt={"breakeven_at_r": be}))
        mc_line(f"breakeven stop @ {be}R", t, cfg)
    for tsb in (6, 10):
        t = drop_orb_shorts(generate_trades(bars, strategies, cfg.instruments,
                                            cfg.costs, mgmt={"time_stop_bars": tsb}))
        mc_line(f"time stop {tsb} bars (<0.5R)", t, cfg)
    mc_line("policy: one-loss-per-day", base, cfg, policy={"stop_after_loss": True})
    mc_line("policy: adaptive timid", base, cfg, policy={"adaptive": "timid"})
    mc_line("policy: adaptive bold", base, cfg, policy={"adaptive": "bold"})
    t = drop_orb_shorts(generate_trades(bars, strategies, cfg.instruments,
                                        cfg.costs, mgmt={"breakeven_at_r": 1.0}))
    mc_line("BE@1R + one-loss-per-day", t, cfg, policy={"stop_after_loss": True})
    mc_line("BE@1R + adaptive bold", t, cfg, policy={"adaptive": "bold"})
    mc_line("BE@1R + bold + one-loss",
            t, cfg, policy={"adaptive": "bold", "stop_after_loss": True})


def run_tuning(cfg, bars):
    import copy
    print(f"\n=== TUNING ({SIMS} sims/cell; all with ORB shorts dropped) ===")

    def trades_with(params_patch, symbols=None):
        params = copy.deepcopy(cfg.strategies)
        for strat, kv in params_patch.items():
            params[strat].update(kv)
        b = {s: v for s, v in bars.items() if symbols is None or s in symbols}
        return drop_orb_shorts(generate_trades(b, build_strategies(params),
                                               cfg.instruments, cfg.costs))

    mc_line("symbols NQ+ES", trades_with({}, ["NQ", "ES"]), cfg)
    mc_line("symbols NQ+ES+YM+RTY", trades_with({}), cfg)
    mc_line("symbols NQ only", trades_with({}, ["NQ"]), cfg)

    for tr in (3.0, 4.0, 100.0):
        label = f"target {tr}R" if tr < 50 else "no target (EOD/stop only)"
        mc_line(label, trades_with({"orb": {"target_r": tr},
                                    "momentum": {"target_r": tr}}, ["NQ", "ES"]), cfg)
    for mins in (45, 60):
        mc_line(f"ORB range {mins}min", trades_with({"orb": {"or_minutes": mins}},
                                                    ["NQ", "ES"]), cfg)
    mc_line("ORB tighter ATR cap 1.0", trades_with({"orb": {"max_or_atr": 1.0}},
                                                   ["NQ", "ES"]), cfg)
    mc_line("momentum gap 0.5 ATR", trades_with({"momentum": {"gap_atr": 0.5}},
                                                ["NQ", "ES"]), cfg)


def run_stack(cfg, bars):
    import copy
    print(f"\n=== STACKED COMBOS ({SIMS} sims/cell, NQ+ES, ORB shorts dropped) ===")

    def trades_with(params_patch):
        params = copy.deepcopy(cfg.strategies)
        for strat, kv in params_patch.items():
            params[strat].update(kv)
        b = {s: v for s, v in bars.items() if s in ("NQ", "ES")}
        return drop_orb_shorts(generate_trades(b, build_strategies(params),
                                               cfg.instruments, cfg.costs))

    # gap-threshold stability scan (looking for a plateau, not a spike)
    for gap in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        mc_line(f"gap {gap:.2f} ATR", trades_with({"momentum": {"gap_atr": gap}}), cfg)
    stacked = trades_with({"momentum": {"gap_atr": 0.5}, "orb": {"max_or_atr": 1.0}})
    mc_line("stack: gap0.5 + ORBcap1.0", stacked, cfg)
    mc_line("stack + one-loss-per-day", stacked, cfg, policy={"stop_after_loss": True})
    mc_line("stack, fine risk sweep", stacked, cfg,
            fracs=(0.18, 0.20, 0.22, 0.25, 0.28))
    best = trades_with({"momentum": {"gap_atr": 0.5}})
    mc_line("gap0.5 + taper near target", best, cfg,
            fracs=(0.20, 0.25, 0.30, 0.35), policy={"taper": True})
    mc_line("gap0.5 + taper + bold", best, cfg,
            fracs=(0.20, 0.25, 0.30, 0.35),
            policy={"taper": True, "adaptive": "bold"})


def run_oos(cfg, bars):
    import copy
    print(f"\n=== ROBUSTNESS: per-period MC with the chosen config ===")
    params = copy.deepcopy(cfg.strategies)
    params["momentum"]["gap_atr"] = 0.5
    b = {s: v for s, v in bars.items() if s in ("NQ", "ES")}
    trades = drop_orb_shorts(generate_trades(b, build_strategies(params),
                                             cfg.instruments, cfg.costs))
    periods = [("2023", (2023,)), ("2024", (2024,)), ("2025 (bad year)", (2025,)),
               ("2026 ytd", (2026,)), ("2023-2024", (2023, 2024)),
               ("2025-2026", (2025, 2026)), ("full sample", None)]
    for label, years in periods:
        sub = [t for t in trades if years is None or t.day.year in years]
        mc_line(label, sub, cfg, fracs=(0.20, 0.25), policy={"taper": True})


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    symbols = ["NQ", "ES", "YM", "RTY"] if mode != "core" else ["NQ", "ES"]
    bars = load_bars(cfg, symbols)
    print(f"cached symbols: {list(bars)} "
          f"({ {s: len(b) for s, b in bars.items()} })")
    if mode in ("variants", "all"):
        run_variants(cfg, bars)
    if mode in ("tuning", "all"):
        run_tuning(cfg, bars)
    if mode in ("stack", "all"):
        run_stack(cfg, bars)
    if mode in ("oos", "all"):
        run_oos(cfg, bars)
    if mode in ("diag", "slices", "all", "core"):
        trades = generate_trades(bars, build_strategies(cfg.strategies),
                                 cfg.instruments, cfg.costs)
        if mode in ("diag", "all"):
            run_diagnostics(cfg, trades)
        if mode in ("slices", "all", "core"):
            run_slices(cfg, trades)


if __name__ == "__main__":
    main()
