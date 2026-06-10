#!/usr/bin/env python3
"""propalgo CLI: fetch | backtest | montecarlo | parity | live"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propalgo.config import load_config  # noqa: E402


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _resolve_risk(args, cfg) -> float:
    """--risk wins; then --profile; then the config's default profile."""
    if getattr(args, "risk", None) is not None:
        return args.risk
    profiles = cfg.sizing.get("profiles", {})
    name = getattr(args, "profile", None) or cfg.sizing.get("profile", "grind")
    if name in profiles:
        return float(profiles[name])
    return float(cfg.sizing.get("risk_frac", 0.085))


def _load_backtest_bars(cfg, symbols, start, end):
    from propalgo.data.dukascopy import DukascopySource
    src = DukascopySource(cfg.instruments)
    bars = {}
    for sym in symbols:
        df = src.fetch(sym, start, end)
        print(f"  {sym}: {len(df)} bars cached "
              f"({df.index[0]} -> {df.index[-1]})" if len(df) else f"  {sym}: no data")
        bars[sym] = df
    return {s: b for s, b in bars.items() if len(b)}


def cmd_fetch(args, cfg):
    from propalgo.data.dukascopy import DukascopySource
    from propalgo.data.yahoo import YahooSource
    start, end = _parse_date(args.start), datetime.now(timezone.utc)
    print(f"Fetching Dukascopy 15m history from {args.start} (proxy backtest data)...")
    _load_backtest_bars(cfg, args.symbols, start, end)
    print("Fetching recent Yahoo futures 15m bars (live/parity data)...")
    ysrc = YahooSource(cfg.instruments)
    ystart = datetime.now(timezone.utc).replace(hour=0) - __import__("datetime").timedelta(days=55)
    for sym in args.symbols:
        try:
            df = ysrc.fetch(sym, ystart)
            print(f"  {sym} ({cfg.instruments[sym].yahoo}): {len(df)} bars")
        except Exception as exc:
            print(f"  {sym}: yahoo fetch failed: {exc}")


def _generate(cfg, args):
    from propalgo.backtest.engine import generate_trades
    from propalgo.strategies import build_strategies
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.now(timezone.utc)
    bars = _load_backtest_bars(cfg, args.symbols, start, end)
    if not bars:
        sys.exit("No data — run `fetch` first (or check network).")
    strategies = build_strategies(cfg.strategies)
    return generate_trades(bars, strategies, cfg.instruments, cfg.costs)


def cmd_backtest(args, cfg):
    from propalgo.backtest.engine import run_eval_sequence
    from propalgo.backtest.report import eval_report, trade_stats, trades_frame
    if args.min_days is not None:
        from dataclasses import replace
        cfg.account = replace(cfg.account, min_trading_days=args.min_days)
    trades = _generate(cfg, args)
    print()
    print(trade_stats(trades))
    print()
    risk = _resolve_risk(args, cfg)
    attempts = run_eval_sequence(trades, cfg.account, risk,
                                 float(cfg.sizing.get("aplus_multiplier", 1.5)))
    print(eval_report(attempts, risk))
    if args.export:
        trades_frame(trades).to_csv(args.export, index=False)
        print(f"\nTrade log written to {args.export}")


def cmd_montecarlo(args, cfg):
    from propalgo.backtest.report import montecarlo_table
    from propalgo.sizing import recommend_risk
    if args.min_days is not None:
        from dataclasses import replace
        cfg.account = replace(cfg.account, min_trading_days=args.min_days)
    trades = _generate(cfg, args)
    print(f"\n{len(trades)} historical trades feed the bootstrap.\n")
    best, results = recommend_risk(
        trades, cfg.account,
        float(cfg.sizing.get("aplus_multiplier", 1.5)),
        horizon_days=args.horizon, n_sims=args.sims)
    print(montecarlo_table(results, args.horizon))


def cmd_parity(args, cfg):
    """Signal parity: proxy (Dukascopy) vs true futures (Yahoo) over ~55 days."""
    from datetime import timedelta

    from propalgo.backtest.engine import generate_trades
    from propalgo.data.yahoo import YahooSource
    from propalgo.strategies import build_strategies
    start = datetime.now(timezone.utc) - timedelta(days=55)
    strategies = build_strategies(cfg.strategies)
    proxy = _load_backtest_bars(cfg, args.symbols, start, datetime.now(timezone.utc))
    ysrc = YahooSource(cfg.instruments)
    fut = {s: ysrc.fetch(s, start) for s in args.symbols}
    fut = {s: b for s, b in fut.items() if len(b)}
    t_proxy = generate_trades(proxy, strategies, cfg.instruments, cfg.costs)
    t_fut = generate_trades(fut, strategies, cfg.instruments, cfg.costs)
    kp = {(t.day, t.symbol, t.strategy, t.side) for t in t_proxy}
    kf = {(t.day, t.symbol, t.strategy, t.side) for t in t_fut}
    both = len(kp & kf)
    print(f"proxy signals: {len(kp)}  futures signals: {len(kf)}  matching: {both}")
    if kp | kf:
        print(f"agreement: {100 * both / len(kp | kf):.0f}% "
              "(differences come from basis offset + Yahoo data gaps)")


def cmd_live(args, cfg):
    from propalgo.live.runner import run_cycle, run_forever
    risk = _resolve_risk(args, cfg)
    if args.once:
        n = run_cycle(cfg, risk, dry_run=args.dry_run)
        print(f"cycle complete, {n} new alert(s)")
    else:
        run_forever(cfg, risk, dry_run=args.dry_run)


def main():
    p = argparse.ArgumentParser(prog="propalgo")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbols", nargs="+", default=["NQ", "ES"])
    common.add_argument("--start", default="2021-01-01")
    common.add_argument("--end", default=None)

    sub.add_parser("fetch", parents=[common], help="download + cache bars")

    bt = sub.add_parser("backtest", parents=[common], help="trade stats + sequential eval replay")
    bt.add_argument("--risk", type=float, default=None,
                    help="per-trade risk as fraction of trailing DD (e.g. 0.085)")
    bt.add_argument("--profile", choices=["sprint", "balanced", "grind"], default=None,
                    help="named risk profile from config.yaml (default: config's choice)")
    bt.add_argument("--min-days", type=int, default=None, help="override min trading days (promo=1)")
    bt.add_argument("--export", default=None, help="CSV path for the trade log")

    mc = sub.add_parser("montecarlo", parents=[common], help="P(pass)/attempts sweep across risk levels")
    mc.add_argument("--horizon", type=int, default=250,
                    help="max trading days per attempt (evals have no time limit; "
                         "fees accrue monthly)")
    mc.add_argument("--sims", type=int, default=10_000)
    mc.add_argument("--min-days", type=int, default=None)

    sub.add_parser("parity", parents=[common], help="proxy vs futures signal agreement")

    lv = sub.add_parser("live", help="online 15m signal loop -> Discord alerts")
    lv.add_argument("--risk", type=float, default=None,
                    help="per-trade risk as fraction of trailing DD (e.g. 0.085)")
    lv.add_argument("--profile", choices=["sprint", "balanced", "grind"], default=None,
                    help="named risk profile from config.yaml (default: config's choice)")
    lv.add_argument("--dry-run", action="store_true")
    lv.add_argument("--once", action="store_true", help="run one cycle and exit")

    args = p.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "montecarlo": cmd_montecarlo,
     "parity": cmd_parity, "live": cmd_live}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
