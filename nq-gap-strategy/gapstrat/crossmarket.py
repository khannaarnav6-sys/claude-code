"""Running one rule set across several index futures.

This is the cheapest out-of-sample test available without paying for data. NQ,
ES, YM and RTY all trade the same New York open and the same macro news, so a
setup that describes something real about how that open behaves should show up
in more than one of them. A result that lives only in NQ is far more likely to
be the sample than the strategy.

Thresholds have to be normalised first. `min_gap_points` and `max_stop_points`
are absolute, and five points means something entirely different on RTY at
2,900 than on YM at 53,000, so each instrument's thresholds are scaled to its
own median session range.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import data as data_module
from .backtest import ExecutionConfig, rth_sessions, run
from .data import CONTRACTS, GAP_FRACTION, MAX_STOP_FRACTION, median_session_range
from .metrics import summarize
from .strategy import StrategyConfig


def scale_to_instrument(config: StrategyConfig, session_range: float) -> StrategyConfig:
    """Restate point thresholds as the same fraction of this market's range."""
    return replace(
        config,
        min_gap_points=round(GAP_FRACTION * session_range, 4),
        min_stop_points=round(GAP_FRACTION * session_range, 4),
        max_stop_points=round(MAX_STOP_FRACTION * session_range, 4),
        max_gap_points=None,
    )


def study(
    config: StrategyConfig,
    symbols: list[str] | None = None,
    execution: ExecutionConfig | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run `config` on each symbol; return the per-market table and the pool.

    The pooled figure is the one to read. Each market on its own carries the
    same small-sample problem as the original NQ result; together they are the
    best estimate of the rule's expectancy that this data can support.
    """
    symbols = symbols or list(CONTRACTS)
    execution = execution or ExecutionConfig()

    rows, pooled = [], []
    for symbol in symbols:
        bars = data_module.load(symbol, "5m", refresh=refresh)
        days = rth_sessions(bars)
        session_range = median_session_range(bars, days)
        scaled = scale_to_instrument(config, session_range)
        result = run(bars, bars, scaled, CONTRACTS[symbol], execution, days=days)
        stats = summarize(result)
        pooled.extend(t.r_multiple for t in result.filled)
        rows.append(
            {
                "symbol": symbol,
                "sessions": stats.sessions,
                "median_range": round(session_range, 1),
                "min_gap_points": scaled.min_gap_points,
                "setups": stats.setups,
                "trades": stats.trades,
                "win_rate": stats.win_rate,
                "expectancy_r": stats.expectancy_r,
                "ci_low": stats.expectancy_r_ci[0],
                "ci_high": stats.expectancy_r_ci[1],
                "total_r": stats.total_r,
            }
        )

    r = np.array(pooled, dtype=float)
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    summary = {
        "trades": int(r.size),
        "win_rate": round(float((r > 0).mean()), 3) if r.size else 0.0,
        "expectancy_r": round(float(r.mean()), 3) if r.size else 0.0,
        "total_r": round(float(r.sum()), 2) if r.size else 0.0,
        "t_stat": round(float(r.mean() / (sd / np.sqrt(r.size))), 2) if sd and r.size > 1 else 0.0,
        "markets_positive": int(sum(1 for row in rows if row["expectancy_r"] > 0)),
        "markets": len(rows),
    }
    return pd.DataFrame(rows), summary


def format_study(table: pd.DataFrame, summary: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(table.to_string(index=False))
    lines.append(
        f"POOLED  {summary['trades']} trades  win {summary['win_rate']:.0%}  "
        f"expectancy {summary['expectancy_r']:+.3f}R  total {summary['total_r']:+.1f}R  "
        f"t={summary['t_stat']:+.2f}  "
        f"({summary['markets_positive']}/{summary['markets']} markets positive)"
    )
    return "\n".join(lines)
