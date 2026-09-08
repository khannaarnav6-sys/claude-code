"""Running the fitted rules, unchanged, over a long out-of-sample period.

Everything else in this project is measured on 49 sessions of NQ, which is not
enough to separate a rule from a run of luck. This module points the same rules
at a year of Nasdaq 100 data from a different source and an earlier period, and
changes nothing about them.

The discipline that makes it a holdout is that no parameter is chosen here. The
only quantity read from the holdout data is each instrument's median session
range, which converts the thresholds into that market's units -- the same
conversion applied to ES, YM and RTY, and not a fitted value.

The instrument is Dukascopy's Nasdaq 100 index CFD, so `PROXY` carries NQ's
tick and point value only to keep the arithmetic comparable. Read R-multiples
from it, not dollars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import ExecutionConfig, rth_sessions, run
from .data import ET, Contract, median_session_range
from .ict import ICTConfig, run_ict
from .ict import scaled_for as scale_ict
from .metrics import summarize
from .strategy import StrategyConfig

# A CFD has no contract size; NQ's specs are borrowed so the tick rounding and
# the cost model match the in-sample run. Dollar figures from this are notional.
PROXY = Contract("USATECHIDXUSD", tick_size=0.25, point_value=20.0, commission_round_turn=4.50)


def load_cached(
    instrument: str = "USATECHIDXUSD", before: str | None = "2026-06-27"
) -> pd.DataFrame:
    """Every day already pulled into the cache, as one frame.

    `before` cuts the frame off ahead of the in-sample window so a holdout run
    cannot accidentally include the sessions the rules were built on.
    """
    import glob

    from .data import DATA_DIR

    frames = []
    for path in sorted(glob.glob(str(DATA_DIR / "dukascopy" / instrument / "*.csv"))):
        day = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
        if day.empty:
            continue
        if day.index.tz is None:
            day.index = day.index.tz_localize("UTC")
        frames.append(day.tz_convert(ET))
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    bars.index.name = "timestamp"
    if before:
        bars = bars[bars.index < pd.Timestamp(before, tz=ET)]
    return bars


def scale_gap_config(config: StrategyConfig, session_range: float) -> StrategyConfig:
    """The same range-relative thresholds used for the cross-market check."""
    from dataclasses import replace

    from .data import GAP_FRACTION, MAX_STOP_FRACTION

    return replace(
        config,
        min_gap_points=round(GAP_FRACTION * session_range, 4),
        min_stop_points=round(GAP_FRACTION * session_range, 4),
        max_stop_points=round(MAX_STOP_FRACTION * session_range, 4),
        max_gap_points=None,
    )


def _stats_row(name: str, result, days: int) -> dict:
    stats = summarize(result)
    r = np.array([t.r_multiple for t in result.filled], dtype=float)
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    return {
        "model": name,
        "sessions": days,
        "setups": stats.setups,
        "trades": stats.trades,
        "win_rate": stats.win_rate,
        "expectancy_r": stats.expectancy_r,
        "ci_low": stats.expectancy_r_ci[0],
        "ci_high": stats.expectancy_r_ci[1],
        "total_r": stats.total_r,
        "profit_factor": stats.profit_factor,
        "max_dd_r": stats.max_drawdown_r,
        "t_stat": round(float(r.mean() / (sd / np.sqrt(r.size))), 2) if sd and r.size > 1 else 0.0,
    }


def evaluate(
    bars: pd.DataFrame,
    contract: Contract = PROXY,
    execution: ExecutionConfig | None = None,
    days: list | None = None,
    models: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run each model over `bars` and return one row per model."""
    execution = execution or ExecutionConfig()
    days = days if days is not None else rth_sessions(bars)
    session_range = median_session_range(bars, days)

    from dataclasses import replace

    models = models or {
        "gap: base rules": ("gap", StrategyConfig()),
        "gap: prior-day bias": ("gap", replace(StrategyConfig(), bias_method="prior_day")),
        "ict: full model": ("ict", ICTConfig()),
        "ict: trimmed": ("ict", replace(ICTConfig(), require_discount=False)),
        "ict: target fixed 2R": ("ict", replace(ICTConfig(), target="fixed")),
    }

    rows, results = [], {}
    for name, (kind, config) in models.items():
        if kind == "gap":
            result = run(bars, bars, scale_gap_config(config, session_range), contract, execution, days=days)
        else:
            result = run_ict(bars, bars, scale_ict(config, session_range), contract, execution, days=days)
        results[name] = result
        rows.append(_stats_row(name, result, len(days)))
    return pd.DataFrame(rows), results


def by_period(result, freq: str = "QE") -> pd.DataFrame:
    """Expectancy period by period -- a single number over a year can hide a lot."""
    filled = result.filled
    if not filled:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {"day": pd.to_datetime([t.day for t in filled]), "r": [t.r_multiple for t in filled]}
    )
    grouped = frame.set_index("day")["r"].resample(freq)
    return pd.DataFrame(
        {
            "trades": grouped.size(),
            "expectancy_r": grouped.mean().round(3),
            "total_r": grouped.sum().round(2),
            "win_rate": grouped.apply(lambda s: float((s > 0).mean()) if len(s) else 0.0).round(3),
        }
    ).dropna(subset=["expectancy_r"])
