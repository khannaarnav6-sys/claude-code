"""Performance statistics for a set of simulated trades.

With one setup per session and two months of data, any summary here rests on a
few dozen trades. The bootstrap interval on expectancy is included for exactly
that reason: it is usually wide enough to contain zero, which is the honest
headline even when the point estimate looks good.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .backtest import BacktestResult


@dataclass
class Stats:
    sessions: int
    setups: int
    trades: int
    fill_rate: float
    wins: int
    losses: int
    scratches: int
    win_rate: float
    expectancy_r: float
    expectancy_r_ci: tuple[float, float]
    median_r: float
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float
    total_r: float
    net_dollars: float
    dollars_per_trade: float
    max_drawdown_r: float
    max_drawdown_dollars: float
    longest_losing_streak: int
    sharpe_per_trade: float
    t_stat: float
    ambiguous_bars: int
    exit_breakdown: dict
    direction_breakdown: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _drawdown(curve: np.ndarray) -> float:
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))
    return float(np.max(peak - np.concatenate([[0.0], curve])))


def _longest_losing_streak(values: np.ndarray) -> int:
    best = run = 0
    for v in values:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


def bootstrap_ci(
    values: np.ndarray,
    draws: int = 10_000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `values`."""
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def summarize(result: BacktestResult) -> Stats:
    filled = result.filled
    r = np.array([t.r_multiple for t in filled], dtype=float)
    dollars = np.array([t.net_dollars for t in filled], dtype=float)

    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    scratches = int((r == 0).sum())
    gross_win = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())

    exits: dict[str, int] = {}
    for t in filled:
        exits[t.exit_reason or "none"] = exits.get(t.exit_reason or "none", 0) + 1

    longs = [t for t in filled if t.direction > 0]
    shorts = [t for t in filled if t.direction < 0]
    direction = {
        "long": {
            "n": len(longs),
            "win_rate": round(float(np.mean([t.r_multiple > 0 for t in longs])), 3) if longs else 0.0,
            "expectancy_r": round(float(np.mean([t.r_multiple for t in longs])), 3) if longs else 0.0,
        },
        "short": {
            "n": len(shorts),
            "win_rate": round(float(np.mean([t.r_multiple > 0 for t in shorts])), 3) if shorts else 0.0,
            "expectancy_r": round(float(np.mean([t.r_multiple for t in shorts])), 3) if shorts else 0.0,
        },
    }

    mean_r = float(r.mean()) if r.size else 0.0
    sd_r = float(r.std(ddof=1)) if r.size > 1 else 0.0

    return Stats(
        sessions=result.sessions_scanned,
        setups=result.sessions_with_setup,
        trades=len(filled),
        fill_rate=round(len(filled) / result.sessions_with_setup, 3) if result.sessions_with_setup else 0.0,
        wins=wins,
        losses=losses,
        scratches=scratches,
        win_rate=round(wins / len(filled), 3) if filled else 0.0,
        expectancy_r=round(mean_r, 3),
        expectancy_r_ci=tuple(round(v, 3) for v in bootstrap_ci(r)),
        median_r=round(float(np.median(r)), 3) if r.size else 0.0,
        avg_win_r=round(float(r[r > 0].mean()), 3) if wins else 0.0,
        avg_loss_r=round(float(r[r < 0].mean()), 3) if losses else 0.0,
        profit_factor=round(gross_win / gross_loss, 3) if gross_loss else float("inf"),
        total_r=round(float(r.sum()), 2),
        net_dollars=round(float(dollars.sum()), 2),
        dollars_per_trade=round(float(dollars.mean()), 2) if dollars.size else 0.0,
        max_drawdown_r=round(_drawdown(np.cumsum(r)), 2),
        max_drawdown_dollars=round(_drawdown(np.cumsum(dollars)), 2),
        longest_losing_streak=_longest_losing_streak(r),
        sharpe_per_trade=round(mean_r / sd_r, 3) if sd_r else 0.0,
        t_stat=round(mean_r / (sd_r / np.sqrt(r.size)), 2) if sd_r and r.size > 1 else 0.0,
        ambiguous_bars=sum(1 for t in filled if t.ambiguous_bar),
        exit_breakdown=exits,
        direction_breakdown=direction,
    )


def equity_curve(result: BacktestResult) -> list[dict]:
    """Cumulative R and dollars after each filled trade, for plotting."""
    curve = []
    total_r = total_d = 0.0
    for t in result.filled:
        total_r += t.r_multiple
        total_d += t.net_dollars
        curve.append(
            {
                "day": str(t.day),
                "r": round(t.r_multiple, 3),
                "cum_r": round(total_r, 3),
                "cum_dollars": round(total_d, 2),
            }
        )
    return curve


def format_stats(stats: Stats, title: str = "") -> str:
    lo, hi = stats.expectancy_r_ci
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    lines += [
        f"sessions {stats.sessions}   setups {stats.setups}   filled {stats.trades} ({stats.fill_rate:.0%} of setups)",
        f"win rate {stats.win_rate:.1%}  ({stats.wins}W / {stats.losses}L)",
        f"expectancy {stats.expectancy_r:+.3f}R   95% CI [{lo:+.3f}, {hi:+.3f}]   t={stats.t_stat:+.2f}",
        f"avg win {stats.avg_win_r:+.2f}R   avg loss {stats.avg_loss_r:+.2f}R   profit factor {stats.profit_factor:.2f}",
        f"total {stats.total_r:+.2f}R   net ${stats.net_dollars:,.0f} on 1 NQ   ${stats.dollars_per_trade:,.0f}/trade",
        f"max drawdown {stats.max_drawdown_r:.2f}R (${stats.max_drawdown_dollars:,.0f})   worst streak {stats.longest_losing_streak}",
        f"exits {stats.exit_breakdown}",
        f"long {stats.direction_breakdown['long']}  short {stats.direction_breakdown['short']}",
    ]
    if stats.ambiguous_bars:
        lines.append(f"note: {stats.ambiguous_bars} trade(s) had stop and target inside one bar (resolved as losses)")
    return "\n".join(lines)
