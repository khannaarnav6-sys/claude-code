"""The full ICT model: sweep, structure shift, imbalance entry, liquidity target.

Every piece of the sequence a discretionary ICT trader waits for, made explicit
so each can be switched off and measured:

  1. DRAW      Before the open, build the liquidity map -- Asia, London and
               overnight ranges, prior day and week, the new week opening gap.
  2. SWEEP     Inside the killzone, price runs one of those pools and fails to
               hold beyond it. Running a low turns the bias up, and vice versa.
  3. SHIFT     Price then closes back through the last swing formed before the
               sweep. That is the market structure shift, and it is what
               separates a reversal from a pause.
  4. IMBALANCE The move that shifts structure leaves a fair value gap.
  5. ENTRY     A limit back into that gap -- or the optimal trade entry zone,
               or the swept level itself -- required to sit in the discount
               half of the dealing range for a long, premium for a short.
  6. STOP      Beyond the wick that swept the pool.
  7. TARGET    The opposing pool. By default the Asia range: run the Asia low,
               target the Asia high.

Higher-timeframe context -- weekly and daily fair value gaps, the new week
opening gap -- is available as a filter, since an intraday long into an
unfilled weekly bearish imbalance is a worse trade than the same long with
nothing in the way.

Every stage is optional, which is the point: a model with nine confluences
tested as one blob cannot tell you which of the nine did the work, and on a
sample this size most of them will be decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import liquidity as liq
from .data import Contract
from .gaps import find_gaps
from .strategy import PlannedTrade, _at, _previous_session

POOLS = ("asia", "london", "overnight", "prior_day", "prior_week")
ENTRY_MODELS = ("fvg", "ote", "level")
TARGETS = ("asia", "london", "overnight", "prior_day", "prior_week", "nearest", "furthest", "fixed")


@dataclass(frozen=True)
class ICTConfig:
    # --- when ---
    killzone_start: str = "08:30"  # NY AM killzone
    killzone_end: str = "11:00"
    entry_expiry: str = "12:00"
    exit_time: str = "15:55"

    # --- the sweep ---
    sweep_pools: tuple = POOLS
    min_penetration: float = 1.0  # points beyond the level before it counts
    require_close_back: bool = True  # the sweep bar must close back inside

    # --- the structure shift ---
    require_mss: bool = True
    confirm_within: int = 8  # bars allowed between sweep and shift
    swing_strength: int = 1  # bars either side of a fractal swing point

    # --- the entry ---
    require_fvg: bool = True
    entry_model: str = "fvg"
    fvg_entry: str = "mid"  # proximal | mid | distal
    ote_level: float = 0.705  # the 62-79% retracement sweet spot
    require_discount: bool = True  # longs only in the lower half of the range

    # --- higher timeframe context ---
    respect_htf_fvg: bool = False  # skip trades running into an opposing weekly/daily gap
    require_nwog_side: bool = False  # only trade in the direction of the new week gap

    # --- the target ---
    target: str = "asia"
    fixed_target_r: float = 2.0
    min_target_r: float = 1.5  # a pool closer than this is not worth the risk
    max_target_r: float | None = 15.0

    # --- risk ---
    stop_buffer_ticks: int = 4
    min_stop_points: float = 5.0
    max_stop_points: float | None = 100.0
    breakeven_at_r: float | None = None

    def validate(self) -> None:
        if self.entry_model not in ENTRY_MODELS:
            raise ValueError(f"bad entry_model {self.entry_model!r}")
        if self.target not in TARGETS:
            raise ValueError(f"bad target {self.target!r}")
        if self.fvg_entry not in ("proximal", "mid", "distal"):
            raise ValueError(f"bad fvg_entry {self.fvg_entry!r}")
        for pool in self.sweep_pools:
            if pool not in POOLS:
                raise ValueError(f"bad sweep pool {pool!r}")
        if not 0.5 <= self.ote_level <= 1.0:
            raise ValueError("ote_level should sit between 0.5 and 1.0")


def swing_points(highs, lows, strength: int) -> tuple[list[int], list[int]]:
    """Fractal swing highs and lows: an extreme with `strength` lower bars either side."""
    swing_highs, swing_lows = [], []
    for k in range(strength, len(highs) - strength):
        window = range(k - strength, k + strength + 1)
        if all(highs[k] >= highs[j] for j in window) and any(highs[k] > highs[j] for j in window):
            swing_highs.append(k)
        if all(lows[k] <= lows[j] for j in window) and any(lows[k] < lows[j] for j in window):
            swing_lows.append(k)
    return swing_highs, swing_lows


def _pool_levels(pools, names) -> list[liq.Level]:
    return [lv for lv in pools if lv.name.rsplit("_", 1)[0] in names]


def plan_ict_session(
    bars: pd.DataFrame,
    day,
    config: ICTConfig,
    contract: Contract,
    prior_week_days: list | None = None,
) -> PlannedTrade | None:
    """The session's first complete A+ sequence, as a resting order."""
    config.validate()
    tz = bars.index.tz
    start = _at(day, config.killzone_start, tz)
    end = _at(day, config.killzone_end, tz)

    # Scan from an hour before the killzone so fractals have context, but only
    # accept a sweep that happens inside it.
    window = bars[(bars.index >= start - pd.Timedelta(hours=1)) & (bars.index <= end)]
    if len(window) < 5:
        return None

    lmap = liq.build(bars, day, _previous_session(bars, day), prior_week_days)

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    swing_highs, swing_lows = swing_points(highs, lows, config.swing_strength)

    low_pools = _pool_levels(lmap.lows(), config.sweep_pools)
    high_pools = _pool_levels(lmap.highs(), config.sweep_pools)

    for i in range(len(window)):
        if window.index[i] < start:
            continue
        # A sweep of a low turns the bias up; a sweep of a high turns it down.
        for direction, pools in ((1, low_pools), (-1, high_pools)):
            for level in pools:
                if direction > 0:
                    took = lows[i] <= level.price - config.min_penetration
                    held = closes[i] > level.price if config.require_close_back else True
                else:
                    took = highs[i] >= level.price + config.min_penetration
                    held = closes[i] < level.price if config.require_close_back else True
                if not (took and held):
                    continue
                trade = _confirm(
                    window, i, direction, level, lmap, day, config, contract, tz,
                    highs, lows, closes, swing_highs, swing_lows,
                )
                if trade is not None:
                    return trade
    return None


def _confirm(
    window, sweep_i, direction, level, lmap, day, config, contract, tz,
    highs, lows, closes, swing_highs, swing_lows,
):
    """Carry a sweep through structure shift, imbalance and target checks."""
    sweep_low, sweep_high = lows[sweep_i], highs[sweep_i]

    # --- 3. the structure shift -------------------------------------------
    # The reference is the last fractal swing formed *before* the sweep: taking
    # that level back is what says the reversal has control.
    if config.require_mss:
        prior = [k for k in (swing_highs if direction > 0 else swing_lows) if k < sweep_i]
        if not prior:
            return None
        reference = highs[prior[-1]] if direction > 0 else lows[prior[-1]]
    else:
        reference = None

    last = min(sweep_i + config.confirm_within, len(window) - 1)
    for j in range(sweep_i + 1, last + 1):
        if config.require_mss:
            shifted = closes[j] > reference if direction > 0 else closes[j] < reference
        else:
            shifted = closes[j] > sweep_high if direction > 0 else closes[j] < sweep_low
        if not shifted:
            continue
        # Rebuilt each attempt: a later bar that passes must not inherit the
        # confluences noted for an earlier one that failed.
        notes = [f"swept {level.name}"]
        if config.require_mss:
            notes.append("mss")

        span = window.iloc[sweep_i : j + 1]

        # --- 4. the imbalance ---------------------------------------------
        gaps = [g for g in find_gaps(span) if g.direction == direction]
        if config.require_fvg and not gaps:
            continue
        gap = gaps[-1] if gaps else None
        if gap is not None:
            notes.append("fvg")

        # --- the dealing range, for premium/discount and the OTE ----------
        range_low = float(span["low"].min())
        range_high = float(span["high"].max())
        if range_high <= range_low:
            continue
        equilibrium = (range_high + range_low) / 2

        # --- 5. the entry --------------------------------------------------
        if config.entry_model == "fvg":
            if gap is None:
                continue
            entry = gap.entry_price(config.fvg_entry)
        elif config.entry_model == "ote":
            entry = (
                range_high - config.ote_level * (range_high - range_low)
                if direction > 0
                else range_low + config.ote_level * (range_high - range_low)
            )
        else:
            entry = level.price

        if config.require_discount:
            # A long belongs below equilibrium, a short above it.
            if direction > 0 and entry > equilibrium:
                continue
            if direction < 0 and entry < equilibrium:
                continue
            notes.append("discount" if direction > 0 else "premium")

        # --- 6. the stop ----------------------------------------------------
        buffer = config.stop_buffer_ticks * contract.tick_size
        stop = sweep_low - buffer if direction > 0 else sweep_high + buffer
        risk = abs(entry - stop)
        if risk < config.min_stop_points:
            continue
        if config.max_stop_points is not None and risk > config.max_stop_points:
            continue

        reference_price = closes[j]  # the market when the order is placed
        # The entry must rest behind the market, or it is a market order.
        if direction > 0 and not (stop < entry <= reference_price):
            continue
        if direction < 0 and not (reference_price <= entry < stop):
            continue

        # --- 7. the target: the opposing pool -------------------------------
        if config.target == "fixed":
            target = entry + direction * config.fixed_target_r * risk
            target_name = f"{config.fixed_target_r:g}R"
        else:
            draw = lmap.draw(direction, entry, config.target)
            if draw is None:
                continue  # nothing left to aim at on that side
            target, target_name = draw.price, draw.name
        notes.append(f"target {target_name}")

        reward_r = abs(target - entry) / risk
        if reward_r < config.min_target_r:
            continue  # the pool is too close to pay for the risk
        if config.max_target_r is not None and reward_r > config.max_target_r:
            continue  # and too far is a fantasy, not a target

        # --- higher timeframe context ---------------------------------------
        if config.respect_htf_fvg and _blocked_by_htf(lmap, direction, entry, target):
            continue
        if config.require_nwog_side and not _nwog_agrees(lmap, direction, reference_price):
            continue

        tick = contract.tick_size
        return PlannedTrade(
            day=day,
            direction=direction,
            entry_type="limit",
            entry_price=round(entry / tick) * tick,
            stop_price=round(stop / tick) * tick,
            target_price=round(target / tick) * tick,
            working_from=window.index[j],
            expires_at=_at(day, config.entry_expiry, tz),
            exit_at=_at(day, config.exit_time, tz),
            gap=gap,
            breakeven_r=config.breakeven_at_r,
            reference_price=reference_price,
            session_bias=direction,
            swept=level.name,
            target_name=target_name,
            notes=" + ".join(notes),
        )
    return None


def _blocked_by_htf(lmap, direction: int, entry: float, target: float) -> bool:
    """Is an opposing weekly or daily imbalance sitting in the path?"""
    lo, hi = (entry, target) if direction > 0 else (target, entry)
    for gap_low, gap_high, gap_dir in list(lmap.weekly_fvgs) + list(lmap.daily_fvgs):
        if gap_dir == direction:
            continue
        if gap_high >= lo and gap_low <= hi:
            return True
    return False


def _nwog_agrees(lmap, direction: int, price: float) -> bool:
    """Trading away from the new week opening gap rather than back into it."""
    if lmap.nwog_high is None or lmap.nwog_low is None:
        return True
    midpoint = (lmap.nwog_high + lmap.nwog_low) / 2
    return price > midpoint if direction > 0 else price < midpoint


def run_ict(bars, exec_bars, config, contract, execution=None, days=None):
    """Backtest the ICT model across every session."""
    from .backtest import BacktestResult, ExecutionConfig, rth_sessions, simulate_trade

    execution = execution or ExecutionConfig()
    days = days if days is not None else rth_sessions(bars)

    out = BacktestResult()
    for n, day in enumerate(days):
        out.sessions_scanned += 1
        plan = plan_ict_session(bars, day, config, contract, prior_week_days=days[max(0, n - 6) : n])
        if plan is None:
            continue
        out.sessions_with_setup += 1
        out.trades.append(simulate_trade(plan, exec_bars, contract, execution))
    return out


def describe(config: ICTConfig) -> str:
    parts = [f"ict/{config.entry_model}", f"target={config.target}"]
    if config.require_mss:
        parts.append("mss")
    if config.require_fvg:
        parts.append("fvg")
    if config.require_discount:
        parts.append("d/p")
    if config.respect_htf_fvg:
        parts.append("htf")
    if config.require_nwog_side:
        parts.append("nwog")
    return " ".join(parts)
