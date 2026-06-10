from propalgo.config import AccountRules
from propalgo.rules.apex import EvalStatus, EvalTracker

RULES = AccountRules(min_trading_days=1)


def make_tracker(**overrides):
    return EvalTracker(AccountRules(min_trading_days=1, **overrides))


def test_initial_threshold():
    t = make_tracker()
    assert t.threshold == 47_500
    assert t.target_balance == 53_000


def test_unrealized_peak_raises_hwm_then_giveback_busts():
    """The classic trap: trade runs +$2,000 unrealized, never realized.
    HWM moves to 52,000, threshold to 49,500 — a later dip to 49,500 busts
    even though closed balance never went negative by 2,500."""
    t = make_tracker()
    t.on_equity_extremes(50_000, 52_000)          # peak +2k unrealized
    assert t.hwm == 52_000 and t.threshold == 49_500
    assert t.on_equity_extremes(49_500, 50_000) is EvalStatus.BUSTED


def test_bust_exactly_at_threshold():
    t = make_tracker()
    assert t.on_equity_extremes(47_500, 50_000) is EvalStatus.BUSTED


def test_one_tick_above_threshold_survives():
    t = make_tracker()
    assert t.on_equity_extremes(47_505, 50_000) is EvalStatus.ACTIVE


def test_conservative_intrabar_order_low_checked_before_high():
    """Within one bar the low must not be saved by that same bar's high."""
    t = make_tracker()
    t.on_equity_extremes(50_000, 52_400)           # threshold now 49,900
    assert t.on_equity_extremes(49_900, 60_000) is EvalStatus.BUSTED


def test_pass_on_realized_target():
    t = make_tracker()
    t.mark_trading_day("d1")
    assert t.on_trade_closed(3_000) is EvalStatus.PASSED


def test_target_needs_min_trading_days():
    t = EvalTracker(AccountRules(min_trading_days=7))
    t.mark_trading_day("d1")
    assert t.on_trade_closed(3_000) is EvalStatus.ACTIVE
    assert t.target_pending()
    for d in range(2, 8):
        t.mark_trading_day(f"d{d}")
    assert t.on_trade_closed(0.0) is EvalStatus.PASSED


def test_threshold_trails_realized_gains_too():
    t = make_tracker()
    t.on_trade_closed(1_000)                       # balance 51k, hwm 51k
    assert t.threshold == 48_500
    assert t.on_trade_closed(-2_500) is EvalStatus.BUSTED


def test_threshold_never_decreases():
    t = make_tracker()
    t.on_equity_extremes(49_000, 51_000)
    thr = t.threshold
    t.on_equity_extremes(48_700, 50_200)           # lower high: hwm unchanged
    assert t.threshold == thr
