"""Apex evaluation rule engine.

The binding constraint is the trailing threshold drawdown: it trails the
REAL-TIME equity high-water mark, including unrealized peaks of open trades
— not the closed balance. Naive simulators that trail only on closed equity
overstate the pass rate, so the engine feeds intra-bar highs/lows here.

Conservative intra-bar ordering: a bar's adverse excursion (low) is checked
against the *current* threshold before the bar's favorable excursion (high)
is allowed to raise the high-water mark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config import AccountRules


class EvalStatus(Enum):
    ACTIVE = "active"
    PASSED = "passed"
    BUSTED = "busted"


@dataclass
class EvalTracker:
    rules: AccountRules
    balance: float = field(init=False)          # realized (closed) balance
    hwm: float = field(init=False)              # real-time equity high-water mark
    status: EvalStatus = field(init=False, default=EvalStatus.ACTIVE)
    trading_days: set = field(init=False, default_factory=set)

    def __post_init__(self):
        self.balance = self.rules.start_balance
        self.hwm = self.rules.start_balance

    @property
    def threshold(self) -> float:
        return self.hwm - self.rules.trailing_drawdown

    @property
    def target_balance(self) -> float:
        return self.rules.start_balance + self.rules.profit_target

    def mark_trading_day(self, day) -> None:
        self.trading_days.add(day)

    def on_equity_extremes(self, equity_low: float, equity_high: float) -> EvalStatus:
        """Process one bar's worth of open-position equity excursion.

        equity_low/high = realized balance +/- worst/best unrealized PnL in the bar.
        """
        if self.status is not EvalStatus.ACTIVE:
            return self.status
        if equity_low <= self.threshold:
            self.status = EvalStatus.BUSTED
            self.balance = self.threshold
            return self.status
        if equity_high > self.hwm:
            self.hwm = equity_high
        return self.status

    def on_trade_closed(self, pnl: float) -> EvalStatus:
        """Realize a closed trade's PnL and check pass/bust on closed balance."""
        if self.status is not EvalStatus.ACTIVE:
            return self.status
        self.balance += pnl
        if self.balance <= self.threshold:
            self.status = EvalStatus.BUSTED
        elif self.balance > self.hwm:
            self.hwm = self.balance
        if (
            self.status is EvalStatus.ACTIVE
            and self.balance >= self.target_balance
            and len(self.trading_days) >= self.rules.min_trading_days
        ):
            self.status = EvalStatus.PASSED
        return self.status

    def target_pending(self) -> bool:
        """Target profit reached but min trading days not yet satisfied."""
        return (
            self.status is EvalStatus.ACTIVE
            and self.balance >= self.target_balance
            and len(self.trading_days) < self.rules.min_trading_days
        )
