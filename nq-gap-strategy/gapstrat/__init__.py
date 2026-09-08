"""Backtesting harness for the New York open unmitigated-gap strategy on NQ."""

from .backtest import BacktestResult, ExecutionConfig, TradeResult, run, simulate_trade
from .data import MNQ, NQ, Contract, load, resample
from .gaps import Gap, find_gaps, mark_mitigation, unmitigated_at
from .metrics import Stats, equity_curve, format_stats, summarize
from .strategy import PlannedTrade, StrategyConfig, describe, plan_session

__all__ = [
    "BacktestResult",
    "Contract",
    "ExecutionConfig",
    "Gap",
    "MNQ",
    "NQ",
    "PlannedTrade",
    "Stats",
    "StrategyConfig",
    "TradeResult",
    "describe",
    "equity_curve",
    "find_gaps",
    "format_stats",
    "load",
    "mark_mitigation",
    "plan_session",
    "resample",
    "run",
    "simulate_trade",
    "summarize",
    "unmitigated_at",
]
