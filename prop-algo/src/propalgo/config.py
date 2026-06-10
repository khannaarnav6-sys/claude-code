"""Load config.yaml into typed objects used across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data_cache"


@dataclass(frozen=True)
class AccountRules:
    name: str = "apex50k"
    start_balance: float = 50_000.0
    profit_target: float = 3_000.0
    trailing_drawdown: float = 2_500.0
    max_contracts: int = 10
    min_trading_days: int = 7
    eval_fee: float = 35.0


@dataclass(frozen=True)
class Instrument:
    symbol: str
    dukascopy: str
    yahoo: str
    point_value: float
    tick_size: float

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size


@dataclass(frozen=True)
class Costs:
    commission_rt: float = 4.0
    slippage_ticks: int = 1


@dataclass
class Config:
    account: AccountRules
    instruments: dict[str, Instrument]
    costs: Costs
    strategies: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)
    live: dict = field(default_factory=dict)


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    instruments = {
        sym: Instrument(symbol=sym, **spec)
        for sym, spec in raw.get("instruments", {}).items()
    }
    return Config(
        account=AccountRules(**raw.get("account", {})),
        instruments=instruments,
        costs=Costs(**raw.get("costs", {})),
        strategies=raw.get("strategies", {}),
        sizing=raw.get("sizing", {}),
        live=raw.get("live", {}),
    )
