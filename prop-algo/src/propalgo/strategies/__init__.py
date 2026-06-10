from .base import Signal, Strategy
from .midday import MiddayBreakout
from .momentum import TrendDayMomentum
from .orb import OpeningRangeBreakout


def build_strategies(cfg: dict) -> list[Strategy]:
    out: list[Strategy] = []
    orb = cfg.get("orb", {})
    if orb.get("enabled", True):
        out.append(OpeningRangeBreakout(orb))
    mom = cfg.get("momentum", {})
    if mom.get("enabled", True):
        out.append(TrendDayMomentum(mom))
    mid = cfg.get("midday", {})
    if mid.get("enabled", False):
        out.append(MiddayBreakout(mid))
    return out
