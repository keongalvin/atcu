import enum

import attrs

from atcu.schemas.assets import INDUSTRY_SECTOR_MAP
from atcu.schemas.assets import Industry
from atcu.schemas.assets import Sector
from atcu.schemas.assets import SubIndustry


class PairType(enum.StrEnum):
    """Classification of trading pair relationships."""

    GPU_AI = enum.auto()  # "GPU/AI"
    MEMORY = enum.auto()  # "Memory"
    ANALOG = enum.auto()  # "Analog"
    RF = enum.auto()  # "RF"
    FOUNDRY = enum.auto()  # "Foundry"
    IP_DESIGN = enum.auto()  # "IP/Design"
    FAB_EQUIPMENT = enum.auto()  # "Fab Equipment"
    NETWORKING = enum.auto()  # "Networking"
    OPTICAL = enum.auto()  # "Optical"
    EDA = enum.auto()  # "EDA"
    AI_INFRA = enum.auto()  # "AI Infrastructure"
    CLOUD = enum.auto()  # "Cloud/Hyperscaler"
    MEMORY_CHAIN = enum.auto()  # "Memory Supply Chain"


@attrs.frozen
class SymbolInfo:
    """Information about a tradeable symbol."""

    symbol: str
    name: str
    primary_industry: Industry
    market_cap_billions: float | None = attrs.field(default=None)
    focus: str = attrs.field(default="")
    sub_industry: SubIndustry | None = attrs.field(default=None)
    secondary_industries: tuple[Industry, ...] = attrs.field(factory=tuple)

    @property
    def sector(self) -> Sector:
        """Get the sector for this symbol."""
        return INDUSTRY_SECTOR_MAP[self.primary_industry]

    @property
    def all_industries(self) -> tuple[Industry, ...]:
        """Get all industries (primary + secondary)."""
        return self.primary_industry, *self.secondary_industries


@attrs.frozen
class TradingPair:
    """A potential pairs trading relationship."""

    symbol_a: str
    symbol_b: str
    pair_type: PairType
    correlation_driver: str

    @property
    def pair_id(self) -> str:
        """Canonical pair identifier (alphabetically ordered)."""
        symbols = sorted([self.symbol_a, self.symbol_b])
        return f"{symbols[0]}_{symbols[1]}"

    def __repr__(self) -> str:
        return f"TradingPair({self.pair_id!r}, type={self.pair_type.value!r})"
