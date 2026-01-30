import pathlib

import attrs
import yaml

from atcu.schemas.assets import Industry
from atcu.schemas.assets import SubIndustry
from atcu.schemas.pairs import PairType
from atcu.schemas.pairs import SymbolInfo
from atcu.schemas.pairs import TradingPair

from .dto import MDUniverse


def _load_symbols(sectors: dict) -> dict[str, SymbolInfo]:
    """Parse sectors hierarchy into SymbolInfo dict."""
    symbols = {}

    for industries in sectors.values():
        for industry_key, industry_data in industries["industries"].items():
            industry_enum = Industry[industry_key.upper()]

            for sym in industry_data["symbols"]:
                symbol_info = _parse_symbol(sym, industry_enum)
                symbols[symbol_info.symbol] = symbol_info

    return symbols


def _parse_symbol(sym: dict, primary_industry: Industry) -> SymbolInfo:
    """Parse a single symbol dict into SymbolInfo."""
    sub_industry = None
    if sub_ind_key := sym.get("sub_industry"):
        sub_industry = SubIndustry[sub_ind_key.upper()]

    secondary_industries = tuple(
        Industry[ind.upper()] for ind in sym.get("secondary_industries", [])
    )

    return SymbolInfo(
        symbol=sym["symbol"],
        name=sym["name"],
        primary_industry=primary_industry,
        market_cap_billions=sym.get("market_cap_billions"),
        focus=sym.get("focus", ""),
        sub_industry=sub_industry,
        secondary_industries=secondary_industries,
    )


def _load_pairs(pairs_data: dict) -> list[TradingPair]:
    """Parse pairs categories into TradingPair list."""
    pairs = []

    for category_pairs in pairs_data.values():
        for pair in category_pairs:
            symbols = pair["symbols"]
            pairs.append(
                TradingPair(
                    symbol_a=symbols[0],
                    symbol_b=symbols[1],
                    pair_type=PairType[pair["type"].upper()],
                    correlation_driver=pair["driver"],
                )
            )

    return pairs


@attrs.frozen
class MDMetadataConfig:
    path: pathlib.Path


class MDMetadata:
    def __init__(self, config: MDMetadataConfig):
        self.config = config

    def get_universe(self) -> MDUniverse:
        with self.config.path.open() as f:
            data = yaml.safe_load(f)

        symbols = _load_symbols(data["sectors"])
        pairs = _load_pairs(data.get("pairs", {}))

        return MDUniverse(symbols=symbols, pairs=pairs)
