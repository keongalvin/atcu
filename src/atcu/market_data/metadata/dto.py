import attrs

from atcu.schemas.pairs import SymbolInfo
from atcu.schemas.pairs import TradingPair


@attrs.frozen
class MDUniverse:
    """Container for loaded universe data."""

    symbols: dict[str, SymbolInfo]
    pairs: list[TradingPair]

    def __repr__(self):
        sym_keys = list(self.symbols.keys())
        sym_preview = sym_keys[:3]
        sym_str = ", ".join(sym_preview)
        if len(sym_keys) > 3:
            sym_str += ", ..."

        pair_preview = [p.pair_id for p in self.pairs[:2]]
        pair_str = ", ".join(pair_preview)
        if len(self.pairs) > 2:
            pair_str += ", ..."

        return (
            f"Universe("
            f"symbols=[{sym_str}] ({len(self.symbols)}), "
            f"pairs=[{pair_str}] ({len(self.pairs)}))"
        )
