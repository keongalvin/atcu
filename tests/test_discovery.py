"""Tests for universe.yaml loading into schemas."""

from pathlib import Path

from atcu.market_data.metadata.dto import MDUniverse
from atcu.market_data.metadata.service import MDMetadata
from atcu.market_data.metadata.service import MDMetadataConfig
from atcu.schemas.assets import Industry
from atcu.schemas.assets import Sector
from atcu.schemas.assets import SubIndustry
from atcu.schemas.pairs import PairType
from atcu.schemas.pairs import SymbolInfo
from atcu.schemas.pairs import TradingPair

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture_universe() -> MDUniverse:
    """Load universe from test fixture."""
    config = MDMetadataConfig(path=FIXTURES_DIR / "universe_minimal.yaml")
    return MDMetadata(config).get_universe()


class TestLoadUniverse:
    """Test loading the full universe from YAML."""

    def test_load_returns_symbols_dict(self):
        """get_universe returns a dict of symbol -> SymbolInfo."""
        universe = load_fixture_universe()

        assert isinstance(universe.symbols, dict)
        assert "NVDA" in universe.symbols
        assert isinstance(universe.symbols["NVDA"], SymbolInfo)

    def test_load_returns_pairs_list(self):
        """get_universe returns a list of TradingPair."""
        universe = load_fixture_universe()

        assert isinstance(universe.pairs, list)
        assert len(universe.pairs) > 0
        assert isinstance(universe.pairs[0], TradingPair)


class TestSymbolLoading:
    """Test individual symbol loading from YAML structure."""

    def test_symbol_with_all_fields(self):
        """Symbol with all optional fields populated."""
        universe = load_fixture_universe()
        nvda = universe.symbols["NVDA"]

        assert nvda.symbol == "NVDA"
        assert nvda.name == "NVIDIA Corporation"
        assert nvda.market_cap_billions == 4170
        assert nvda.focus == "GPUs, AI accelerators"
        assert nvda.sub_industry == SubIndustry.GPU_AI

    def test_symbol_primary_industry_from_hierarchy(self):
        """primary_industry comes from containing industry in YAML."""
        universe = load_fixture_universe()
        nvda = universe.symbols["NVDA"]

        assert nvda.primary_industry == Industry.SEMICONDUCTORS

    def test_symbol_sector_derived_from_industry(self):
        """sector property derives from primary_industry."""
        universe = load_fixture_universe()
        nvda = universe.symbols["NVDA"]

        assert nvda.sector == Sector.ELECTRONIC_TECHNOLOGY

    def test_symbol_without_sub_industry(self):
        """Symbol without sub_industry field has None."""
        universe = load_fixture_universe()
        avgo = universe.symbols["AVGO"]

        assert avgo.sub_industry is None

    def test_symbol_with_secondary_industries(self):
        """Symbol with secondary_industries list."""
        universe = load_fixture_universe()
        klac = universe.symbols["KLAC"]

        assert klac.primary_industry == Industry.SEMICONDUCTORS
        assert Industry.ELECTRONIC_PRODUCTION_EQUIPMENT in klac.secondary_industries

    def test_symbol_all_industries_property(self):
        """all_industries includes primary + secondary."""
        universe = load_fixture_universe()
        klac = universe.symbols["KLAC"]

        all_ind = klac.all_industries
        assert Industry.SEMICONDUCTORS in all_ind
        assert Industry.ELECTRONIC_PRODUCTION_EQUIPMENT in all_ind


class TestPairLoading:
    """Test trading pair loading from YAML structure."""

    def test_pair_symbols_mapped(self):
        """Pair symbols list maps to symbol_a and symbol_b."""
        universe = load_fixture_universe()

        # Find NVDA/AVGO pair
        nvda_avgo = next(
            (p for p in universe.pairs if {p.symbol_a, p.symbol_b} == {"NVDA", "AVGO"}),
            None,
        )
        assert nvda_avgo is not None

    def test_pair_type_converted_to_enum(self):
        """Pair type string converts to PairType enum."""
        universe = load_fixture_universe()

        nvda_avgo = next(
            (p for p in universe.pairs if {p.symbol_a, p.symbol_b} == {"NVDA", "AVGO"}),
            None,
        )
        assert nvda_avgo.pair_type == PairType.GPU_AI

    def test_pair_driver_mapped(self):
        """Pair driver field maps to correlation_driver."""
        universe = load_fixture_universe()

        nvda_avgo = next(
            (p for p in universe.pairs if {p.symbol_a, p.symbol_b} == {"NVDA", "AVGO"}),
            None,
        )
        assert nvda_avgo.correlation_driver == "Direct GPU competition"

    def test_pair_from_different_categories(self):
        """Pairs from different YAML categories all loaded."""
        universe = load_fixture_universe()

        # Equipment pair
        amat_lrcx = next(
            (p for p in universe.pairs if {p.symbol_a, p.symbol_b} == {"AMAT", "LRCX"}),
            None,
        )
        assert amat_lrcx is not None
        assert amat_lrcx.pair_type == PairType.FAB_EQUIPMENT

        # EDA pair
        snps_cdns = next(
            (p for p in universe.pairs if {p.symbol_a, p.symbol_b} == {"SNPS", "CDNS"}),
            None,
        )
        assert snps_cdns is not None
        assert snps_cdns.pair_type == PairType.EDA


class TestSymbolCount:
    """Test that all symbols are loaded."""

    def test_loads_all_sectors(self):
        """Symbols from all sectors are loaded."""
        universe = load_fixture_universe()

        # Check we have symbols from both sectors
        sectors = {s.sector for s in universe.symbols.values()}
        assert Sector.ELECTRONIC_TECHNOLOGY in sectors
        assert Sector.TECHNOLOGY_SERVICES in sectors

    def test_loads_symbols_from_multiple_industries(self):
        """Symbols from multiple industries within a sector."""
        universe = load_fixture_universe()

        industries = {s.primary_industry for s in universe.symbols.values()}
        assert Industry.SEMICONDUCTORS in industries
        assert Industry.ELECTRONIC_PRODUCTION_EQUIPMENT in industries
        assert Industry.PACKAGED_SOFTWARE in industries

    def test_loads_correct_symbol_count(self):
        """All 7 symbols in fixture are loaded."""
        universe = load_fixture_universe()

        assert len(universe.symbols) == 7

    def test_loads_correct_pair_count(self):
        """All 3 pairs in fixture are loaded."""
        universe = load_fixture_universe()

        assert len(universe.pairs) == 3


class TestTradingPairRepr:
    """Test TradingPair __repr__ method."""

    def test_repr_includes_pair_id(self):
        """Repr should include pair_id for easy identification."""
        pair = TradingPair(
            symbol_a="BTC",
            symbol_b="ETH",
            pair_type=PairType.GPU_AI,
            correlation_driver="Test driver",
        )

        repr_str = repr(pair)

        assert "BTC_ETH" in repr_str
