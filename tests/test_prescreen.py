"""Tests for pair prescreening."""

import random

import polars as pl

from atcu.stats.prescreen import PrescreenResult
from atcu.stats.prescreen import prescreen_pair


class TestPrescreenPair:
    """Test prescreen_pair function."""

    def test_returns_prescreen_result(self):
        """Function returns a PrescreenResult instance."""
        lf = pl.LazyFrame(
            {
                "a": [100.0, 101.0, 102.0, 101.0, 100.0] * 20,
                "b": [50.0, 50.5, 51.0, 50.5, 50.0] * 20,
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert isinstance(result, PrescreenResult)

    def test_highly_correlated_prices_have_high_correlation(self):
        """Perfectly comoving prices have correlation near 1."""
        # Prices that move together proportionally
        lf = pl.LazyFrame(
            {
                "a": [100.0, 110.0, 105.0, 115.0, 110.0] * 20,
                "b": [50.0, 55.0, 52.5, 57.5, 55.0] * 20,
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert result.correlation > 0.99

    def test_uncorrelated_prices_have_low_correlation(self):
        """Uncorrelated prices have low correlation."""
        # One trends up, other oscillates
        lf = pl.LazyFrame(
            {
                "a": list(range(100, 200)),
                "b": [50 + (i % 10) for i in range(100)],
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert result.correlation < 0.5

    def test_spread_statistics_computed(self):
        """Spread mean, std, and range are computed."""
        lf = pl.LazyFrame(
            {
                "a": [100.0, 102.0, 104.0, 102.0, 100.0] * 20,
                "b": [100.0, 101.0, 102.0, 101.0, 100.0] * 20,
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert result.spread_mean is not None
        assert result.spread_std is not None
        assert result.spread_range > 0

    def test_zero_crossings_counted(self):
        """Zero crossings are counted for mean-reverting spread."""
        # Create data where normalized spread oscillates around zero
        n = 100
        a_prices = [100.0 + (i % 4) for i in range(n)]
        b_prices = [100.0 + ((i + 2) % 4) for i in range(n)]

        lf = pl.LazyFrame({"a": a_prices, "b": b_prices})

        result = prescreen_pair(lf, "a", "b")

        assert result.zero_crossings > 0

    def test_n_observations_matches_input_length(self):
        """n_observations reflects input data length."""
        n = 50
        lf = pl.LazyFrame(
            {
                "a": [100.0 + i * 0.1 for i in range(n)],
                "b": [100.0 + i * 0.1 for i in range(n)],
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert result.n_observations == n

    def test_crossing_rate_property(self):
        """crossing_rate computes zero crossings per 100 observations."""
        n = 200
        a_prices = [100.0 + (i % 4) for i in range(n)]
        b_prices = [100.0 + ((i + 2) % 4) for i in range(n)]

        lf = pl.LazyFrame({"a": a_prices, "b": b_prices})

        result = prescreen_pair(lf, "a", "b")

        expected_rate = result.zero_crossings * 100 / result.n_observations
        assert result.crossing_rate == expected_rate


class TestPrescreenPairHalfLife:
    """Test half-life computation in prescreen_pair."""

    def test_mean_reverting_spread_has_finite_half_life(self):
        """Spread that reverts to mean has finite half-life."""
        # Create mean-reverting spread with AR(1) beta = 0.9
        # Both series vary so correlation is defined
        n = 500
        random.seed(42)

        # Base price with some randomness
        base = [100.0]
        for _ in range(n - 1):
            base.append(base[-1] + random.gauss(0, 0.5))

        # Mean-reverting spread around zero
        spread = [0.0]
        beta = 0.9
        for _ in range(n - 1):
            spread.append(beta * spread[-1] + random.gauss(0, 0.1))

        a_prices = [b + s for b, s in zip(base, spread, strict=False)]
        b_prices = base

        lf = pl.LazyFrame({"a": a_prices, "b": b_prices})

        result = prescreen_pair(lf, "a", "b")

        assert result.half_life is not None
        assert result.half_life > 0

    def test_explosive_spread_has_null_half_life(self):
        """Spread that explodes (beta > 1) has null half-life."""
        # Create explosive series where spread grows exponentially
        n = 100

        # Base price
        base = [100.0 + i * 0.1 for i in range(n)]

        # Explosive spread: each step multiplies by 1.05
        spread = [0.1]
        for _ in range(n - 1):
            spread.append(spread[-1] * 1.05)

        a_prices = [b + s for b, s in zip(base, spread, strict=False)]
        b_prices = base

        lf = pl.LazyFrame({"a": a_prices, "b": b_prices})

        result = prescreen_pair(lf, "a", "b")

        assert result.half_life is None


class TestPrescreenPairPassed:
    """Test passed flag logic."""

    def test_passes_when_all_criteria_met(self):
        """Pair passes when correlation, half-life, and crossings are good."""
        # Create highly correlated, mean-reverting pair
        n = 500
        base = [100.0]
        for i in range(n - 1):
            base.append(base[-1] + (0.5 if i % 10 < 5 else -0.5))

        a_prices = base
        b_prices = [p * 0.5 + 0.01 * (i % 3 - 1) for i, p in enumerate(base)]

        lf = pl.LazyFrame({"a": a_prices, "b": b_prices})

        result = prescreen_pair(
            lf,
            "a",
            "b",
            min_correlation=0.7,
            max_half_life=2000.0,
            min_zero_crossings=5,
        )

        if (
            result.correlation >= 0.7
            and result.half_life
            and result.half_life <= 2000.0
            and result.zero_crossings >= 5
        ):
            assert result.passed is True

    def test_fails_when_correlation_too_low(self):
        """Pair fails when correlation below threshold."""
        # Uncorrelated prices
        lf = pl.LazyFrame(
            {
                "a": list(range(100, 200)),
                "b": [50 + (i % 10) for i in range(100)],
            }
        )

        result = prescreen_pair(lf, "a", "b", min_correlation=0.9)

        assert result.passed is False

    def test_fails_when_half_life_null(self):
        """Pair fails when half-life is null (explosive spread)."""
        # Explosive spread (beta > 1)
        n = 100
        base = [100.0 + i * 0.1 for i in range(n)]
        spread = [0.1]
        for _ in range(n - 1):
            spread.append(spread[-1] * 1.05)

        lf = pl.LazyFrame(
            {
                "a": [b + s for b, s in zip(base, spread, strict=False)],
                "b": base,
            }
        )

        result = prescreen_pair(lf, "a", "b")

        assert result.half_life is None
        assert result.passed is False

    def test_fails_when_zero_crossings_too_few(self):
        """Pair fails when zero crossings below threshold."""
        # Prices that don't cross much
        lf = pl.LazyFrame(
            {
                "a": [100.0 + i * 0.01 for i in range(100)],
                "b": [100.0 + i * 0.01 for i in range(100)],
            }
        )

        result = prescreen_pair(lf, "a", "b", min_zero_crossings=50)

        assert result.passed is False
