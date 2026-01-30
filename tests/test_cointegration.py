"""Tests for cointegration analysis module.

Tests for Augmented Dickey-Fuller (ADF) test and Engle-Granger cointegration test.

References:
- MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution Functions for
  Unit-Root and Cointegration Tests". Journal of Business & Economic Statistics.
- Engle, R.F. & Granger, C.W.J. (1987). "Co-Integration and Error Correction:
  Representation, Estimation, and Testing". Econometrica, 55(2), 251-276.
"""

import numpy as np

from atcu.stats.cointegration import ADFResult
from atcu.stats.cointegration import CointegrationResult
from atcu.stats.cointegration import EngleGrangerResult
from atcu.stats.cointegration import adf_test
from atcu.stats.cointegration import cointegration_test
from atcu.stats.cointegration import engle_granger_test
from atcu.stats.cointegration import get_adf_critical_values


class TestGetADFCriticalValues:
    """Test ADF critical value lookup."""

    def test_returns_tuple_of_three_floats(self):
        """Critical values returned as (1%, 5%, 10%) tuple."""
        cv = get_adf_critical_values(100)

        assert isinstance(cv, tuple)
        assert len(cv) == 3
        assert all(isinstance(v, float) for v in cv)

    def test_small_sample_uses_conservative_values(self):
        """Smaller samples get more conservative (larger magnitude) critical values."""
        cv_small = get_adf_critical_values(25)
        cv_large = get_adf_critical_values(500)

        # 1% critical value should be more negative for small samples
        assert cv_small[0] < cv_large[0]

    def test_very_large_sample_uses_asymptotic_values(self):
        """Samples larger than table use asymptotic values."""
        cv_large = get_adf_critical_values(10000)
        cv_huge = get_adf_critical_values(100000)

        assert cv_large == cv_huge


class TestADFTest:
    """Test Augmented Dickey-Fuller test for stationarity."""

    def test_returns_adf_result(self):
        """Function returns an ADFResult instance."""
        series = np.random.randn(100)

        result = adf_test(series)

        assert isinstance(result, ADFResult)

    def test_stationary_series_rejects_unit_root(self):
        """White noise series should reject unit root hypothesis."""
        np.random.seed(42)
        # White noise is stationary
        series = np.random.randn(500)

        result = adf_test(series)

        # Should reject at 5% level (statistic < critical value)
        assert result.is_stationary_5pct is True

    def test_random_walk_fails_to_reject_unit_root(self):
        """Random walk should fail to reject unit root hypothesis."""
        np.random.seed(42)
        # Random walk: y_t = y_{t-1} + e_t
        n = 500
        series = np.cumsum(np.random.randn(n))

        result = adf_test(series)

        # Should NOT reject at 5% level
        assert result.is_stationary_5pct is False

    def test_mean_reverting_series_is_stationary(self):
        """AR(1) with |beta| < 1 should be stationary."""
        np.random.seed(42)
        n = 500
        series = np.zeros(n)
        beta = 0.5  # Mean-reverting
        for t in range(1, n):
            series[t] = beta * series[t - 1] + np.random.randn()

        result = adf_test(series)

        assert result.is_stationary_5pct is True

    def test_critical_values_in_result(self):
        """Result contains critical values dict."""
        series = np.random.randn(100)

        result = adf_test(series)

        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values

    def test_pvalue_approx_is_string_category(self):
        """Approximate p-value is categorical string."""
        series = np.random.randn(100)

        result = adf_test(series)

        valid_pvalues = {"< 0.01", "< 0.05", "< 0.10", "> 0.10"}
        assert result.pvalue_approx in valid_pvalues

    def test_raises_on_short_series(self):
        """Raises ValueError for series too short."""
        series = np.array([1.0, 2.0, 3.0])

        try:
            adf_test(series)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "too short" in str(e).lower()


class TestEngleGrangerTest:
    """Test Engle-Granger two-step cointegration test."""

    def test_returns_engle_granger_result(self):
        """Function returns an EngleGrangerResult instance."""
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = engle_granger_test(prices_a, prices_b)

        assert isinstance(result, EngleGrangerResult)

    def test_cointegrated_series_detected(self):
        """Two cointegrated series should be detected."""
        np.random.seed(42)
        n = 500

        # Create cointegrated pair: y_t = 2*x_t + stationary_error
        x = np.cumsum(np.random.randn(n))  # Random walk
        stationary_error = np.zeros(n)
        for t in range(1, n):
            stationary_error[t] = 0.5 * stationary_error[t - 1] + np.random.randn()
        y = 2 * x + stationary_error

        result = engle_granger_test(y, x)

        assert result.cointegrated is True

    def test_non_cointegrated_series_detected(self):
        """Two independent random walks should not be cointegrated."""
        np.random.seed(42)
        n = 500

        # Two independent random walks
        prices_a = np.cumsum(np.random.randn(n)) + 100
        prices_b = np.cumsum(np.random.randn(n)) + 50

        result = engle_granger_test(prices_a, prices_b)

        assert result.cointegrated is False

    def test_hedge_ratio_computed(self):
        """Hedge ratio (beta) is computed from OLS regression."""
        np.random.seed(42)
        n = 500

        # y = 2*x + noise
        x = np.cumsum(np.random.randn(n)) + 100
        y = 2 * x + np.random.randn(n) * 0.1

        result = engle_granger_test(y, x)

        # Hedge ratio should be close to 2
        assert abs(result.hedge_ratio - 2.0) < 0.1

    def test_intercept_computed(self):
        """Intercept (alpha) is computed from OLS regression."""
        np.random.seed(42)
        n = 500

        x = np.linspace(100, 200, n)
        y = 50 + 1.5 * x + np.random.randn(n) * 0.1

        result = engle_granger_test(y, x)

        # Intercept should be close to 50
        assert abs(result.intercept - 50.0) < 5.0

    def test_residual_std_computed(self):
        """Residual standard deviation is computed."""
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = engle_granger_test(prices_a, prices_b)

        assert result.residual_std > 0

    def test_raises_on_length_mismatch(self):
        """Raises ValueError when series lengths differ."""
        prices_a = np.random.randn(100)
        prices_b = np.random.randn(50)

        try:
            engle_granger_test(prices_a, prices_b)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "length" in str(e).lower()

    def test_significance_level_affects_result(self):
        """Different significance levels affect cointegration decision."""
        np.random.seed(123)
        n = 200

        # Create marginally cointegrated pair
        x = np.cumsum(np.random.randn(n))
        stationary_error = np.zeros(n)
        for t in range(1, n):
            stationary_error[t] = 0.8 * stationary_error[t - 1] + np.random.randn()
        y = 1.5 * x + stationary_error * 2

        result_1pct = engle_granger_test(y, x, significance=0.01)
        result_10pct = engle_granger_test(y, x, significance=0.10)

        # More lenient threshold should be at least as likely to find cointegration
        if result_1pct.cointegrated:
            assert result_10pct.cointegrated


class TestCointegrationTest:
    """Test main cointegration_test function (integration of ADF + E-G)."""

    def test_returns_cointegration_result(self):
        """Function returns a CointegrationResult instance."""
        np.random.seed(42)
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = cointegration_test(prices_a, prices_b)

        assert isinstance(result, CointegrationResult)

    def test_result_contains_engle_granger_result(self):
        """Result contains full EngleGrangerResult."""
        np.random.seed(42)
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = cointegration_test(prices_a, prices_b)

        assert isinstance(result.engle_granger, EngleGrangerResult)

    def test_residuals_stored_for_analysis(self):
        """Regression residuals are stored for further analysis."""
        np.random.seed(42)
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = cointegration_test(prices_a, prices_b)

        assert result.residuals is not None
        assert len(result.residuals) == 100

    def test_cointegrated_property_delegates(self):
        """cointegrated property delegates to engle_granger result."""
        np.random.seed(42)
        n = 500

        x = np.cumsum(np.random.randn(n))
        stationary_error = np.zeros(n)
        for t in range(1, n):
            stationary_error[t] = 0.5 * stationary_error[t - 1] + np.random.randn()
        y = 2 * x + stationary_error

        result = cointegration_test(y, x)

        assert result.cointegrated == result.engle_granger.cointegrated

    def test_hedge_ratio_property_delegates(self):
        """hedge_ratio property delegates to engle_granger result."""
        np.random.seed(42)
        prices_a = np.random.randn(100) + 100
        prices_b = np.random.randn(100) + 50

        result = cointegration_test(prices_a, prices_b)

        assert result.hedge_ratio == result.engle_granger.hedge_ratio

    def test_spread_computed_from_residuals(self):
        """Spread is computed as y - hedge_ratio * x - intercept."""
        np.random.seed(42)
        n = 100
        prices_a = np.random.randn(n) + 100
        prices_b = np.random.randn(n) + 50

        result = cointegration_test(prices_a, prices_b)

        expected_spread = (
            prices_a
            - result.engle_granger.hedge_ratio * prices_b
            - result.engle_granger.intercept
        )
        np.testing.assert_array_almost_equal(result.residuals, expected_spread)


class TestCointegrationResultPlot:
    """Test CointegrationResult plotting functionality."""

    def test_plot_returns_figure(self):
        """plot() method returns a matplotlib Figure."""
        from matplotlib.figure import Figure

        np.random.seed(42)
        n = 100
        prices_a = np.random.randn(n) + 100
        prices_b = np.random.randn(n) + 50

        result = cointegration_test(prices_a, prices_b)

        fig = result.plot()

        assert isinstance(fig, Figure)

    def test_plot_with_labels(self):
        """plot() accepts custom labels for the series."""
        from matplotlib.figure import Figure

        np.random.seed(42)
        n = 100
        prices_a = np.random.randn(n) + 100
        prices_b = np.random.randn(n) + 50

        result = cointegration_test(prices_a, prices_b)

        fig = result.plot(label_a="NVDA", label_b="AMD")

        assert isinstance(fig, Figure)

    def test_plot_with_timestamps(self):
        """plot() can use timestamps for x-axis."""
        from matplotlib.figure import Figure

        np.random.seed(42)
        n = 100
        prices_a = np.random.randn(n) + 100
        prices_b = np.random.randn(n) + 50
        timestamps = list(range(n))

        result = cointegration_test(prices_a, prices_b)

        fig = result.plot(timestamps=timestamps)

        assert isinstance(fig, Figure)

    def test_prices_stored_for_plotting(self):
        """CointegrationResult stores prices for plotting."""
        np.random.seed(42)
        n = 100
        prices_a = np.random.randn(n) + 100
        prices_b = np.random.randn(n) + 50

        result = cointegration_test(prices_a, prices_b)

        assert result.prices_a is not None
        assert result.prices_b is not None
        assert len(result.prices_a) == n
        assert len(result.prices_b) == n
