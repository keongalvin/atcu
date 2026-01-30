"""Cointegration testing for pairs trading using Engle-Granger method.

Pure NumPy implementation without statsmodels dependency.

References:
- MacKinnon, J.G. (1994). "Approximate Asymptotic Distribution Functions for
  Unit-Root and Cointegration Tests". Journal of Business & Economic Statistics.
  Table used: "c" case (constant, no trend).
- Engle, R.F. & Granger, C.W.J. (1987). "Co-Integration and Error Correction:
  Representation, Estimation, and Testing". Econometrica, 55(2), 251-276.
"""

from __future__ import annotations

from typing import Sequence

import attrs
import numpy as np
from matplotlib.figure import Figure

# ADF critical values (MacKinnon 1994) for "c" (constant, no trend)
# Key: sample size threshold, value: (1%, 5%, 10%)
ADF_CRITICAL_VALUES: dict[int, tuple[float, float, float]] = {
    25: (-3.75, -3.00, -2.63),
    50: (-3.58, -2.93, -2.60),
    100: (-3.51, -2.89, -2.58),
    250: (-3.46, -2.88, -2.57),
    500: (-3.44, -2.87, -2.57),
    10000: (-3.43, -2.86, -2.57),
}


@attrs.frozen
class ADFResult:
    """Result of Augmented Dickey-Fuller test for stationarity.

    The ADF test tests the null hypothesis that a unit root is present
    in a time series (i.e., the series is non-stationary). A rejection
    of the null hypothesis indicates the series is stationary.
    """

    statistic: float
    pvalue_approx: str  # "< 0.01", "< 0.05", "< 0.10", "> 0.10"
    critical_values: dict[str, float]
    is_stationary_1pct: bool
    is_stationary_5pct: bool
    is_stationary_10pct: bool


@attrs.frozen
class EngleGrangerResult:
    """Result of Engle-Granger two-step cointegration test.

    Step 1: OLS regression y = alpha + beta*x + epsilon
    Step 2: ADF test on residuals

    If residuals are stationary, the series are cointegrated with
    cointegrating vector (1, -beta).
    """

    cointegrated: bool
    adf_statistic: float
    pvalue_approx: str
    hedge_ratio: float  # beta from y = alpha + beta*x + epsilon
    intercept: float  # alpha from y = alpha + beta*x + epsilon
    residual_std: float


@attrs.frozen
class CointegrationResult:
    """Full cointegration analysis result.

    Contains the Engle-Granger test result plus additional data
    useful for pairs trading strategy implementation.
    """

    engle_granger: EngleGrangerResult
    residuals: np.ndarray = attrs.field(eq=False, repr=False)
    prices_a: np.ndarray = attrs.field(eq=False, repr=False)
    prices_b: np.ndarray = attrs.field(eq=False, repr=False)

    @property
    def cointegrated(self) -> bool:
        """Whether the series are cointegrated."""
        return self.engle_granger.cointegrated

    @property
    def hedge_ratio(self) -> float:
        """Hedge ratio (beta) from cointegrating regression."""
        return self.engle_granger.hedge_ratio

    def plot(
        self,
        label_a: str = "Series A",
        label_b: str = "Series B",
        timestamps: Sequence | None = None,
    ) -> Figure:
        """Plot prices and spread for visual inspection.

        Args:
            label_a: Label for first price series
            label_b: Label for second price series
            timestamps: Optional x-axis values (e.g., dates)

        Returns:
            matplotlib Figure with two subplots:
            - Top: Both price series
            - Bottom: Spread (residuals) with mean and ±1σ bands
        """
        x = timestamps if timestamps is not None else range(len(self.prices_a))
        spread = self.residuals
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        fig = Figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        # Top: price series
        ax1.plot(x, self.prices_a, label=label_a, alpha=0.8)
        ax1.plot(x, self.prices_b, label=label_b, alpha=0.8)
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        status = "COINTEGRATED" if self.cointegrated else "NOT COINTEGRATED"
        ax1.set_title(
            f"{label_a}/{label_b} | {status} | "
            f"ADF={self.engle_granger.adf_statistic:.3f} "
            f"(p{self.engle_granger.pvalue_approx}) | "
            f"hedge={self.hedge_ratio:.4f}"
        )
        ax1.grid(True, alpha=0.3)

        # Bottom: spread with bands
        ax2.plot(x, spread, label="Spread", color="purple", alpha=0.8)
        ax2.axhline(spread_mean, color="black", linestyle="--", label="Mean")
        ax2.axhline(spread_mean + spread_std, color="gray", linestyle=":", alpha=0.7)
        ax2.axhline(
            spread_mean - spread_std,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="±1σ",
        )
        ax2.fill_between(
            x,
            spread_mean - spread_std,
            spread_mean + spread_std,
            alpha=0.1,
            color="gray",
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel(f"Spread ({label_a} - {self.hedge_ratio:.2f}×{label_b})")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig


def get_adf_critical_values(n: int) -> tuple[float, float, float]:
    """Get ADF critical values for sample size n.

    Returns critical values for 1%, 5%, and 10% significance levels
    based on MacKinnon (1994) tables.

    Args:
        n: Sample size

    Returns:
        Tuple of (1% cv, 5% cv, 10% cv)
    """
    for threshold in sorted(ADF_CRITICAL_VALUES.keys()):
        if n <= threshold:
            return ADF_CRITICAL_VALUES[threshold]
    return ADF_CRITICAL_VALUES[10000]


def adf_test(series: np.ndarray, max_lags: int = 1) -> ADFResult:
    """Augmented Dickey-Fuller test for stationarity.

    Tests H0: series has a unit root (non-stationary)
    vs H1: series is stationary

    Uses regression: delta_y_t = alpha + gamma*y_{t-1} + sum(beta_i * delta_y_{t-i}) + epsilon_t
    Test statistic is t-statistic for gamma.

    Args:
        series: Time series to test
        max_lags: Number of lagged differences to include (default 1)

    Returns:
        ADFResult with test statistic and stationarity flags

    Raises:
        ValueError: If series is too short for reliable test
    """
    n = len(series)
    if n < 20:
        msg = f"Series too short for ADF test: {n} < 20"
        raise ValueError(msg)

    # Compute differences
    dy = np.diff(series)
    y_lag = series[:-1]

    # Adjust lags if series is short
    effective_n = len(dy) - max_lags
    if effective_n < 10:
        max_lags = max(0, len(dy) - 10)
        effective_n = len(dy) - max_lags

    # Trim to account for lags
    dy_trimmed = dy[max_lags:]
    y_lag_trimmed = y_lag[max_lags:]

    # Build design matrix: [1, y_{t-1}, delta_y_{t-1}, ..., delta_y_{t-p}]
    x_cols = [np.ones(effective_n), y_lag_trimmed]

    # Add lagged differences
    for lag in range(1, max_lags + 1):
        lagged_diff = dy[max_lags - lag : -lag] if lag > 0 else dy[max_lags:]
        x_cols.append(lagged_diff)

    x = np.column_stack(x_cols)

    # OLS: (X'X)^{-1} X'y
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x, dy_trimmed, rcond=None)
    except np.linalg.LinAlgError:
        return ADFResult(
            statistic=0.0,
            pvalue_approx="> 0.10",
            critical_values={"1%": 0.0, "5%": 0.0, "10%": 0.0},
            is_stationary_1pct=False,
            is_stationary_5pct=False,
            is_stationary_10pct=False,
        )

    # Compute residuals and standard errors
    fitted = x @ coeffs
    resid = dy_trimmed - fitted
    sigma2 = np.sum(resid**2) / (effective_n - len(coeffs))

    # Standard errors of coefficients
    try:
        var_covar = sigma2 * np.linalg.inv(x.T @ x)
        se = np.sqrt(np.diag(var_covar))
    except np.linalg.LinAlgError:
        se = np.ones(len(coeffs))

    # t-statistic for gamma (coefficient on y_{t-1})
    gamma = coeffs[1]
    t_stat = gamma / se[1] if se[1] > 0 else 0.0

    # Get critical values
    cv_1, cv_5, cv_10 = get_adf_critical_values(effective_n)

    # Determine approximate p-value
    if t_stat < cv_1:
        pvalue_approx = "< 0.01"
    elif t_stat < cv_5:
        pvalue_approx = "< 0.05"
    elif t_stat < cv_10:
        pvalue_approx = "< 0.10"
    else:
        pvalue_approx = "> 0.10"

    return ADFResult(
        statistic=float(t_stat),
        pvalue_approx=pvalue_approx,
        critical_values={"1%": cv_1, "5%": cv_5, "10%": cv_10},
        is_stationary_1pct=bool(t_stat < cv_1),
        is_stationary_5pct=bool(t_stat < cv_5),
        is_stationary_10pct=bool(t_stat < cv_10),
    )


def engle_granger_test(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    significance: float = 0.05,
) -> EngleGrangerResult:
    """Engle-Granger two-step cointegration test.

    Step 1: OLS regression y = alpha + beta*x + epsilon
    Step 2: ADF test on residuals

    If residuals are stationary, the series are cointegrated.

    Args:
        prices_a: First price series (dependent variable y)
        prices_b: Second price series (independent variable x)
        significance: Significance level for ADF test (0.01, 0.05, or 0.10)

    Returns:
        EngleGrangerResult with cointegration test results

    Raises:
        ValueError: If price series have different lengths
    """
    n = len(prices_a)
    if n != len(prices_b):
        msg = f"Price series must have same length: {n} != {len(prices_b)}"
        raise ValueError(msg)

    # Step 1: OLS regression y = alpha + beta*x
    x = np.column_stack([np.ones(n), prices_b])
    coeffs, _, _, _ = np.linalg.lstsq(x, prices_a, rcond=None)
    intercept, hedge_ratio = coeffs[0], coeffs[1]

    # Compute residuals (spread)
    residuals = prices_a - (intercept + hedge_ratio * prices_b)
    residual_std = float(np.std(residuals))

    # Step 2: ADF test on residuals
    adf_result = adf_test(residuals)

    # Determine cointegration based on significance level
    if significance <= 0.01:
        cointegrated = adf_result.is_stationary_1pct
    elif significance <= 0.05:
        cointegrated = adf_result.is_stationary_5pct
    else:
        cointegrated = adf_result.is_stationary_10pct

    return EngleGrangerResult(
        cointegrated=bool(cointegrated),
        adf_statistic=adf_result.statistic,
        pvalue_approx=adf_result.pvalue_approx,
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        residual_std=residual_std,
    )


def cointegration_test(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    significance: float = 0.05,
) -> CointegrationResult:
    """Full cointegration analysis for a pair of price series.

    Performs Engle-Granger cointegration test and returns comprehensive
    results including residuals for pairs trading strategy implementation.

    Args:
        prices_a: First price series (dependent variable)
        prices_b: Second price series (independent variable)
        significance: Significance level for ADF test (default 0.05)

    Returns:
        CointegrationResult with test results and residuals
    """
    eg_result = engle_granger_test(prices_a, prices_b, significance)

    # Compute residuals for storage
    residuals = prices_a - eg_result.hedge_ratio * prices_b - eg_result.intercept

    return CointegrationResult(
        engle_granger=eg_result,
        residuals=residuals,
        prices_a=prices_a,
        prices_b=prices_b,
    )
