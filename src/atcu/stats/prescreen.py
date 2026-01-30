from typing import TYPE_CHECKING

import attrs
import polars as pl

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@attrs.frozen
class PrescreenResult:
    """Result of pair prescreening based on normalized prices."""

    correlation: float  # Correlation of normalized prices
    spread_mean: float  # Mean of normalized spread
    spread_std: float  # Std of normalized spread
    spread_range: float  # Max - min of normalized spread
    zero_crossings: int  # Number of times spread crosses zero
    half_life: float | None  # Estimated half-life of mean reversion (None if divergent)
    passed: bool  # Whether pair passes prescreening
    n_observations: int  # Number of observations used

    @property
    def crossing_rate(self) -> float:
        """Zero crossings per 100 observations."""
        return self.zero_crossings * 100 / max(1, self.n_observations)


def prescreen_pair(
    lf: pl.LazyFrame,
    col_a: str,
    col_b: str,
    min_correlation: float = 0.7,
    max_half_life: float = 2000.0,  # ~5 trading days for minute data
    min_zero_crossings: int = 5,
) -> PrescreenResult:
    """
    Prescreen a pair based on normalized price characteristics.

    Criteria for passing:
    1. High correlation of normalized prices (co-movement)
    2. Reasonable half-life (mean reversion speed)
    3. Sufficient zero crossings (actual mean-reverting behavior)
    """
    ln2 = 0.6931471805599453  # math.log(2)

    # Normalize prices: divide by first value and compute spread
    lf_norm = lf.with_columns(
        (pl.col(col_a) / pl.col(col_a).first()).alias("norm_a"),
        (pl.col(col_b) / pl.col(col_b).first()).alias("norm_b"),
    ).with_columns(
        (pl.col("norm_a") - pl.col("norm_b")).alias("spread"),
    )

    # Add spread lag for AR(1) and spread mean for zero crossings
    lf_with_lag = lf_norm.with_columns(
        pl.col("spread").shift(1).alias("spread_lag"),
        pl.col("spread").mean().alias("spread_mean_val"),
    )

    # Zero crossings: count sign changes in demeaned spread
    lf_crossings = (
        lf_with_lag.with_columns(
            (pl.col("spread") - pl.col("spread_mean_val")).sign().alias("sign"),
        )
        .with_columns(
            pl.col("sign").shift(1).alias("prev_sign"),
        )
        .with_columns(
            (
                (pl.col("sign") != pl.col("prev_sign"))
                & (pl.col("sign") != 0)
                & (pl.col("prev_sign") != 0)
            ).alias("crossed"),
        )
    )

    # Aggregate all metrics including AR(1) beta for half-life
    # β = Cov(spread, spread_lag) / Var(spread_lag)
    # half_life = -ln(2) / ln(β) if 0 < β < 1 else None
    row = (
        lf_crossings.select(
            pl.corr("norm_a", "norm_b").alias("correlation"),
            pl.col("spread").mean().alias("spread_mean"),
            pl.col("spread").std().alias("spread_std"),
            (pl.col("spread").max() - pl.col("spread").min()).alias("spread_range"),
            pl.col("crossed").sum().alias("zero_crossings"),
            pl.len().alias("n_observations"),
            # AR(1) beta for half-life calculation
            (pl.cov("spread", "spread_lag") / pl.col("spread_lag").var()).alias("beta"),
        )
        .with_columns(
            # Half-life: -ln(2) / ln(β) if 0 < β < 1, else null
            pl.when((pl.col("beta") > 0) & (pl.col("beta") < 1))
            .then(-ln2 / pl.col("beta").log())
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("half_life"),
        )
        .collect()
        .row(0, named=True)
    )

    passed = (
        row["correlation"] >= min_correlation
        and row["half_life"] is not None
        and row["half_life"] <= max_half_life
        and row["zero_crossings"] >= min_zero_crossings
    )

    return PrescreenResult(
        correlation=row["correlation"],
        spread_mean=row["spread_mean"],
        spread_std=row["spread_std"],
        spread_range=row["spread_range"],
        zero_crossings=row["zero_crossings"],
        half_life=row["half_life"],
        passed=passed,
        n_observations=row["n_observations"],
    )


def plot_prescreen(
    lf: pl.LazyFrame,
    col_a: str,
    col_b: str,
    result: PrescreenResult,
) -> "Figure":
    """
    Plot normalized prices and spread for visual inspection.

    Returns a matplotlib Figure with two subplots:
    - Top: Normalized prices for both series
    - Bottom: Spread with mean and ±1σ bands
    """
    import matplotlib.pyplot as plt

    # Compute normalized data
    df = (
        lf.with_columns(
            (pl.col(col_a) / pl.col(col_a).first()).alias("norm_a"),
            (pl.col(col_b) / pl.col(col_b).first()).alias("norm_b"),
        )
        .with_columns(
            (pl.col("norm_a") - pl.col("norm_b")).alias("spread"),
        )
        .collect()
    )

    norm_a = df["norm_a"].to_numpy()
    norm_b = df["norm_b"].to_numpy()
    spread = df["spread"].to_numpy()
    x = range(len(norm_a))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: normalized prices
    ax1.plot(x, norm_a, label=col_a, alpha=0.8)
    ax1.plot(x, norm_b, label=col_b, alpha=0.8)
    ax1.set_ylabel("Normalized Price")
    ax1.legend(loc="upper left")
    half_life_str = f"{result.half_life:.1f}" if result.half_life else "None"
    ax1.set_title(
        f"Prescreen: {'PASS' if result.passed else 'FAIL'} | "
        f"corr={result.correlation:.3f} | "
        f"half_life={half_life_str} | "
        f"crossings={result.zero_crossings}"
    )
    ax1.grid(True, alpha=0.3)

    # Bottom: spread with bands
    ax2.plot(x, spread, label="Spread", color="purple", alpha=0.8)
    ax2.axhline(result.spread_mean, color="black", linestyle="--", label="Mean")
    ax2.axhline(
        result.spread_mean + result.spread_std,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )
    ax2.axhline(
        result.spread_mean - result.spread_std,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="±1σ",
    )
    ax2.fill_between(
        x,
        result.spread_mean - result.spread_std,
        result.spread_mean + result.spread_std,
        alpha=0.1,
        color="gray",
    )
    ax2.set_xlabel("Observation")
    ax2.set_ylabel("Spread (norm_a - norm_b)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
