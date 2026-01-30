import attrs
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from atcu.schemas.pairs import TradingPair


@attrs.frozen
class PrescreenResult:
    """Result of pair prescreening based on normalized prices."""

    pair: TradingPair
    correlation: float
    spread_mean: float
    spread_std: float
    spread_range: float
    zero_crossings: int
    half_life: float | None
    passed: bool
    n_observations: int
    # Data for plotting
    timestamps: list = attrs.field(factory=list)
    norm_a: list[float] = attrs.field(factory=list)
    norm_b: list[float] = attrs.field(factory=list)
    spread: list[float] = attrs.field(factory=list)

    @property
    def crossing_rate(self) -> float:
        """Zero crossings per 100 observations."""
        return self.zero_crossings * 100 / max(1, self.n_observations)

    def plot(self) -> Figure:
        """Plot normalized prices and spread for visual inspection."""
        norm_a = np.array(self.norm_a)
        norm_b = np.array(self.norm_b)
        spread = np.array(self.spread)
        x = self.timestamps if self.timestamps else range(len(norm_a))

        fig = Figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        # Top: normalized prices
        ax1.plot(x, norm_a, label=self.pair.symbol_a, alpha=0.8)
        ax1.plot(x, norm_b, label=self.pair.symbol_b, alpha=0.8)
        ax1.set_ylabel("Normalized Price")
        ax1.legend(loc="upper left")
        half_life_str = f"{self.half_life:.1f}" if self.half_life else "None"
        ax1.set_title(
            f"{self.pair.pair_id} | "
            f"{'PASS' if self.passed else 'FAIL'} | "
            f"corr={self.correlation:.3f} | "
            f"half_life={half_life_str} | "
            f"crossings={self.zero_crossings}"
        )
        ax1.grid(True, alpha=0.3)

        # Bottom: spread with bands
        ax2.plot(x, spread, label="Spread", color="purple", alpha=0.8)
        ax2.axhline(self.spread_mean, color="black", linestyle="--", label="Mean")
        ax2.axhline(
            self.spread_mean + self.spread_std, color="gray", linestyle=":", alpha=0.7
        )
        ax2.axhline(
            self.spread_mean - self.spread_std,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="±1σ",
        )
        ax2.fill_between(
            x,
            self.spread_mean - self.spread_std,
            self.spread_mean + self.spread_std,
            alpha=0.1,
            color="gray",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Spread (norm_a - norm_b)")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


def prescreen_pair(
    lf: pl.LazyFrame,
    pair: TradingPair,
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
    col_a = pair.symbol_a.upper()
    col_b = pair.symbol_b.upper()
    ln2 = np.log(2)

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

    # Get plot data
    df_plot = lf_norm.select("timestamp", "norm_a", "norm_b", "spread").collect()

    return PrescreenResult(
        pair=pair,
        correlation=row["correlation"],
        spread_mean=row["spread_mean"],
        spread_std=row["spread_std"],
        spread_range=row["spread_range"],
        zero_crossings=row["zero_crossings"],
        half_life=row["half_life"],
        passed=passed,
        n_observations=row["n_observations"],
        timestamps=df_plot["timestamp"].to_list(),
        norm_a=df_plot["norm_a"].to_list(),
        norm_b=df_plot["norm_b"].to_list(),
        spread=df_plot["spread"].to_list(),
    )
