from pathlib import Path

import polars as pl


class MarketDataService:
    """Service to read market data from consolidated parquet file.

    Expects file from IngestPipeline with columns:
        ticker, window_start, open, high, low, close, volume, transactions
    """

    def __init__(self, data_file: Path):
        self.data_file = data_file

    def get_closes(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pl.LazyFrame:
        """Load close prices for a single symbol."""
        # Filter ticker first for predicate pushdown on sorted parquet
        return (
            pl.scan_parquet(self.data_file)
            .filter(pl.col("ticker") == symbol)
            .select(
                pl.col("window_start").cast(pl.Datetime("ns")).alias("timestamp"),
                pl.col("close").alias(symbol.upper()),
            )
            .filter(
                pl.col("timestamp").is_between(
                    pl.lit(start_date).str.to_datetime(),
                    pl.lit(end_date).str.to_datetime(),
                )
            )
        )

    def get_pair_closes(
        self,
        sym_a: str,
        sym_b: str,
        start_date: str,
        end_date: str,
    ) -> pl.LazyFrame:
        """Load and align close prices for a pair."""
        lf_a = self.get_closes(sym_a, start_date, end_date)
        lf_b = self.get_closes(sym_b, start_date, end_date)
        return lf_a.join(lf_b, on="timestamp", how="inner")
