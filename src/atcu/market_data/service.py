import pathlib

import polars as pl


class MarketDataService:
    """Service to load market data from Hive-partitioned parquet files."""

    def __init__(self, data_dir: pathlib.Path):
        self.data_dir = data_dir

    def get_closes(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Load close prices for a single symbol."""
        pattern = f"{self.data_dir}/ticker={symbol}/**/*.parquet"

        try:
            df = (
                pl.scan_parquet(pattern)
                .select(
                    pl.col("window_start").cast(pl.Datetime("ns")).alias("timestamp"),
                    pl.col("close"),
                )
                .filter(
                    pl.col("timestamp").is_between(
                        pl.lit(start_date).str.to_datetime(),
                        pl.lit(end_date).str.to_datetime(),
                    )
                )
                .collect()
            )
        except Exception:
            return pl.DataFrame()

        return df.rename({"close": symbol.upper()})

    def get_pair_closes(
        self,
        sym_a: str,
        sym_b: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Load and align close prices for a pair."""
        df_a = self.get_closes(sym_a, start_date, end_date)
        df_b = self.get_closes(sym_b, start_date, end_date)

        if df_a.is_empty() or df_b.is_empty():
            return pl.DataFrame()

        # Join on timestamp
        return df_a.join(df_b, on="timestamp", how="inner")
