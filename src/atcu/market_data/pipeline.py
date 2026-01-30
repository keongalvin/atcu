"""Market data ingestion pipeline.

Stages:
    fetch  -> DataFrame | None
    stage  -> Path
    consolidate -> Path

Usage:
    pipeline = IngestPipeline(fetcher, staging_dir, output_file)
    pipeline.run(symbols, start_date, end_date)
"""

from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path

import polars as pl

from atcu.market_data.providers.massive import MassiveDataDownloader


def fetch(
    fetcher: MassiveDataDownloader,
    symbols: list[str],
    d: date,
) -> pl.DataFrame | None:
    """Fetch minute aggregates for a single date."""
    try:
        return fetcher.get_minute_aggs(symbols=symbols, date=d.isoformat())
    except Exception:
        return None


def _staging_path(staging_dir: Path, ticker: str, d: date) -> Path:
    return staging_dir / ticker / f"{d.year}-{d.month:02d}.parquet"


def _existing_dates(staging_dir: Path, ticker: str, d: date) -> set[date]:
    path = _staging_path(staging_dir, ticker, d)
    if not path.exists():
        return set()
    df = pl.read_parquet(path, columns=["window_start"])
    dates = df["window_start"].cast(pl.Datetime("ns")).dt.date().unique().to_list()
    return set(dates)


def is_staged(symbols: list[str], staging_dir: Path, d: date) -> bool:
    """Check if all symbols have data staged for a date."""
    return all(d in _existing_dates(staging_dir, s, d) for s in symbols)


def stage(df: pl.DataFrame, staging_dir: Path, d: date) -> list[Path]:
    """Write dataframe to staging, partitioned by ticker into monthly files."""
    paths = []
    for (ticker,), group in df.group_by("ticker"):
        path = _staging_path(staging_dir, ticker, d)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pl.read_parquet(path)
            group = pl.concat([existing, group]).unique().sort("window_start")

        group.sort("window_start").write_parquet(path)
        paths.append(path)
    return paths


def consolidate(staging_dir: Path, output_file: Path) -> Path | None:
    """Consolidate staging files into single optimized parquet."""
    if not staging_dir.exists():
        return None

    files = list(staging_dir.glob("*/*.parquet"))
    if not files:
        return None

    dfs = [
        pl.read_parquet(f).with_columns(pl.lit(f.parent.name).alias("ticker"))
        for f in files
    ]

    combined = pl.concat(dfs).unique().sort(["ticker", "window_start"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(
        output_file,
        statistics=True,
        row_group_size=100_000,
    )
    return output_file


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@dataclass
class IngestConfig:
    staging_dir: Path
    output_file: Path


class IngestPipeline:
    """Orchestrates fetch -> stage -> consolidate."""

    def __init__(
        self,
        config: IngestConfig,
        fetcher: MassiveDataDownloader,
    ):
        self.config = config
        self.fetcher = fetcher

    def run(
        self,
        symbols: list[str],
        start: date,
        end: date,
        skip_weekends: bool = True,
    ) -> Path | None:
        """Run full pipeline: fetch, stage, consolidate."""
        for d in _date_range(start, end):
            if skip_weekends and d.weekday() >= 5:
                continue

            if is_staged(symbols, self.config.staging_dir, d):
                continue

            df = fetch(self.fetcher, symbols, d)
            if df is not None and len(df) > 0:
                stage(df, self.config.staging_dir, d)

        return consolidate(self.config.staging_dir, self.config.output_file)
