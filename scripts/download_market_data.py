import os
from datetime import date
from datetime import timedelta
from pathlib import Path

import polars as pl

from atcu.market_data.metadata.dto import MDUniverse
from atcu.market_data.metadata.service import MDMetadata
from atcu.market_data.metadata.service import MDMetadataConfig
from atcu.market_data.providers.massive import MassiveDataDownloader
from atcu.market_data.providers.massive import MassiveDataDownloaderConfig

REPO_ROOT = Path(__file__).parent.parent
UNIVERSE_PATH = REPO_ROOT / "data" / "universe.yaml"
RAW_DATA_DIR = REPO_ROOT / "data" / "minute_aggs"

MASSIVE_API_KEY = os.environ["MASSIVE_API_KEY"]
MASSIVE_FLAT_FILE_ACCESS_KEY_ID = os.environ["MASSIVE_FLAT_FILE_ACCESS_KEY_ID"]
MASSIVE_FLAT_FILE_SECRET_ACCESS_KEY = os.environ["MASSIVE_FLAT_FILE_SECRET_ACCESS_KEY"]

START_DATE = date(2022, 1, 1)
END_DATE = date(2025, 12, 31)

# pl.Config(verbose=True)


def date_range(start: date, end: date):
    """Yield dates from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_minute_aggs_for_date(
    symbols: list[str],
    massive: MassiveDataDownloader,
    d: date,
) -> pl.DataFrame | None:
    """Download minute aggregates for symbols on a given date."""
    try:
        return massive.get_minute_aggs(symbols=symbols, date=d.isoformat())
    except Exception:
        return None


def hive_path(output_dir: Path, ticker: str, d: date) -> Path:
    """Build Hive-partitioned path: ticker=X/year=Y/month=M/day.parquet"""
    return (
        output_dir
        / f"ticker={ticker}"
        / f"year={d.year}"
        / f"month={d.month:02d}"
        / f"{d.isoformat()}.parquet"
    )


def check_date_complete(symbols: list[str], output_dir: Path, d: date) -> bool:
    """Check if all ticker files exist for a date."""
    return all(hive_path(output_dir, ticker, d).exists() for ticker in symbols)


def save_by_ticker(df: pl.DataFrame, output_dir: Path, d: date):
    """Partition dataframe by ticker and save with Hive partitioning."""
    for (ticker,), group in df.group_by("ticker"):
        output_file = hive_path(output_dir, ticker, d)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Drop ticker column since it's in the path
        group.drop("ticker").write_parquet(output_file)


def download_all_minute_aggs(
    universe: MDUniverse,
    massive: MassiveDataDownloader,
    start: date,
    end: date,
    output_dir: Path,
):
    """Download minute aggregates for all dates, partitioned by ticker."""
    symbols = list(universe.symbols.keys())
    (end - start).days + 1

    for _i, d in enumerate(date_range(start, end), 1):
        # Skip weekends
        if d.weekday() >= 5:
            continue

        if check_date_complete(symbols, output_dir, d):
            continue

        df = download_minute_aggs_for_date(symbols, massive, d)

        if df is not None and len(df) > 0:
            save_by_ticker(df, output_dir, d)
            df["ticker"].n_unique()


def main():
    # Load universe
    metadata_config = MDMetadataConfig(path=UNIVERSE_PATH)
    metadata = MDMetadata(metadata_config)
    universe = metadata.get_universe()

    # Setup massive service
    massive_config = MassiveDataDownloaderConfig(
        api_key=MASSIVE_API_KEY,
        flat_file_access_key_id=MASSIVE_FLAT_FILE_ACCESS_KEY_ID,
        flat_file_secret_access_key=MASSIVE_FLAT_FILE_SECRET_ACCESS_KEY,
    )
    massive = MassiveDataDownloader(config=massive_config)

    # Download all minute aggs from 2022 to 2025
    download_all_minute_aggs(universe, massive, START_DATE, END_DATE, RAW_DATA_DIR)


if __name__ == "__main__":
    main()
