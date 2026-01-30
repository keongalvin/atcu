import os
import pathlib

import attrs
import massive as m
import polars as pl

S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"


def _default_cache_dir() -> pathlib.Path:
    """Return XDG_CACHE_HOME/atcu or ~/.cache/atcu."""
    base = os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "atcu"


@attrs.frozen
class MassiveDataDownloaderConfig:
    api_key: str | None = attrs.field(default=None)
    flat_file_access_key_id: str | None = attrs.field(default=None)
    flat_file_secret_access_key: str | None = attrs.field(default=None)
    cache_dir: pathlib.Path = attrs.field(factory=_default_cache_dir)


class MassiveDataDownloader:
    def __init__(self, config: MassiveDataDownloaderConfig):
        self.config = config
        self._client: m.RESTClient | None = None

    @property
    def client(self) -> m.RESTClient:
        if self.config.api_key is None:
            msg = "api_key required to use massive.RESTClient"
            raise ValueError(msg)
        if not self._client:
            self._client = m.RESTClient(self.config.api_key)
        return self._client

    @property
    def _storage_options(self) -> dict[str, str]:
        if self.config.flat_file_access_key_id is None:
            msg = "flat_file_access_key_id required for flat file access"
            raise ValueError(msg)
        if self.config.flat_file_secret_access_key is None:
            msg = "flat_file_secret_access_key required for flat file access"
            raise ValueError(msg)
        return {
            "aws_access_key_id": self.config.flat_file_access_key_id,
            "aws_secret_access_key": self.config.flat_file_secret_access_key,
            "aws_endpoint_url": S3_ENDPOINT,
        }

    def get_minute_aggs(self, symbols: str | list[str], date: str) -> pl.DataFrame:
        """Get minute aggregates for symbol(s) on a given date.

        Args:
            symbols: Single ticker or list of tickers (e.g., "NVDA" or ["NVDA", "AMD"])
            date: Date string in YYYY-MM-DD format

        Returns:
            DataFrame with minute aggregate data for the symbol(s)
        """
        year, month, _ = date.split("-")
        s3_path = f"s3://{S3_BUCKET}/us_stocks_sip/minute_aggs_v1/{year}/{month}/{date}.csv.gz"

        if isinstance(symbols, str):
            symbols = [symbols]

        return (
            pl.scan_csv(s3_path, storage_options=self._storage_options)
            .filter(pl.col("ticker").is_in(symbols))
            .collect()
        )
