import os
from datetime import date
from pathlib import Path

from atcu.market_data.metadata.service import MDMetadata
from atcu.market_data.metadata.service import MDMetadataConfig
from atcu.market_data.pipeline import IngestConfig
from atcu.market_data.pipeline import IngestPipeline
from atcu.market_data.providers.massive import MassiveDataDownloader
from atcu.market_data.providers.massive import MassiveDataDownloaderConfig

REPO_ROOT = Path(__file__).parent.parent


def main():
    config = IngestConfig(
        staging_dir=REPO_ROOT / "data" / "staging",
        output_file=REPO_ROOT / "data" / "ohlcv.parquet",
    )

    metadata = MDMetadata(MDMetadataConfig(path=REPO_ROOT / "data" / "universe.yaml"))
    symbols = list(metadata.get_universe().symbols.keys())

    fetcher = MassiveDataDownloader(
        MassiveDataDownloaderConfig(
            api_key=os.environ["MASSIVE_API_KEY"],
            flat_file_access_key_id=os.environ["MASSIVE_FLAT_FILE_ACCESS_KEY_ID"],
            flat_file_secret_access_key=os.environ[
                "MASSIVE_FLAT_FILE_SECRET_ACCESS_KEY"
            ],
        )
    )

    pipeline = IngestPipeline(config, fetcher)
    result = pipeline.run(
        symbols=symbols, start=date(2022, 1, 1), end=date(2025, 12, 31)
    )

    if result:
        pass


if __name__ == "__main__":
    main()
