#!/usr/bin/env python3
"""
Mediatech to Albert API synchronization script.

Syncs HuggingFace datasets to Albert API collections for RAG.
"""

import argparse
import logging
import sys

from src.albert_client import AlbertClient
from src.config import DATASETS, get_settings
from src.huggingface_source import HuggingFaceSource
from src.state_store import StateStore
from src.sync_service import SyncService


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync HuggingFace datasets to Albert API",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Sync only this dataset (e.g., AgentPublic/legi)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List configured datasets and exit",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show sync status for all datasets and exit",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force sync even if dataset hasn't changed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides env var)",
    )
    return parser.parse_args()


def print_status(state_store: StateStore) -> None:
    """Print sync status for all datasets."""
    collections = state_store.get_all_collections()

    if not collections:
        print("No datasets synced yet.")
        return

    print("\nDataset Sync Status:")
    print("-" * 80)
    print(f"{'Dataset':<45} {'Status':<10} {'Last Modified':<25}")
    print("-" * 80)

    for coll in collections:
        print(
            f"{coll.dataset_name:<45} "
            f"{coll.status:<10} "
            f"{coll.last_modified or 'Never':<25}"
        )

    print("-" * 80)


def print_results(result) -> None:
    """Print sync results."""
    print("\n" + "=" * 80)
    print("SYNC RESULTS")
    print("=" * 80)

    for ds_result in result.datasets:
        status = "OK" if ds_result.success else "FAILED"
        print(f"\n{ds_result.dataset_name}: {status}")

        if ds_result.error:
            print(f"  Error: {ds_result.error}")
        else:
            print(f"  Documents created:   {ds_result.documents_created}")
            print(f"  Documents updated:   {ds_result.documents_updated}")
            print(f"  Documents deleted:   {ds_result.documents_deleted}")
            print(f"  Documents unchanged: {ds_result.documents_unchanged}")
            print(f"  Documents failed:    {ds_result.documents_failed}")
            print(f"  Chunks uploaded:     {ds_result.chunks_uploaded}")
            print(f"  Duration:            {ds_result.duration_seconds:.2f}s")

    print("\n" + "-" * 80)
    print(f"Total duration: {result.total_duration_seconds:.2f}s")
    print(f"Overall status: {'SUCCESS' if result.success else 'FAILED'}")
    print("=" * 80)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # List datasets and exit
    if args.list:
        print("Configured datasets:")
        for ds in DATASETS:
            print(f"  - {ds}")
        return 0

    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        print("Make sure ALBERT_API_TOKEN and HUGGINGFACE_TOKEN are set.")
        return 1

    # Setup logging
    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Mediatech to Albert API sync")
    logger.info(f"Albert API: {settings.albert_api_url}")
    logger.info(f"SQLite path: {settings.sqlite_path}")

    # Initialize components
    with StateStore(settings.sqlite_path) as state_store:
        # Show status and exit
        if args.status:
            print_status(state_store)
            return 0

        with AlbertClient(
            settings.albert_api_url,
            settings.albert_api_token,
        ) as albert_client:
            hf_source = HuggingFaceSource(settings.huggingface_token)

            sync_service = SyncService(
                settings=settings,
                state_store=state_store,
                albert_client=albert_client,
                hf_source=hf_source,
            )

            # Determine which datasets to sync
            if args.dataset:
                if args.dataset not in DATASETS:
                    logger.warning(
                        f"Dataset {args.dataset} not in configured list, "
                        "but will try to sync anyway"
                    )
                datasets_to_sync = [args.dataset]
            else:
                datasets_to_sync = DATASETS

            logger.info(f"Datasets to sync: {datasets_to_sync}")

            # Handle force flag by clearing last_modified
            if args.force:
                logger.info("Force flag set, clearing last_modified dates")
                for ds_name in datasets_to_sync:
                    coll = state_store.get_collection(ds_name)
                    if coll:
                        state_store.update_collection_last_modified(coll, "")
                state_store.commit()

            # Run sync
            result = sync_service.sync_all(datasets_to_sync)

            print_results(result)

            return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
