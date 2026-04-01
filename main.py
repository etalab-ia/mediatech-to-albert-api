#!/usr/bin/env python3
"""Syncs HuggingFace datasets to Albert API collections for RAG."""

import argparse
import logging
import sys

from src.albert_client import AlbertClient
from src.config import DATASETS, get_settings
from src.display import print_results, print_status
from src.huggingface_source import HuggingFaceSource
from src.notifier import TchapNotifier
from src.state_store import StateStore
from src.sync_service import SyncService


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync HuggingFace datasets to Albert API")
    parser.add_argument("--dataset", "-d", type=str, nargs="+", help="Sync only these datasets (space-separated)")
    parser.add_argument("--status", "-s", action="store_true", help="Show sync status and exit")
    parser.add_argument("--force", "-f", action="store_true", help="Force sync even if dataset unchanged")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        print("Make sure ALBERT_API_TOKEN and HUGGINGFACE_TOKEN are set.")
        return 1

    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    with (
        StateStore(settings.database_url) as state_store,
        AlbertClient(settings.albert_api_url, settings.albert_api_token, requests_per_second=settings.requests_per_second) as albert_client,
    ):
        hf_source = HuggingFaceSource(settings.huggingface_token)

        if args.status:
            print_status(state_store, albert_client, hf_source)
            return 0

        logger.info("Starting Mediatech to Albert API sync")
        logger.info(f"Albert API url: {settings.albert_api_url}")

        if args.dataset:
            for ds in args.dataset:
                if ds not in DATASETS:
                    raise ValueError(f"Unknown dataset `{ds}`. Configured datasets: {DATASETS}")
            datasets_to_sync = args.dataset
        else:
            datasets_to_sync = DATASETS

        if args.force:
            logger.info("Force flag set, clearing last_modified dates")
            for ds_name in datasets_to_sync:
                coll = state_store.get_collection(ds_name)
                if coll:
                    state_store.update_collection_last_modified(coll, "")
            state_store.commit()

        sync_service = SyncService(
            settings=settings,
            state_store=state_store,
            albert_client=albert_client,
            hf_source=hf_source,
        )
        result = sync_service.sync_all(datasets_to_sync)
        print_results(result)

        if settings.tchap_access_token and settings.tchap_room_id:
            notifier = TchapNotifier(
                homeserver=settings.tchap_homeserver,
                access_token=settings.tchap_access_token,
                room_id=settings.tchap_room_id,
            )
            notifier.send(TchapNotifier.format_sync_result(result, settings.albert_api_url))

        return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
