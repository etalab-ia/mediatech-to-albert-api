#!/usr/bin/env python3
"""Syncs HuggingFace datasets to Albert API collections for RAG."""

import argparse
import logging
import sys

from src.albert_client import AlbertClient
from src.config import DATASETS, get_settings
from src.huggingface_source import HuggingFaceSource
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
    parser.add_argument("--dataset", "-d", type=str, help="Sync only this dataset")
    parser.add_argument("--list", "-l", action="store_true", help="List configured datasets and exit")
    parser.add_argument("--status", "-s", action="store_true", help="Show sync status and exit")
    parser.add_argument("--force", "-f", action="store_true", help="Force sync even if dataset unchanged")
    parser.add_argument(
        "--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def print_status(
    state_store: StateStore,
    albert_client: AlbertClient,
    hf_source: HuggingFaceSource,
) -> None:
    coll_by_name = {c.dataset_name: c for c in state_store.get_all_collections()}

    name_w, sync_w, hf_w, local_w, albert_w = 33, 16, 12, 22, 12

    header = (
        f"{'Dataset':<{name_w}} {'Last sync':<{sync_w}}"
        f" {'HF chunks':>{hf_w}} {'Local docs (chunks)':<{local_w}} {'Albert docs':>{albert_w}} St."
    )
    sep = "-" * len(header)

    print("\nSYNC STATUS")
    print(sep)
    print(header)
    print(sep)

    for dataset_name in DATASETS:
        short_name = dataset_name.split("/")[-1]
        collection = coll_by_name.get(dataset_name)

        last_sync = collection.last_modified[:16] if collection and collection.last_modified else "never"

        if collection:
            doc_count, chunk_count = state_store.get_collection_counts(collection.id)
            local_str = f"{doc_count:,} ({chunk_count:,})"
        else:
            local_str = "0 (0)"

        hf_chunks = hf_source.get_chunk_count(dataset_name)
        hf_str = f"{hf_chunks:,}" if hf_chunks is not None else "?"

        albert_info = albert_client.get_collection_by_name(short_name)
        albert_str = f"{albert_info.documents_count:,}" if albert_info else "0"

        status_icon = "🟢" if (collection and collection.status == "success") else "🟠"

        print(
            f"{short_name:<{name_w}} {last_sync:<{sync_w}}"
            f" {hf_str:>{hf_w}} {local_str:<{local_w}} {albert_str:>{albert_w}} {status_icon}"
        )

    print(sep)


def print_results(result) -> None:
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
    args = parse_args()

    if args.list:
        print("Configured datasets:")
        for ds in DATASETS:
            print(f"  - {ds}")
        return 0

    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        print("Make sure ALBERT_API_TOKEN and HUGGINGFACE_TOKEN are set.")
        return 1

    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    with StateStore(settings.sqlite_path) as state_store:
        with AlbertClient(settings.albert_api_url, settings.albert_api_token) as albert_client:
            hf_source = HuggingFaceSource(settings.huggingface_token)

            if args.status:
                print_status(state_store, albert_client, hf_source)
                return 0

            logger.info("Starting Mediatech to Albert API sync")
            logger.info(f"Albert API: {settings.albert_api_url}")

            if args.dataset:
                if args.dataset not in DATASETS:
                    logger.warning(f"Dataset {args.dataset} not in configured list, will try anyway")
                datasets_to_sync = [args.dataset]
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
            return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
