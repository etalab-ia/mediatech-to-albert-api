from dataclasses import dataclass

from .albert_client import AlbertClient
from .config import DATASET_COLLECTION_NAMES, DATASETS
from .huggingface_source import HuggingFaceSource
from .models import CollectionStatus
from .state_store import StateStore
from .sync_service import SyncResult


@dataclass
class _DatasetStatusRow:
    short_name: str
    last_sync: str
    hf_str: str
    local_str: str
    albert_str: str
    status_icon: str


def _collect_status_rows(
    state_store: StateStore,
    albert_client: AlbertClient,
    hf_source: HuggingFaceSource,
) -> list[_DatasetStatusRow]:
    coll_by_name = {c.dataset_name: c for c in state_store.get_all_collections()}
    rows = []

    for dataset_name in DATASETS:
        collection_name = DATASET_COLLECTION_NAMES[dataset_name]
        collection = coll_by_name.get(dataset_name)

        last_sync = collection.last_modified[:16] if collection and collection.last_modified else "never"

        if collection:
            doc_count, chunk_count = state_store.get_collection_counts(collection.id)
            local_str = f"{doc_count:,} ({chunk_count:,})"
        else:
            local_str = "0 (0)"

        hf_chunks = hf_source.get_chunk_count(dataset_name)
        hf_str = f"{hf_chunks:,}" if hf_chunks is not None else "?"

        albert_info = albert_client.get_collection_by_name(collection_name)
        albert_str = f"{albert_info.documents_count:,}" if albert_info else "0"

        status_icon = "🟢" if (collection and collection.status == CollectionStatus.SUCCESS.value) else "🟠"

        rows.append(
            _DatasetStatusRow(
                short_name=collection_name,
                last_sync=last_sync,
                hf_str=hf_str,
                local_str=local_str,
                albert_str=albert_str,
                status_icon=status_icon,
            )
        )

    return rows


def print_status(
    state_store: StateStore,
    albert_client: AlbertClient,
    hf_source: HuggingFaceSource,
) -> None:
    rows = _collect_status_rows(state_store, albert_client, hf_source)

    name_w, sync_w, hf_w, local_w, albert_w = 48, 16, 12, 22, 12

    header = (
        f"{'Dataset':<{name_w}} {'Last sync':<{sync_w}}" f" {'HF chunks':>{hf_w}} {'Local docs (chunks)':<{local_w}} {'Albert docs':>{albert_w}} St."
    )
    sep = "-" * len(header)

    print("\nSYNC STATUS")
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        print(
            f"{row.short_name:<{name_w}} {row.last_sync:<{sync_w}}"
            f" {row.hf_str:>{hf_w}} {row.local_str:<{local_w}} {row.albert_str:>{albert_w}} {row.status_icon}"
        )

    print(sep)


def print_results(result: SyncResult) -> None:
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
