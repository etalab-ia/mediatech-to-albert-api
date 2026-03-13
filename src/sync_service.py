"""Synchronization service orchestrating the sync process."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from .albert_client import AlbertClient, ChunkData, AlbertAPIError
from .config import Settings
from .huggingface_source import HuggingFaceSource, DocumentInfo
from .models import CollectionStatus
from .state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass
class DocumentSyncResult:
    """Result of syncing a single document."""

    doc_id: str
    action: str  # "created", "updated", "unchanged", "deleted", "failed"
    chunks_count: int = 0
    error: str | None = None


@dataclass
class DatasetSyncResult:
    """Result of syncing a single dataset."""

    dataset_name: str
    success: bool
    documents_created: int = 0
    documents_updated: int = 0
    documents_deleted: int = 0
    documents_unchanged: int = 0
    documents_failed: int = 0
    chunks_uploaded: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class SyncResult:
    """Result of the full sync operation."""

    success: bool
    datasets: list[DatasetSyncResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


class SyncService:
    """Orchestrator for synchronizing HuggingFace datasets to Albert API."""

    def __init__(
        self,
        settings: Settings,
        state_store: StateStore,
        albert_client: AlbertClient,
        hf_source: HuggingFaceSource,
    ):
        self.settings = settings
        self.state = state_store
        self.albert = albert_client
        self.hf = hf_source

    def sync_all(self, datasets: list[str]) -> SyncResult:
        """Sync all configured datasets."""
        start_time = datetime.now()
        results: list[DatasetSyncResult] = []
        all_success = True

        for dataset_name in datasets:
            logger.info(f"=== Syncing dataset: {dataset_name} ===")
            try:
                result = self.sync_dataset(dataset_name)
                results.append(result)
                if not result.success:
                    all_success = False
            except Exception as e:
                logger.error(f"Unexpected error syncing {dataset_name}: {e}")
                results.append(
                    DatasetSyncResult(dataset_name=dataset_name, success=False, error=str(e))
                )
                all_success = False

        total_duration = (datetime.now() - start_time).total_seconds()
        return SyncResult(success=all_success, datasets=results, total_duration_seconds=total_duration)

    def sync_dataset(self, dataset_name: str) -> DatasetSyncResult:
        """Sync a single dataset."""
        start_time = datetime.now()
        result = DatasetSyncResult(dataset_name=dataset_name, success=True)
        collection = None

        try:
            collection = self.state.get_collection(dataset_name)
            if collection is None:
                collection = self.state.create_collection(dataset_name)
                self.state.commit()
                logger.info(f"Created new collection record for {dataset_name}")

            # Check if dataset has changed
            dataset_info = self.hf.get_dataset_info(dataset_name)
            remote_modified = (
                dataset_info.last_modified.isoformat() if dataset_info.last_modified else None
            )

            dataset_unchanged = (
                collection.last_modified
                and remote_modified
                and collection.last_modified >= remote_modified
            )

            collection_name = self._make_collection_name(dataset_name)

            if dataset_unchanged:
                if collection.albert_collection_id:
                    # Verify Albert collection still exists before skipping
                    existing = self.albert.get_collection_by_name(collection_name)
                    if existing is not None:
                        logger.info(f"Dataset {dataset_name} unchanged, skipping")
                        result.duration_seconds = (datetime.now() - start_time).total_seconds()
                        return result
                    # Albert collection was deleted externally — force re-sync
                    logger.warning(
                        f"Albert collection '{collection_name}' was deleted externally, forcing re-sync"
                    )
                    self.state.reset_collection_documents(collection.id)
                    collection.albert_collection_id = None
                    self.state.commit()
                # else: no Albert collection yet, fall through to create it

            logger.info(
                f"Dataset {dataset_name} changed: "
                f"local={collection.last_modified}, remote={remote_modified}"
            )

            self.state.update_collection_status(collection, CollectionStatus.SYNCING)
            self.state.commit()

            # Resolve Albert collection
            albert_collection_id = self._resolve_albert_collection(collection, collection_name)

            # Track which documents we see in this sync
            seen_doc_ids: set[str] = set()

            for doc_info in self.hf.iter_documents(dataset_name):
                seen_doc_ids.add(doc_info.doc_id)
                doc_result = self._sync_document(collection.id, albert_collection_id, doc_info)

                if doc_result.action == "created":
                    result.documents_created += 1
                    result.chunks_uploaded += doc_result.chunks_count
                elif doc_result.action == "updated":
                    result.documents_updated += 1
                    result.chunks_uploaded += doc_result.chunks_count
                elif doc_result.action == "unchanged":
                    result.documents_unchanged += 1
                elif doc_result.action == "failed":
                    result.documents_failed += 1
                    logger.error(f"Failed to sync document {doc_info.doc_id}: {doc_result.error}")

            # Handle deletions
            existing_doc_ids = self.state.get_document_ids_set(collection.id)
            deleted_doc_ids = existing_doc_ids - seen_doc_ids

            if deleted_doc_ids:
                logger.info(f"Deleting {len(deleted_doc_ids)} removed documents")
                albert_ids_to_delete = self.state.delete_documents_by_ids(
                    collection.id, deleted_doc_ids
                )
                for albert_id in albert_ids_to_delete:
                    try:
                        self.albert.delete_document(int(albert_id))
                    except AlbertAPIError as e:
                        logger.warning(f"Failed to delete document {albert_id} from Albert: {e}")
                result.documents_deleted = len(deleted_doc_ids)

            if remote_modified:
                self.state.update_collection_last_modified(collection, remote_modified)
            self.state.update_collection_status(collection, CollectionStatus.SUCCESS)
            self.state.commit()

        except Exception as e:
            logger.error(f"Error syncing {dataset_name}: {e}")
            result.success = False
            result.error = str(e)

            if collection:
                self.state.update_collection_status(
                    collection, CollectionStatus.FAILED, error_message=str(e)[:1000]
                )
                self.state.commit()

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def _make_collection_name(self, dataset_name: str) -> str:
        return dataset_name.split("/")[-1]

    def _resolve_albert_collection(self, collection, collection_name: str) -> int:
        """
        Ensure an Albert collection exists and return its ID.

        Handles three cases:
        - Normal: local albert_collection_id set and collection exists in Albert → use it
        - External deletion: local albert_collection_id set but collection gone from Albert
          → reset local document state, create new collection
        - state.db lost: no local albert_collection_id but collection exists in Albert
          → delete Albert collection to avoid duplicates, create new one
        """
        existing_albert = self.albert.get_collection_by_name(collection_name)

        if collection.albert_collection_id:
            if existing_albert is None:
                logger.warning(
                    f"Albert collection '{collection_name}' no longer exists, resetting local state"
                )
                self.state.reset_collection_documents(collection.id)
                collection.albert_collection_id = None
                self.state.commit()
            else:
                return int(collection.albert_collection_id)

        # Need to create a new Albert collection
        if existing_albert:
            # state.db was lost — delete existing Albert collection to avoid duplicates
            logger.warning(
                f"Found existing Albert collection '{collection_name}' but no local state "
                f"(state.db may have been lost). Deleting and recreating."
            )
            self.albert.delete_collection(existing_albert.id)

        albert_id = self.albert.create_collection(name=collection_name)
        self.state.set_collection_albert_id(collection, str(albert_id))
        self.state.commit()
        return albert_id

    def _sync_document(
        self,
        collection_id: int,
        albert_collection_id: int,
        doc_info: DocumentInfo,
    ) -> DocumentSyncResult:
        try:
            existing_doc = self.state.get_document(collection_id, doc_info.doc_id)

            if existing_doc is None:
                return self._create_document(collection_id, albert_collection_id, doc_info)

            if self._document_needs_update(existing_doc.id, doc_info):
                return self._update_document(existing_doc, albert_collection_id, doc_info)

            return DocumentSyncResult(doc_id=doc_info.doc_id, action="unchanged")

        except Exception as e:
            return DocumentSyncResult(doc_id=doc_info.doc_id, action="failed", error=str(e))

    def _document_needs_update(self, document_id: int, doc_info: DocumentInfo) -> bool:
        existing_hashes = self.state.get_chunk_hashes(document_id)

        for chunk in doc_info.chunks:
            if existing_hashes.get(chunk.chunk_id) != chunk.chunk_hash:
                return True

        new_chunk_ids = {c.chunk_id for c in doc_info.chunks}
        if set(existing_hashes) != new_chunk_ids:
            return True

        return False

    def _create_document(
        self,
        collection_id: int,
        albert_collection_id: int,
        doc_info: DocumentInfo,
    ) -> DocumentSyncResult:
        doc_name = doc_info.name or doc_info.doc_id
        albert_doc_id = self.albert.create_document(albert_collection_id, doc_name)

        document = self.state.create_document(
            collection_id=collection_id,
            doc_id_source=doc_info.doc_id,
            name=doc_name,
            albert_document_id=str(albert_doc_id),
        )

        chunk_ids = self._upload_chunks(albert_doc_id, doc_info)
        chunks_data = [
            (chunk.chunk_id, chunk.chunk_hash, str(chunk_id) if chunk_id else None)
            for chunk, chunk_id in zip(doc_info.chunks, chunk_ids)
        ]
        self.state.create_chunks_bulk(document.id, chunks_data)
        self.state.commit()

        logger.debug(f"Created document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")
        return DocumentSyncResult(
            doc_id=doc_info.doc_id, action="created", chunks_count=len(doc_info.chunks)
        )

    def _update_document(self, document, albert_collection_id: int, doc_info: DocumentInfo) -> DocumentSyncResult:
        if document.albert_document_id:
            try:
                self.albert.delete_document(int(document.albert_document_id))
            except AlbertAPIError as e:
                logger.warning(f"Failed to delete document from Albert: {e}")

        self.state.delete_chunks(document.id)

        doc_name = doc_info.name or doc_info.doc_id
        albert_doc_id = self.albert.create_document(albert_collection_id, doc_name)
        self.state.update_document_albert_id(document, str(albert_doc_id))

        chunk_ids = self._upload_chunks(albert_doc_id, doc_info)
        chunks_data = [
            (chunk.chunk_id, chunk.chunk_hash, str(chunk_id) if chunk_id else None)
            for chunk, chunk_id in zip(doc_info.chunks, chunk_ids)
        ]
        self.state.create_chunks_bulk(document.id, chunks_data)
        self.state.commit()

        logger.debug(f"Updated document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")
        return DocumentSyncResult(
            doc_id=doc_info.doc_id, action="updated", chunks_count=len(doc_info.chunks)
        )

    def _upload_chunks(self, albert_doc_id: int, doc_info: DocumentInfo) -> list[int]:
        chunks_data = [
            ChunkData(content=chunk.content, metadata=chunk.metadata or None)
            for chunk in doc_info.chunks
        ]
        return self.albert.upload_chunks_batched(
            albert_doc_id, chunks_data, batch_size=self.settings.chunk_batch_size
        )
