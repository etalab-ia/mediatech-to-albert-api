"""Synchronization service orchestrating the sync process."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .albert_client import AlbertClient, ChunkData, AlbertAPIError
from .config import Settings, DATASET_COLLECTION_NAMES, DATASET_COLLECTION_DESCRIPTIONS
from .huggingface_source import HuggingFaceSource, DocumentInfo
from .models import Collection, CollectionStatus, Document
from .state_store import StateStore

logger = logging.getLogger(__name__)


class DocumentAction(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
    FAILED = "failed"


@dataclass
class DocumentSyncResult:
    doc_id: str
    action: DocumentAction
    chunks_count: int = 0
    error: str | None = None


@dataclass
class DatasetSyncResult:
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
    success: bool
    datasets: list[DatasetSyncResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


class SyncService:
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
        start_time = datetime.now()
        result = DatasetSyncResult(dataset_name=dataset_name, success=True)
        collection = None

        try:
            collection = self.state.get_collection(dataset_name)
            if collection is None:
                collection = self.state.create_collection(dataset_name)
                self.state.commit()
                logger.info(f"Created new collection record for {dataset_name}")

            dataset_info = self.hf.get_dataset_info(dataset_name)
            remote_modified = (
                dataset_info.last_modified.isoformat() if dataset_info.last_modified else None
            )

            collection_name = self._make_collection_name(dataset_name)

            if self._is_dataset_unchanged(collection, remote_modified):
                existing = self.albert.get_collection_by_name(collection_name)
                if existing is not None:
                    logger.info(f"Dataset {dataset_name} unchanged, skipping")
                    result.duration_seconds = (datetime.now() - start_time).total_seconds()
                    return result
                logger.warning(
                    f"Albert collection '{collection_name}' was deleted externally, forcing re-sync"
                )
                # _ensure_albert_collection_id will handle the state reset below

            logger.info(
                f"Syncing {dataset_name}: local={collection.last_modified}, remote={remote_modified}"
            )

            self.state.update_collection_status(collection, CollectionStatus.SYNCING)
            self.state.commit()

            albert_collection_id = self._ensure_albert_collection_id(collection, dataset_name, collection_name)

            seen_doc_ids: set[str] = set()

            for doc_info in self.hf.iter_documents(dataset_name):
                seen_doc_ids.add(doc_info.doc_id)
                doc_result = self._sync_document(collection.id, albert_collection_id, doc_info)

                if doc_result.action == DocumentAction.CREATED:
                    result.documents_created += 1
                    result.chunks_uploaded += doc_result.chunks_count
                elif doc_result.action == DocumentAction.UPDATED:
                    result.documents_updated += 1
                    result.chunks_uploaded += doc_result.chunks_count
                elif doc_result.action == DocumentAction.UNCHANGED:
                    result.documents_unchanged += 1
                elif doc_result.action == DocumentAction.FAILED:
                    result.documents_failed += 1
                    logger.error(f"Failed to sync document {doc_info.doc_id}: {doc_result.error}")

            result.documents_deleted = self._delete_removed_documents(collection.id, seen_doc_ids)

            if remote_modified and result.documents_failed == 0:
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
        return DATASET_COLLECTION_NAMES[dataset_name]

    def _is_dataset_unchanged(self, collection: Collection, remote_modified: str | None) -> bool:
        """Return True if the local state is at least as recent as the remote dataset."""
        return bool(
            collection.last_modified
            and remote_modified
            and collection.last_modified >= remote_modified
            and collection.albert_collection_id
        )

    def _ensure_albert_collection_id(self, collection: Collection, dataset_name: str, collection_name: str) -> int:
        """
        Return the Albert collection ID, creating one if needed.

        Handles three cases:
        - Normal: local state has an ID and the collection exists in Albert → return it
        - External deletion: local state has an ID but Albert collection is gone
          → reset local document state, create a new collection
        - Lost state.db: no local ID but collection exists in Albert
          → delete the orphaned Albert collection (to avoid duplicates), create a new one
        """
        existing_albert = self.albert.get_collection_by_name(collection_name)

        if collection.albert_collection_id:
            if existing_albert is not None:
                return int(collection.albert_collection_id)
            # Albert collection was deleted externally — reset local state and recreate
            logger.warning(
                f"Albert collection '{collection_name}' no longer exists, resetting local state"
            )
            self.state.reset_collection_documents(collection.id)
            collection.albert_collection_id = None
            self.state.commit()

        # No local albert_collection_id — need to create one
        if existing_albert:
            # state.db was lost — delete orphaned Albert collection to avoid duplicates
            logger.warning(
                f"Found existing Albert collection '{collection_name}' but no local state "
                f"(state.db may have been lost). Deleting and recreating."
            )
            self.albert.delete_collection(existing_albert.id)

        albert_id = self.albert.create_collection(
            name=collection_name,
            description=DATASET_COLLECTION_DESCRIPTIONS[dataset_name],
        )
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

            if self._has_document_changed(existing_doc.id, doc_info):
                return self._replace_document(existing_doc, albert_collection_id, doc_info)

            return DocumentSyncResult(doc_id=doc_info.doc_id, action=DocumentAction.UNCHANGED)

        except Exception as e:
            logger.warning(f"Failed to sync document {doc_info.doc_id}: {e}", exc_info=True)
            return DocumentSyncResult(doc_id=doc_info.doc_id, action=DocumentAction.FAILED, error=str(e))

    def _has_document_changed(self, document_id: int, doc_info: DocumentInfo) -> bool:
        chunk_id_to_chunk_hash = self.state.get_document_chunk_hashes_by_chunk_id(document_id)
        if set(chunk_id_to_chunk_hash) != {c.chunk_id for c in doc_info.chunks}:
            return True
        return any(chunk_id_to_chunk_hash.get(c.chunk_id) != c.chunk_hash for c in doc_info.chunks)

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

        self._upload_and_save_chunks(document, albert_doc_id, doc_info)

        logger.debug(f"Created document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")
        return DocumentSyncResult(
            doc_id=doc_info.doc_id, action=DocumentAction.CREATED, chunks_count=len(doc_info.chunks)
        )

    def _replace_document(self, document: Document, albert_collection_id: int, doc_info: DocumentInfo) -> DocumentSyncResult:
        if document.albert_document_id:
            try:
                self.albert.delete_document(int(document.albert_document_id))
            except AlbertAPIError as e:
                logger.warning(f"Failed to delete document from Albert: {e}")

        self.state.delete_chunks(document.id)

        doc_name = doc_info.name or doc_info.doc_id
        albert_doc_id = self.albert.create_document(albert_collection_id, doc_name)
        self.state.update_document_albert_id(document, str(albert_doc_id))

        self._upload_and_save_chunks(document, albert_doc_id, doc_info)

        logger.debug(f"Updated document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")
        return DocumentSyncResult(
            doc_id=doc_info.doc_id, action=DocumentAction.UPDATED, chunks_count=len(doc_info.chunks)
        )

    def _upload_and_save_chunks(self, document: Document, albert_doc_id: int, doc_info: DocumentInfo) -> None:
        chunk_ids = self._upload_chunks(albert_doc_id, doc_info)
        self.state.create_chunks_bulk(document.id, [
            (chunk.chunk_id, chunk.chunk_hash, str(chunk_id))
            for chunk, chunk_id in zip(doc_info.chunks, chunk_ids)
        ])
        self.state.commit()

    def _delete_removed_documents(self, collection_id: int, seen_doc_ids: set[str]) -> int:
        deleted_doc_ids = self.state.get_document_ids_set(collection_id) - seen_doc_ids
        if not deleted_doc_ids:
            return 0
        logger.info(f"Deleting {len(deleted_doc_ids)} removed documents")
        albert_ids_to_delete = self.state.delete_documents_by_ids(collection_id, deleted_doc_ids)
        for albert_id in albert_ids_to_delete:
            try:
                self.albert.delete_document(int(albert_id))
            except AlbertAPIError as e:
                logger.warning(f"Failed to delete document {albert_id} from Albert: {e}")
        return len(deleted_doc_ids)

    def _upload_chunks(self, albert_doc_id: int, doc_info: DocumentInfo) -> list[int]:
        chunks_data = [
            ChunkData(
                content=chunk.content,
                metadata={
                    **(chunk.metadata or {}),
                    "_doc_id": doc_info.doc_id,
                    "_chunk_id": chunk.chunk_id,
                    "_chunk_hash": chunk.chunk_hash,
                },
            )
            for chunk in doc_info.chunks
        ]
        return self.albert.upload_chunks_batched(
            albert_doc_id, chunks_data, batch_size=self.settings.chunk_batch_size
        )
