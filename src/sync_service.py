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
    """
    Orchestrator for synchronizing HuggingFace datasets to Albert API.

    Handles:
    - Change detection via last_modified dates
    - Document creation/update/deletion
    - Chunk upload with batching
    - State tracking in SQLite
    """

    def __init__(
        self,
        settings: Settings,
        state_store: StateStore,
        albert_client: AlbertClient,
        hf_source: HuggingFaceSource,
    ):
        """
        Initialize the sync service.

        Args:
            settings: Application settings
            state_store: SQLite state store
            albert_client: Albert API client
            hf_source: HuggingFace data source
        """
        self.settings = settings
        self.state = state_store
        self.albert = albert_client
        self.hf = hf_source

    def sync_all(self, datasets: list[str]) -> SyncResult:
        """
        Sync all configured datasets.

        Args:
            datasets: List of dataset names to sync

        Returns:
            Overall sync result
        """
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
                    DatasetSyncResult(
                        dataset_name=dataset_name,
                        success=False,
                        error=str(e),
                    )
                )
                all_success = False

        total_duration = (datetime.now() - start_time).total_seconds()

        return SyncResult(
            success=all_success,
            datasets=results,
            total_duration_seconds=total_duration,
        )

    def sync_dataset(self, dataset_name: str) -> DatasetSyncResult:
        """
        Sync a single dataset.

        Args:
            dataset_name: HuggingFace dataset name

        Returns:
            Sync result for this dataset
        """
        start_time = datetime.now()
        result = DatasetSyncResult(dataset_name=dataset_name, success=True)

        try:
            # Get or create local collection record
            collection = self.state.get_collection(dataset_name)
            if collection is None:
                collection = self.state.create_collection(dataset_name)
                self.state.commit()
                logger.info(f"Created new collection record for {dataset_name}")

            # Check if dataset has changed
            dataset_info = self.hf.get_dataset_info(dataset_name)
            remote_modified = (
                dataset_info.last_modified.isoformat()
                if dataset_info.last_modified
                else None
            )

            if (
                collection.last_modified
                and remote_modified
                and collection.last_modified >= remote_modified
            ):
                logger.info(f"Dataset {dataset_name} unchanged, skipping")
                result.duration_seconds = (datetime.now() - start_time).total_seconds()
                return result

            logger.info(
                f"Dataset {dataset_name} changed: "
                f"local={collection.last_modified}, remote={remote_modified}"
            )

            # Mark as syncing
            self.state.update_collection_status(collection, CollectionStatus.SYNCING)
            self.state.commit()

            # Verify or create Albert collection
            collection_name = self._make_collection_name(dataset_name)
            if collection.albert_collection_id:
                # Verify it still exists in Albert (handles external deletions)
                existing = self.albert.get_collection_by_name(collection_name)
                if existing is None:
                    logger.warning(
                        f"Albert collection '{collection_name}' (ID: {collection.albert_collection_id}) "
                        "no longer exists in API — resetting local state to force re-upload"
                    )
                    self.state.reset_collection_documents(collection.id)
                    collection.albert_collection_id = None
                    self.state.commit()

            if not collection.albert_collection_id:
                albert_id = self._get_or_create_albert_collection(collection_name)
                self.state.set_collection_albert_id(collection, str(albert_id))
                self.state.commit()

            albert_collection_id = int(collection.albert_collection_id)

            # Track which documents we see in this sync
            seen_doc_ids: set[str] = set()

            # Process documents from HuggingFace
            for doc_info in self.hf.iter_documents(dataset_name):
                seen_doc_ids.add(doc_info.doc_id)
                doc_result = self._sync_document(
                    collection.id,
                    albert_collection_id,
                    doc_info,
                )

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

            # Handle deletions - documents in local state but not in HF
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

            # Update last_modified and status
            if remote_modified:
                self.state.update_collection_last_modified(collection, remote_modified)
            self.state.update_collection_status(collection, CollectionStatus.SUCCESS)
            self.state.commit()

        except Exception as e:
            logger.error(f"Error syncing {dataset_name}: {e}")
            result.success = False
            result.error = str(e)

            # Mark as failed
            if collection:
                self.state.update_collection_status(
                    collection,
                    CollectionStatus.FAILED,
                    error_message=str(e)[:1000],
                )
                self.state.commit()

        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def _make_collection_name(self, dataset_name: str) -> str:
        """
        Create Albert collection name from dataset name.

        Args:
            dataset_name: Full dataset name (e.g., "AgentPublic/legi")

        Returns:
            Collection name (e.g., "legi")
        """
        # Use the part after the slash
        return dataset_name.split("/")[-1]

    def _get_or_create_albert_collection(self, name: str) -> int:
        """
        Get existing or create new Albert collection.

        Args:
            name: Collection name

        Returns:
            Collection ID
        """
        existing = self.albert.get_collection_by_name(name)
        if existing:
            logger.info(f"Found existing Albert collection: {name} (ID: {existing.id})")
            return existing.id

        logger.info(f"Creating new Albert collection: {name}")
        return self.albert.create_collection(
            name=name,
            visibility="public",
            description=f"Synced from HuggingFace dataset",
        )

    def _sync_document(
        self,
        collection_id: int,
        albert_collection_id: int,
        doc_info: DocumentInfo,
    ) -> DocumentSyncResult:
        """
        Sync a single document.

        Args:
            collection_id: Local collection ID
            albert_collection_id: Albert collection ID
            doc_info: Document information from HuggingFace

        Returns:
            Document sync result
        """
        try:
            existing_doc = self.state.get_document(collection_id, doc_info.doc_id)

            if existing_doc is None:
                # New document - create it
                return self._create_document(
                    collection_id, albert_collection_id, doc_info
                )

            # Check if document needs update
            if self._document_needs_update(existing_doc.id, doc_info):
                return self._update_document(
                    existing_doc, albert_collection_id, doc_info
                )

            return DocumentSyncResult(
                doc_id=doc_info.doc_id,
                action="unchanged",
            )

        except Exception as e:
            return DocumentSyncResult(
                doc_id=doc_info.doc_id,
                action="failed",
                error=str(e),
            )

    def _document_needs_update(
        self,
        document_id: int,
        doc_info: DocumentInfo,
    ) -> bool:
        """
        Check if a document needs to be updated based on chunk hashes.

        Args:
            document_id: Local document ID
            doc_info: Document info from HuggingFace

        Returns:
            True if document needs update
        """
        existing_hashes = self.state.get_chunk_hashes(document_id)

        # Check for new or changed chunks
        for chunk in doc_info.chunks:
            existing_hash = existing_hashes.get(chunk.chunk_id)
            if existing_hash is None or existing_hash != chunk.chunk_hash:
                return True

        # Check for deleted chunks
        new_chunk_ids = {c.chunk_id for c in doc_info.chunks}
        for existing_id in existing_hashes:
            if existing_id not in new_chunk_ids:
                return True

        return False

    def _create_document(
        self,
        collection_id: int,
        albert_collection_id: int,
        doc_info: DocumentInfo,
    ) -> DocumentSyncResult:
        """
        Create a new document in Albert and local state.

        Args:
            collection_id: Local collection ID
            albert_collection_id: Albert collection ID
            doc_info: Document info from HuggingFace

        Returns:
            Document sync result
        """
        # Create document in Albert
        doc_name = doc_info.name or doc_info.doc_id
        albert_doc_id = self.albert.create_document(albert_collection_id, doc_name)

        # Create local document record
        document = self.state.create_document(
            collection_id=collection_id,
            doc_id_source=doc_info.doc_id,
            name=doc_name,
            albert_document_id=str(albert_doc_id),
        )

        # Upload chunks
        chunk_ids = self._upload_chunks(albert_doc_id, doc_info)

        # Save chunk records
        chunks_data = [
            (chunk.chunk_id, chunk.chunk_hash, str(chunk_id) if chunk_id else None)
            for chunk, chunk_id in zip(doc_info.chunks, chunk_ids)
        ]
        self.state.create_chunks_bulk(document.id, chunks_data)
        self.state.commit()

        logger.debug(f"Created document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")

        return DocumentSyncResult(
            doc_id=doc_info.doc_id,
            action="created",
            chunks_count=len(doc_info.chunks),
        )

    def _update_document(
        self,
        document,
        albert_collection_id: int,
        doc_info: DocumentInfo,
    ) -> DocumentSyncResult:
        """
        Update a document by deleting and recreating it.

        Args:
            document: Existing document record
            albert_collection_id: Albert collection ID
            doc_info: New document info from HuggingFace

        Returns:
            Document sync result
        """
        # Delete existing document in Albert
        if document.albert_document_id:
            try:
                self.albert.delete_document(int(document.albert_document_id))
            except AlbertAPIError as e:
                logger.warning(f"Failed to delete document from Albert: {e}")

        # Delete local chunks
        self.state.delete_chunks(document.id)

        # Create new document in Albert
        doc_name = doc_info.name or doc_info.doc_id
        albert_doc_id = self.albert.create_document(albert_collection_id, doc_name)

        # Update local document record
        self.state.update_document_albert_id(document, str(albert_doc_id))

        # Upload chunks
        chunk_ids = self._upload_chunks(albert_doc_id, doc_info)

        # Save chunk records
        chunks_data = [
            (chunk.chunk_id, chunk.chunk_hash, str(chunk_id) if chunk_id else None)
            for chunk, chunk_id in zip(doc_info.chunks, chunk_ids)
        ]
        self.state.create_chunks_bulk(document.id, chunks_data)
        self.state.commit()

        logger.debug(f"Updated document {doc_info.doc_id} with {len(doc_info.chunks)} chunks")

        return DocumentSyncResult(
            doc_id=doc_info.doc_id,
            action="updated",
            chunks_count=len(doc_info.chunks),
        )

    def _upload_chunks(
        self,
        albert_doc_id: int,
        doc_info: DocumentInfo,
    ) -> list[int | None]:
        """
        Upload chunks to Albert in batches.

        Args:
            albert_doc_id: Albert document ID
            doc_info: Document info containing chunks

        Returns:
            List of Albert chunk IDs (None for failed uploads)
        """
        chunks_data = [
            ChunkData(
                content=chunk.content,
                metadata=chunk.metadata if chunk.metadata else None,
            )
            for chunk in doc_info.chunks
        ]

        return self.albert.upload_chunks_batched(
            albert_doc_id,
            chunks_data,
            batch_size=self.settings.chunk_batch_size,
        )
