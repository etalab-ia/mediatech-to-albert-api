"""State store for tracking sync state in SQLite."""

import logging
from datetime import datetime

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session

from .models import Base, Collection, Document, Chunk, CollectionStatus

logger = logging.getLogger(__name__)


class StateStore:
    """
    Repository for managing sync state in SQLite.

    Handles CRUD operations for collections, documents, and chunks.
    """

    def __init__(self, sqlite_path: str):
        """
        Initialize the state store.

        Args:
            sqlite_path: Path to the SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._session: Session | None = None

    @property
    def session(self) -> Session:
        """Get or create the database session."""
        if self._session is None:
            self._session = Session(self.engine)
        return self._session

    def close(self) -> None:
        """Close the database session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def commit(self) -> None:
        """Commit the current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()

    # --- Collections ---

    def get_collection(self, dataset_name: str) -> Collection | None:
        """
        Get a collection by dataset name.

        Args:
            dataset_name: HuggingFace dataset name

        Returns:
            Collection if found, None otherwise
        """
        stmt = select(Collection).where(Collection.dataset_name == dataset_name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all_collections(self) -> list[Collection]:
        """Get all collections."""
        stmt = select(Collection)
        return list(self.session.execute(stmt).scalars().all())

    def create_collection(
        self,
        dataset_name: str,
        albert_collection_id: str | None = None,
    ) -> Collection:
        """
        Create a new collection.

        Args:
            dataset_name: HuggingFace dataset name
            albert_collection_id: Albert API collection ID

        Returns:
            Created collection
        """
        collection = Collection(
            dataset_name=dataset_name,
            albert_collection_id=albert_collection_id,
            status=CollectionStatus.IDLE.value,
        )
        self.session.add(collection)
        self.session.flush()
        return collection

    def update_collection_status(
        self,
        collection: Collection,
        status: CollectionStatus,
        error_message: str | None = None,
    ) -> None:
        """
        Update collection status.

        Args:
            collection: Collection to update
            status: New status
            error_message: Optional error message if failed
        """
        collection.status = status.value
        collection.error_message = error_message
        collection.updated_at = datetime.utcnow()
        self.session.flush()

    def update_collection_last_modified(
        self,
        collection: Collection,
        last_modified: str,
    ) -> None:
        """
        Update collection last modified date.

        Args:
            collection: Collection to update
            last_modified: ISO format date string
        """
        collection.last_modified = last_modified
        collection.updated_at = datetime.utcnow()
        self.session.flush()

    def set_collection_albert_id(
        self,
        collection: Collection,
        albert_collection_id: str,
    ) -> None:
        """
        Set the Albert collection ID.

        Args:
            collection: Collection to update
            albert_collection_id: Albert API collection ID
        """
        collection.albert_collection_id = albert_collection_id
        self.session.flush()

    # --- Documents ---

    def get_document(
        self,
        collection_id: int,
        doc_id_source: str,
    ) -> Document | None:
        """
        Get a document by source ID within a collection.

        Args:
            collection_id: Local collection ID
            doc_id_source: Document ID from parquet

        Returns:
            Document if found, None otherwise
        """
        stmt = select(Document).where(
            Document.collection_id == collection_id,
            Document.doc_id_source == doc_id_source,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all_documents(self, collection_id: int) -> list[Document]:
        """
        Get all documents in a collection.

        Args:
            collection_id: Local collection ID

        Returns:
            List of documents
        """
        stmt = select(Document).where(Document.collection_id == collection_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_document_ids_set(self, collection_id: int) -> set[str]:
        """
        Get set of all doc_id_source values in a collection.

        Useful for detecting deletions.

        Args:
            collection_id: Local collection ID

        Returns:
            Set of document source IDs
        """
        stmt = select(Document.doc_id_source).where(
            Document.collection_id == collection_id
        )
        return set(self.session.execute(stmt).scalars().all())

    def create_document(
        self,
        collection_id: int,
        doc_id_source: str,
        name: str | None = None,
        albert_document_id: str | None = None,
    ) -> Document:
        """
        Create a new document.

        Args:
            collection_id: Local collection ID
            doc_id_source: Document ID from parquet
            name: Document name
            albert_document_id: Albert API document ID

        Returns:
            Created document
        """
        document = Document(
            collection_id=collection_id,
            doc_id_source=doc_id_source,
            name=name,
            albert_document_id=albert_document_id,
        )
        self.session.add(document)
        self.session.flush()
        return document

    def update_document_albert_id(
        self,
        document: Document,
        albert_document_id: str,
    ) -> None:
        """
        Set the Albert document ID.

        Args:
            document: Document to update
            albert_document_id: Albert API document ID
        """
        document.albert_document_id = albert_document_id
        document.updated_at = datetime.utcnow()
        self.session.flush()

    def delete_document(self, document: Document) -> None:
        """
        Delete a document and its chunks.

        Args:
            document: Document to delete
        """
        self.session.delete(document)
        self.session.flush()

    def delete_documents_by_ids(
        self,
        collection_id: int,
        doc_id_sources: set[str],
    ) -> list[str]:
        """
        Delete multiple documents by their source IDs.

        Args:
            collection_id: Local collection ID
            doc_id_sources: Set of document source IDs to delete

        Returns:
            List of Albert document IDs that need to be deleted
        """
        if not doc_id_sources:
            return []

        # First get Albert IDs for cleanup
        stmt = select(Document.albert_document_id).where(
            Document.collection_id == collection_id,
            Document.doc_id_source.in_(doc_id_sources),
            Document.albert_document_id.isnot(None),
        )
        albert_ids = list(self.session.execute(stmt).scalars().all())

        # Delete documents
        stmt = delete(Document).where(
            Document.collection_id == collection_id,
            Document.doc_id_source.in_(doc_id_sources),
        )
        self.session.execute(stmt)
        self.session.flush()

        return albert_ids

    # --- Chunks ---

    def get_chunks(self, document_id: int) -> list[Chunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: Local document ID

        Returns:
            List of chunks
        """
        stmt = select(Chunk).where(Chunk.document_id == document_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_chunk_hashes(self, document_id: int) -> dict[str, str]:
        """
        Get chunk hashes for a document.

        Args:
            document_id: Local document ID

        Returns:
            Dictionary mapping chunk_id_source to chunk_hash
        """
        stmt = select(Chunk.chunk_id_source, Chunk.chunk_hash).where(
            Chunk.document_id == document_id
        )
        return dict(self.session.execute(stmt).all())

    def create_chunk(
        self,
        document_id: int,
        chunk_id_source: str,
        chunk_hash: str,
        albert_chunk_id: str | None = None,
    ) -> Chunk:
        """
        Create a new chunk.

        Args:
            document_id: Local document ID
            chunk_id_source: Chunk ID from parquet
            chunk_hash: Chunk hash from parquet
            albert_chunk_id: Albert API chunk ID

        Returns:
            Created chunk
        """
        chunk = Chunk(
            document_id=document_id,
            chunk_id_source=chunk_id_source,
            chunk_hash=chunk_hash,
            albert_chunk_id=albert_chunk_id,
        )
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def create_chunks_bulk(
        self,
        document_id: int,
        chunks_data: list[tuple[str, str, str | None]],
    ) -> list[Chunk]:
        """
        Create multiple chunks efficiently.

        Args:
            document_id: Local document ID
            chunks_data: List of (chunk_id_source, chunk_hash, albert_chunk_id) tuples

        Returns:
            List of created chunks
        """
        chunks = [
            Chunk(
                document_id=document_id,
                chunk_id_source=chunk_id,
                chunk_hash=chunk_hash,
                albert_chunk_id=albert_id,
            )
            for chunk_id, chunk_hash, albert_id in chunks_data
        ]
        self.session.add_all(chunks)
        self.session.flush()
        return chunks

    def delete_chunks(self, document_id: int) -> None:
        """
        Delete all chunks for a document.

        Args:
            document_id: Local document ID
        """
        stmt = delete(Chunk).where(Chunk.document_id == document_id)
        self.session.execute(stmt)
        self.session.flush()

    def reset_collection_documents(self, collection_id: int) -> None:
        """
        Delete all documents and chunks for a collection from local state.

        Used when the Albert collection has been deleted externally and we
        need to force a full re-upload.

        Args:
            collection_id: Local collection ID
        """
        doc_ids = list(
            self.session.execute(
                select(Document.id).where(Document.collection_id == collection_id)
            ).scalars().all()
        )
        if doc_ids:
            self.session.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
            self.session.execute(delete(Document).where(Document.collection_id == collection_id))
        self.session.flush()
