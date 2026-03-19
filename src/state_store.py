import logging
from datetime import datetime

from sqlalchemy import create_engine, select, delete, func
from sqlalchemy.orm import Session

from .models import Base, Collection, Document, Chunk, CollectionStatus

logger = logging.getLogger(__name__)


class StateStore:
    """Repository for managing sync state in SQLite."""

    def __init__(self, sqlite_path: str):
        self.engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._session: Session | None = None

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = Session(self.engine)
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def commit(self) -> None:
        self.session.commit()

    # --- Collections ---

    def get_collection(self, dataset_name: str) -> Collection | None:
        stmt = select(Collection).where(Collection.dataset_name == dataset_name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all_collections(self) -> list[Collection]:
        stmt = select(Collection)
        return list(self.session.execute(stmt).scalars().all())

    def create_collection(self, dataset_name: str) -> Collection:
        collection = Collection(
            dataset_name=dataset_name,
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
        collection.status = status.value
        collection.error_message = error_message
        collection.updated_at = datetime.utcnow()
        self.session.flush()

    def update_collection_last_modified(self, collection: Collection, last_modified: str) -> None:
        collection.last_modified = last_modified
        collection.updated_at = datetime.utcnow()
        self.session.flush()

    def set_collection_albert_id(self, collection: Collection, albert_collection_id: str) -> None:
        collection.albert_collection_id = albert_collection_id
        self.session.flush()

    # --- Documents ---

    def get_document(self, collection_id: int, doc_id_source: str) -> Document | None:
        stmt = select(Document).where(
            Document.collection_id == collection_id,
            Document.doc_id_source == doc_id_source,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_collection_counts(self, collection_id: int) -> tuple[int, int]:
        doc_count = self.session.execute(
            select(func.count(Document.id)).where(Document.collection_id == collection_id)
        ).scalar()
        chunk_count = self.session.execute(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.collection_id == collection_id)
        ).scalar()
        return doc_count, chunk_count

    def get_document_ids_set(self, collection_id: int) -> set[str]:
        stmt = select(Document.doc_id_source).where(Document.collection_id == collection_id)
        return set(self.session.execute(stmt).scalars().all())

    def create_document(
        self,
        collection_id: int,
        doc_id_source: str,
        name: str | None = None,
        albert_document_id: str | None = None,
    ) -> Document:
        document = Document(
            collection_id=collection_id,
            doc_id_source=doc_id_source,
            name=name,
            albert_document_id=albert_document_id,
        )
        self.session.add(document)
        self.session.flush()
        return document

    def update_document_albert_id(self, document: Document, albert_document_id: str) -> None:
        document.albert_document_id = albert_document_id
        document.updated_at = datetime.utcnow()
        self.session.flush()

    def delete_documents_by_ids(
        self,
        collection_id: int,
        doc_id_sources: set[str],
    ) -> list[str]:
        """Delete documents by source IDs. Returns Albert document IDs to delete."""
        if not doc_id_sources:
            return []

        stmt = select(Document.albert_document_id).where(
            Document.collection_id == collection_id,
            Document.doc_id_source.in_(doc_id_sources),
            Document.albert_document_id.isnot(None),
        )
        albert_ids = list(self.session.execute(stmt).scalars().all())

        stmt = delete(Document).where(
            Document.collection_id == collection_id,
            Document.doc_id_source.in_(doc_id_sources),
        )
        self.session.execute(stmt)
        self.session.flush()

        return albert_ids

    # --- Chunks ---

    def get_document_chunk_hashes_by_chunk_id(self, document_id: int) -> dict[str, str]:
        stmt = select(Chunk.chunk_id_source, Chunk.chunk_hash).where(
            Chunk.document_id == document_id
        )
        return dict(self.session.execute(stmt).all())

    def create_chunks_bulk(
        self,
        document_id: int,
        chunks_data: list[tuple[str, str, str]],
    ) -> None:
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

    def delete_chunks(self, document_id: int) -> None:
        stmt = delete(Chunk).where(Chunk.document_id == document_id)
        self.session.execute(stmt)
        self.session.flush()

    def reset_collection_documents(self, collection_id: int) -> None:
        """Delete all documents and chunks for a collection from local state."""
        doc_ids = list(
            self.session.execute(
                select(Document.id).where(Document.collection_id == collection_id)
            ).scalars().all()
        )
        if doc_ids:
            self.session.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
            self.session.execute(
                delete(Document).where(Document.collection_id == collection_id)
            )
        self.session.flush()
