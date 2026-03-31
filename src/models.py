from datetime import datetime
from enum import Enum

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
)


class Base(DeclarativeBase):
    pass


class CollectionStatus(str, Enum):
    IDLE = "idle"
    SYNCING = "syncing"
    SUCCESS = "success"
    FAILED = "failed"


class Collection(Base):
    """
    Represents a HuggingFace dataset synced as an Albert collection.

    One dataset = one collection.
    """

    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # HuggingFace dataset name (e.g., "AgentPublic/legi")
    dataset_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Albert collection ID (returned by API)
    albert_collection_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Last modification date from HuggingFace (ISO format)
    last_modified: Mapped[str | None] = mapped_column(String(50), nullable=True)

    status: Mapped[str] = mapped_column(
        String(20),
        default=CollectionStatus.IDLE.value,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Collection(id={self.id}, dataset={self.dataset_name}, status={self.status})>"


class Document(Base):
    """
    Represents a document within a collection.

    Maps doc_id from HF parquet to Albert document ID.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Document ID from parquet (doc_id field)
    doc_id_source: Mapped[str] = mapped_column(String(255), nullable=False)

    # Document name (usually from title field)
    name: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Albert document ID (returned by API)
    albert_document_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    collection: Mapped["Collection"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "doc_id_source",
            name="uq_document_per_collection",
        ),
        Index("idx_doc_id_source", "doc_id_source"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, doc_id={self.doc_id_source})>"


class Chunk(Base):
    """
    Represents a chunk within a document.

    Stores the hash for change detection.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Chunk ID from parquet (chunk_id field)
    chunk_id_source: Mapped[str] = mapped_column(String(255), nullable=False)

    # Hash from parquet (chunk_xxh64 field) for change detection
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # Albert chunk ID (returned by API)
    albert_chunk_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_id_source",
            name="uq_chunk_per_document",
        ),
        Index("idx_chunk_hash", "chunk_hash"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, chunk_id={self.chunk_id_source})>"


