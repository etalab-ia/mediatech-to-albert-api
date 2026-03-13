"""HuggingFace dataset source for loading parquet data."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

from .config import DATASET_METADATA_FIELDS

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a HuggingFace dataset."""

    name: str
    last_modified: datetime | None
    size_bytes: int | None = None


@dataclass
class ChunkInfo:
    """Information about a single chunk from a dataset."""

    chunk_id: str
    chunk_index: int
    chunk_hash: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentInfo:
    """Information about a document (group of chunks)."""

    doc_id: str
    name: str | None
    chunks: list[ChunkInfo] = field(default_factory=list)


class HuggingFaceSource:
    """
    Source for loading datasets from HuggingFace.

    Handles:
    - Fetching dataset metadata (last modified date)
    - Loading dataset in streaming mode
    - Grouping chunks by document
    """

    def __init__(self, token: str | None = None):
        """
        Initialize the HuggingFace source.

        Args:
            token: HuggingFace API token for private datasets
        """
        self.token = token
        self.api = HfApi(token=token)

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """
        Get metadata about a dataset.

        Args:
            dataset_name: Full dataset name (e.g., "AgentPublic/legi")

        Returns:
            Dataset information including last modified date
        """
        try:
            info = self.api.dataset_info(dataset_name)
            return DatasetInfo(
                name=dataset_name,
                last_modified=info.last_modified,
                size_bytes=info.card_data.get("download_size") if info.card_data else None,
            )
        except Exception as e:
            logger.warning(f"Could not fetch info for {dataset_name}: {e}")
            return DatasetInfo(
                name=dataset_name,
                last_modified=None,
                size_bytes=None,
            )

    def _extract_metadata(
        self,
        row: dict[str, Any],
        dataset_name: str,
    ) -> dict[str, Any]:
        """
        Extract relevant metadata fields from a dataset row.

        Args:
            row: Raw row from the dataset
            dataset_name: Dataset name for field mapping

        Returns:
            Dictionary of metadata fields (non-null values only)
        """
        fields = DATASET_METADATA_FIELDS.get(dataset_name, ["title", "url"])
        metadata = {}

        for field_name in fields:
            if field_name in row and row[field_name] is not None:
                value = row[field_name]
                # Convert dates to ISO strings
                if isinstance(value, datetime):
                    value = value.isoformat()
                # Skip empty strings
                if isinstance(value, str) and not value.strip():
                    continue
                # Limit string length for metadata (Albert has limits)
                if isinstance(value, str) and len(value) > 255:
                    value = value[:252] + "..."
                metadata[field_name] = value

        return metadata

    def _get_document_name(self, row: dict[str, Any]) -> str | None:
        """Extract document name from a row (tries title fields)."""
        for field_name in ["title", "full_title", "name"]:
            if field_name in row and row[field_name]:
                title = str(row[field_name])
                # Limit length for document name
                if len(title) > 255:
                    return title[:252] + "..."
                return title
        return None

    def iter_documents(
        self,
        dataset_name: str,
        config: str = "latest",
        split: str = "train",
    ) -> Iterator[DocumentInfo]:
        """
        Iterate over documents in a dataset, grouping chunks.

        Uses streaming to handle large datasets without loading everything
        into memory. Yields one DocumentInfo per unique doc_id.

        Args:
            dataset_name: Full dataset name (e.g., "AgentPublic/legi")
            config: Dataset configuration name
            split: Dataset split name

        Yields:
            DocumentInfo objects containing all chunks for each document
        """
        logger.info(f"Loading dataset {dataset_name} (streaming mode)")

        try:
            dataset = load_dataset(
                dataset_name,
                config,
                split=split,
                streaming=True,
                token=self.token,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        current_doc_id: str | None = None
        current_document: DocumentInfo | None = None
        chunk_count = 0

        for row in dataset:
            doc_id = row.get("doc_id")
            if not doc_id:
                logger.warning(f"Row missing doc_id, skipping: {row.get('chunk_id')}")
                continue

            # New document detected
            if doc_id != current_doc_id:
                # Yield previous document if exists
                if current_document is not None:
                    yield current_document

                # Start new document
                current_doc_id = doc_id
                current_document = DocumentInfo(
                    doc_id=doc_id,
                    name=self._get_document_name(row),
                    chunks=[],
                )

            # Extract chunk info
            chunk = ChunkInfo(
                chunk_id=row.get("chunk_id", f"{doc_id}_{row.get('chunk_index', 0)}"),
                chunk_index=row.get("chunk_index", 0),
                chunk_hash=row.get("chunk_xxh64", ""),
                content=row.get("chunk_text", row.get("text", "")),
                metadata=self._extract_metadata(row, dataset_name),
            )

            if current_document is not None:
                current_document.chunks.append(chunk)

            chunk_count += 1
            if chunk_count % 1000 == 0:
                logger.info(f"Processed {chunk_count} chunks...")

        # Yield last document
        if current_document is not None:
            yield current_document

        logger.info(f"Finished processing {chunk_count} chunks from {dataset_name}")

    def count_documents(
        self,
        dataset_name: str,
        config: str = "latest",
        split: str = "train",
    ) -> tuple[int, int]:
        """
        Count documents and chunks in a dataset (loads full dataset).

        Args:
            dataset_name: Full dataset name

        Returns:
            Tuple of (document_count, chunk_count)
        """
        doc_ids: set[str] = set()
        chunk_count = 0

        for doc in self.iter_documents(dataset_name, config, split):
            doc_ids.add(doc.doc_id)
            chunk_count += len(doc.chunks)

        return len(doc_ids), chunk_count
