"""HuggingFace dataset source for loading parquet data."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from datasets import load_dataset, load_dataset_builder, disable_progress_bar, enable_progress_bar
from huggingface_hub import HfApi

from .config import DATASET_METADATA_FIELDS, DATASET_TITLE_FIELD

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
    """Source for loading datasets from HuggingFace."""

    def __init__(self, token: str | None = None):
        self.token = token
        self.api = HfApi(token=token)

    def get_chunk_count(self, dataset_name: str, config: str = "latest") -> int | None:
        """Get total chunk count from dataset metadata (no download needed)."""
        try:
            disable_progress_bar()
            try:
                builder = load_dataset_builder(dataset_name, config, token=self.token)
            finally:
                enable_progress_bar()
            splits = builder.info.splits
            if splits and "train" in splits:
                return splits["train"].num_examples
        except Exception as e:
            logger.debug(f"Could not get chunk count for {dataset_name}: {e}")
        return None

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Get metadata about a dataset."""
        try:
            info = self.api.dataset_info(dataset_name)
            return DatasetInfo(
                name=dataset_name,
                last_modified=info.last_modified,
                size_bytes=info.card_data.get("download_size") if info.card_data else None,
            )
        except Exception as e:
            logger.warning(f"Could not fetch info for {dataset_name}: {e}")
            return DatasetInfo(name=dataset_name, last_modified=None)

    def _extract_metadata(self, row: dict[str, Any], dataset_name: str) -> dict[str, Any]:
        """Extract relevant metadata fields from a dataset row (non-null scalar values only)."""
        fields = DATASET_METADATA_FIELDS.get(dataset_name, ["title", "url"])
        metadata = {}

        for field_name in fields:
            if field_name not in row or row[field_name] is None:
                continue
            value = row[field_name]
            if isinstance(value, datetime):
                value = value.isoformat()
            if isinstance(value, str) and not value.strip():
                continue
            # TODO: verify whether Albert API has a 255-char limit on metadata values
            if isinstance(value, str) and len(value) > 255:
                value = value[:252] + "..."
            metadata[field_name] = value

        return metadata

    def _get_document_name(self, row: dict[str, Any], dataset_name: str) -> str | None:
        """Extract document name using the dataset-specific title field."""
        field_name = DATASET_TITLE_FIELD.get(dataset_name, "title")
        value = row.get(field_name)
        if not value:
            return None
        title = str(value)
        # TODO: verify whether Albert API has a 255-char limit on document names
        if len(title) > 255:
            return title[:252] + "..."
        return title

    def iter_documents(
        self,
        dataset_name: str,
        config: str = "latest",
        split: str = "train",
    ) -> Iterator[DocumentInfo]:
        """
        Iterate over documents in a dataset, grouping chunks by doc_id.

        Uses streaming to handle large datasets without loading everything into memory.
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

            if doc_id != current_doc_id:
                if current_document is not None:
                    yield current_document

                current_doc_id = doc_id
                current_document = DocumentInfo(
                    doc_id=doc_id,
                    name=self._get_document_name(row, dataset_name),
                    chunks=[],
                )

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

        if current_document is not None:
            yield current_document

        logger.info(f"Finished processing {chunk_count} chunks from {dataset_name}")
