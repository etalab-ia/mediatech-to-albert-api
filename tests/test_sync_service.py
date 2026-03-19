"""Tests for SyncService logic."""

from unittest.mock import MagicMock

from src.sync_service import SyncService
from src.huggingface_source import DocumentInfo, ChunkInfo


def make_service():
    settings = MagicMock()
    settings.chunk_batch_size = 64
    return SyncService(
        settings=settings,
        state_store=MagicMock(),
        albert_client=MagicMock(),
        hf_source=MagicMock(),
    )


def make_doc(chunks: list[tuple[str, str]]) -> DocumentInfo:
    """Helper: create a DocumentInfo with (chunk_id, chunk_hash) pairs."""
    return DocumentInfo(
        doc_id="doc1",
        name="Test Doc",
        chunks=[
            ChunkInfo(chunk_id=cid, chunk_index=i, chunk_hash=chash, content="text")
            for i, (cid, chash) in enumerate(chunks)
        ],
    )


def test_make_collection_name():
    svc = make_service()
    assert svc._make_collection_name("AgentPublic/legi") == "legi"
    assert svc._make_collection_name("AgentPublic/travail-emploi") == "travail-emploi"


def test_has_document_changed_changed_hash():
    svc = make_service()
    svc.state.get_document_chunk_hashes_by_chunk_id.return_value = {"chunk1": "oldhash"}
    assert svc._has_document_changed(1, make_doc([("chunk1", "newhash")])) is True


def test_has_document_changed_unchanged():
    svc = make_service()
    svc.state.get_document_chunk_hashes_by_chunk_id.return_value = {"chunk1": "samehash"}
    assert svc._has_document_changed(1, make_doc([("chunk1", "samehash")])) is False


def test_has_document_changed_deleted_chunk():
    svc = make_service()
    svc.state.get_document_chunk_hashes_by_chunk_id.return_value = {"chunk1": "h1", "chunk2": "h2"}
    assert svc._has_document_changed(1, make_doc([("chunk1", "h1")])) is True


def test_has_document_changed_new_chunk():
    svc = make_service()
    svc.state.get_document_chunk_hashes_by_chunk_id.return_value = {"chunk1": "h1"}
    assert svc._has_document_changed(1, make_doc([("chunk1", "h1"), ("chunk2", "h2")])) is True
