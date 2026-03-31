"""Integration tests for Albert API retrieval — converted from test_retrieval.sh."""

import pytest

pytestmark = pytest.mark.integration

COLLECTION_NAME = "travail-emploi"
SEARCH_QUERY = "médaille d'honneur du travail"


def test_travail_emploi_collection_exists(albert_client):
    """The travail-emploi collection must exist and contain documents."""
    collection = albert_client.get_collection_by_name(COLLECTION_NAME)
    if collection is None:
        pytest.skip(f"'{COLLECTION_NAME}' collection not found — run sync first")
    assert collection.name == COLLECTION_NAME
    assert collection.documents_count > 0, f"'{COLLECTION_NAME}' collection has no documents"


def test_semantic_search_returns_chunks(albert_client):
    """Semantic search on travail-emploi returns non-empty chunks with content."""
    collection = albert_client.get_collection_by_name(COLLECTION_NAME)
    if collection is None:
        pytest.skip(f"'{COLLECTION_NAME}' collection not found — run sync first")

    results = albert_client.search(
        prompt=SEARCH_QUERY,
        collection_ids=[collection.id],
        k=6,
    )

    assert len(results) > 0, f"No chunks returned for query: `{SEARCH_QUERY}`"
    for result in results:
        assert result.chunk["content"], "Each result must have non-empty chunk content"
