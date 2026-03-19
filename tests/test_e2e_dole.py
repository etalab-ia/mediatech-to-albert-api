import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DATASET = "AgentPublic/dole"
COLLECTION_NAME = "dole"
# Query derived from the known content of the dataset (French legislative dossiers).
SEARCH_QUERY = "organismes génétiquement modifiés utilisation confinée agrément"

REPO_ROOT = Path(__file__).parent.parent


def test_dole_e2e_sync_and_retrieval(albert_client, settings, tmp_path):
    """Full E2E: no results before sync, results after sync."""

    # --- Phase 1: ensure clean state ---
    existing = albert_client.get_collection_by_name(COLLECTION_NAME)
    if existing:
        albert_client.delete_collection(existing.id)

    collection = albert_client.get_collection_by_name(COLLECTION_NAME)
    assert collection is None, (
        f"Expected no '{COLLECTION_NAME}' collection before sync, "
        f"but it still exists in Albert (id={getattr(collection, 'id', '?')})"
    )

    # --- Phase 2: run sync ---
    env = {
        **os.environ,
        "SQLITE_PATH": str(tmp_path / "state.db"),
    }
    proc = subprocess.run(
        ["python", "main.py", "--dataset", DATASET],
        cwd=str(REPO_ROOT),
        env=env,
        timeout=3600,
    )
    assert proc.returncode == 0, f"main.py exited with code {proc.returncode}"

    # --- Phase 3: verify collection was created ---
    collection = albert_client.get_collection_by_name(COLLECTION_NAME)
    assert collection is not None, (
        f"'{COLLECTION_NAME}' collection not found in Albert after sync"
    )
    assert collection.documents_count > 0, (
        f"'{COLLECTION_NAME}' collection has 0 documents after sync"
    )

    # --- Phase 4: verify RAG retrieval returns chunks ---
    results = albert_client.search(
        prompt=SEARCH_QUERY,
        collection_ids=[collection.id],
        k=6,
    )
    assert len(results) > 0, (
        f"No chunks returned for query `{SEARCH_QUERY}` after syncing `{COLLECTION_NAME}`"
    )
    for result in results:
        assert result.chunk["content"], "Each search result must have non-empty chunk content"

    # --- Cleanup ---
    albert_client.delete_collection(collection.id)
