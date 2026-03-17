"""HTTP client for the Albert API."""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AlbertAPIError(Exception):
    """Exception raised when Albert API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Albert API error {status_code}: {message}")


@dataclass
class CollectionInfo:
    """Information about an Albert collection."""

    id: int
    name: str
    documents_count: int = 0


@dataclass
class ChunkData:
    """Data for a chunk to upload."""

    content: str
    metadata: dict[str, Any] | None = None


class AlbertClient:
    """HTTP client for interacting with the Albert API."""

    def __init__(self, base_url: str, api_token: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._json_headers = {"Content-Type": "application/json"}
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "AlbertClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                error_detail = response.json()
                message = error_detail.get("detail", response.text)
            except Exception:
                message = response.text
            raise AlbertAPIError(response.status_code, message)

        if response.status_code == 204:
            return None

        return response.json()

    def get_collection_by_name(self, name: str) -> CollectionInfo | None:
        """Get a collection by its exact name, or None if not found."""
        response = self.client.get(
            "/v1/collections", params={"name": name}, headers=self._json_headers
        )
        data = self._handle_response(response)
        for c in data.get("data", []):
            if c["name"] == name:
                return CollectionInfo(
                    id=c["id"],
                    name=c["name"],
                    documents_count=c.get("documents", 0),
                )
        return None

    def create_collection(self, name: str) -> int:
        """Create a new public collection. Returns the collection ID."""
        logger.info(f"Creating collection: {name}")
        response = self.client.post(
            "/v1/collections",
            json={"name": name, "visibility": "public"},
            headers=self._json_headers,
        )
        data = self._handle_response(response)
        collection_id = data["id"]
        logger.info(f"Created collection {name} with ID {collection_id}")
        return collection_id

    def delete_collection(self, collection_id: int) -> None:
        """Delete a collection and all its documents."""
        logger.info(f"Deleting collection {collection_id}")
        response = self.client.delete(f"/v1/collections/{collection_id}")
        self._handle_response(response)

    def create_document(self, collection_id: int, name: str) -> int:
        """Create an empty document in a collection. Returns the document ID."""
        response = self.client.post(
            "/v1/documents",
            data={
                "collection_id": str(collection_id),
                "name": name,
            },
        )
        data = self._handle_response(response)
        return data["id"]

    def delete_document(self, document_id: int) -> None:
        """Delete a document and all its chunks."""
        logger.debug(f"Deleting document {document_id}")
        response = self.client.delete(f"/v1/documents/{document_id}")
        self._handle_response(response)

    def create_chunks(self, document_id: int, chunks: list[ChunkData]) -> list[int]:
        """Upload chunks to a document (max 64 per request). Returns chunk IDs."""
        if len(chunks) > 64:
            raise ValueError("Maximum 64 chunks per request")

        payload = {
            "chunks": [
                {
                    "content": c.content,
                    **({"metadata": c.metadata} if c.metadata else {}),
                }
                for c in chunks
            ]
        }

        response = self.client.post(
            f"/v1/documents/{document_id}/chunks",
            json=payload,
            headers=self._json_headers,
        )
        data = self._handle_response(response)
        return data.get("ids", [])

    def upload_chunks_batched(
        self,
        document_id: int,
        chunks: list[ChunkData],
        batch_size: int = 64,
    ) -> list[int]:
        """Upload chunks in batches. Returns all created chunk IDs."""
        all_ids: list[int] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = self.create_chunks(document_id, batch)
            all_ids.extend(ids)
            logger.debug(f"Uploaded batch {i // batch_size + 1}: {len(batch)} chunks")

        return all_ids

    def search(
        self,
        prompt: str,
        collection_ids: list[int],
        k: int = 6,
        method: str = "semantic",
    ) -> list[dict]:
        """Perform semantic search across specified collections. Returns list of results."""
        payload = {
            "prompt": prompt,
            "collections": collection_ids,
            "k": k,
            "method": method,
        }
        response = self.client.post(
            "/v1/search",
            json=payload,
            headers=self._json_headers,
        )
        data = self._handle_response(response)
        return data.get("data", [])
