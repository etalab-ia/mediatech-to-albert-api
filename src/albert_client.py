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
    owner: str
    visibility: str
    documents_count: int


@dataclass
class ChunkData:
    """Data for a chunk to upload."""

    content: str
    metadata: dict[str, Any] | None = None


class AlbertClient:
    """
    HTTP client for interacting with the Albert API.

    Handles collections, documents, and chunks operations.
    """

    def __init__(self, base_url: str, api_token: str, timeout: float = 60.0):
        """
        Initialize the Albert client.

        Args:
            base_url: Base URL of the Albert API (e.g., "https://albert.api.etalab.gouv.fr")
            api_token: Bearer token for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self._json_headers = {"Content-Type": "application/json"}
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "AlbertClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise errors if needed."""
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

    def list_collections(
        self,
        name: str | None = None,
        visibility: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CollectionInfo]:
        """
        List collections.

        Args:
            name: Filter by collection name
            visibility: Filter by visibility ("private" or "public")
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of collection info objects
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if name:
            params["name"] = name
        if visibility:
            params["visibility"] = visibility

        response = self.client.get("/v1/collections", params=params, headers=self._json_headers)
        data = self._handle_response(response)

        return [
            CollectionInfo(
                id=c["id"],
                name=c["name"],
                owner=c["owner"],
                visibility=c["visibility"],
                documents_count=c.get("documents", 0),
            )
            for c in data.get("data", [])
        ]

    def get_collection_by_name(self, name: str) -> CollectionInfo | None:
        """
        Get a collection by its exact name.

        Args:
            name: Collection name to search for

        Returns:
            Collection info if found, None otherwise
        """
        collections = self.list_collections(name=name)
        for c in collections:
            if c.name == name:
                return c
        return None

    def create_collection(
        self,
        name: str,
        visibility: str = "public",
        description: str | None = None,
    ) -> int:
        """
        Create a new collection.

        Args:
            name: Collection name
            visibility: "public" or "private"
            description: Optional description

        Returns:
            The created collection ID
        """
        payload: dict[str, Any] = {
            "name": name,
            "visibility": visibility,
        }
        if description:
            payload["description"] = description

        logger.info(f"Creating collection: {name}")
        response = self.client.post("/v1/collections", json=payload, headers=self._json_headers)
        data = self._handle_response(response)
        collection_id = data["id"]
        logger.info(f"Created collection {name} with ID {collection_id}")
        return collection_id

    def delete_collection(self, collection_id: int) -> None:
        """
        Delete a collection and all its documents.

        Args:
            collection_id: ID of the collection to delete
        """
        logger.info(f"Deleting collection {collection_id}")
        response = self.client.delete(f"/v1/collections/{collection_id}")
        self._handle_response(response)

    def create_document(self, collection_id: int, name: str) -> int:
        """
        Create an empty document in a collection.

        Args:
            collection_id: ID of the parent collection
            name: Document name

        Returns:
            The created document ID
        """
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
        """
        Delete a document and all its chunks.

        Args:
            document_id: ID of the document to delete
        """
        logger.debug(f"Deleting document {document_id}")
        response = self.client.delete(f"/v1/documents/{document_id}")
        self._handle_response(response)

    def list_documents(
        self,
        collection_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List documents in a collection.

        Args:
            collection_id: ID of the collection
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of document dictionaries
        """
        params = {
            "collection_id": collection_id,
            "limit": limit,
            "offset": offset,
        }
        response = self.client.get("/v1/documents", params=params, headers=self._json_headers)
        data = self._handle_response(response)
        return data.get("data", [])

    def create_chunks(
        self,
        document_id: int,
        chunks: list[ChunkData],
    ) -> list[int]:
        """
        Upload chunks to a document.

        Args:
            document_id: ID of the parent document
            chunks: List of chunk data (max 64 per request)

        Returns:
            List of created chunk IDs

        Raises:
            ValueError: If more than 64 chunks are provided
        """
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
        """
        Upload chunks in batches of max batch_size.

        Args:
            document_id: ID of the parent document
            chunks: List of all chunks to upload
            batch_size: Maximum chunks per request (default 64)

        Returns:
            List of all created chunk IDs
        """
        all_ids: list[int] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            ids = self.create_chunks(document_id, batch)
            all_ids.extend(ids)
            logger.debug(
                f"Uploaded batch {i // batch_size + 1}: {len(batch)} chunks"
            )

        return all_ids
