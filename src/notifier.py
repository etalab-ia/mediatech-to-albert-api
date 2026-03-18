"""Tchap (Matrix) notification sender."""

import logging

import httpx

from .sync_service import SyncResult

logger = logging.getLogger(__name__)


class TchapNotifier:
    """Sends notifications to a Tchap room via the Matrix API."""

    def __init__(self, homeserver: str, access_token: str, room_id: str):
        self.homeserver = homeserver.rstrip("/")
        self.room_id = room_id
        self._headers = {"Authorization": f"Bearer {access_token}"}

    def send(self, message: str) -> None:
        """Send a plain-text message to the configured room. Logs a warning on failure."""
        url = f"{self.homeserver}/_matrix/client/v3/rooms/{self.room_id}/send/m.room.message"
        payload = {"msgtype": "m.text", "body": message}

        try:
            response = httpx.post(url, json=payload, headers=self._headers, timeout=10)
            response.raise_for_status()
            logger.debug("Tchap notification sent")
        except Exception as e:
            logger.warning(f"Failed to send Tchap notification: {e}")

    @staticmethod
    def format_sync_result(result: SyncResult) -> str:
        """Format a SyncResult into a human-readable message."""
        status = "✅ Success sync" if result.success else "❌ Errors in sync"
        lines = [status, ""]

        for ds in result.datasets:
            if not ds.success:
                lines.append(f"• {ds.dataset_name}: FAILED — {ds.error}")
            else:
                parts = []
                if ds.documents_created:
                    parts.append(f"+{ds.documents_created}")
                if ds.documents_updated:
                    parts.append(f"~{ds.documents_updated}")
                if ds.documents_deleted:
                    parts.append(f"-{ds.documents_deleted}")
                if ds.documents_unchanged:
                    parts.append(f"={ds.documents_unchanged} unchanged")
                summary = ", ".join(parts) if parts else "already up-to date, nothing to do"
                lines.append(
                    f"• {ds.dataset_name}: {summary}"
                    f" ({ds.chunks_uploaded:,} chunks, {ds.duration_seconds:.0f}s)"
                )

        lines.append(f"\nDuration : {result.total_duration_seconds:.0f}s")
        return "\n".join(lines)
