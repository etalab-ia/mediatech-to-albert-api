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
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    @staticmethod
    def _env_from_url(albert_api_url: str) -> str:
        if "staging" in albert_api_url:
            return "staging"
        if "dev" in albert_api_url:
            return "dev"
        if albert_api_url.rstrip("/") == "https://albert.api.etalab.gouv.fr":
            return "prod"
        return albert_api_url

    @staticmethod
    def format_sync_result(result: SyncResult, albert_api_url: str = "") -> str:
        env = TchapNotifier._env_from_url(albert_api_url)
        status = "✅ Mediatech sync success" if result.success else "❌ Mediatech sync errors"
        lines = [f"{status} [{env}]", ""]

        for ds in result.datasets:
            if not ds.success:
                lines.append(f"• {ds.dataset_name}: FAILED — {ds.error}")
            else:
                parts = []
                if ds.documents_created:
                    parts.append(f"+{ds.documents_created:,} docs created")
                if ds.documents_updated:
                    parts.append(f"~{ds.documents_updated:,} docs updated")
                if ds.documents_deleted:
                    parts.append(f"-{ds.documents_deleted:,} docs deleted")
                if ds.documents_unchanged:
                    parts.append(f"{ds.documents_unchanged:,} docs unchanged")
                summary = ", ".join(parts) if parts else "already up-to-date, nothing to do"
                lines.append(
                    f"• {ds.dataset_name}: {summary}"
                    f" ({ds.chunks_uploaded:,} chunks uploaded, {TchapNotifier._format_duration(ds.duration_seconds)})"
                )

        lines.append(f"\nTotal duration: {TchapNotifier._format_duration(result.total_duration_seconds)}")
        return "\n".join(lines)
