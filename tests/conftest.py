"""Shared fixtures for integration tests."""

import pytest

from src.albert_client import AlbertClient
from src.config import get_settings


@pytest.fixture(scope="session")
def settings():
    try:
        return get_settings()
    except Exception:
        pytest.skip("Missing required environment variables (ALBERT_API_TOKEN, HUGGINGFACE_TOKEN)")


@pytest.fixture(scope="session")
def albert_client(settings):
    with AlbertClient(settings.albert_api_url, settings.albert_api_token) as client:
        yield client
