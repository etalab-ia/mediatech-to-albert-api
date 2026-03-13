"""Configuration module for Mediatech to Albert API sync."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Albert API
    albert_api_url: str = Field(
        default="https://albert.api.etalab.gouv.fr",
        description="Base URL for the Albert API",
    )
    albert_api_token: str = Field(
        description="Bearer token for Albert API authentication",
    )

    # HuggingFace
    huggingface_token: str = Field(
        description="HuggingFace API token for dataset access",
    )

    # SQLite database
    sqlite_path: str = Field(
        default="./state.db",
        description="Path to the SQLite database file for state tracking",
    )

    # Sync settings
    chunk_batch_size: int = Field(
        default=64,
        description="Maximum number of chunks per API request",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Datasets to sync
DATASETS = [
    "AgentPublic/legi",
    "AgentPublic/travail-emploi",
    "AgentPublic/service-public",
    "AgentPublic/local-administrations-directory",
    "AgentPublic/state-administrations-directory",
    "AgentPublic/dole",
    "AgentPublic/constit",
    "AgentPublic/cnil",
]

# Field used as document name for each dataset
DATASET_TITLE_FIELD: dict[str, str] = {
    "AgentPublic/legi": "title",
    "AgentPublic/travail-emploi": "title",
    "AgentPublic/service-public": "title",
    "AgentPublic/dole": "title",
    "AgentPublic/constit": "title",
    "AgentPublic/cnil": "title",
    "AgentPublic/local-administrations-directory": "name",
    "AgentPublic/state-administrations-directory": "name",
}

# Metadata fields to extract from each dataset (string/scalar fields only)
DATASET_METADATA_FIELDS: dict[str, list[str]] = {
    "AgentPublic/legi": [
        "title",
        "full_title",
        "nature",
        "category",
        "ministry",
        "status",
        "start_date",
        "end_date",
        "number",
        "url",
    ],
    "AgentPublic/service-public": [
        "title",
        "audience",
        "theme",
        "surtitle",
        "url",
        "introduction",
    ],
    "AgentPublic/travail-emploi": [
        "title",
        "url",
    ],
    "AgentPublic/local-administrations-directory": [
        "name",
        "types",
        "mission_description",
        "additional_information",
        "directory_url",
    ],
    "AgentPublic/state-administrations-directory": [
        "name",
        "types",
        "mission_description",
        "additional_information",
        "directory_url",
    ],
    "AgentPublic/dole": [
        "title",
        "url",
    ],
    "AgentPublic/constit": [
        "title",
        "url",
    ],
    "AgentPublic/cnil": [
        "title",
        "url",
    ],
}


def get_settings() -> Settings:
    return Settings()
