# mediatech-to-albert-api

Syncs HuggingFace datasets (Mediatech parquet files) into Albert API public collections for RAG.

## How it works

Each HuggingFace dataset maps to one Albert collection. The script streams the parquet data, groups chunks by `doc_id`, and creates/updates/deletes documents in Albert to keep the collection in sync.

**Change detection** happens at two levels:
- Dataset level: compares `last_modified` from HuggingFace — if unchanged and the Albert collection still exists, the dataset is skipped entirely.
- Document level: compares `chunk_xxh64` hashes stored in the local SQLite state. Only modified or new documents are re-uploaded.

**State** is tracked in a local SQLite database (`state.db`) mapping HuggingFace doc/chunk IDs to Albert API IDs. This enables incremental syncs and safe resume after interruption — each document is committed individually, so a crash mid-sync only loses at most one document.

**If `state.db` is lost**, the script detects the orphaned Albert collection, deletes and recreates it, then re-uploads everything. For this reason it's recommended to mount `state.db` on a persistent volume in production.

## Datasets

The datasets currently synchronized are :

- [AgentPublic/legi](https://huggingface.co/datasets/AgentPublic/legi)
- [AgentPublic/travail-emploi](https://huggingface.co/datasets/AgentPublic/travail-emploi)
- [AgentPublic/service-public](https://huggingface.co/datasets/AgentPublic/service-public)
- [AgentPublic/local-administrations-directory](https://huggingface.co/datasets/AgentPublic/local-administrations-directory)
- [AgentPublic/state-administrations-directory](https://huggingface.co/datasets/AgentPublic/state-administrations-directory)
- [AgentPublic/dole](https://huggingface.co/datasets/AgentPublic/dole)
- [AgentPublic/constit](https://huggingface.co/datasets/AgentPublic/constit)
- [AgentPublic/cnil](https://huggingface.co/datasets/AgentPublic/cnil)

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
# Fill in ALBERT_API_TOKEN and HUGGINGFACE_TOKEN
```

## Usage

```bash
# Sync all datasets
python main.py

# Sync only specific datasets
python main.py --dataset AgentPublic/travail-emploi AgentPublic/dole

# Show sync status (HF chunks vs local vs Albert)
python main.py --status

# Force re-sync even if dataset is unchanged
python main.py --force
```

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALBERT_API_TOKEN` | yes | — | Bearer token for Albert API |
| `HUGGINGFACE_TOKEN` | yes | — | HuggingFace API token |
| `ALBERT_API_URL` | no | `https://albert.api.dev.etalab.gouv.fr` | Albert API base URL |
| `SQLITE_PATH` | no | `./state.db` | Path to state database |
| `LOG_LEVEL` | no | `INFO` | Logging level |

## Docker

The Docker image uses `/data` as the mount point for the persistent `state.db`.

### Build

```bash
docker build -t mediatech-to-albert-api .
```

### Run with docker compose

```bash
# Sync all datasets
docker compose run --rm sync

# Sync a single dataset
docker compose run --rm sync --dataset AgentPublic/dole

# Show status
docker compose run --rm sync --status
```

The named volume `mediatech_state` persists `state.db` across container restarts and rebuilds.

### Backup state.db

```bash
# Find the volume path on the host
docker volume inspect mediatech-to-albert-api_state
# → "Mountpoint": "/var/lib/docker/volumes/mediatech-to-albert-api_state/_data"

# Copy directly from the host filesystem
cp /var/lib/docker/volumes/mediatech-to-albert-api_state/_data/state.db ~/state.db.backup
```

## Tests

### Unit tests

```bash
pytest tests/test_sync_service.py tests/test_huggingface_source.py -v
```

### Integration tests

Require a live Albert API and the `travail-emploi` collection to already be synced.
Credentials must be set in `.env` or as environment variables.

```bash
pytest tests/test_retrieval.py -v -m integration
```

### End-to-end test (dole)

Runs a full sync of `AgentPublic/dole` from scratch and verifies RAG retrieval works afterwards.

**Warning:** this test makes real API calls and takes 10 to 20 minutes.
It deletes and recreates the `dole` collection in Albert.

```bash
pytest tests/test_e2e_dole.py -v -m integration -s
```
- The test uses an isolated temporary `state.db` and cleans up the Albert collection after it completes.

## TODO
- see if perf can be improved
- deployment and scheduling (ansible, gitlab CI scheduled pipelines)
