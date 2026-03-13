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
pip install -r requirements.txt
cp .env.example .env
# Fill in ALBERT_API_TOKEN and HUGGINGFACE_TOKEN
```

## Usage

```bash
# Sync all datasets
python sync.py

# Sync a single dataset
python sync.py --dataset AgentPublic/travail-emploi

# Show sync status (HF chunks vs local vs Albert)
python sync.py --status

# Force re-sync even if dataset is unchanged
python sync.py --force
```

## Environment variables

| Variable | Required | Default                                 | Description |
|----------|----------|-----------------------------------------|-------------|
| `ALBERT_API_TOKEN` | yes | —                                       | Bearer token for Albert API |
| `HUGGINGFACE_TOKEN` | yes | —                                       | HuggingFace API token |
| `ALBERT_API_URL` | no | `https://albert.api.dev.etalab.gouv.fr` | Albert API base URL |
| `SQLITE_PATH` | no | `./state.db`                            | Path to state database |
| `LOG_LEVEL` | no | `INFO`                                  | Logging level |


## TODO
- see if perf can be improved (should take 9h for biggest collection)
- handle state.db persistence (docker volume?)
- deployment and scheduling (docker, ansible, gitlab CI scheduled pipelines)
- add other tests
- make sure it doesn't slow down the API too much for other users
- add matrix/tchap notifications