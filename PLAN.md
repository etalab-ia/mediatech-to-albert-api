# Mediatech to Albert API - Synchronisation

## Contexte

Ce projet synchronise des datasets Hugging Face (format parquet) vers l'API Albert pour alimenter le RAG.

### Datasets sources (Hugging Face)

| Dataset | Description | Volume estimé |
|---------|-------------|---------------|
| `AgentPublic/legi` | Législation française consolidée | ~2.14M rows |
| `AgentPublic/travail-emploi` | Droit du travail | - |
| `AgentPublic/service-public` | Procédures administratives | ~34.6k rows |
| `AgentPublic/local-administrations-directory` | Annuaire administrations locales | - |
| `AgentPublic/state-administrations-directory` | Annuaire administrations d'État | - |
| `AgentPublic/dole` | - | - |
| `AgentPublic/constit` | Constitution | - |
| `AgentPublic/cnil` | CNIL | - |

**Volume total** : ~15 Go

### Structure des datasets

Colonnes communes à tous les datasets :
- `chunk_id` : Identifiant unique du chunk
- `doc_id` : Identifiant du document parent
- `chunk_index` : Position du chunk dans le document
- `chunk_xxh64` : Hash XXH64 du contenu (pour détection des changements)
- `chunk_text` : Contenu textuel du chunk
- `embeddings_bge-m3` : Embeddings pré-calculés (NON UTILISÉS - Albert regénère)

Colonnes de métadonnées (varient selon le dataset) :
- `title`, `full_title` : Titre du document
- `url` : Lien vers la source
- `ministry` : Ministère responsable
- `category`, `nature` : Classification
- `status` : Statut (ex: "VIGUEUR")
- `start_date`, `end_date` : Dates de validité

### API Albert (OpenGateLLM)

**Endpoints utilisés** :

```
POST /v1/collections
  - name: string
  - visibility: "public"
  → Retourne: { "id": collection_id }

POST /v1/documents
  - name: string
  - collection_id: int
  → Retourne: { "id": document_id }

POST /v1/documents/{document_id}/chunks
  - chunks: [{ "content": string, "metadata": object }]
  - Max 64 chunks par requête
  → Retourne: { "ids": [chunk_ids] }

DELETE /v1/documents/{document_id}
  - Supprime le document et tous ses chunks

GET /v1/collections
  - Lister les collections existantes
```

**Authentification** : Bearer token

**Owner des collections publiques** : `albert.api@numerique.gouv.fr`

---

## Architecture

### Structure du projet

```
mediatech-to-albert-api/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (datasets, credentials)
│   ├── models.py              # Modèles SQLAlchemy (état local)
│   ├── albert_client.py       # Client HTTP pour l'API Albert
│   ├── huggingface_source.py  # Récupération des datasets HF
│   ├── state_store.py         # Repository SQLite
│   └── sync_service.py        # Orchestrateur de synchronisation
├── sync.py                    # Point d'entrée CLI
├── state.db                   # Base SQLite (état local, gitignored)
├── requirements.txt
├── PLAN.md                    # Ce fichier
└── README.md
```

### Modèle de données (état local SQLite)

```
┌─────────────────────────────────────────────────────────────┐
│ collections                                                 │
├─────────────────────────────────────────────────────────────┤
│ id                  INTEGER PRIMARY KEY                     │
│ dataset_name        TEXT UNIQUE        (ex: "AgentPublic/legi") │
│ albert_collection_id TEXT              (ID retourné par Albert) │
│ last_modified       TEXT               (date HF au format ISO)  │
│ status              TEXT               (idle/syncing/success/failed) │
│ created_at          TEXT                                    │
│ updated_at          TEXT                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ documents                                                   │
├─────────────────────────────────────────────────────────────┤
│ id                  INTEGER PRIMARY KEY                     │
│ collection_id       INTEGER FOREIGN KEY                     │
│ doc_id_source       TEXT               (doc_id du parquet)  │
│ albert_document_id  TEXT               (ID retourné par Albert) │
│ created_at          TEXT                                    │
│ updated_at          TEXT                                    │
│ UNIQUE(collection_id, doc_id_source)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ chunks                                                      │
├─────────────────────────────────────────────────────────────┤
│ id                  INTEGER PRIMARY KEY                     │
│ document_id         INTEGER FOREIGN KEY                     │
│ chunk_id_source     TEXT               (chunk_id du parquet) │
│ chunk_hash          TEXT               (chunk_xxh64 du parquet) │
│ albert_chunk_id     TEXT               (ID retourné par Albert) │
│ created_at          TEXT                                    │
│ UNIQUE(document_id, chunk_id_source)                        │
└─────────────────────────────────────────────────────────────┘
```

### Classes principales

```python
class Config:
    """Configuration chargée depuis variables d'environnement"""
    albert_api_url: str
    albert_api_token: str
    huggingface_token: str
    datasets: list[str]
    sqlite_path: str

class AlbertClient:
    """Client HTTP pour l'API Albert"""
    def create_collection(name: str, visibility: str) -> int
    def get_collections() -> list[dict]
    def create_document(collection_id: int, name: str) -> int
    def delete_document(document_id: int) -> None
    def create_chunks(document_id: int, chunks: list[dict]) -> list[int]

class HuggingFaceSource:
    """Récupération des datasets depuis Hugging Face"""
    def get_dataset_info(dataset_name: str) -> DatasetInfo
    def iter_documents(dataset_name: str) -> Iterator[Document]
    # Streaming pour gérer les gros volumes

class StateStore:
    """Repository pour l'état local (SQLite)"""
    def get_collection(dataset_name: str) -> Collection | None
    def upsert_collection(...) -> Collection
    def get_documents(collection_id: int) -> list[Document]
    def get_chunks(document_id: int) -> list[Chunk]
    # etc.

class SyncService:
    """Orchestrateur de synchronisation"""
    def sync_all() -> SyncResult
    def sync_dataset(dataset_name: str) -> DatasetSyncResult
```

---

## Logique de synchronisation

### Algorithme principal

```
Pour chaque dataset configuré:
  1. Récupérer les métadonnées HF (date de modification)
  2. Comparer avec last_modified stocké localement

  Si pas de changement:
    → Skip

  Si changement détecté:
    3. Marquer la collection comme "syncing"
    4. Charger le dataset en streaming
    5. Grouper les chunks par doc_id

    Pour chaque document:
      a. Vérifier si le document existe localement

      Si nouveau document:
        → Créer le document dans Albert
        → Uploader les chunks (par batch de 64)
        → Sauvegarder les mappings d'IDs

      Si document existant:
        → Comparer les hashes des chunks

        Si changement détecté:
          → Supprimer le document dans Albert
          → Recréer le document + chunks
          → Mettre à jour les mappings

    6. Détecter les documents supprimés (présents localement mais absents de HF)
       → Supprimer dans Albert
       → Supprimer localement

    7. Mettre à jour last_modified
    8. Marquer la collection comme "success"
```

### Gestion des métadonnées

Pour chaque chunk uploadé vers Albert, inclure les métadonnées pertinentes :

```python
METADATA_FIELDS = {
    "legi": ["title", "full_title", "nature", "category", "ministry", "status", "start_date", "end_date", "url"],
    "service-public": ["title", "audience", "theme", "url"],
    "travail-emploi": ["title", "url"],
    # ... à compléter pour chaque dataset
}
```

### Gestion des volumes (15 Go)

- **Streaming** : Utiliser `datasets` en mode streaming pour ne pas charger tout en mémoire
- **Batches** : Traiter les documents par lots
- **Chunks** : Envoyer max 64 chunks par requête à Albert (contrainte API)

---

## Variables d'environnement

```bash
# Obligatoires
ALBERT_API_URL=https://albert.api.etalab.gouv.fr
ALBERT_API_TOKEN=<token>
HUGGINGFACE_TOKEN=<token>

# Optionnels
SQLITE_PATH=./state.db
LOG_LEVEL=INFO
BATCH_SIZE=64
```

---

## Plan d'implémentation

### Phase 1 - MVP (actuel)

- [x] Documentation du contexte et du plan
- [x] Configuration (`config.py`)
- [x] Modèles SQLAlchemy (`models.py`)
- [x] Client Albert (`albert_client.py`)
- [x] Source HuggingFace (`huggingface_source.py`)
- [x] State store (`state_store.py`)
- [x] Service de sync (`sync_service.py`)
- [x] Point d'entrée CLI (`sync.py`)
- [ ] Tests manuels

### Phase 2 - Robustesse

- [ ] Retry avec backoff exponentiel (tenacity)
- [ ] Gestion des erreurs plus fine
- [ ] Mode dry-run
- [ ] Tests unitaires
- [ ] Tests d'intégration (mock Albert API)

### Phase 3 - Observabilité

- [ ] Métriques (durée, nb chunks, erreurs)
- [ ] Notifications Matrix en cas d'erreur
- [ ] Dashboard de monitoring

---

## Dépendances

```
# requirements.txt
httpx>=0.27.0          # Client HTTP async
datasets>=2.18.0       # Chargement datasets HF
huggingface-hub>=0.21.0  # API HuggingFace
sqlalchemy>=2.0.0      # ORM
pydantic>=2.0.0        # Validation config
pydantic-settings>=2.0.0  # Chargement env vars
python-dotenv>=1.0.0   # Fichier .env local
```

---

## Exécution

```bash
# Installation
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec les tokens

# Lancer la synchronisation
python sync.py

# Synchroniser un seul dataset
python sync.py --dataset AgentPublic/legi

# Mode dry-run (Phase 2)
python sync.py --dry-run
```

---

## Notes et décisions

1. **Embeddings** : Albert regénère ses propres embeddings. On n'utilise PAS les `embeddings_bge-m3` des datasets HF pour ne pas être dépendants du modèle.

2. **Détection des changements** : On utilise le champ `chunk_xxh64` présent dans les datasets comme hash de référence. Pas besoin de recalculer.

3. **Suppression de documents** : Si un document est modifié (hash d'un chunk différent), on supprime le document entier dans Albert et on le recrée. C'est plus simple que de gérer des updates partiels.

4. **SQLite vs in-memory** : On utilise un fichier SQLite persistant pour conserver l'état entre les exécutions (contrairement à ce que suggéraient les specs originales avec "in-memory").

5. **Pas d'Alembic** : Le schéma est simple et stable, SQLAlchemy `create_all()` suffit.

6. **Pas de Matrix pour le MVP** : Les logs GitLab suffisent pour le debugging initial.
