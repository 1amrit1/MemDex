"""The on-disk vector store: one embedded Chroma collection; MemDex 'collections' (rules, status, ...) are a metadata field.

One physical collection keeps scores comparable across categories, and a `where` filter gives the per-category view.
Embeddings are always passed in explicitly, so Chroma's own embedding function is never invoked.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings

COLLECTION = "memdex"


def open_client(index_dir: Path) -> chromadb.ClientAPI:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


def reset_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    return client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection | None:
    if COLLECTION not in [c.name for c in client.list_collections()]:
        return None
    return client.get_collection(COLLECTION)


def summarize(collection: chromadb.Collection) -> dict:
    """Counts straight from the store (not from the last ingest report)."""
    metas = collection.get(include=["metadatas"])["metadatas"] or []
    by_collection = Counter(m["collection"] for m in metas)
    files = {m["path"] for m in metas}
    superseded_files = {m["path"] for m in metas if m.get("superseded")}
    return {
        "chunks": len(metas),
        "files": len(files),
        "superseded_files": len(superseded_files),
        "by_collection": dict(sorted(by_collection.items())),
    }
