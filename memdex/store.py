"""The on-disk vector store: one embedded Chroma collection; MemDex 'collections' (rules, status, ...) and
'projects' are metadata fields.

One physical collection keeps scores comparable across categories, and a `where` filter gives the
per-category / per-project view. Embeddings are always passed in explicitly, so Chroma's own embedding
function is never invoked.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings

COLLECTION = "memdex"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


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
    by_project = Counter(m.get("project", "") for m in metas)
    files = {m["path"] for m in metas}
    superseded_files = {m["path"] for m in metas if m.get("superseded")}
    return {
        "chunks": len(metas),
        "files": len(files),
        "superseded_files": len(superseded_files),
        "by_collection": dict(sorted(by_collection.items())),
        "by_project": dict(sorted(by_project.items())),
        "chunks_without_project": by_project.get("", 0),  # > 0 means the index predates projects: re-ingest
    }


def index_size_bytes(index_dir: Path) -> int:
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())


def vector_segment_folders(index_dir: Path) -> dict:
    """Which HNSW folders in the index belong to a live collection and which were left behind by a rebuild."""
    index_dir = Path(index_dir)
    folders = sorted(p.name for p in index_dir.iterdir() if p.is_dir() and _UUID.match(p.name)) if index_dir.is_dir() else []
    live: set[str] = set()
    known = False
    db = index_dir / "chroma.sqlite3"
    if db.is_file():
        try:
            with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
                live = {row[0] for row in con.execute("SELECT id FROM segments WHERE scope = 'VECTOR'")}
            known = True
        except sqlite3.Error:
            known = False
    return {
        "live": [f for f in folders if f in live],
        "orphan": [f for f in folders if f not in live] if known else [],
        "known": known,
        "orphan_bytes": sum(index_size_bytes(index_dir / f) for f in folders if f not in live) if known else 0,
    }
