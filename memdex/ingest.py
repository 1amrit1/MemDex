"""Walk the configured sources, chunk every file, embed the chunks locally, and write them to the store.

Milestones 1-2 rebuild the whole index on each run. Incremental (hash-based) re-indexing is Milestone 3.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from memdex.chunker import chunk_markdown, doc_date, is_superseded
from memdex.config import Config, Source
from memdex.embedder import Embedder, ModelMissing
from memdex import store

BATCH = 64


class NothingToIngest(RuntimeError):
    pass


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    truncated: int = 0
    by_collection: Counter = field(default_factory=Counter)
    by_project: Counter = field(default_factory=Counter)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    max_chars: int = 0
    finished_at: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["by_collection"] = dict(sorted(self.by_collection.items()))
        d["by_project"] = dict(sorted(self.by_project.items()))
        d["seconds"] = round(self.seconds, 1)
        return d


def discover(source: Source) -> Iterator[tuple[Path, str]]:
    for path in sorted(source.path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source.path).as_posix()
        if source.wants(rel):
            yield path, rel


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def run_ingest(cfg: Config, embedder: Embedder, client) -> IngestReport:
    if not cfg.sources:
        raise NothingToIngest("no sources: add folders to memdex.yaml or create a project in the UI")
    if not embedder.is_present():  # check before wiping the old index
        raise ModelMissing(f"embedding model not found in {embedder.model_path} - run: memdex fetch-model")

    started = time.perf_counter()
    report = IngestReport(max_chars=cfg.max_chars)
    collection = store.reset_collection(client)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    def flush() -> None:
        if ids:
            collection.add(ids=list(ids), documents=list(docs), metadatas=list(metas), embeddings=embedder.embed(docs))
            ids.clear(), docs.clear(), metas.clear()

    for s_idx, source in enumerate(cfg.sources):
        for path, rel in discover(source):
            try:
                text = _read(path)
            except OSError as exc:
                report.skipped.append((str(path), str(exc)))
                continue
            chunks = chunk_markdown(text, title=rel, max_chars=cfg.max_chars)
            if not chunks:
                report.skipped.append((str(path), "no content"))
                continue

            coll = source.collection_for(rel)
            file_meta = {
                "path": str(path),
                "rel_path": rel,
                "source": str(source.path),
                "project": source.name,
                "collection": coll,
                "doc_date": doc_date(text),
                "superseded": is_superseded(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            report.files += 1
            for i, ch in enumerate(chunks):
                if embedder.truncated(ch.text):
                    report.truncated += 1
                ids.append(f"{s_idx}:{rel}#{i}")
                docs.append(ch.text)
                metas.append(
                    {**file_meta, "heading_path": ch.heading_path, "start_line": ch.start_line, "end_line": ch.end_line, "chunk": i}
                )
                report.chunks += 1
                report.by_collection[coll] += 1
                report.by_project[source.name] += 1
                if len(ids) >= BATCH:
                    flush()
    flush()
    report.seconds = time.perf_counter() - started
    report.finished_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    try:
        cfg.last_ingest_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    except OSError:
        pass  # the report is a convenience for the UI, never a reason to fail an ingest
    return report


def last_ingest(cfg: Config) -> dict | None:
    try:
        return json.loads(cfg.last_ingest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
