"""Retrieval: embed the question with the same local model, ask Chroma for the nearest chunks, return hits
with their citation (file, heading path, line range). `rules` is `search` restricted to the rules collection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Hit:
    score: float  # 1 - cosine distance; higher is closer
    rel_path: str
    path: str
    heading_path: str
    start_line: int
    end_line: int
    collection: str
    project: str
    text: str
    doc_date: str
    superseded: bool
    id: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_where(collection: str | None = None, project: str | None = None, include_superseded: bool = False) -> dict | None:
    conds: list[dict] = []
    if collection:
        conds.append({"collection": collection})
    if project:
        conds.append({"project": project})
    if not include_superseded:
        conds.append({"superseded": False})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(
    embedder,
    chroma_collection,
    query: str,
    k: int = 5,
    collection: str | None = None,
    project: str | None = None,
    include_superseded: bool = False,
) -> list[Hit]:
    q = (query or "").strip()
    if not q:
        return []
    res = chroma_collection.query(
        query_embeddings=embedder.embed([q]),
        n_results=max(1, int(k)),
        where=build_where(collection, project, include_superseded),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[Hit] = []
    for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            Hit(
                score=round(1.0 - float(dist), 4),
                rel_path=str(meta.get("rel_path", "")),
                path=str(meta.get("path", "")),
                heading_path=str(meta.get("heading_path", "")),
                start_line=int(meta.get("start_line", 0)),
                end_line=int(meta.get("end_line", 0)),
                collection=str(meta.get("collection", "")),
                project=str(meta.get("project", "")),
                text=doc or "",
                doc_date=str(meta.get("doc_date", "")),
                superseded=bool(meta.get("superseded", False)),
                id=str(id_),
            )
        )
    return hits


def as_context(hits: list[Hit], query: str) -> str:
    """Markdown block meant to be pasted into an assistant's context (the UI's 'copy as context').

    `heading_path` already begins with the file's rel_path, because ingest passes it to the chunker as the
    title, so it is printed alone. Until 2026-10-20 both were printed and ASOH's injected citations read
    `CLAUDE.md > CLAUDE.md > PART 1 > ...`, which is wasted tokens on every hook-injected prompt.
    """
    if not hits:
        return f"MemDex: no results for: {query}"
    out = [f"## MemDex results for: {query}", ""]
    for n, h in enumerate(hits, 1):
        flag = " (SUPERSEDED)" if h.superseded else ""
        where = f"score {h.score:.3f} · lines {h.start_line}-{h.end_line} · {h.collection}"
        if h.project:
            where += f" · project {h.project}"
        out += [f"### {n}. {h.heading_path}{flag}", where, "", h.text, ""]
    return "\n".join(out)


def as_rules_text(hits: list[Hit], query: str) -> str:
    """Compact text for the Claude Code hook: injected into context on every prompt.

    The header carries the score alone WHEN the body already opens with the heading path, which it does
    for every chunk ingest produces: it passes `rel_path` to the chunker as the title and stores
    `ch.text` verbatim. Printing both repeated the citation twice per hit - LIVE READ 2026-10-20 on
    ASOH's hook, 490 of 3,274 injected bytes, 15.0%, about 122 tokens, paid on every single prompt.

    When the body does NOT carry it, the header prints the full path. Dropping it unconditionally was
    the first attempt and the suite caught it: such a hit would have been injected with no citation at
    all, silently, which is the same shape of defect as the two found on 2026-10-20.
    """
    out = [f"Rules from MemDex that may apply to this prompt (top {len(hits)}):"]
    for h in hits:
        # Print the path only when the body does not already open with it, so the citation is a
        # guarantee of this function rather than an inherited side effect of how ingest titles chunks.
        carried = h.text.lstrip().startswith(h.heading_path)
        head = f"score {h.score:.2f}" if carried else f"{h.heading_path} (score {h.score:.2f})"
        out.append(f"\n--- {head} ---\n{h.text}")
    return "\n".join(out)


def as_table(hits: list[Hit]) -> str:
    if not hits:
        return "no results"
    lines = []
    for n, h in enumerate(hits, 1):
        tail = h.heading_path.split(" > ")[-1]
        flag = " [SUPERSEDED]" if h.superseded else ""
        proj = f" ({h.project})" if h.project else ""
        lines.append(f"{n}. {h.score:.3f}  {h.collection:<11}{h.rel_path}:{h.start_line}-{h.end_line}  | {tail}{proj}{flag}")
    return "\n".join(lines)
