"""Fetch a NAMED region of a document, rather than the k nearest chunks.

Why this exists, from a live failure: ASOH typed "list doc creation rules" into a fresh session. Retrieval
was perfect - all five hits were inside `Claude_Rules/Doc_Creation_Rules.md`, scoring 0.58-0.62 - and the
session still read the file from disk, correctly, because 5 chunks x 700 chars cannot enumerate an 11,700
byte document. Top-k answers "where is this?"; it can never answer "give me all of X". That is this module.

Two commands sit on top of it:
  outline  - the sections of a file, with line ranges and a ready-to-run `show` for each
  show     - one named region, read from the FILE ON DISK

`show` reads the file, not the stored chunk text, and the distinction matters: the chunker drops blank
lines, `---` rules and the heading lines themselves, so stored text is NOT a verbatim copy of the source.
Anything that must be quoted back has to come from disk. The price is that the file may have moved on since
the index was built, so every read compares `content_hash` and says so when they differ.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Section:
    heading_path: str
    start_line: int
    end_line: int
    chunks: int
    chars: int
    first_chunk: int

    @property
    def title(self) -> str:
        """The last segment - the heading itself, without the file and its ancestors."""
        return self.heading_path.split(" > ")[-1]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Region:
    rel_path: str
    path: str
    heading_path: str
    start_line: int
    end_line: int
    text: str
    chars: int
    truncated: bool
    stale: bool

    def as_dict(self) -> dict:
        return asdict(self)


class RegionError(Exception):
    """Raised with a message meant to be shown to the user as-is."""


def _rows(chroma_collection, rel_path: str | None = None) -> list[dict]:
    """Metadata rows for one file (or all), ordered as they appear in the document."""
    where = {"rel_path": rel_path} if rel_path else None
    got = chroma_collection.get(where=where, include=["documents", "metadatas"])
    metas = got.get("metadatas") or []
    docs = got.get("documents") or []
    rows = []
    for meta, doc in zip(metas, docs):
        row = dict(meta)
        row["_text"] = doc or ""
        rows.append(row)
    rows.sort(key=lambda r: int(r.get("chunk", 0)))
    return rows


def list_files(chroma_collection, needle: str = "") -> list[str]:
    """Distinct `rel_path`s, optionally filtered by a case-insensitive substring."""
    metas = chroma_collection.get(include=["metadatas"])["metadatas"] or []
    seen = {str(m.get("rel_path", "")) for m in metas}
    n = (needle or "").strip().lower()
    out = sorted(p for p in seen if p and (not n or n in p.lower()))
    return out


def resolve_file(chroma_collection, needle: str) -> str:
    """One file, or an error naming the candidates - never a silent guess at which one was meant."""
    n = (needle or "").strip()
    if not n:
        raise RegionError("name a file, e.g.: memdex outline CLAUDE.md")
    matches = list_files(chroma_collection, n)
    if not matches:
        raise RegionError(f"no indexed file matches {n!r} - try: memdex outline --list")
    if len(matches) > 1:
        exact = [m for m in matches if m.lower() == n.lower() or Path(m).name.lower() == n.lower()]
        if len(exact) == 1:
            return exact[0]
        joined = "\n  ".join(matches[:12])
        more = f"\n  ... and {len(matches) - 12} more" if len(matches) > 12 else ""
        raise RegionError(f"{len(matches)} files match {n!r}:\n  {joined}{more}")
    return matches[0]


def outline(chroma_collection, rel_path: str) -> list[Section]:
    """The file's sections in document order, merged from its chunks.

    Consecutive chunks sharing a heading path are one section, which is the unit a reader actually wants:
    the corpus averages 2.57 chunks per section, so a chunk list would be nearly three times as long and
    far less useful for deciding where to look.
    """
    sections: list[Section] = []
    for row in _rows(chroma_collection, rel_path):
        hp = str(row.get("heading_path", ""))
        start, end = int(row.get("start_line", 0)), int(row.get("end_line", 0))
        chars, ordinal = len(row["_text"]), int(row.get("chunk", 0))
        if sections and sections[-1].heading_path == hp:
            prev = sections[-1]
            sections[-1] = Section(
                heading_path=hp,
                start_line=min(prev.start_line, start),
                end_line=max(prev.end_line, end),
                chunks=prev.chunks + 1,
                chars=prev.chars + chars,
                first_chunk=prev.first_chunk,
            )
        else:
            sections.append(Section(hp, start, end, 1, chars, ordinal))
    return sections


def _read_lines(path: Path, start: int, end: int) -> tuple[str, str]:
    """Returns (region text, sha256 of the WHOLE file) - the hash is what detects drift since ingest."""
    try:
        whole = io.open(path, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        raise RegionError(f"cannot read {path}: {exc}") from None
    lines = whole.splitlines()
    lo = max(1, int(start))
    hi = min(len(lines), int(end) if end else len(lines))
    if lo > len(lines):
        raise RegionError(f"{path.name} has {len(lines)} lines; line {lo} is past the end")
    return "\n".join(lines[lo - 1 : hi]), hashlib.sha256(whole.encode("utf-8")).hexdigest()


def show(
    chroma_collection,
    rel_path: str,
    heading: str | None = None,
    lines: tuple[int, int] | None = None,
    chunk: int | None = None,
    max_chars: int = 4000,
) -> Region:
    """One region of one file, read from disk. Exactly one selector, or the whole file when none is given."""
    rows = _rows(chroma_collection, rel_path)
    if not rows:
        raise RegionError(f"{rel_path} is not in the index")
    path = Path(str(rows[0].get("path", "")))
    stored_hash = str(rows[0].get("content_hash", ""))

    if chunk is not None:
        picked = [r for r in rows if int(r.get("chunk", -1)) == int(chunk)]
        if not picked:
            raise RegionError(f"{rel_path} has no chunk {chunk} (it has {len(rows)})")
        row = picked[0]
        hp, lo, hi = str(row.get("heading_path", "")), int(row["start_line"]), int(row["end_line"])
    elif heading:
        needle = heading.strip().lower()
        secs = [s for s in outline(chroma_collection, rel_path) if needle in s.heading_path.lower()]
        if not secs:
            raise RegionError(f"no section of {rel_path} matches {heading!r} - run: memdex outline {rel_path}")
        hp, lo, hi = secs[0].heading_path, secs[0].start_line, secs[0].end_line
    elif lines:
        hp, lo, hi = f"{rel_path} > lines {lines[0]}-{lines[1]}", int(lines[0]), int(lines[1])
    else:
        hp, lo, hi = rel_path, 1, 0  # 0 = to the end of the file

    text, live_hash = _read_lines(path, lo, hi)
    truncated = len(text) > max_chars > 0
    if truncated:
        text = text[:max_chars]
    return Region(
        rel_path=rel_path,
        path=str(path),
        heading_path=hp,
        start_line=lo,
        end_line=hi if hi else len(io.open(path, encoding="utf-8", errors="replace").read().splitlines()),
        text=text,
        chars=len(text),
        truncated=truncated,
        stale=bool(stored_hash) and stored_hash != live_hash,
    )


def as_outline_text(rel_path: str, sections: list[Section]) -> str:
    """Printed so the next command is visible rather than guessed at - that is what makes digging findable."""
    if not sections:
        return f"{rel_path}: no sections indexed"
    total_chunks = sum(s.chunks for s in sections)
    total_chars = sum(s.chars for s in sections)
    out = [f"{rel_path} - {len(sections)} sections, {total_chunks} chunks, {total_chars:,} chars", ""]
    for s in sections:
        out.append(f"  lines {s.start_line:>5}-{s.end_line:<5} {s.chunks:>2} chunk(s) {s.chars:>6} chars  {s.title}")
    # Suggest the meatiest section, not the first: the first is usually the title line, which makes the
    # example look useless at exactly the moment the reader is deciding whether the command is worth using.
    biggest = max(sections, key=lambda x: x.chars)
    out += ["", "fetch one with:", f'  memdex show {rel_path} --heading "{biggest.title}"']
    return "\n".join(out)


def as_region_text(region: Region) -> str:
    head = f"--- {region.heading_path}  (lines {region.start_line}-{region.end_line}, {region.chars:,} chars) ---"
    body = [head, region.text]
    if region.truncated:
        body.append(
            f"\n[truncated at {region.chars:,} chars - continue with: "
            f'memdex show {region.rel_path} --lines {region.start_line}-{region.end_line} --max-chars 0]'
        )
    return "\n".join(body)
