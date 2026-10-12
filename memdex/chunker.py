"""Heading-aware Markdown chunking.

A file is split into sections at H1-H3 headings. Each section is packed into chunks of at most
`max_chars`, cutting only between blocks (paragraphs, tables, fenced code), so a table or code
block is never split unless that one block alone is larger than `max_chars`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SECTION_LEVEL = 3  # H1-H3 start a new section; H4+ stay inside their parent section
DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
SUPERSEDED = re.compile(r"\bSUPERSEDED\b")


@dataclass(frozen=True)
class Chunk:
    text: str  # heading path + section content — this is what gets embedded and returned
    heading_path: str
    start_line: int  # 1-based, inclusive
    end_line: int


@dataclass
class _Block:
    lines: list[str]
    start: int  # 1-based
    end: int

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def doc_date(text: str) -> str:
    """The first ISO date in the first 5 lines (the byline convention), or ''."""
    for line in text.splitlines()[:5]:
        m = DATE.search(line)
        if m:
            return m.group(1)
    return ""


def is_superseded(text: str) -> bool:
    """True when the header (first 10 lines) carries an upper-case SUPERSEDED marker."""
    return any(SUPERSEDED.search(line) for line in text.splitlines()[:10])


def _blocks(lines: list[str]) -> list[tuple[int, str, _Block]]:
    """Group lines into (heading_level, heading_title, block); level 0 means a content block."""
    out: list[tuple[int, str, _Block]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip() or line.strip() == "---":
            i += 1
            continue
        if FENCE.match(line):
            marker = FENCE.match(line).group(1)
            j = i + 1
            while j < n and not lines[j].lstrip().startswith(marker):
                j += 1
            j = min(j, n - 1)
            out.append((0, "", _Block(lines[i : j + 1], i + 1, j + 1)))
            i = j + 1
            continue
        h = HEADING.match(line)
        if h:
            level = len(h.group(1))
            out.append((level, h.group(2), _Block([line], i + 1, i + 1)))
            i += 1
            continue
        is_table = line.lstrip().startswith("|")
        j = i
        while j + 1 < n:
            nxt = lines[j + 1]
            if not nxt.strip() or nxt.strip() == "---" or FENCE.match(nxt) or HEADING.match(nxt):
                break
            if nxt.lstrip().startswith("|") != is_table:
                break
            j += 1
        out.append((0, "", _Block(lines[i : j + 1], i + 1, j + 1)))
        i = j + 1
    return out


def _split_oversize(block: _Block, max_chars: int) -> list[_Block]:
    """Last resort for a single block larger than max_chars: cut between lines, then inside a line."""
    pieces: list[_Block] = []
    cur: list[str] = []
    cur_start = last = block.start
    for offset, line in enumerate(block.lines):
        lineno = block.start + offset
        for seg in [line[k : k + max_chars] for k in range(0, len(line), max_chars)] or [""]:
            if cur and len("\n".join(cur + [seg])) > max_chars:
                pieces.append(_Block(cur, cur_start, last))
                cur = []
            if not cur:
                cur_start = lineno
            cur.append(seg)
            last = lineno
    if cur:
        pieces.append(_Block(cur, cur_start, last))
    return pieces


def chunk_markdown(text: str, title: str, max_chars: int = 1000) -> list[Chunk]:
    lines = text.splitlines()
    stack: list[str] = []
    sections: list[tuple[str, list[_Block]]] = []
    current: list[_Block] = []
    current_path = title

    for level, heading, block in _blocks(lines):
        if 1 <= level <= SECTION_LEVEL:
            if current:
                sections.append((current_path, current))
            stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [heading]
            current_path = " > ".join([title] + [s for s in stack if s])
            current = []
            continue
        current.append(block)
    if current:
        sections.append((current_path, current))

    chunks: list[Chunk] = []
    for heading_path, blocks in sections:
        budget = max(200, max_chars - len(heading_path) - 2)
        parts: list[_Block] = []
        for b in blocks:
            parts.extend(_split_oversize(b, budget) if len(b.text) > budget else [b])

        group: list[_Block] = []
        size = 0
        for b in parts:
            added = len(b.text) + (2 if group else 0)
            if group and size + added > budget:
                chunks.append(_make_chunk(heading_path, group))
                group, size = [], 0
                added = len(b.text)
            group.append(b)
            size += added
        if group:
            chunks.append(_make_chunk(heading_path, group))
    return chunks


def _make_chunk(heading_path: str, group: list[_Block]) -> Chunk:
    body = "\n\n".join(b.text for b in group)
    return Chunk(
        text=f"{heading_path}\n\n{body}",
        heading_path=heading_path,
        start_line=group[0].start,
        end_line=group[-1].end,
    )
