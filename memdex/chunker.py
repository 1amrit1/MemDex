"""Heading-aware Markdown chunking.

A file is split into sections at H1-H3 headings. Each section is packed into chunks of at most
`max_chars`, cutting only between blocks (paragraphs, tables, fenced code), so a table or code
block is never split unless that one block alone is larger than `max_chars`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# A fence line, split into indent + the whole run of backticks/tildes + the info string. Getting any part
# of the CommonMark close rule wrong lets a fence end early, and every line of code after it is then read
# as Markdown - which is how a shell comment becomes an H1 and silently re-roots every citation below it
# (LIVE READ 2026-10-20: with a fixed 3-character marker, 16 chunks of ASOH's CLAUDE.md carried a heading
# path rooted at "# 2. confirm - must print the project path"). `_opens` and `_closes` carry the rules.
FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
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


def _opens(line: str) -> tuple[str, str] | None:
    """(indent, marker) when the line opens a fenced code block, else None."""
    m = FENCE.match(line)
    if not m:
        return None
    indent, marker, info = m.groups()
    if marker[0] == "`" and "`" in info:
        return None  # a backtick fence's info string may not contain a backtick, so this is not a fence
    return indent, marker


def _closes(line: str, indent: str, marker: str) -> bool:
    """CommonMark's three conditions for closing the fence opened by (indent, marker)."""
    m = FENCE.match(line)
    if not m:
        return False
    c_indent, c_marker, c_info = m.groups()
    return (
        c_marker[0] == marker[0]  # same character ...
        and len(c_marker) >= len(marker)  # ... repeated at least as many times ...
        and not c_info.strip()  # ... with nothing after it (a closing fence carries no info string) ...
        and len(c_indent) - len(indent) <= 3  # ... indented no more than 3 further than the opener
    )


def _blocks(lines: list[str]) -> list[tuple[int, str, _Block]]:
    """Group lines into (heading_level, heading_title, block); level 0 means a content block."""
    out: list[tuple[int, str, _Block]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip() or line.strip() == "---":
            i += 1
            continue
        opened = _opens(line)
        if opened:
            indent, marker = opened
            j = i + 1
            while j < n and not _closes(lines[j], indent, marker):
                j += 1
            j = min(j, n - 1)
            # A fence is the one block that may hold blank lines, so end_line is pulled back to the last
            # line with content: the text is unchanged, the reported range just stops pointing at a blank.
            end = j
            while end > i and not lines[end].strip():
                end -= 1
            out.append((0, "", _Block(lines[i : j + 1], i + 1, end + 1)))
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
            if not nxt.strip() or nxt.strip() == "---" or _opens(nxt) or HEADING.match(nxt):
                break
            if nxt.lstrip().startswith("|") != is_table:
                break
            j += 1
        out.append((0, "", _Block(lines[i : j + 1], i + 1, j + 1)))
        i = j + 1
    return out


def _split_oversize(block: _Block, max_chars: int) -> list[_Block]:
    """Last resort for a single block larger than max_chars: cut between lines, then inside a line.

    Only a fenced block reaches here with blank lines in it, and a blank line must never become a piece's
    reported start or end - `lines 6-7` where line 7 is empty is a citation pointing past its own content
    (LIVE READ 2026-10-20: 60 chunks of the live index ended on a blank line this way). So each piece
    reports the first and last line it holds that actually has content.
    """
    pieces: list[_Block] = []
    cur: list[str] = []
    content: list[int] = []  # the lines in `cur` that are not blank
    fallback = block.start  # for a piece that is all blank: the last line with content before it

    def flush() -> None:
        nonlocal cur, content, fallback
        start = content[0] if content else fallback
        end = content[-1] if content else fallback
        pieces.append(_Block(cur, start, end))
        fallback = end
        cur, content = [], []

    for offset, line in enumerate(block.lines):
        lineno = block.start + offset
        for seg in [line[k : k + max_chars] for k in range(0, len(line), max_chars)] or [""]:
            if cur and len("\n".join(cur + [seg])) > max_chars:
                flush()
            cur.append(seg)
            if line.strip():
                content.append(lineno)
    if cur:
        flush()
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
