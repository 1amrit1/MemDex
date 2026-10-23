"""`outline` and `show`: fetching a NAMED region instead of the k nearest chunks.

The case these exist for: retrieval put all five hits inside the right file and the assistant still had to
read the file, because top-k cannot enumerate a document. So the tests that matter here are the ones about
addressing a region by name and reading it from DISK - not about scores.
"""

from __future__ import annotations

import hashlib

import pytest

from memdex import store
from memdex.regions import (
    RegionError,
    as_outline_text,
    list_files,
    outline,
    resolve_file,
    show,
)

# A real file on disk, because `show` reads the source rather than the stored chunk text.
SOURCE = """# Guide

Intro line.

## Alpha

Alpha one.

Alpha two after a blank line.

## Beta

Beta body.
"""


def _seed(tmp_path, emb):
    """Two files. `guide.md` exists on disk; its stored chunks deliberately differ from the source, the
    way the real chunker's output does (it drops blank lines and the heading lines themselves)."""
    src = tmp_path / "guide.md"
    src.write_text(SOURCE, encoding="utf-8")
    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()

    client = store.open_client(tmp_path / ".index")
    coll = store.reset_collection(client)
    rows = [
        # id, text, meta
        ("g0", "guide.md > Guide\nIntro line.",
         dict(rel_path="guide.md", path=str(src), heading_path="guide.md > Guide",
              start_line=3, end_line=3, chunk=0, content_hash=digest)),
        ("g1", "guide.md > Guide > Alpha\nAlpha one.",
         dict(rel_path="guide.md", path=str(src), heading_path="guide.md > Guide > Alpha",
              start_line=7, end_line=7, chunk=1, content_hash=digest)),
        ("g2", "guide.md > Guide > Alpha\nAlpha two after a blank line.",
         dict(rel_path="guide.md", path=str(src), heading_path="guide.md > Guide > Alpha",
              start_line=9, end_line=9, chunk=2, content_hash=digest)),
        ("g3", "guide.md > Guide > Beta\nBeta body.",
         dict(rel_path="guide.md", path=str(src), heading_path="guide.md > Guide > Beta",
              start_line=13, end_line=13, chunk=3, content_hash=digest)),
        ("o0", "other.md > Other\nsomething else",
         dict(rel_path="notes/other.md", path=str(tmp_path / "other.md"),
              heading_path="notes/other.md > Other", start_line=1, end_line=1, chunk=0, content_hash="deadbeef")),
    ]
    coll.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        metadatas=[{**r[2], "collection": "docs", "project": "P", "superseded": False} for r in rows],
        embeddings=emb.embed([r[1] for r in rows]),
    )
    return coll, src


def test_outline_merges_consecutive_chunks_of_one_section(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    secs = outline(coll, "guide.md")
    assert [s.title for s in secs] == ["Guide", "Alpha", "Beta"]
    alpha = secs[1]
    assert alpha.chunks == 2  # g1 + g2 merged
    assert (alpha.start_line, alpha.end_line) == (7, 9)  # span covers both
    assert alpha.chars == len("guide.md > Guide > Alpha\nAlpha one.") + len("guide.md > Guide > Alpha\nAlpha two after a blank line.")


def test_outline_text_names_the_next_command(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    text = as_outline_text("guide.md", outline(coll, "guide.md"))
    assert "3 sections, 4 chunks" in text
    assert "memdex show guide.md --heading" in text  # discoverability is the point of outline


def test_resolve_file_is_exact_unique_or_an_error(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    assert resolve_file(coll, "guide.md") == "guide.md"
    assert resolve_file(coll, "other") == "notes/other.md"  # unique substring
    assert sorted(list_files(coll)) == ["guide.md", "notes/other.md"]
    with pytest.raises(RegionError, match="no indexed file"):
        resolve_file(coll, "nope.md")
    with pytest.raises(RegionError, match="name a file"):
        resolve_file(coll, "   ")
    with pytest.raises(RegionError) as exc:  # ".md" matches both -> must list, never guess
        resolve_file(coll, ".md")
    assert "guide.md" in str(exc.value) and "notes/other.md" in str(exc.value)


def test_show_reads_the_file_not_the_stored_chunk(tmp_path, fake_embedder):
    """The whole reason `show` exists: stored text is not verbatim source."""
    coll, _ = _seed(tmp_path, fake_embedder)
    region = show(coll, "guide.md", heading="Alpha")
    assert region.start_line == 7 and region.end_line == 9
    # The blank line between the two Alpha paragraphs is in the FILE but in no stored chunk.
    assert region.text == "Alpha one.\n\nAlpha two after a blank line."
    assert "guide.md > Guide > Alpha" not in region.text  # the chunker's title prefix is not in the source
    assert region.stale is False


def test_show_by_lines_chunk_and_whole_file(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    assert show(coll, "guide.md", lines=(5, 7)).text == "## Alpha\n\nAlpha one."
    assert show(coll, "guide.md", chunk=3).text == "Beta body."
    whole = show(coll, "guide.md")
    assert whole.text.startswith("# Guide") and whole.text.rstrip().endswith("Beta body.")
    assert whole.end_line == len(SOURCE.splitlines())


def test_show_truncates_to_the_budget_and_flags_it(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    region = show(coll, "guide.md", max_chars=12)
    assert region.truncated is True and region.chars == 12
    assert show(coll, "guide.md", max_chars=0).truncated is False  # 0 = no limit


def test_show_flags_a_file_that_moved_on_since_ingest(tmp_path, fake_embedder):
    coll, src = _seed(tmp_path, fake_embedder)
    assert show(coll, "guide.md", heading="Beta").stale is False
    src.write_text(SOURCE + "\n## Gamma\n\nAdded later.\n", encoding="utf-8")
    assert show(coll, "guide.md", heading="Beta").stale is True  # content_hash no longer matches


def test_show_rejects_what_it_cannot_address(tmp_path, fake_embedder):
    coll, _ = _seed(tmp_path, fake_embedder)
    with pytest.raises(RegionError, match="no section"):
        show(coll, "guide.md", heading="Delta")
    with pytest.raises(RegionError, match="no chunk 99"):
        show(coll, "guide.md", chunk=99)
    with pytest.raises(RegionError, match="not in the index"):
        show(coll, "missing.md")
