from memdex.chunker import chunk_markdown, doc_date, is_superseded

DOC = """# Title

2026-10-12 · Someone

Intro paragraph.

---

## Part A

Para A1.

| col | col |
|---|---|
| a | b |

### Sub A.1

```powershell
# a comment that looks like a heading
Get-Location
```

#### Deep heading stays inside Sub A.1

Deep text.

## Part B

Para B.
"""


def test_sections_and_heading_paths():
    paths = [c.heading_path for c in chunk_markdown(DOC, "doc.md", max_chars=2000)]
    assert paths == [
        "doc.md > Title",
        "doc.md > Title > Part A",
        "doc.md > Title > Part A > Sub A.1",
        "doc.md > Title > Part B",
    ]


def test_fence_comment_is_not_a_heading_and_h4_stays_in_section():
    sub = [c for c in chunk_markdown(DOC, "doc.md", 2000) if c.heading_path.endswith("Sub A.1")][0]
    assert "# a comment that looks like a heading" in sub.text
    assert "#### Deep heading stays inside Sub A.1" in sub.text


def test_table_kept_whole_and_rules_dropped():
    part_a = [c for c in chunk_markdown(DOC, "doc.md", 2000) if c.heading_path.endswith("Part A")][0]
    assert "| col | col |\n|---|---|\n| a | b |" in part_a.text
    assert "\n---\n" not in part_a.text


def test_line_numbers_point_at_source():
    lines = DOC.splitlines()
    for c in chunk_markdown(DOC, "doc.md", 2000):
        assert lines[c.start_line - 1].strip() and lines[c.end_line - 1].strip()


def test_max_chars_respected():
    big = "## H\n\n" + "\n\n".join(f"paragraph {i} " + "x" * 300 for i in range(20)) + "\n\n" + "y" * 5000
    chunks = chunk_markdown(big, "big.md", max_chars=1000)
    assert len(chunks) > 5
    assert all(len(c.text) <= 1000 for c in chunks)


def test_byline_date_and_superseded():
    assert doc_date(DOC) == "2026-10-12"
    assert not is_superseded(DOC)
    assert is_superseded("# X\n\n> **SUPERSEDED 2026-10-12** - see Y")
    assert not is_superseded("# X\n\n**Supersedes:** old file")
