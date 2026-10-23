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


NESTED_FENCE = """# Doc

## 2.4b Comments inside the block

A four-backtick block that wraps three-backtick blocks:

````
```powershell
# 1. move to the project directory
Set-Location C:\\project

# 2. confirm - must print the project path
Get-Location
```
````

## 3.1 Next real section

Body of 3.1.
"""


def test_nested_fence_does_not_close_early():
    """A ```` fence closes only on 4+ backticks, so comments inside it never become headings."""
    chunks = chunk_markdown(NESTED_FENCE, "doc.md", max_chars=2000)
    paths = [c.heading_path for c in chunks]
    assert paths == ["doc.md > Doc > 2.4b Comments inside the block", "doc.md > Doc > 3.1 Next real section"]
    assert not any("confirm" in p or "move to the" in p for p in paths)
    block = chunks[0].text
    assert "# 1. move to the project directory" in block and "Get-Location" in block


INFO_STRING_CLOSER = """# Doc

## S1

```
code
```python
# not a heading - a closing fence carries no info string, so the block is still open
```

## S2

Body of S2.
"""


def test_closing_fence_may_not_carry_an_info_string():
    """```python inside an open ``` block is content, not a close (CommonMark).

    Before this was enforced the fence ended on that line, the comment below it became an H1, and S2 -
    a real heading - was swallowed by the fence the chunker then opened. Same silent re-rooting as the
    four-backtick bug, reached through a different wrong close.
    """
    paths = [c.heading_path for c in chunk_markdown(INFO_STRING_CLOSER, "doc.md", max_chars=2000)]
    assert paths == ["doc.md > Doc > S1", "doc.md > Doc > S2"]
    assert not any("not a heading" in p for p in paths)


INDENTED_CLOSER = """# Doc

## S1

```
code
    ```
# not a heading - that closer is indented 4 further than the opener
```

## S2

Body of S2.
"""


def test_closing_fence_indented_four_further_does_not_close():
    """A closing fence may be indented at most 3 further than its opener, so the 4-space one is content."""
    paths = [c.heading_path for c in chunk_markdown(INDENTED_CLOSER, "doc.md", max_chars=2000)]
    assert paths == ["doc.md > Doc > S1", "doc.md > Doc > S2"]
    assert not any("not a heading" in p for p in paths)


def test_indented_fence_in_a_list_item_still_pairs():
    """Control for the rule above: a list-item fence is itself indented, and its opener and closer must
    still match - ASOH's 7D_AI_STEPS.MD has 8 of them, and if they stop pairing the shell comments
    inside them become headings."""
    doc = "# Doc\n\n## S1\n\n- item\n\n  ```\n  # not a heading\n  ```\n\n## S2\n\nBody.\n"
    paths = [c.heading_path for c in chunk_markdown(doc, "doc.md", max_chars=2000)]
    assert paths == ["doc.md > Doc > S1", "doc.md > Doc > S2"]


def test_backtick_in_info_string_is_not_a_fence():
    """A backtick fence's info string may not contain a backtick, so that line opens nothing and the
    heading under it is real. Before the fix it opened a fence that ran to end of file and swallowed it.
    The expected path has no `Doc` in it because an H1 resets the heading stack to its own level."""
    doc = "# Doc\n\n## S1\n\n```a`b\n# Real heading\n\nBody under the real heading.\n"
    paths = [c.heading_path for c in chunk_markdown(doc, "doc.md", max_chars=2000)]
    assert "doc.md > Real heading" in paths


def test_line_numbers_never_point_at_a_blank_line_inside_a_split_fence():
    """A fence is the one block that holds blank lines, and splitting it must not report one.

    LIVE READ 2026-10-20: 60 chunks of the live index ended on a blank line - `lines 6-7` where line 7
    was empty - because the splitter advanced its end marker over blank lines. The text was right, so
    nothing failed; the citation just pointed one line past its own content.
    """
    doc = "# Doc\n\n```\n" + "\n\n".join(f"content line {i} " + "x" * 200 for i in range(6)) + "\n```\n"
    src = doc.splitlines()
    chunks = chunk_markdown(doc, "doc.md", max_chars=400)
    assert len(chunks) > 2, "the oversize fence must actually have been split"
    for c in chunks:
        assert src[c.start_line - 1].strip(), f"start_line {c.start_line} points at a blank line"
        assert src[c.end_line - 1].strip(), f"end_line {c.end_line} points at a blank line"
        body = c.text.split("\n\n", 1)[1]
        first = next(x.strip() for x in body.split("\n") if x.strip())
        last = next(x.strip() for x in reversed(body.split("\n")) if x.strip())
        assert first in src[c.start_line - 1]
        assert last in src[c.end_line - 1]


def test_byline_date_and_superseded():
    assert doc_date(DOC) == "2026-10-12"
    assert not is_superseded(DOC)
    assert is_superseded("# X\n\n> **SUPERSEDED 2026-10-12** - see Y")
    assert not is_superseded("# X\n\n**Supersedes:** old file")
