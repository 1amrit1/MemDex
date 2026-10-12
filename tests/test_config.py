from pathlib import Path

import pytest

from memdex.config import ConfigError, glob_match, load_config


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("CLAUDE.md", "CLAUDE.md", True),
        ("sub/CLAUDE.md", "CLAUDE.md", False),
        ("Claude_Rules/Planning_Rules.md", "Claude_Rules/**", True),
        ("STATUS.MD", "status*.md", True),
        ("Personal/Career/WEEKLY_TASKS/ARCHIVED/Week_01.md", "**/ARCHIVED/**", True),
        ("ARCHIVED/x.md", "**/ARCHIVED/**", True),
        ("PROJECTS/main_RAG.md", "PROJECTS/*", True),
        ("PROJECTS/a/b.md", "PROJECTS/*", False),
        (".claude/settings.md", ".claude/**", True),
    ],
)
def test_glob_match(path, pattern, expected):
    assert glob_match(path, pattern) is expected


def _write(tmp: Path, body: str) -> Path:
    (tmp / "docs").mkdir(exist_ok=True)
    cfg = tmp / "memdex.yaml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_load_and_route(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
index_dir: .index
sources:
  - path: docs
    types: [md]
    exclude: ["_BACKUPS/**"]
    collections:
      - { match: "CLAUDE.md", collection: rules }
      - { match: "**/ARCHIVED/**", collection: archive }
""",
        )
    )
    src = cfg.sources[0]
    assert cfg.index_dir == (tmp_path / ".index").resolve()
    assert src.wants("notes/a.MD")
    assert not src.wants("notes/a.txt")
    assert not src.wants("_BACKUPS/2026/STATUS.MD")
    assert src.collection_for("CLAUDE.md") == "rules"
    assert src.collection_for("x/ARCHIVED/y.md") == "archive"
    assert src.collection_for("x/y.md") == "references"


def test_missing_source_folder(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "sources:\n  - path: nope\n"))
