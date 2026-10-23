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
        ("PROJECTS/MemDex_RAG.md", "PROJECTS/*", True),
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
    name: CLAUDE
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
    assert cfg.data_dir == (tmp_path / "data").resolve() and cfg.log_path == cfg.data_dir / "logs" / "queries.jsonl"
    assert src.name == "CLAUDE" and src.kind == "config"
    assert src.wants("notes/a.MD")
    assert not src.wants("notes/a.txt")
    assert not src.wants("_BACKUPS/2026/STATUS.MD")
    assert src.collection_for("CLAUDE.md") == "rules"
    assert src.collection_for("x/ARCHIVED/y.md") == "archive"
    assert src.collection_for("x/y.md") == "references"


def test_source_name_defaults_to_folder_and_projects_are_appended(tmp_path):
    (tmp_path / "data" / "sources" / "Travel").mkdir(parents=True)
    (tmp_path / "data" / "sources" / ".hidden").mkdir()
    cfg = load_config(_write(tmp_path, "sources:\n  - path: docs\n"))
    assert [s.name for s in cfg.sources] == ["docs", "Travel"]
    assert cfg.sources[1].kind == "project" and cfg.sources[1].default_collection == "docs"


def test_missing_source_folder(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "sources:\n  - path: nope\n"))


def test_explicit_missing_config_errors_but_default_missing_runs_on_defaults(tmp_path, monkeypatch):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
    monkeypatch.setenv("MEMDEX_CONFIG", str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError):
        load_config()
    monkeypatch.delenv("MEMDEX_CONFIG")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.path is None and cfg.sources == () and cfg.index_dir == (tmp_path / ".index").resolve() and cfg.max_chars == 700
