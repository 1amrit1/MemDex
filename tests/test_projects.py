import pytest

from memdex.projects import ProjectError, create_folder, create_project, list_projects, save_upload, validate_filename, validate_folder, validate_name


def test_validate_names():
    assert validate_name("  My Notes 2026 ") == "My Notes 2026"
    for bad in ("", ".hidden", "a/b", "x" * 64, "trailing.", "a..b", "semi;colon"):
        with pytest.raises(ProjectError):
            validate_name(bad)
    assert validate_folder("") == "" and validate_folder("\\notes\\2026\\") == "notes/2026"
    with pytest.raises(ProjectError):
        validate_folder("notes/../etc")


def test_validate_filename_types():
    assert validate_filename("C:\\Users\\me\\Plan.MD") == "Plan.MD"
    assert validate_filename("notes (final).txt") == "notes (final).txt"
    for bad in ("report.pdf", "sheet.xlsx", "doc.docx", ".env", "x.md.", "bad:name.md"):
        with pytest.raises(ProjectError):
            validate_filename(bad)


def test_projects_folders_and_uploads(tmp_cfg):
    _, load = tmp_cfg
    cfg = load()
    assert list_projects(cfg) == []
    path, created = create_project(cfg, "Travel")
    assert created and path == cfg.sources_dir / "Travel"
    assert create_project(cfg, "Travel")[1] is False

    fpath, created = create_folder(cfg, "Travel", "2026/sunshine coast")
    assert created and fpath == path / "2026" / "sunshine coast"
    with pytest.raises(ProjectError):
        create_folder(cfg, "Nope", "x")

    saved = save_upload(cfg, "Travel", "2026/sunshine coast", "plan.md", b"# Plan\n\nferry at 9\n")
    assert saved["rel_path"] == "2026/sunshine coast/plan.md" and saved["replaced"] is False
    assert save_upload(cfg, "Travel", "2026/sunshine coast", "plan.md", b"# Plan v2\n")["replaced"] is True
    save_upload(cfg, "Travel", "", "root.txt", b"plain text")
    with pytest.raises(ProjectError):
        save_upload(cfg, "Travel", "", "empty.md", b"   \n")
    with pytest.raises(ProjectError):
        save_upload(cfg, "Travel", "", "huge.md", b"x" * (10 * 1024 * 1024 + 1))

    [p] = list_projects(cfg)
    assert p["name"] == "Travel" and p["files"] == 2 and p["folders"] == ["2026", "2026/sunshine coast"]

    # the project is now a source, picked up on the next config load
    cfg2 = load()
    [src] = cfg2.project_sources()
    assert src.name == "Travel" and src.kind == "project" and src.types == ("md", "txt") and src.collection_for("any/file.md") == "docs"
