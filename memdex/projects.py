"""Projects = folders under data_dir/sources, created and filled through the web UI.

The UI never edits anything outside data_dir/sources. Sub-folders are plain folders; nesting is allowed.
Only md/txt are accepted until Milestone 7 adds PDF/Word/Excel extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

from memdex.config import PROJECT_TYPES, Config

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,62}$")
FILENAME = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,150}$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class ProjectError(ValueError):
    pass


def validate_name(name: str, what: str = "project") -> str:
    name = (name or "").strip()
    if not NAME.match(name) or name.endswith(".") or ".." in name:
        raise ProjectError(f"{what} name must be 1-63 characters: letters, digits, space, _ . - (no leading dot)")
    return name


def validate_folder(folder: str) -> str:
    """'' = project root; 'a/b' = nested. Each part is validated like a project name."""
    folder = (folder or "").strip().replace("\\", "/").strip("/")
    if not folder:
        return ""
    return "/".join(validate_name(part, "folder") for part in folder.split("/"))


def validate_filename(filename: str) -> str:
    name = Path((filename or "").replace("\\", "/")).name.strip()
    if not FILENAME.match(name) or name.startswith(".") or name.endswith("."):
        raise ProjectError(f"invalid file name: {filename!r}")
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in PROJECT_TYPES:
        accepted = ", ".join("." + t for t in PROJECT_TYPES)
        raise ProjectError(f"only {accepted} files are accepted today ({name}); PDF/Word/Excel come in Milestone 7")
    return name


def project_dir(cfg: Config, name: str) -> Path:
    return cfg.sources_dir / validate_name(name)


def list_projects(cfg: Config) -> list[dict]:
    root = cfg.sources_dir
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        folders = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_dir())
        files = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in PROJECT_TYPES)
        out.append({"name": d.name, "path": str(d), "folders": folders, "files": files})
    return out


def create_project(cfg: Config, name: str) -> tuple[Path, bool]:
    path = project_dir(cfg, name)
    created = not path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return path, created


def create_folder(cfg: Config, project: str, folder: str) -> tuple[Path, bool]:
    base = project_dir(cfg, project)
    if not base.is_dir():
        raise ProjectError(f"project not found: {project}")
    rel = validate_folder(folder)
    if not rel:
        raise ProjectError("folder name is empty")
    path = base / rel
    created = not path.exists()
    path.mkdir(parents=True, exist_ok=True)
    return path, created


def save_upload(cfg: Config, project: str, folder: str, filename: str, data: bytes) -> dict:
    base = project_dir(cfg, project)
    if not base.is_dir():
        raise ProjectError(f"project not found: {project}")
    rel_folder = validate_folder(folder)
    name = validate_filename(filename)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ProjectError(f"{name} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data.strip():
        raise ProjectError(f"{name} is empty")
    target = (base / rel_folder / name) if rel_folder else (base / name)
    target.parent.mkdir(parents=True, exist_ok=True)
    replaced = target.exists()
    target.write_bytes(data)
    return {
        "project": base.name,
        "folder": rel_folder,
        "name": name,
        "rel_path": target.relative_to(base).as_posix(),
        "path": str(target),
        "bytes": len(data),
        "replaced": replaced,
    }
