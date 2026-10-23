"""Load settings and sources.

Three inputs, merged into one `Config`:
- `memdex.yaml`  - configured source folders + collection routing (power users; optional)
- `.env` / env   - scalar settings (index/model/data folders, chunk size)
- `data_dir/sources/<project>/` - projects created through the web UI (uploads); discovered automatically
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG = "memdex.yaml"
FALLBACK_COLLECTION = "references"
PROJECT_COLLECTION = "docs"  # collection for files uploaded through the UI
PROJECT_TYPES = ("md", "txt")  # what the UI accepts today; PDF/Word/Excel extraction is Milestone 7
# all-MiniLM-L6-v2 reads 256 tokens; on ASOH's Markdown corpus 700 chars truncated 0.4% of chunks vs 14% at 1000
DEFAULT_MAX_CHARS = 700


class ConfigError(ValueError):
    pass


@lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob to a regex. `**/` = any number of folders, `**` = anything, `*` = within one folder."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z", re.IGNORECASE)


def glob_match(rel_path: str, pattern: str) -> bool:
    """Case-insensitive match of a forward-slash relative path against a glob."""
    return _glob_regex(pattern.replace("\\", "/")).match(rel_path) is not None


@dataclass(frozen=True)
class CollectionRule:
    match: str
    collection: str


@dataclass(frozen=True)
class Source:
    path: Path
    name: str  # "project" name shown in the UI and stored on every chunk
    types: tuple[str, ...]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    collections: tuple[CollectionRule, ...]
    default_collection: str = FALLBACK_COLLECTION
    kind: str = "config"  # "config" = from memdex.yaml · "project" = created in the UI under data_dir/sources

    def wants(self, rel_path: str) -> bool:
        if Path(rel_path).suffix.lower().lstrip(".") not in self.types:
            return False
        if not any(glob_match(rel_path, p) for p in self.include):
            return False
        return not any(glob_match(rel_path, p) for p in self.exclude)

    def collection_for(self, rel_path: str) -> str:
        for rule in self.collections:
            if glob_match(rel_path, rule.match):
                return rule.collection
        return self.default_collection


@dataclass(frozen=True)
class Config:
    path: Path | None  # the YAML file, or None when running on defaults + UI projects only
    base: Path  # folder relative paths resolve against
    index_dir: Path
    model_dir: Path
    data_dir: Path
    max_chars: int
    sources: tuple[Source, ...]

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "queries.jsonl"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def last_ingest_path(self) -> Path:
        return self.index_dir / "last_ingest.json"

    def project_sources(self) -> tuple[Source, ...]:
        return tuple(s for s in self.sources if s.kind == "project")


def project_sources(sources_dir: Path) -> tuple[Source, ...]:
    """Every folder directly under data_dir/sources is a project (md + txt, everything included)."""
    if not sources_dir.is_dir():
        return ()
    return tuple(
        Source(
            path=d.resolve(),
            name=d.name,
            types=PROJECT_TYPES,
            include=("**",),
            exclude=(),
            collections=(),
            default_collection=PROJECT_COLLECTION,
            kind="project",
        )
        for d in sorted(sources_dir.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    )


def _resolve(base: Path, value: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (base / p).resolve()


def _str_list(raw: object, field: str, default: list[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(default)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ConfigError(f"'{field}' must be a list of strings")
    return tuple(raw)


def load_config(path: str | Path | None = None) -> Config:
    """Precedence for scalar settings: process env > .env (loaded by the CLI) > memdex.yaml > defaults.

    A config file named explicitly (argument or MEMDEX_CONFIG) must exist. With nothing named and no
    ./memdex.yaml, MemDex runs on defaults with UI-created projects as its only sources.
    """
    explicit = path or os.environ.get("MEMDEX_CONFIG")
    cfg_path = Path(explicit or DEFAULT_CONFIG).resolve()
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg_file: Path | None = cfg_path
        base = cfg_path.parent
    elif explicit:
        raise ConfigError(f"config not found: {cfg_path} (copy memdex.example.yaml to memdex.yaml)")
    else:
        data, cfg_file, base = {}, None, Path.cwd().resolve()

    raw_sources = data.get("sources") or []
    if not isinstance(raw_sources, list):
        raise ConfigError("'sources' must be a list")

    sources = []
    for n, s in enumerate(raw_sources, 1):
        if not isinstance(s, dict) or "path" not in s:
            raise ConfigError(f"source #{n} has no 'path'")
        src_path = _resolve(base, s["path"])
        if not src_path.is_dir():
            raise ConfigError(f"source #{n} path is not a folder: {src_path}")
        rules = tuple(
            CollectionRule(match=r["match"], collection=r["collection"]) for r in (s.get("collections") or [])
        )
        sources.append(
            Source(
                path=src_path,
                name=str(s.get("name") or src_path.name),
                types=tuple(t.lower().lstrip(".") for t in _str_list(s.get("types"), "types", ["md"])),
                include=_str_list(s.get("include"), "include", ["**"]),
                exclude=_str_list(s.get("exclude"), "exclude", []),
                collections=rules,
            )
        )

    max_chars = _setting("MEMDEX_MAX_CHARS", data, "max_chars", DEFAULT_MAX_CHARS)
    try:
        max_chars = int(max_chars)
    except ValueError:
        raise ConfigError(f"max_chars must be a whole number, got {max_chars!r}") from None
    if max_chars < 200:
        raise ConfigError(f"max_chars must be at least 200, got {max_chars}")

    data_dir = _resolve(base, str(_setting("MEMDEX_DATA_DIR", data, "data_dir", "data")))
    return Config(
        path=cfg_file,
        base=base,
        index_dir=_resolve(base, str(_setting("MEMDEX_INDEX_DIR", data, "index_dir", ".index"))),
        model_dir=_resolve(base, str(_setting("MEMDEX_MODEL_DIR", data, "model_dir", "models"))),
        data_dir=data_dir,
        max_chars=max_chars,
        sources=tuple(sources) + project_sources(data_dir / "sources"),
    )


def _setting(env_key: str, data: dict, yaml_key: str, default: object) -> object:
    """Scalar settings: environment / .env first, then memdex.yaml, then the built-in default."""
    value = os.environ.get(env_key)
    if value not in (None, ""):
        return value
    return data.get(yaml_key, default)
