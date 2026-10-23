"""Minimal .env loader (no extra dependency).

KEY=VALUE per line; blank lines and `#` comments ignored; optional surrounding quotes stripped.
Variables already set in the process environment win over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV = ".env"


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Read the .env file (default: ./.env) into os.environ without overriding existing variables."""
    env_path = Path(path or DEFAULT_ENV)
    if not env_path.is_file():
        return {}
    values = parse_env(env_path.read_text(encoding="utf-8-sig"))
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


# The scalar settings that memdex.yaml also offers; an env / .env value for one of these silently beats the YAML.
SETTING_KEYS = ("MEMDEX_INDEX_DIR", "MEMDEX_MODEL_DIR", "MEMDEX_DATA_DIR", "MEMDEX_MAX_CHARS")


def env_overrides(cwd: str | Path, config_base: str | Path | None, environ: dict[str, str] | None = None) -> dict:
    """Which settings currently come from the environment / a ./.env instead of memdex.yaml.

    ASOH's live test: a .env in the working folder beat the --config he named, so MEMDEX_MODEL_DIR resolved
    against the wrong folder. This only reports; the documented precedence is unchanged.
    A key counts as "from .env" when the cwd .env defines it and the process value equals it (load_env uses
    setdefault, so a different value means a real environment variable won).
    """
    environ = os.environ if environ is None else environ
    env_file = Path(cwd) / DEFAULT_ENV
    file_values = parse_env(env_file.read_text(encoding="utf-8-sig")) if env_file.is_file() else {}
    settings: dict[str, dict] = {}
    for key in SETTING_KEYS:
        value = environ.get(key)
        if value in (None, ""):
            continue
        settings[key] = {"value": value, "source": ".env" if file_values.get(key) == value else "environment"}
    outside = False
    if file_values and config_base is not None:
        try:
            outside = env_file.resolve().parent != Path(config_base).resolve()
        except OSError:
            outside = True
    from_env_file = sorted(k for k, v in settings.items() if v["source"] == ".env")
    return {
        "env_file": str(env_file.resolve()) if file_values else None,
        "env_file_outside_config": bool(outside and from_env_file),
        "settings": settings,
        "warning": (
            f"{env_file.resolve()} is supplying {', '.join(from_env_file)} - these override the config file you named "
            f"(a .env in the working folder beats memdex.yaml). Run from the config's own folder or remove those keys."
            if outside and from_env_file else None
        ),
    }
