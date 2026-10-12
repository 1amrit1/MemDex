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
