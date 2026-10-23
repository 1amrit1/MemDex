"""Query log: one JSON line per search/rules call, whoever made it (cli, hook, mcp, ui).

This is how the UI shows "Claude Code pulled these chunks for that prompt". Lives under data_dir/logs,
outside the index, so a rebuild never touches it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from memdex.config import Config
from memdex.search import Hit

CALLERS = ("cli", "hook", "mcp", "ui", "eval")


def log_query(
    cfg: Config, caller: str, query: str, hits: list[Hit], ms: float, k: int, n_hits: int | None = None, **filters
) -> dict:
    """`n_hits` is only passed by callers that return something other than Hit objects: `outline` and
    `show` return sections and file regions. The log is meant to be the record of everything an
    assistant pulled from the index, not only the vector searches."""
    entry = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "caller": caller,
        "query": query,
        "k": k,
        "filters": {key: value for key, value in filters.items() if value not in (None, "", False)},
        "ms": round(ms, 1),
        "n_hits": len(hits) if n_hits is None else int(n_hits),
        "hits": [{"rel_path": h.rel_path, "heading_path": h.heading_path, "score": h.score} for h in hits],
    }
    path: Path = cfg.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_log(cfg: Config, n: int = 5, caller: str | None = None, contains: str | None = None) -> list[dict]:
    """Newest first. `caller` filters exactly; `contains` is a case-insensitive substring of the query."""
    path: Path = cfg.log_path
    if not path.is_file():
        return []
    needle = (contains or "").strip().lower()
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if caller and e.get("caller") != caller:
            continue
        if needle and needle not in str(e.get("query", "")).lower():
            continue
        out.append(e)
    out.reverse()
    return out[: max(0, int(n))]


def count_log(cfg: Config) -> int:
    path: Path = cfg.log_path
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
