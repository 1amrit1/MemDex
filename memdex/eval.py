"""Retrieval eval: a YAML list of questions with the file (and optionally heading) a good answer lives in.

recall@k = share of questions whose expected chunk appears in the top k. MRR = mean of 1/rank of the
first match. This is the number that decides chunk size and embedding model choices.

Question file (YAML):
    k: 5
    questions:
      - q: what must happen before any write step
        collection: rules            # optional filter, like the CLI
        project: CLAUDE              # optional filter
        expect:                      # any one match counts
          - { rel_path: CLAUDE.md, heading: "1.4" }
          - { rel_path: CLAUDE.md, heading: "1.3" }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from memdex.config import Config
from memdex.search import Hit


class EvalError(ValueError):
    pass


@dataclass(frozen=True)
class Expectation:
    rel_path: str
    heading: str = ""


@dataclass(frozen=True)
class Question:
    q: str
    expect: tuple[Expectation, ...]
    collection: str | None = None
    project: str | None = None
    note: str = ""


@dataclass
class EvalResult:
    k: int
    rows: list[dict] = field(default_factory=list)
    ran_at: str = ""
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def hits(self) -> int:
        return sum(1 for r in self.rows if r["hit"])

    @property
    def recall(self) -> float:
        return round(self.hits / self.n, 4) if self.n else 0.0

    @property
    def mrr(self) -> float:
        return round(sum(1.0 / r["rank"] for r in self.rows if r["rank"]) / self.n, 4) if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "n": self.n,
            "hits": self.hits,
            "recall_at_k": self.recall,
            "mrr": self.mrr,
            "ran_at": self.ran_at,
            "source": self.source,
            "rows": self.rows,
        }


def load_questions(path: Path) -> tuple[int, list[Question]]:
    if not Path(path).is_file():
        raise EvalError(f"eval file not found: {path} (see eval.example.yaml)")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = data.get("questions") or []
    if not raw:
        raise EvalError(f"no questions in {path}")
    questions = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict) or not item.get("q"):
            raise EvalError(f"question #{n} has no q")
        exp = []
        for e in item.get("expect") or []:
            if isinstance(e, str):
                exp.append(Expectation(rel_path=e))
            elif isinstance(e, dict) and e.get("rel_path"):
                exp.append(Expectation(rel_path=str(e["rel_path"]), heading=str(e.get("heading") or "")))
            else:
                raise EvalError(f"question #{n}: each expect needs rel_path")
        if not exp:
            raise EvalError(f"question #{n} has no expect")
        questions.append(
            Question(
                q=str(item["q"]).strip(),
                expect=tuple(exp),
                collection=item.get("collection"),
                project=item.get("project"),
                note=str(item.get("note") or ""),
            )
        )
    return int(data.get("k") or 5), questions


def matches(hit: Hit, exp: Expectation) -> bool:
    want = exp.rel_path.replace("\\", "/").lower()
    got = hit.rel_path.replace("\\", "/").lower()
    if not (got == want or got.endswith("/" + want)):
        return False
    return (exp.heading.lower() in hit.heading_path.lower()) if exp.heading else True


SearchFn = Callable[[str, int, "str | None", "str | None"], list[Hit]]


def run_eval(questions: list[Question], search_fn: SearchFn, k: int, source: str = "") -> EvalResult:
    result = EvalResult(k=k, ran_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), source=source)
    for q in questions:
        hits = search_fn(q.q, k, q.collection, q.project)
        rank = next((i for i, h in enumerate(hits, 1) if any(matches(h, e) for e in q.expect)), None)
        result.rows.append(
            {
                "q": q.q,
                "collection": q.collection,
                "project": q.project,
                "expect": [e.rel_path + (f" > *{e.heading}*" if e.heading else "") for e in q.expect],
                "hit": rank is not None,
                "rank": rank,
                "top": [{"rel_path": h.rel_path, "heading_path": h.heading_path, "score": h.score} for h in hits[:3]],
                "note": q.note,
            }
        )
    return result


def save_result(cfg: Config, result: EvalResult) -> Path:
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.eval_dir / "last_result.json"
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_last(cfg: Config) -> dict | None:
    try:
        return json.loads((cfg.eval_dir / "last_result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
