"""The CLI must write UTF-8 whatever the terminal's codepage is.

Regression test for the defect found 2026-10-20. Windows opens a REDIRECTED stdout in the locale codepage
(cp1252), which cannot encode characters the real documents contain: box drawing, the EXECUTION ORDER
arrow, the tick and cross marks in CLAUDE.md 2.4a. On ASOH's index 285 of 1,793 chunks (48 of 70 files)
were unprintable, so `search --json` exited 1 with no output at all, and `rules --hook` exited 0 having
injected nothing - silently dropping the very rule the prompt was breaking. The web UI never had the bug
because it encodes UTF-8 itself, which is why every other M2 check passed.

These tests drive the real console entry point in a child process whose streams start as cp1252, forced
with PYTHONIOENCODING so the test reproduces the failure on any machine, not only a Windows one. The child
swaps in the fake embedder before the command runs, so no model and no network are needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memdex import store
from memdex.config import load_config
from memdex.ingest import run_ingest

# One character from each class that broke: box drawing, arrow, tick, cross, em dash, section sign, middot.
HOSTILE = "─ ▶ ✅ ❌ — § ·"
DOC = f"""# Hostile characters

2026-10-20 · Someone

## 2.4a Marks that cp1252 cannot encode

{HOSTILE}

- ✅ allowed, never a table
- ❌ refused

▶ ORDER: T1 → T2
"""

# Runs in the child: point `Embedder` at the fake, then hand control to the CLI. `cli.py` resolves
# `from memdex.embedder import Embedder` at call time, so replacing the module attribute is enough.
CHILD = """
import sys
sys.path.insert(0, {tests_dir!r})
import memdex.embedder as embedder_module
from fake_embedder import FakeEmbedder
embedder_module.Embedder = FakeEmbedder
from memdex.cli import app
app()
"""


@pytest.fixture
def indexed(tmp_path):
    """A real on-disk index holding one file full of characters cp1252 cannot encode."""
    with pytest.raises(UnicodeEncodeError):  # if the corpus is not hostile, these tests prove nothing
        DOC.encode("cp1252")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "HOSTILE.md").write_text(DOC, encoding="utf-8")
    cfg_path = tmp_path / "memdex.yaml"
    cfg_path.write_text(
        "index_dir: .index\n"
        "model_dir: models\n"
        "data_dir: data\n"
        "sources:\n"
        "  - path: docs\n"
        "    name: hostile\n"
        "    types: [md]\n"
        '    collections: [{ match: "**", collection: rules }]\n',
        encoding="utf-8",
    )
    from fake_embedder import FakeEmbedder

    cfg = load_config(cfg_path)
    report = run_ingest(cfg, FakeEmbedder(), store.open_client(cfg.index_dir))
    assert report.chunks > 0
    return cfg_path


def _run(cfg_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run the CLI in a child whose streams default to cp1252, as a redirected Windows stdout does."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMDEX_")}
    env["PYTHONIOENCODING"] = "cp1252"
    script = CHILD.format(tests_dir=str(Path(__file__).parent))
    return subprocess.run(
        [sys.executable, "-c", script, *args, "--config", str(cfg_path)],
        input=stdin.encode("utf-8") if stdin is not None else None,
        capture_output=True,
        env=env,
        cwd=str(cfg_path.parent),
        timeout=180,
    )


def test_search_json_survives_a_cp1252_stdout(indexed):
    """`search --json` is the path Claude Code consumes: it must exit 0 and parse."""
    out = _run(indexed, ["search", "marks that cp1252 cannot encode", "-k", "3", "--json"])
    stderr = out.stderr.decode("utf-8", "replace")
    assert out.returncode == 0, stderr
    assert out.stdout, f"no output at all; stderr: {stderr}"
    payload = json.loads(out.stdout.decode("utf-8"))
    assert payload["hits"], "no hits, so nothing hostile was encoded and the test proves nothing"
    out.stdout.decode("ascii")  # ensure_ascii=True: the JSON bytes are plain ASCII whatever the codepage
    # Which chunk ranks first depends on the fake embedder, so assert across the hits, not on the top one
    assert any("✅" in h["text"] for h in payload["hits"]), "the hostile text must survive the round trip"


def test_search_table_output_survives_a_cp1252_stdout(indexed):
    """The human-readable path prints the heading, which also carries hostile characters."""
    out = _run(indexed, ["search", "marks that cp1252 cannot encode", "-k", "3"])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert b"2.4a" in out.stdout


def test_rules_hook_injects_rather_than_silently_dropping(indexed):
    """The hook must emit the rule, not exit 0 with an empty stdout."""
    out = _run(
        indexed,
        ["rules", "--hook", "--min-score", "0.0"],
        stdin=json.dumps({"prompt": "marks that cp1252 cannot encode, never a table"}),
    )
    assert out.returncode == 0
    text = out.stdout.decode("utf-8")
    assert text.strip(), f"hook injected nothing; stderr: {out.stderr.decode('utf-8', 'replace')}"
    assert "Rules from MemDex" in text
    assert "2.4a" in text
    # Claude Code reports a hook whose stdout starts with `{` but is not valid JSON as a hook error
    assert not text.lstrip().startswith("{")


def test_citation_does_not_repeat_the_file_name():
    """heading_path already starts with rel_path, so printing both wasted tokens on every injected prompt."""
    from memdex.search import Hit, as_context, as_rules_text

    hit = Hit(0.5, "CLAUDE.md", "x/CLAUDE.md", "CLAUDE.md > PART 1 > 1.4", 1, 2, "rules", "P", "body", "", False, "0:CLAUDE.md#0")
    for rendered in (as_context([hit], "q"), as_rules_text([hit], "q")):
        assert "CLAUDE.md > PART 1 > 1.4" in rendered
        assert "CLAUDE.md > CLAUDE.md" not in rendered
