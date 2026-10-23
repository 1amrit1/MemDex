"""End-to-end through the HTTP API with the fake embedder: projects -> upload -> ingest -> search -> log -> eval."""

import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from memdex.ui.server import AppState, make_server, parse_multipart


@pytest.fixture
def server(tmp_cfg, fake_embedder):
    _, load = tmp_cfg
    state = AppState(load, embedder_factory=lambda cfg: fake_embedder)
    srv = make_server(state, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{state.port}", state
    finally:
        srv.shutdown()
        srv.server_close()


def _call(base, path, method="GET", body=None, ctype="application/json"):
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = Request(base + path, data=data, method=method, headers={"Content-Type": ctype} if data else {})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _multipart(fields: dict, files: list[tuple[str, bytes]]):
    b = b"----memdex-test-boundary"
    out = []
    for k, v in fields.items():
        out.append(b"--" + b + b"\r\nContent-Disposition: form-data; name=\"" + k.encode() + b"\"\r\n\r\n" + v.encode() + b"\r\n")
    for name, data in files:
        out.append(b"--" + b + b"\r\nContent-Disposition: form-data; name=\"files\"; filename=\"" + name.encode() + b"\"\r\nContent-Type: application/octet-stream\r\n\r\n" + data + b"\r\n")
    out.append(b"--" + b + b"--\r\n")
    return b"".join(out), f"multipart/form-data; boundary={b.decode()}"


def test_parse_multipart_roundtrip():
    body, ctype = _multipart({"project": "P", "folder": "a/b"}, [("x.md", b"# hi\n\xe2\x9c\x93 utf8 and \x00 binary")])
    fields, files = parse_multipart(ctype, body)
    assert fields == {"project": "P", "folder": "a/b"} and files == [("x.md", b"# hi\n\xe2\x9c\x93 utf8 and \x00 binary")]


def test_full_flow(server, tmp_path):
    base, state = server
    status, page = 200, urlopen(base + "/", timeout=10).read()
    assert b"<title>MemDex</title>" in page

    code, h = _call(base, "/api/health")
    assert code == 200 and h["chunks"] == 0 and h["index_open"] is True and h["projects"] == 0 and h["setup"]["hook_json"]

    assert _call(base, "/api/search?q=anything")[0] == 409  # no index yet
    assert _call(base, "/api/search?q=")[0] == 400

    code, r = _call(base, "/api/projects", "POST", {"name": "Travel"})
    assert code == 201 and r["created"] is True
    assert _call(base, "/api/projects", "POST", {"name": "../x"})[0] == 400
    code, r = _call(base, "/api/projects/folders", "POST", {"project": "Travel", "folder": "2026"})
    assert code == 201

    body, ctype = _multipart({"project": "Travel", "folder": "2026", "reindex": "true"},
                             [("ferry.md", b"# Ferry\n\nThe ferry to Langdale leaves at nine from Horseshoe Bay.\n"),
                              ("bad.pdf", b"%PDF-1.4"),
                              ("notes.txt", b"Skookumchuck rapids are best at a big flood tide.\n")])
    code, r = _call(base, "/api/upload", "POST", body, ctype)
    assert code == 200 and [s["rel_path"] for s in r["saved"]] == ["2026/ferry.md", "2026/notes.txt"]
    assert r["errors"][0]["name"] == "bad.pdf" and "Milestone 7" in r["errors"][0]["error"]
    assert r["reindex_started"] is True

    for _ in range(100):
        if not state.ingest_status["running"]:
            break
        time.sleep(0.1)
    st = state.ingest_status
    assert st["error"] is None, st
    assert st["report"]["files"] == 2 and st["report"]["by_project"] == {"Travel": st["report"]["chunks"]}

    code, p = _call(base, "/api/projects")
    assert p["projects"][0]["files"] == 2 and p["projects"][0]["folders"] == ["2026"]

    code, s = _call(base, "/api/search?q=ferry+to+Langdale+leaves+at+nine+from+Horseshoe+Bay&k=5&project=Travel")
    assert code == 200 and s["hits"][0]["rel_path"] == "2026/ferry.md" and s["hits"][0]["project"] == "Travel" and s["hits"][0]["collection"] == "docs"
    assert "## MemDex results for" in s["context"]
    assert _call(base, "/api/search?q=ferry&project=Nope")[1]["hits"] == []

    code, lg = _call(base, "/api/log?n=5")
    assert lg["total"] == 2 and lg["entries"][0]["caller"] == "ui" and lg["entries"][0]["query"] == "ferry"
    assert _call(base, "/api/log?n=5&caller=bogus")[0] == 400

    code, h = _call(base, "/api/health")
    assert h["files"] == 2 and h["by_project"] == {"Travel": h["chunks"]} and h["needs_reingest"] is False and h["last_ingest"]["files"] == 2

    cfg = state.cfg
    cfg.eval_dir.mkdir(parents=True)
    (cfg.eval_dir / "questions.yaml").write_text(
        "k: 3\nquestions:\n  - q: ferry to Langdale leaves at nine from Horseshoe Bay\n    project: Travel\n    expect: [{rel_path: 2026/ferry.md, heading: Ferry}]\n"
        "  - q: Skookumchuck rapids are best at a big flood tide\n    expect: [2026/notes.txt]\n  - q: completely unrelated zebra\n    expect: [nothing.md]\n", encoding="utf-8")
    code, ev = _call(base, "/api/eval/run", "POST", {})
    assert code == 200 and ev["result"]["n"] == 3 and ev["result"]["hits"] == 2 and ev["result"]["rows"][0]["rank"] == 1
    assert _call(base, "/api/eval")[1]["result"]["recall_at_k"] == round(2 / 3, 4)

    assert _call(base, "/api/nope")[0] == 404


def test_search_tab_has_download_button_disabled_until_results(server):
    base, _ = server
    page = urlopen(base + "/", timeout=10).read().decode()
    assert 'id="s-dl" disabled' in page and 'id="s-copy" disabled' in page
    assert "function contextFilename" in page


def test_health_env_overrides_quiet_when_nothing_set(server, tmp_path, monkeypatch):
    base, _ = server
    monkeypatch.chdir(tmp_path)
    status, h = _call(base, "/api/health")
    assert status == 200 and h["env_overrides"]["settings"] == {} and h["env_overrides"]["warning"] is None
    assert h["env_overrides"]["env_file"] is None


def test_health_warns_about_cwd_env_outside_config_folder(server, tmp_path, monkeypatch):
    base, _ = server
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("MEMDEX_MODEL_DIR=models\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("MEMDEX_MODEL_DIR", "models")  # what load_env() does at CLI start-up
    _, h = _call(base, "/api/health")
    ov = h["env_overrides"]
    assert ov["settings"]["MEMDEX_MODEL_DIR"]["source"] == ".env" and ov["env_file_outside_config"] is True
    assert "MEMDEX_MODEL_DIR" in ov["warning"] and ov["env_file"].endswith(".env")


def test_health_no_warning_when_env_is_in_config_folder_or_is_real_env(server, tmp_path, monkeypatch):
    base, _ = server
    # .env beside memdex.yaml (tmp_path) is the supported setup -> reported but no warning
    (tmp_path / ".env").write_text("MEMDEX_MAX_CHARS=800\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMDEX_MAX_CHARS", "800")
    _, h = _call(base, "/api/health")
    assert h["env_overrides"]["settings"]["MEMDEX_MAX_CHARS"]["source"] == ".env" and h["env_overrides"]["warning"] is None
    # a real environment variable that differs from the cwd .env is labelled "environment", not blamed on the file
    other = tmp_path / "other"
    other.mkdir()
    (other / ".env").write_text("MEMDEX_MAX_CHARS=900\n", encoding="utf-8")
    monkeypatch.chdir(other)
    _, h = _call(base, "/api/health")
    assert h["env_overrides"]["settings"]["MEMDEX_MAX_CHARS"]["source"] == "environment" and h["env_overrides"]["warning"] is None
