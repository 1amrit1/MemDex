"""The web UI's server: one HTML page + a small JSON API, served by Python's standard library on loopback.

No framework: `ThreadingHTTPServer` is enough for one user on one machine, needs no extra dependency, and
keeps the offline guarantee simple (the guard allows loopback only). If MemDex ever serves remotely
(remote MCP for claude.ai, multi-user), that is the moment to move to FastAPI/uvicorn — not before.
"""

from __future__ import annotations

import email.policy
import json
import os
import socketserver
import sys
import threading
import time
import traceback
import webbrowser
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from memdex import __version__, store
from memdex.config import Config, ConfigError
from memdex.env import env_overrides
from memdex.embedder import ModelMissing, model_present
from memdex.eval import EvalError, load_questions, load_last, run_eval, save_result
from memdex.ingest import NothingToIngest, last_ingest, run_ingest
from memdex.offline import is_blocked
from memdex.projects import ProjectError, create_folder, create_project, list_projects, save_upload
from memdex.querylog import CALLERS, count_log, log_query, read_log
from memdex.search import as_context, search

STATIC = Path(__file__).parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY = 64 * 1024 * 1024


class NoIndex(RuntimeError):
    pass


class AppState:
    """Everything the request handlers share. `cfg_loader` is re-run per request so new projects show up."""

    def __init__(self, cfg_loader: Callable[[], Config], embedder_factory: Callable[[Config], object] | None = None) -> None:
        self._load = cfg_loader
        self._embedder_factory = embedder_factory
        self._embedder = None
        self._client = None
        self.lock = threading.RLock()
        self.host = DEFAULT_HOST
        self.port = DEFAULT_PORT
        self.ingest_status: dict = {"running": False, "started": None, "finished": None, "report": None, "error": None}

    @property
    def cfg(self) -> Config:
        return self._load()

    def embedder(self):
        if self._embedder is None:
            cfg = self.cfg
            if self._embedder_factory is not None:
                self._embedder = self._embedder_factory(cfg)
            else:
                from memdex.embedder import Embedder

                self._embedder = Embedder(cfg.model_dir)
        return self._embedder

    def client(self):
        if self._client is None:
            self._client = store.open_client(self.cfg.index_dir)
        return self._client

    def collection(self):
        return store.get_collection(self.client())

    # ---- search -------------------------------------------------------------------------------------------
    def do_search(self, caller: str, q: str, k: int, collection: str | None, project: str | None, superseded: bool):
        with self.lock:
            coll = self.collection()
            if coll is None:
                raise NoIndex("index is empty - run an ingest first")
            t0 = time.perf_counter()
            hits = search(self.embedder(), coll, q, k=k, collection=collection, project=project, include_superseded=superseded)
            ms = (time.perf_counter() - t0) * 1000
            log_query(self.cfg, caller, q, hits, ms, k, collection=collection, project=project, superseded=superseded)
            return hits, ms

    # ---- ingest (background thread, one at a time) -----------------------------------------------------------
    def start_ingest(self) -> bool:
        with self.lock:
            if self.ingest_status["running"]:
                return False
            self.ingest_status = {"running": True, "started": _now(), "finished": None, "report": None, "error": None}
        threading.Thread(target=self._run_ingest, name="memdex-ingest", daemon=True).start()
        return True

    def _run_ingest(self) -> None:
        try:
            with self.lock:
                report = run_ingest(self.cfg, self.embedder(), self.client())
            self.ingest_status.update(report=report.as_dict())
        except (ModelMissing, NothingToIngest, ConfigError) as exc:
            self.ingest_status.update(error=str(exc))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never swallowed silently
            self.ingest_status.update(error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            self.ingest_status.update(running=False, finished=_now())

    # ---- eval ---------------------------------------------------------------------------------------------
    def run_eval(self, path: Path | None = None) -> dict:
        cfg = self.cfg
        file = Path(path) if path else cfg.eval_dir / "questions.yaml"
        k, questions = load_questions(file)

        def fn(q: str, kk: int, collection: str | None, project: str | None):
            hits, _ = self.do_search("eval", q, kk, collection, project, False)
            return hits

        result = run_eval(questions, fn, k, source=str(file))
        save_result(cfg, result)
        return result.as_dict()

    # ---- health -------------------------------------------------------------------------------------------
    def health(self) -> dict:
        cfg = self.cfg
        out: dict = {
            "version": __version__,
            "config_path": str(cfg.path) if cfg.path else None,
            "base": str(cfg.base),
            "data_dir": str(cfg.data_dir),
            "index_dir": str(cfg.index_dir),
            "model_dir": str(cfg.model_dir),
            "max_chars": cfg.max_chars,
            "model_present": model_present(cfg.model_dir),
            "index_exists": (cfg.index_dir / "chroma.sqlite3").is_file(),
            "index_open": False,
            "index_error": None,
            "files": 0,
            "chunks": 0,
            "superseded_files": 0,
            "by_collection": {},
            "by_project": {},
            "chunks_without_project": 0,
            "index_size_mb": round(store.index_size_bytes(cfg.index_dir) / (1024 * 1024), 1),
            "segments": store.vector_segment_folders(cfg.index_dir),
            "last_ingest": last_ingest(cfg),
            "ingest": self.ingest_status,
            "sources": [
                {"name": s.name, "kind": s.kind, "path": str(s.path), "exists": s.path.is_dir(), "types": list(s.types),
                 "default_collection": s.default_collection}
                for s in cfg.sources
            ],
            "projects": len(cfg.project_sources()),
            "log_entries": count_log(cfg),
            "offline_guard": is_blocked(),
            "server": {"host": self.host, "port": self.port},
            "setup": self.setup_info(cfg),
            "env_overrides": env_overrides(Path.cwd(), cfg.base if cfg.path else None),
        }
        try:
            with self.lock:
                coll = self.collection()
                if coll is not None:
                    out.update(store.summarize(coll))
                out["index_open"] = True
        except Exception as exc:  # noqa: BLE001
            out["index_error"] = f"{type(exc).__name__}: {exc}"
        out["needs_reingest"] = bool(out["chunks"] and out["chunks_without_project"])
        return out

    @staticmethod
    def setup_info(cfg: Config) -> dict:
        """What to paste where so Claude Code (and later Claude Desktop) can call MemDex."""
        exe = Path(sys.executable).with_name("memdex.exe" if sys.platform == "win32" else "memdex")
        exe_s = exe.as_posix()
        cfg_s = cfg.path.as_posix() if cfg.path else "(no memdex.yaml - defaults)"
        cfg_flag = f" --config {cfg_s}" if cfg.path else ""
        hook = {
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": f"{exe_s} rules --hook{cfg_flag}", "timeout": 15}]}
                ]
            }
        }
        return {
            "memdex_exe": exe_s,
            "config": cfg_s,
            "claude_md": (
                "Before acting on a task, query MemDex and read what comes back:\n"
                f"  {exe_s} rules \"<the task in one line>\"{cfg_flag}\n"
                f"  {exe_s} search \"<question>\" --json{cfg_flag}"
            ),
            "hook_json": json.dumps(hook, indent=2),
            "hook_where": "Claude Code > ~/.claude/settings.json (every project) or <project>/.claude/settings.json (one project). "
                          "Merge the 'hooks' key into the existing file; restart the Claude Code session.",
            "mcp": "Milestone 6 adds an MCP server (`memdex mcp`) for Claude Code and Claude Desktop; until then use the CLI line or the hook.",
            "settings_button": "A button that writes these into your Claude settings for you is planned for Milestone 4 (hook) - for now copy and paste.",
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    """multipart/form-data -> (fields, files) using the stdlib email parser."""
    msg = BytesParser(policy=email.policy.HTTP).parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("expected multipart/form-data")
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append((filename, payload))
        elif name:
            fields[str(name)] = payload.decode("utf-8", "replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    state: AppState  # set by make_server
    server_version = f"MemDex/{__version__}"

    def log_message(self, fmt, *args):  # quiet by default; errors still go through log_error
        return

    # ---- helpers -------------------------------------------------------------------------------------------
    def _json(self, obj: object, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"request too large ({length} bytes)")
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    @staticmethod
    def _q(qs: dict, key: str, default: str = "") -> str:
        return (qs.get(key) or [default])[0].strip()

    # ---- routing -------------------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html", "text/html; charset=utf-8")
            if url.path == "/api/health":
                return self._json(self.state.health())
            if url.path == "/api/search":
                return self._search(qs)
            if url.path == "/api/log":
                n = int(self._q(qs, "n", "5") or 5)
                caller = self._q(qs, "caller") or None
                if caller and caller not in CALLERS:
                    return self._error(f"caller must be one of {', '.join(CALLERS)}")
                return self._json({"entries": read_log(self.state.cfg, n=n, caller=caller, contains=self._q(qs, "q") or None),
                                   "total": count_log(self.state.cfg), "callers": list(CALLERS)})
            if url.path == "/api/projects":
                return self._json({"projects": list_projects(self.state.cfg)})
            if url.path == "/api/ingest/status":
                return self._json(self.state.ingest_status)
            if url.path == "/api/eval":
                return self._json({"result": load_last(self.state.cfg),
                                   "questions_file": str(self.state.cfg.eval_dir / "questions.yaml"),
                                   "questions_exists": (self.state.cfg.eval_dir / "questions.yaml").is_file()})
            if url.path == "/api/setup":
                return self._json(self.state.setup_info(self.state.cfg))
            return self._error("not found", 404)
        except NoIndex as exc:
            return self._error(str(exc), 409)
        except (ValueError, ConfigError) as exc:
            return self._error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/api/projects":
                name = str(self._json_body().get("name", ""))
                path, created = create_project(self.state.cfg, name)
                return self._json({"name": path.name, "path": str(path), "created": created}, 201 if created else 200)
            if url.path == "/api/projects/folders":
                data = self._json_body()
                path, created = create_folder(self.state.cfg, str(data.get("project", "")), str(data.get("folder", "")))
                return self._json({"path": str(path), "created": created}, 201 if created else 200)
            if url.path == "/api/upload":
                return self._upload()
            if url.path == "/api/ingest":
                started = self.state.start_ingest()
                return self._json({"started": started, "status": self.state.ingest_status}, 202 if started else 409)
            if url.path == "/api/eval/run":
                data = self._json_body()
                return self._json({"result": self.state.run_eval(data.get("file") or None)})
            return self._error("not found", 404)
        except NoIndex as exc:
            return self._error(str(exc), 409)
        except (ProjectError, EvalError, ValueError, ConfigError) as exc:
            return self._error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # ---- handlers ------------------------------------------------------------------------------------------
    def _static(self, name: str, content_type: str) -> None:
        data = (STATIC / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _search(self, qs: dict) -> None:
        q = self._q(qs, "q")
        if not q:
            return self._error("q is required")
        k = min(50, max(1, int(self._q(qs, "k", "5") or 5)))
        collection = self._q(qs, "collection") or None
        project = self._q(qs, "project") or None
        superseded = self._q(qs, "superseded") in ("1", "true", "yes")
        hits, ms = self.state.do_search("ui", q, k, collection, project, superseded)
        return self._json({"query": q, "k": k, "ms": round(ms, 1), "hits": [h.as_dict() for h in hits], "context": as_context(hits, q)})

    def _upload(self) -> None:
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return self._error("upload must be multipart/form-data")
        fields, files = parse_multipart(ctype, self._body())
        project = fields.get("project", "")
        folder = fields.get("folder", "")
        reindex = fields.get("reindex", "false").lower() in ("1", "true", "on", "yes")
        if not files:
            return self._error("no files in the upload")
        saved, errors = [], []
        for filename, data in files:
            try:
                saved.append(save_upload(self.state.cfg, project, folder, filename, data))
            except ProjectError as exc:
                errors.append({"name": filename, "error": str(exc)})
        started = self.state.start_ingest() if (reindex and saved) else False
        return self._json({"saved": saved, "errors": errors, "reindex_started": started}, 200 if saved else 400)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:  # skip BaseHTTPServer's reverse-DNS lookup of the bind address
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]


def make_server(state: AppState, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> _Server:
    handler = type("MemDexHandler", (Handler,), {"state": state})
    srv = _Server((host, port), handler)
    state.host, state.port = srv.server_address[0], srv.server_address[1]
    return srv


def serve(cfg_loader: Callable[[], Config], host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    state = AppState(cfg_loader)
    srv = make_server(state, host, port)
    url = f"http://{state.host}:{state.port}/"
    print(f"MemDex UI  {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
