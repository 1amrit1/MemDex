"""MemDex command line. Every command runs with the network blocked (loopback allowed) except `fetch-model`."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

# Windows opens a REDIRECTED stdout in the locale codepage (cp1252), which cannot encode the characters the
# docs actually contain: the box-drawing rules in ASOH's code blocks, the arrow in the EXECUTION ORDER line,
# the tick and cross marks in CLAUDE.md 2.4a. LIVE READ 2026-10-20 on ASOH's index: 285 of 1,793 chunks
# (48 of 70 files; 11 of 154 rules chunks) could not be written, so `search --json` exited 1 with no output
# and `rules --hook` exited 0 having injected NOTHING - silently dropping the rule the prompt was breaking.
# This must run before anything writes to a stream. The web UI was never affected: it encodes UTF-8 itself.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # a stream that cannot be reconfigured is left as it is
        pass

import typer

from memdex import __version__
from memdex.config import ConfigError, load_config
from memdex.env import load_env
from memdex.offline import block_network

app = typer.Typer(add_completion=False, no_args_is_help=True, help="MemDex - local, offline index over your docs.")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to memdex.yaml (default: ./memdex.yaml or $MEMDEX_CONFIG)")


def _config(path: Optional[str]):
    load_env()  # ./.env fills settings not already set in the environment
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.secho(f"config error: {exc}", fg="red", err=True)
        raise typer.Exit(2)


def _open(cfg):
    """(embedder, chroma collection) for read commands; exits with a clear message when either is missing."""
    from memdex import store
    from memdex.embedder import Embedder

    coll = store.get_collection(store.open_client(cfg.index_dir))
    if coll is None:
        typer.secho("index is empty - run: memdex ingest", fg="red", err=True)
        raise typer.Exit(1)
    emb = Embedder(cfg.model_dir)
    if not emb.is_present():
        typer.secho(f"embedding model not found in {emb.model_path} - run: memdex fetch-model", fg="red", err=True)
        raise typer.Exit(1)
    return emb, coll


def _search(cfg, caller: str, q: str, k: int, collection: Optional[str], project: Optional[str], superseded: bool):
    from memdex.querylog import log_query
    from memdex.search import search

    emb, coll = _open(cfg)
    t0 = time.perf_counter()
    hits = search(emb, coll, q, k=k, collection=collection, project=project, include_superseded=superseded)
    ms = (time.perf_counter() - t0) * 1000
    log_query(cfg, caller, q, hits, ms, k, collection=collection, project=project, superseded=superseded)
    return hits, ms


@app.command()
def version() -> None:
    """Print the MemDex version."""
    typer.echo(__version__)


@app.command("fetch-model")
def fetch_model(config: Optional[str] = ConfigOpt) -> None:
    """Download the embedding model once (the ONLY command that uses the network)."""
    from memdex.embedder import Embedder

    cfg = _config(config)
    emb = Embedder(cfg.model_dir)
    if emb.is_present():
        typer.echo(f"model already present: {emb.model_path}")
        return
    typer.echo(f"downloading model to {emb.model_path} ...")
    emb.fetch()
    typer.echo("OK - model present" if emb.is_present() else "FAILED - model files missing after download")


@app.command()
def ingest(config: Optional[str] = ConfigOpt) -> None:
    """Rebuild the index from all configured sources and UI projects (offline)."""
    block_network()
    from memdex import store
    from memdex.embedder import Embedder, ModelMissing
    from memdex.ingest import NothingToIngest, run_ingest

    cfg = _config(config)
    try:
        report = run_ingest(cfg, Embedder(cfg.model_dir), store.open_client(cfg.index_dir))
    except (ModelMissing, NothingToIngest) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1)

    typer.echo(f"index:     {cfg.index_dir}")
    typer.echo(f"files:     {report.files}")
    typer.echo(f"chunks:    {report.chunks}  (max_chars {cfg.max_chars}; truncated at 256 tokens: {report.truncated})")
    for name, n in sorted(report.by_collection.items()):
        typer.echo(f"  {name:<12}{n}")
    typer.echo("projects:")
    for name, n in sorted(report.by_project.items()):
        typer.echo(f"  {name:<24}{n}")
    for path, reason in report.skipped:
        typer.echo(f"skipped:   {path} - {reason}")
    typer.echo(f"time:      {report.seconds:.1f}s")


@app.command()
def stats(config: Optional[str] = ConfigOpt) -> None:
    """Show what is in the index, read from the store itself (offline)."""
    block_network()
    from memdex import store

    cfg = _config(config)
    coll = store.get_collection(store.open_client(cfg.index_dir))
    if coll is None:
        typer.echo("index is empty - run: memdex ingest")
        raise typer.Exit(1)
    s = store.summarize(coll)
    typer.echo(f"index:     {cfg.index_dir}")
    typer.echo(f"files:     {s['files']}  (superseded: {s['superseded_files']})")
    typer.echo(f"chunks:    {s['chunks']}")
    for name, n in s["by_collection"].items():
        typer.echo(f"  {name:<12}{n}")
    typer.echo("projects:")
    for name, n in s["by_project"].items():
        typer.echo(f"  {name or '(none - re-ingest)':<24}{n}")
    if s["chunks_without_project"]:
        typer.secho(f"note: {s['chunks_without_project']} chunks predate projects - run: memdex ingest", fg="yellow")


@app.command()
def search(
    query: list[str] = typer.Argument(..., help="The question (quotes optional)"),
    k: int = typer.Option(5, "-k", "--k", help="How many chunks"),
    collection: Optional[str] = typer.Option(None, "--collection", "-C", help="rules, status, projects, references, archive, docs ..."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Limit to one project / source name"),
    superseded: bool = typer.Option(False, "--superseded", help="Include chunks from files marked SUPERSEDED"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output (for Claude Code)"),
    context: bool = typer.Option(False, "--context", help="Markdown block ready to paste into an assistant"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Semantic search over the index; prints file, heading path, line range and score per chunk (offline)."""
    block_network()
    from memdex.search import as_context, as_table

    cfg = _config(config)
    q = " ".join(query).strip()
    hits, ms = _search(cfg, "cli", q, k, collection, project, superseded)
    if json_out:
        typer.echo(json.dumps({"query": q, "k": k, "ms": round(ms, 1), "hits": [h.as_dict() for h in hits]}, ensure_ascii=True, indent=2))
    elif context:
        typer.echo(as_context(hits, q))
    else:
        typer.echo(as_table(hits))
        typer.echo(f"({len(hits)} hits in {ms:.0f} ms)")


@app.command()
def rules(
    query: Optional[list[str]] = typer.Argument(None, help="The task or prompt (omit with --hook)"),
    k: int = typer.Option(5, "-k", "--k"),
    min_score: float = typer.Option(
        0.25,
        "--min-score",
        # Do not raise this. 0.35 was tried on 2026-10-20 and injected ZERO bytes for "delete an old
        # policy rule from the server config" - whose correct answer, 1.4 CAPTURE-BEFORE-CHANGE, scores
        # 0.327. That is the silent-nothing failure of 4.4b, reintroduced by a flag.
        #
        # Measured over 20 real hook calls in the query log, useful and useless overlap completely:
        #   0.620 list doc creation rules          -> Doc_Creation_Rules      USEFUL
        #   0.562 task list with steps as a row    -> 2.4a never a table      USEFUL
        #   0.419 rename a runbook and its pointer -> 4.3 one file/workstream USEFUL
        #   0.369 rename a config on a remote      -> 1.1a LOCAL vs REMOTE    USEFUL
        #   0.329 create a pdf with a diagram      -> doc rules               USEFUL
        #   0.327 delete an old policy rule        -> 1.4 capture-before      USEFUL  <- the best one
        #   0.325 "i see that now..."              -> 2.5 one-line            noise
        #   0.314 "ok lets proceed..."             -> Planning Rules title    noise
        #   0.252 "do i have to load memdex?"      -> 1.2 execution priority  noise
        # The median top-hit is 0.349, so 0.35 drops half of ALL prompts. A score floor cannot separate
        # these; deciding whether a conversational prompt should fire a rules lookup at all is a design
        # question, not a threshold. 0.25 keeps every useful hit and is the conservative default.
        help="Drop weaker matches (0..1)",
    ),
    json_out: bool = typer.Option(False, "--json"),
    hook: bool = typer.Option(False, "--hook", help="Claude Code UserPromptSubmit mode: read the hook JSON on stdin, print matching rules, never fail"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """The rules that apply to a task: `search` restricted to the rules collection (offline)."""
    block_network()
    from memdex.search import as_rules_text, as_table

    if hook:
        # A hook must never block the user's prompt: any problem -> print nothing, exit 0, note on stderr.
        try:
            payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
            q = str(payload.get("prompt") or "").strip()
            if len(q) < 8:
                raise typer.Exit(0)
            cfg = _config(config)
            hits, _ = _search(cfg, "hook", q, k, "rules", None, False)
            hits = [h for h in hits if h.score >= min_score]
            if hits:
                typer.echo(as_rules_text(hits, q))
        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"memdex hook: {type(exc).__name__}: {exc}", fg="yellow", err=True)
        raise typer.Exit(0)

    q = " ".join(query or []).strip()
    if not q:
        typer.secho("give the task as an argument, e.g.: memdex rules \"rename a config file\"", fg="red", err=True)
        raise typer.Exit(2)
    cfg = _config(config)
    hits, ms = _search(cfg, "cli", q, k, "rules", None, False)
    hits = [h for h in hits if h.score >= min_score]
    if json_out:
        typer.echo(json.dumps({"query": q, "k": k, "ms": round(ms, 1), "hits": [h.as_dict() for h in hits]}, ensure_ascii=True, indent=2))
    else:
        typer.echo(as_table(hits))
        typer.echo(f"({len(hits)} rules >= {min_score} in {ms:.0f} ms)")


@app.command()
def outline(
    file: Optional[str] = typer.Argument(None, help="File name or part of its path, e.g. CLAUDE.md"),
    list_files_: bool = typer.Option(False, "--list", help="List every indexed file instead"),
    json_out: bool = typer.Option(False, "--json"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """The sections of one file, with line ranges and a ready-to-run `show` (offline)."""
    block_network()
    from memdex.querylog import log_query
    from memdex.regions import RegionError, as_outline_text, list_files, outline as build_outline, resolve_file

    cfg = _config(config)
    _, coll = _open(cfg)
    t0 = time.perf_counter()
    try:
        if list_files_:
            names = list_files(coll, file or "")
            ms = (time.perf_counter() - t0) * 1000
            log_query(cfg, "cli", f"outline --list {file or ''}".strip(), [], ms, 0, n_hits=len(names), kind="outline")
            if json_out:
                typer.echo(json.dumps({"files": names}, ensure_ascii=True, indent=2))
            else:
                typer.echo("\n".join(names) or "index is empty")
                typer.echo(f"({len(names)} files)")
            return
        rel = resolve_file(coll, file or "")
        sections = build_outline(coll, rel)
    except RegionError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2)
    ms = (time.perf_counter() - t0) * 1000
    log_query(cfg, "cli", f"outline {rel}", [], ms, 0, n_hits=len(sections), kind="outline", file=rel)
    if json_out:
        typer.echo(json.dumps({"file": rel, "sections": [x.as_dict() for x in sections]}, ensure_ascii=True, indent=2))
    else:
        typer.echo(as_outline_text(rel, sections))


@app.command()
def show(
    file: str = typer.Argument(..., help="File name or part of its path"),
    heading: Optional[str] = typer.Option(None, "--heading", "-H", help="Substring of the section heading"),
    lines: Optional[str] = typer.Option(None, "--lines", "-L", help="Line range, e.g. 120-180"),
    chunk: Optional[int] = typer.Option(None, "--chunk", help="Chunk ordinal within the file"),
    max_chars: int = typer.Option(4000, "--max-chars", help="Budget; 0 means no limit"),
    json_out: bool = typer.Option(False, "--json"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """One named region of a file, read from disk - what top-k retrieval cannot give you (offline)."""
    block_network()
    from memdex.querylog import log_query
    from memdex.regions import RegionError, as_region_text, resolve_file, show as fetch_region

    picked = [name for name, value in (("--heading", heading), ("--lines", lines), ("--chunk", chunk)) if value is not None]
    if len(picked) > 1:
        typer.secho(f"use one selector at a time, not {' and '.join(picked)}", fg="red", err=True)
        raise typer.Exit(2)

    span = None
    if lines:
        try:
            lo, _, hi = lines.partition("-")
            span = (int(lo), int(hi or lo))
        except ValueError:
            typer.secho(f"--lines wants a range like 120-180, got {lines!r}", fg="red", err=True)
            raise typer.Exit(2)

    cfg = _config(config)
    _, coll = _open(cfg)
    t0 = time.perf_counter()
    try:
        rel = resolve_file(coll, file)
        region = fetch_region(coll, rel, heading=heading, lines=span, chunk=chunk, max_chars=max_chars)
    except RegionError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2)
    ms = (time.perf_counter() - t0) * 1000
    log_query(cfg, "cli", f"show {rel}", [], ms, 0, n_hits=1, kind="show", file=rel, heading=heading, lines=lines, chunk=chunk)

    # Staleness goes to stderr so it is visible to a human but never corrupts piped or --json output.
    if region.stale:
        typer.secho(f"stale: {rel} has changed on disk since the last ingest - run: memdex ingest", fg="yellow", err=True)
    if json_out:
        typer.echo(json.dumps(region.as_dict(), ensure_ascii=True, indent=2))
    else:
        typer.echo(as_region_text(region))


@app.command()
def eval(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Questions YAML (default: <data_dir>/eval/questions.yaml)"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Run the retrieval eval set and report recall@k and MRR (offline)."""
    block_network()
    from memdex.eval import EvalError, load_questions, run_eval, save_result

    cfg = _config(config)
    path = file or (cfg.eval_dir / "questions.yaml")
    try:
        k, questions = load_questions(path)
    except EvalError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2)

    def fn(q: str, kk: int, collection: Optional[str], project: Optional[str]):
        hits, _ = _search(cfg, "eval", q, kk, collection, project, False)
        return hits

    result = run_eval(questions, fn, k, source=str(path))
    for n, row in enumerate(result.rows, 1):
        mark = "HIT " if row["hit"] else "MISS"
        top = row["top"][0] if row["top"] else None
        top_s = f"{top['score']:.2f} {top['rel_path']} > {top['heading_path'].split(' > ')[-1]}" if top else "-"
        typer.echo(f"{mark} #{n:<3} rank {row['rank'] or '-':<3} {row['q'][:60]:<60} | top: {top_s}")
    typer.echo(f"\nrecall@{k}: {result.recall:.0%} ({result.hits}/{result.n})   MRR: {result.mrr:.2f}")
    typer.echo(f"saved: {save_result(cfg, result)}")


@app.command()
def log(
    n: int = typer.Option(5, "-n", help="How many entries (newest first)"),
    caller: Optional[str] = typer.Option(None, "--caller", help="cli, hook, mcp, ui, eval"),
    grep: Optional[str] = typer.Option(None, "--grep", help="Only queries containing this text"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Show the query log: who asked what, and which chunks came back."""
    from memdex.querylog import count_log, read_log

    cfg = _config(config)
    entries = read_log(cfg, n=n, caller=caller, contains=grep)
    if not entries:
        typer.echo(f"no entries ({count_log(cfg)} logged in total) - {cfg.log_path}")
        return
    for e in entries:
        top = e["hits"][0] if e["hits"] else None
        top_s = f"{top['score']:.2f} {top['rel_path']} > {top['heading_path'].split(' > ')[-1]}" if top else "no hits"
        typer.echo(f"{e['ts'][:19]}  {e['caller']:<5} {e['ms']:>6.0f}ms  {e['query'][:60]:<60} | {top_s}")
    typer.echo(f"({len(entries)} shown of {count_log(cfg)} logged)")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback only; MemDex is not meant to be exposed"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser automatically"),
    config: Optional[str] = ConfigOpt,
) -> None:
    """Start the local web UI (search, upload, health, query log, eval, ingest) on http://127.0.0.1:8765."""
    block_network()
    from memdex.offline import is_loopback
    from memdex.ui.server import serve

    if not is_loopback(host):
        typer.secho("MemDex serves on loopback only (127.0.0.1). Remote access is a later milestone with auth.", fg="red", err=True)
        raise typer.Exit(2)
    _config(config)  # fail early with a clear message; the server re-reads the config per request
    serve(lambda: load_config(config), host=host, port=port, open_browser=not no_browser)
