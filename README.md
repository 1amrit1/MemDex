# MemDex

A local, offline index over your own docs — and, later, your code — so an AI assistant (Claude Code today, a local model later) can **search** them and pull the **rules** that apply to a task, instead of loading every file into context.

- **Fully local.** Embeddings are computed on your machine; the index is a file on your disk.
- **Offline by construction.** Every command except `fetch-model` runs with Python networking blocked (loopback stays open for the local web UI) — an accidental network call fails loudly instead of happening quietly.
- **Config-driven.** Point it at any folders; route files into collections (`rules`, `status`, `projects`, `references`, `archive`) by path pattern.
- **Or no config at all.** Create a **project** in the web UI and drop files into it; MemDex indexes it without a YAML file.
- **Bring your own assistant.** MemDex only retrieves. It never calls a cloud model, so it costs nothing to run and needs no API key.

## Status

Milestone 2 — `search`, `rules`, `eval`, `log`, the local web **UI** with uploads, and a measured retrieval baseline.

| Phase | Scope |
|---|---|
| 1 | Docs: Markdown → txt → PDF → Excel; search + rules retrieval; Claude Code integration; MCP server |
| 2 | Code as a second source type (AST-aware chunking) |
| 3 | Local LLM over the index; later, fine-tuning |

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"

Copy-Item .env.example .env                 # settings: index/model/data folders, chunk size
Copy-Item memdex.example.yaml memdex.yaml   # OPTIONAL: source folders to index. Skip it and use UI projects
memdex fetch-model                          # one-time download (~90 MB), the only network step
memdex ui                                   # the web UI on http://127.0.0.1:8765 - upload, search, health
```

Or stay on the command line:

```powershell
memdex ingest                                       # build/rebuild the index (offline)
memdex stats                                        # what is in the index
memdex search "which embedding model is used" -k 5  # semantic search with citations
memdex rules "rename a config file"                 # just the rules that apply to a task
memdex eval                                         # recall@k over your own question set
memdex log -n 20                                    # who queried what, and what came back
```

## The web UI

`memdex ui` serves a single page on `127.0.0.1` (loopback only — there is no auth, and remote access is deliberately not supported yet). Tabs:

| Tab | What it does |
|---|---|
| **Search** (default) | Query box with collection / project / top-k filters and a *Copy as context* button that yields a Markdown block to paste into Claude |
| **Upload** | Create a **project**, create nested **folders**, upload `.md` / `.txt` files, optionally re-index straight away |
| **Health / connection** | Index path, whether it opens, file/chunk counts per collection and project, chunk size, size on disk, orphan HNSW folders, last ingest, plus **copy-paste snippets to connect Claude Code** |
| **Query log** | The last N queries (default 5) with the caller — `cli`, `hook`, `mcp`, `ui`, `eval` — latency and top hits. This is where you watch Claude Code pull from the index |
| **Eval** | recall@k and MRR per question, with the top 3 results for every miss |
| **Ingest** | Rebuild the index and read the report |

The server is Python's standard-library `ThreadingHTTPServer` — no framework, no extra dependency, nothing to install or keep running. One user, one machine, loopback only; a framework would buy nothing until MemDex serves remotely.

## Connecting Claude Code

Two ways today (the Health tab prints both, filled in with your own paths):

1. **A line in your `CLAUDE.md`** — Claude Code runs the CLI through its shell when a task needs the rules:
   `memdex rules "<the task>"` / `memdex search "<question>" --json`
2. **A `UserPromptSubmit` hook** — `memdex rules --hook` reads Claude Code's JSON on stdin and prints the matching rules, which Claude Code injects into context on every prompt. It never blocks a prompt: on any error it prints nothing and exits 0.

An MCP server (`memdex mcp`), which also works with Claude Desktop, is Milestone 6.

## Configuration

| File | Holds | In git |
|---|---|---|
| `.env` (from `.env.example`) | `MEMDEX_CONFIG`, `MEMDEX_INDEX_DIR`, `MEMDEX_MODEL_DIR`, `MEMDEX_DATA_DIR`, `MEMDEX_MAX_CHARS` | No |
| `memdex.yaml` (from `memdex.example.yaml`) | Optional sources: folders, file types, include/exclude, collection routing | No |
| `data/sources/<project>/` | Files uploaded through the UI — one folder per project, nested folders allowed | No |
| `data/logs/queries.jsonl` | The query log | No |
| `data/eval/questions.yaml` | Your eval set (template: `eval.example.yaml`) | No |

Precedence for settings: real environment variables > `.env` > `memdex.yaml` > built-in defaults.

> **`.env` is read from the folder you run `memdex` in.** From elsewhere, pass `--config` and set what you need explicitly — a stray `.env` in the current folder overrides the config file you named.

## How it works

1. **Discover** — walk each source folder (and every UI project); keep files matching `types` / `include` / `exclude`.
2. **Chunk** — split Markdown at H1–H3 headings; pack each section into chunks of at most `max_chars`, cutting only between paragraphs, tables and code blocks. Every chunk carries its heading path (`file > Part 1 > 1.3`), source line range, project and collection.
3. **Embed** — `all-MiniLM-L6-v2` (ONNX, 384 dims) runs locally. Chunks longer than the model's 256-token window are counted and reported; the default `max_chars` of 700 keeps that near zero for Markdown.
4. **Store** — one embedded Chroma collection (cosine); MemDex's collection and project are metadata fields, so scores stay comparable and any filter is a `where` clause.
5. **Retrieve** — the question is embedded the same way; the nearest chunks come back with file, heading path and line range as the citation. Files marked `SUPERSEDED` are excluded unless asked for.

## Measuring retrieval

Retrieval quality is the thing worth measuring, so it ships with an eval set rather than a claim:

```powershell
Copy-Item eval.example.yaml data\eval\questions.yaml   # then write questions about YOUR docs
memdex eval
```

Each question names the file (and optionally a heading) where a good answer lives; the report gives recall@k, MRR and the top 3 hits for every miss. Changing the chunk size or the embedding model is then a measurement, not a guess.

## Tests

```powershell
pytest     # 38 tests, no network, no model download (a fake embedder stands in)
```

## Licence

Not yet chosen.
