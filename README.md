# MemDex

A local, offline index over your own docs — and, later, your code — so an AI assistant (Claude Code today, a local model later) can **search** them and pull the **rules** that apply to a task, instead of loading every file into context.

- **Fully local.** Embeddings are computed on your machine; the index is a file on your disk.
- **Offline by construction.** Every command except `fetch-model` runs with Python networking blocked — an accidental network call fails loudly instead of happening quietly.
- **Config-driven.** Point it at any folders; route files into collections (`rules`, `status`, `projects`, `references`, `archive`) by path pattern.

## Status

Milestone 1 — Markdown ingest + `stats`. Search and rules retrieval arrive in Milestone 2.

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

Copy-Item .env.example .env                 # settings: index/model folders, chunk size
Copy-Item memdex.example.yaml memdex.yaml   # sources: edit the folder paths
memdex fetch-model                          # one-time download (~90 MB), the only network step
memdex ingest                               # offline
memdex stats                                # offline
```

## Configuration

| File | Holds | In git |
|---|---|---|
| `.env` (from `.env.example`) | `MEMDEX_CONFIG`, `MEMDEX_INDEX_DIR`, `MEMDEX_MODEL_DIR`, `MEMDEX_MAX_CHARS` | No |
| `memdex.yaml` (from `memdex.example.yaml`) | Sources: folders, file types, include/exclude, collection routing | No |

Precedence for settings: real environment variables > `.env` > `memdex.yaml` > built-in defaults. `.env` is read from the folder you run `memdex` in.

## How it works

1. **Discover** — walk each source folder; keep files matching `types` / `include` / `exclude`.
2. **Chunk** — split Markdown at H1–H3 headings; pack each section into chunks of at most `max_chars`, cutting only between paragraphs, tables and code blocks. Every chunk carries its heading path (`file > Part 1 > 1.3`) and source line range.
3. **Embed** — `all-MiniLM-L6-v2` (ONNX, 384 dims) runs locally. Chunks longer than the model's 256-token window are counted and reported; the default `max_chars` of 700 keeps that near zero for Markdown.
4. **Store** — one embedded Chroma collection (cosine); the MemDex collection is a metadata field, so scores stay comparable across collections.

## Tests

```powershell
pytest
```

## Licence

Not yet chosen.
