"""MemDex command line. Every command runs with the network blocked except `fetch-model`."""

from __future__ import annotations

from typing import Optional

import typer

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
    """Rebuild the index from all configured sources (offline)."""
    block_network()
    from memdex import store
    from memdex.embedder import Embedder, ModelMissing
    from memdex.ingest import run_ingest

    cfg = _config(config)
    try:
        report = run_ingest(cfg, Embedder(cfg.model_dir), store.open_client(cfg.index_dir))
    except ModelMissing as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1)

    typer.echo(f"index:     {cfg.index_dir}")
    typer.echo(f"files:     {report.files}")
    typer.echo(f"chunks:    {report.chunks}  (max_chars {cfg.max_chars}; truncated at 256 tokens: {report.truncated})")
    for name, n in sorted(report.by_collection.items()):
        typer.echo(f"  {name:<12}{n}")
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
