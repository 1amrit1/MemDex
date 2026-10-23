"""Shared test helpers: a deterministic fake embedder (no model download, no ONNX) and a temp config."""

from __future__ import annotations

import pytest

from memdex.config import load_config

# The class itself lives in its own module so a child process can import it too (test_cli_encoding.py).
from fake_embedder import DIMS, FakeEmbedder  # noqa: F401  (re-exported for tests that import it from here)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("MEMDEX_CONFIG", "MEMDEX_INDEX_DIR", "MEMDEX_MODEL_DIR", "MEMDEX_MAX_CHARS", "MEMDEX_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def tmp_cfg(tmp_path):
    """A config with no YAML sources: projects created under data/sources are the only inputs."""
    yaml_path = tmp_path / "memdex.yaml"
    yaml_path.write_text("index_dir: .index\nmodel_dir: models\ndata_dir: data\n", encoding="utf-8")
    return yaml_path, lambda: load_config(yaml_path)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
