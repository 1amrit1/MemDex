"""A deterministic stand-in for the ONNX embedder, so tests need no model download and no network.

Importable on its own (not just through conftest) because `tests/test_cli_encoding.py` has to hand it to a
child process that runs the real CLI entry point.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

DIMS = 16


class FakeEmbedder:
    """Bag-of-words hashed into a small vector; identical text -> identical vector, shared words -> closer."""

    model_path = Path("fake-model")

    def __init__(self, model_dir=None) -> None:  # accepts Embedder's signature so it can replace it
        self.model_dir = Path(model_dir) if model_dir else Path("fake-model")

    def is_present(self) -> bool:
        return True

    def truncated(self, text: str) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * DIMS
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            v[h % DIMS] += 1.0
            v[(h >> 8) % DIMS] += 0.5
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
