"""Local embeddings: all-MiniLM-L6-v2 (ONNX, 384 dims), stored inside the project's model_dir.

Reuses Chroma's ONNX runner but points it at our own folder and never lets it auto-download:
`fetch()` is the one and only network step. chromadb is pinned (requirements.txt) because this
relies on that class's internals (DOWNLOAD_PATH, _forward, tokenizer).
"""

from __future__ import annotations

from pathlib import Path

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_TOKENS = 256  # longer text is silently truncated by the model — counted, not hidden
_FILES = ("config.json", "model.onnx", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt")


class ModelMissing(RuntimeError):
    pass


def model_present(model_dir: Path) -> bool:
    """True when every model file is on disk — checked without loading anything."""
    folder = Path(model_dir) / MODEL_NAME / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME
    return all((folder / f).is_file() for f in _FILES)


class Embedder:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / MODEL_NAME
        runner = type("_LocalMiniLM", (ONNXMiniLM_L6_V2,), {"DOWNLOAD_PATH": self.model_path})
        self._onnx = runner()

    def is_present(self) -> bool:
        return model_present(self.model_dir)

    def fetch(self) -> None:
        """Download + checksum-verify the model into model_dir. Needs the network; nothing else does."""
        self._onnx._download_model_if_not_exists()

    def truncated(self, text: str) -> bool:
        return bool(self._onnx.tokenizer.encode(text).overflowing)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.is_present():
            raise ModelMissing(f"embedding model not found in {self.model_path} - run: memdex fetch-model")
        return self._onnx._forward(texts).tolist()
