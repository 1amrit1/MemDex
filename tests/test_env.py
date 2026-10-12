import pytest

from memdex.config import ConfigError, load_config
from memdex.env import load_env, parse_env

ENV_KEYS = ("MEMDEX_CONFIG", "MEMDEX_INDEX_DIR", "MEMDEX_MODEL_DIR", "MEMDEX_MAX_CHARS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_parse_env():
    values = parse_env(
        """
# comment
MEMDEX_MAX_CHARS=700
export MEMDEX_INDEX_DIR=".index"
MEMDEX_MODEL_DIR='models'
WITH_COMMENT=abc  # trailing note
not a pair
"""
    )
    assert values == {
        "MEMDEX_MAX_CHARS": "700",
        "MEMDEX_INDEX_DIR": ".index",
        "MEMDEX_MODEL_DIR": "models",
        "WITH_COMMENT": "abc",
    }


def test_existing_environment_wins_over_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MEMDEX_MAX_CHARS=700\n", encoding="utf-8")
    monkeypatch.setenv("MEMDEX_MAX_CHARS", "800")
    load_env(env)
    import os

    assert os.environ["MEMDEX_MAX_CHARS"] == "800"


def _yaml(tmp_path, extra=""):
    (tmp_path / "docs").mkdir(exist_ok=True)
    cfg = tmp_path / "memdex.yaml"
    cfg.write_text(f"{extra}sources:\n  - path: docs\n", encoding="utf-8")
    return cfg


def test_precedence_env_over_yaml_over_default(tmp_path, monkeypatch):
    assert load_config(_yaml(tmp_path)).max_chars == 700
    assert load_config(_yaml(tmp_path, "max_chars: 900\n")).max_chars == 900
    monkeypatch.setenv("MEMDEX_MAX_CHARS", "650")
    monkeypatch.setenv("MEMDEX_INDEX_DIR", "idx")
    cfg = load_config(_yaml(tmp_path, "max_chars: 900\n"))
    assert cfg.max_chars == 650
    assert cfg.index_dir == (tmp_path / "idx").resolve()


def test_bad_max_chars(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMDEX_MAX_CHARS", "seven hundred")
    with pytest.raises(ConfigError):
        load_config(_yaml(tmp_path))
    monkeypatch.setenv("MEMDEX_MAX_CHARS", "50")
    with pytest.raises(ConfigError):
        load_config(_yaml(tmp_path))
