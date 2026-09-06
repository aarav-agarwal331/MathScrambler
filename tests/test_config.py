from __future__ import annotations

from pathlib import Path

import pytest

from mathscrambler import paths
from mathscrambler.config import ConfigError, load_config, parse_model_overrides

EXAMPLE = paths.config_example_path()


def test_example_config_loads_with_expected_defaults():
    cfg = load_config(EXAMPLE)
    assert cfg.profile == "default"
    assert cfg.ollama.mode == "private"
    assert cfg.ollama.port == 11435
    assert cfg.ollama.global_port == 11434
    assert cfg.ollama.max_loaded_models == 3
    assert cfg.dashboard.port == 8765
    assert cfg.sampling.max_samples == 2000
    assert cfg.memory.max_fraction == 0.85


def test_fast_role_aliases_to_vision():
    cfg = load_config(EXAMPLE)
    roles = cfg.active_roles()
    assert roles.resolve("fast").tag == roles.resolve("vision").tag
    assert roles.resolve("reasoner").tag == "gpt-oss:120b"
    assert roles.resolve("reasoner").num_ctx == 16384
    assert roles.resolve("vision").num_ctx == 8192


def test_lite_profile_swaps_reasoner():
    cfg = load_config(EXAMPLE)
    lite = cfg.active_roles(profile="lite")
    # qwen3.6:27b-mlx, not qwen3.8: the 3.8 tag 412s on this machine's Ollama (see PLAN.md)
    assert lite.resolve("reasoner").tag == "qwen3.6:27b-mlx"
    assert lite.resolve("fast").tag == lite.resolve("vision").tag


def test_model_override_wins_and_concretizes_alias():
    cfg = load_config(EXAMPLE)
    roles = cfg.active_roles(model_overrides={"reasoner": "deepseek-r1:70b", "fast": "gemma4"})
    assert roles.resolve("reasoner").tag == "deepseek-r1:70b"
    # overriding an aliased role makes it concrete but keeps the target's other settings
    assert roles.resolve("fast").tag == "gemma4"
    assert roles.resolve("fast").num_ctx == roles.resolve("vision").num_ctx


def test_parse_model_overrides():
    assert parse_model_overrides("reasoner=a:1, vision=b:2") == {"reasoner": "a:1", "vision": "b:2"}
    assert parse_model_overrides(None) == {}
    with pytest.raises(ConfigError):
        parse_model_overrides("nonsense")


def test_unknown_override_role_rejected():
    cfg = load_config(EXAMPLE)
    with pytest.raises(ConfigError, match="unknown role"):
        cfg.active_roles(model_overrides={"writer": "x"})


def test_missing_config_points_at_setup(tmp_path: Path):
    with pytest.raises(ConfigError, match="mathscramble setup"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_is_a_config_error(tmp_path: Path):
    bad = tmp_path / "config.toml"
    bad.write_text("profile = [unclosed")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)


def test_bad_alias_rejected_at_load_time(tmp_path: Path):
    """A typo'd alias must be a load-time ConfigError (doctor shows it red), never a
    use-time crash halfway through a command."""
    bad = tmp_path / "config.toml"
    # anchor to line start: the example's comments also contain the literal string
    bad.write_text(EXAMPLE.read_text().replace('\nfast = "vision"', '\nfast = "visoin"', 1))
    with pytest.raises(ConfigError, match="visoin"):
        load_config(bad)


def test_missing_profile_table_rejected_at_load_time(tmp_path: Path):
    bad = tmp_path / "config.toml"
    bad.write_text(EXAMPLE.read_text().replace('profile = "default"', 'profile = "lite"').replace(
        "[roles.lite]", "[roles.zzz]"
    ))
    with pytest.raises(ConfigError):
        load_config(bad)


def test_config_ignores_poisoned_ollama_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:6666")
    monkeypatch.setenv("OLLAMA_MAX_LOADED_MODELS", "99")
    cfg = load_config(EXAMPLE)
    assert cfg.ollama.port == 11435
    assert cfg.ollama.max_loaded_models == 3
