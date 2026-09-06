"""config.toml loading and validation.

Models are configured only here (Section 0). Config loading never touches the
network and never reads OLLAMA_* environment variables — the private server's
env is scoped at spawn time in ollama_server.py, not inherited from config.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mathscrambler import paths

Role = Literal["vision", "reasoner", "fast"]
ROLES: tuple[Role, ...] = ("vision", "reasoner", "fast")


class ConfigError(RuntimeError):
    pass


class RoleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str
    num_ctx: int = 8192
    temperature: float = 0.2
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    think: bool | None = None


class ProfileRoles(BaseModel):
    """One profile's role table. A role may be a string naming another role (an alias)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vision: RoleConfig | str
    reasoner: RoleConfig | str
    fast: RoleConfig | str

    def resolve(self, role: Role) -> RoleConfig:
        value = getattr(self, role)
        if isinstance(value, RoleConfig):
            return value
        target = value
        if target not in ROLES or target == role:
            raise ConfigError(f"role alias {role} = {target!r} must name a different role")
        resolved = getattr(self, target)
        if not isinstance(resolved, RoleConfig):
            raise ConfigError(f"role alias {role} -> {target} points at another alias; chain not allowed")
        return resolved

    def resolved_tags(self) -> dict[Role, str]:
        return {role: self.resolve(role).tag for role in ROLES}


class OllamaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["private", "shared"] = "private"
    port: int = 11435
    global_port: int = 11434
    keep_alive: str = "30m"
    max_loaded_models: int = 3


class DashboardConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    port: int = 8765
    open_browser: bool = True


class SamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_samples: int = 2000
    relaxed_extra_samples: int = 1000
    keep_nice_answers: bool = True
    reject_degenerate: bool = True
    blueprint_max_retries: int = 3
    verify_max_retries: int = 3


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_s: float = 5.0
    memory_mb: int = 1024


class MemoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_fraction: float = Field(default=0.85, gt=0.0, le=1.0)


class BenchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_variants: int = 2
    candidates: list[str] = Field(default_factory=list)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    profile: Literal["default", "lite"] = "default"
    ollama: OllamaConfig = OllamaConfig()
    dashboard: DashboardConfig = DashboardConfig()
    roles: dict[str, ProfileRoles]
    sampling: SamplingConfig = SamplingConfig()
    sandbox: SandboxConfig = SandboxConfig()
    memory: MemoryConfig = MemoryConfig()
    bench: BenchConfig = BenchConfig()

    def active_roles(
        self,
        profile: str | None = None,
        model_overrides: dict[str, str] | None = None,
    ) -> ProfileRoles:
        """The role table for `profile` (default: config's), with per-role tag overrides applied."""
        name = profile or self.profile
        if name not in self.roles:
            raise ConfigError(f"profile {name!r} not defined in config.toml (have: {sorted(self.roles)})")
        table = self.roles[name]
        if not model_overrides:
            return table
        updates: dict[str, RoleConfig | str] = {}
        for role, tag in model_overrides.items():
            if role not in ROLES:
                raise ConfigError(f"--models: unknown role {role!r} (expected one of {', '.join(ROLES)})")
            base = table.resolve(role)  # an override turns an alias into a concrete role
            updates[role] = base.model_copy(update={"tag": tag})
        return table.model_copy(update=updates)


def parse_model_overrides(spec: str | None) -> dict[str, str]:
    """Parse ``reasoner=gpt-oss:120b,vision=...`` from --models."""
    if not spec:
        return {}
    overrides: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ConfigError(f"--models: expected role=tag, got {part!r}")
        role, _, tag = part.partition("=")
        overrides[role.strip()] = tag.strip()
    return overrides


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or paths.config_path()
    if not cfg_path.is_file():
        raise ConfigError(f"no config file at {cfg_path} — run `mathscramble setup` to create it")
    try:
        raw = tomllib.loads(cfg_path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"invalid TOML in {cfg_path}: {e}") from e
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ConfigError(f"invalid config {cfg_path}: {details}") from e
