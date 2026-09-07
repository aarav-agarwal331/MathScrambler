"""Async client for the (private) Ollama server's native API.

One client, three roles. Every call pins the role's num_ctx and temperature in
``options`` — Ollama reloads a model's runner whenever num_ctx changes, so a
role's context size must be identical on every call (PLAN "Live finding").

Structured outputs use ``format`` = a JSON schema. Some model families degrade
when schema-constrained decoding runs while thinking is enabled (PLAN decision
5); the client handles this with a per-tag mode cached in state.json:

- ``direct``:   one call, schema + thinking together (tried first);
- ``two_call``: a free-form reasoning call, then a schema-constrained
  extraction call with thinking off.

A direct-mode tag that exhausts its retries falls back to two_call within the
same invocation and the switch is remembered. Validation errors are fed back
to the model verbatim, at most ``max_retries`` attempts per phase.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from mathscrambler import ollama_server
from mathscrambler.config import ProfileRoles, Role, RoleConfig

log = logging.getLogger("mathscrambler.ollama_client")

CONNECT_TIMEOUT_S = 10.0
DEFAULT_READ_TIMEOUT_S = 900.0  # cold-loading a 65 GB model under load takes minutes
DEFAULT_MAX_RETRIES = 3

M = TypeVar("M", bound=BaseModel)


class OllamaClientError(RuntimeError):
    pass


class StructuredCallError(OllamaClientError):
    """All attempts (including any two-call fallback) failed; carries every error."""

    def __init__(self, tag: str, attempts: list[str], calls: int | None = None):
        self.attempts = attempts
        # Show both ends: the first error is usually the root cause that forced a
        # mode fallback; the last ones are what finally gave up.
        shown = attempts if len(attempts) <= 3 else [attempts[0], "…", *attempts[-2:]]
        calls_note = f" over {calls} model call(s)" if calls else ""
        super().__init__(
            f"{tag}: no valid structured output after {len(attempts)} failed attempt(s)"
            f"{calls_note}: {'; '.join(shown)}"
        )


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class TimingRecord:
    """Server-side timings from the response's eval counters (nanoseconds → seconds)."""

    model: str
    total_s: float
    load_s: float
    prompt_tokens: int
    prompt_eval_s: float
    eval_tokens: int
    eval_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.eval_tokens / self.eval_s if self.eval_s > 0 else 0.0

    @classmethod
    def from_response(cls, data: dict) -> TimingRecord:
        ns = 1e9

        def num(key: str) -> float:
            value = data.get(key, 0)
            return float(value) if isinstance(value, int | float) else 0.0

        return cls(
            model=str(data.get("model", "?")),
            total_s=num("total_duration") / ns,
            load_s=num("load_duration") / ns,
            prompt_tokens=int(num("prompt_eval_count")),
            prompt_eval_s=num("prompt_eval_duration") / ns,
            eval_tokens=int(num("eval_count")),
            eval_s=num("eval_duration") / ns,
        )


@dataclass(frozen=True)
class ChatResult:
    content: str
    thinking: str | None
    timing: TimingRecord


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    mode: str  # "direct" | "two_call"
    timings: list[TimingRecord] = field(default_factory=list)


# --------------------------------------------------------------------------- family adapter


class _Unset:
    pass


_UNSET = _Unset()


def think_param(rc: RoleConfig) -> bool | str | None:
    """Map a role's config onto Ollama's ``think`` field, per model family.

    gpt-oss takes effort levels (``think: "low"|"medium"|"high"``, configured as
    reasoning_effort); qwen and friends take a boolean. None = omit the field
    entirely (let the model default).
    """
    if rc.reasoning_effort is not None:
        return rc.reasoning_effort
    return rc.think


def _cached_structured_mode(tag: str) -> str | None:
    mode = ollama_server.read_state().get("structured_mode", {})
    value = mode.get(tag) if isinstance(mode, dict) else None
    return value if value in ("direct", "two_call") else None


def _cache_structured_mode(tag: str, mode: str) -> None:
    modes = ollama_server.read_state().get("structured_mode", {})
    if not isinstance(modes, dict):
        modes = {}
    modes[tag] = mode
    ollama_server.write_state({"structured_mode": modes})


def _think_unsupported(tag: str) -> bool:
    tags = ollama_server.read_state().get("think_unsupported", [])
    return isinstance(tags, list) and tag in tags


def _mark_think_unsupported(tag: str) -> None:
    tags = ollama_server.read_state().get("think_unsupported", [])
    if not isinstance(tags, list):
        tags = []
    if tag not in tags:
        tags.append(tag)
    ollama_server.write_state({"think_unsupported": tags})


# --------------------------------------------------------------------------- client


RETRY_FEEDBACK = (
    "Your previous reply was not valid for the required JSON schema.\n"
    "Error: {error}\n"
    "Reply again with ONLY a JSON object matching the schema — no prose, no code fences."
)

EXTRACT_PROMPT = (
    "Below is a worked solution. Extract the final result as a JSON object matching the "
    "required schema. Use only what the solution states; do not re-solve.\n\n"
    "--- solution ---\n{solution}"
)


class OllamaClient:
    """Async wrapper over the native API for one server (private by default)."""

    def __init__(
        self,
        base_url: str,
        roles: ProfileRoles,
        keep_alive: str = "30m",
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.roles = roles
        self.keep_alive = keep_alive
        # trust_env=False: loopback traffic must never route through a user proxy.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(CONNECT_TIMEOUT_S, read=read_timeout_s, write=30.0, pool=30.0),
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ raw chat

    async def chat(
        self,
        role: Role,
        messages: Sequence[dict[str, Any]],
        *,
        schema: dict | None = None,
        images: Sequence[str] | None = None,
        keep_alive: str | int | None = None,
        think: bool | str | _Unset | None = _UNSET,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """One /api/chat call for `role`. `images` are base64 strings attached to the
        last user message. `on_token` switches to streaming and receives content deltas."""
        rc = self.roles.resolve(role)
        msgs = [dict(m) for m in messages]
        if images:
            for m in reversed(msgs):
                if m.get("role") == "user":
                    m["images"] = list(images)
                    break
            else:
                raise OllamaClientError("images given but no user message to attach them to")
        payload: dict[str, Any] = {
            "model": rc.tag,
            "messages": msgs,
            "stream": on_token is not None,
            "options": {"num_ctx": rc.num_ctx, "temperature": rc.temperature},
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        if schema is not None:
            payload["format"] = schema
        effective_think = think_param(rc) if isinstance(think, _Unset) else think
        if effective_think is not None and not _think_unsupported(rc.tag):
            payload["think"] = effective_think
        try:
            return await self._post_chat(payload, on_token)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500]
            if e.response.status_code == 400 and "think" in payload and "think" in body.lower():
                # Family adapter self-heal: retry once without the think field. Only
                # a body saying the FIELD is unsupported marks the tag persistently —
                # a 400 about a bad think VALUE also heals by dropping the field, but
                # that's a config problem to surface, not a tag capability to cache.
                payload.pop("think")
                try:
                    result = await self._post_chat(payload, on_token)
                except httpx.HTTPStatusError as e2:
                    raise OllamaClientError(
                        f"{rc.tag}: retry without think failed: HTTP {e2.response.status_code}: "
                        f"{e2.response.text[:300]} (original 400: {body})"
                    ) from e2
                if "does not support" in body.lower():
                    _mark_think_unsupported(rc.tag)
                else:
                    log.warning(
                        "%s rejected think=%r (%s); healed this call without persisting — "
                        "check the role's think/reasoning_effort in config.toml",
                        rc.tag,
                        effective_think,
                        body,
                    )
                return result
            raise OllamaClientError(f"{rc.tag}: HTTP {e.response.status_code}: {body}") from e
        except httpx.HTTPError as e:
            raise OllamaClientError(f"{rc.tag}: {self.base_url} unreachable: {e}") from e

    async def _post_chat(self, payload: dict, on_token: Callable[[str], None] | None) -> ChatResult:
        if on_token is None:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise OllamaClientError(f"{payload['model']}: non-object /api/chat response")
            message = data.get("message") or {}
            return ChatResult(
                content=str(message.get("content", "")),
                thinking=message.get("thinking") or None,
                timing=TimingRecord.from_response(data),
            )
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        final: dict = {}
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.is_error:
                # Buffer the error body while the stream is open: without this,
                # e.response.text in chat()'s handlers raises ResponseNotRead and
                # the think self-heal is unreachable for streamed calls.
                await resp.aread()
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError as e:
                    raise OllamaClientError(f"unparseable stream line: {line[:120]!r}") from e
                if event.get("error"):
                    raise OllamaClientError(f"{payload['model']}: {event['error']}")
                message = event.get("message") or {}
                if message.get("thinking"):
                    thinking_parts.append(message["thinking"])
                if message.get("content"):
                    content_parts.append(message["content"])
                    on_token(message["content"])
                if event.get("done"):
                    final = event
        if not final:
            # A cleanly-closed body without a done event is a truncated generation,
            # not a success — partial content must never masquerade as complete.
            raise OllamaClientError(
                f"{payload['model']}: stream ended without a done event "
                f"({len(content_parts)} content chunk(s) received)"
            )
        return ChatResult(
            content="".join(content_parts),
            thinking="".join(thinking_parts) or None,
            timing=TimingRecord.from_response(final | {"model": payload["model"]}),
        )

    # ------------------------------------------------------------------ structured

    async def structured(
        self,
        role: Role,
        messages: Sequence[dict[str, Any]],
        model_cls: type[M],
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        images: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> StructuredResult[M]:
        """A validated `model_cls` instance from `role`, or StructuredCallError.

        Tries the tag's cached mode (default: direct schema+thinking). Retries
        feed the validation error back. A direct-mode failure falls back to the
        two-call pattern once and caches the switch.
        """
        rc = self.roles.resolve(role)
        schema = model_cls.model_json_schema()
        mode = _cached_structured_mode(rc.tag) or "direct"
        errors: list[str] = []
        timings: list[TimingRecord] = []

        if mode == "direct":
            value = await self._structured_attempts(
                role, list(messages), model_cls, schema, max_retries, errors, timings,
                images=images, seed=seed, think=_UNSET, phase="direct",
            )
            if value is not None:
                if _cached_structured_mode(rc.tag) is None:
                    _cache_structured_mode(rc.tag, "direct")
                return StructuredResult(value=value, mode="direct", timings=timings)

        value = await self._structured_two_call(
            role, list(messages), model_cls, schema, max_retries, errors, timings,
            images=images, seed=seed,
        )
        if value is not None:
            if mode == "direct":  # fell back and it worked: remember for this tag
                _cache_structured_mode(rc.tag, "two_call")
            return StructuredResult(value=value, mode="two_call", timings=timings)
        raise StructuredCallError(rc.tag, errors, calls=len(timings))

    async def _structured_attempts(
        self,
        role: Role,
        msgs: list[dict[str, Any]],
        model_cls: type[M],
        schema: dict,
        max_retries: int,
        errors: list[str],
        timings: list[TimingRecord],
        *,
        images: Sequence[str] | None,
        seed: int | None,
        think: bool | str | _Unset | None,
        phase: str,
    ) -> M | None:
        for _attempt in range(max_retries):
            # Images ride along on EVERY attempt (chat attaches them to the latest
            # user message) — a retry that can't see the image would let the model
            # fabricate schema-valid values it can no longer ground.
            result = await self.chat(role, msgs, schema=schema, images=images, seed=seed, think=think)
            timings.append(result.timing)
            try:
                return model_cls.model_validate_json(result.content)
            except ValidationError as e:
                first = e.errors()[0].get("msg", "invalid")
                errors.append(f"{phase}: validation: {first} ({e.error_count()} error(s))")
            except ValueError:
                errors.append(f"{phase}: not JSON: {result.content[:80]!r}")
            msgs = [
                *msgs,
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": RETRY_FEEDBACK.format(error=errors[-1])},
            ]
        return None

    async def _structured_two_call(
        self,
        role: Role,
        msgs: list[dict[str, Any]],
        model_cls: type[M],
        schema: dict,
        max_retries: int,
        errors: list[str],
        timings: list[TimingRecord],
        *,
        images: Sequence[str] | None,
        seed: int | None,
    ) -> M | None:
        # Call 1: free-form, thinking as configured — the model reasons in prose.
        free = await self.chat(role, msgs, images=images, seed=seed)
        timings.append(free.timing)
        if not free.content.strip():
            errors.append("two_call: free-form reasoning call returned empty content")
            return None
        # Call 2..n: schema-constrained extraction with minimal thinking (the
        # conflict this mode avoids), retrying on validation errors as usual.
        # Effort-level families (gpt-oss) can't disable thinking — use "low".
        rc = self.roles.resolve(role)
        extract_think: bool | str = "low" if rc.reasoning_effort is not None else False
        extract_msgs: list[dict[str, Any]] = [
            {"role": "user", "content": EXTRACT_PROMPT.format(solution=free.content)}
        ]
        return await self._structured_attempts(
            role, extract_msgs, model_cls, schema, max_retries, errors, timings,
            images=None, seed=seed, think=extract_think, phase="extract",
        )

    # ------------------------------------------------------------------ load management

    async def preload(self, role: Role, keep_alive: str | int | None = None) -> TimingRecord:
        """Load a role's model (empty message list) with its pinned num_ctx.

        Callers loading several roles should preload largest-first: the 0.30
        scheduler admits against current free memory, so the big model must
        claim its slice before the small ones fill the gap (PLAN "Live finding").
        """
        rc = self.roles.resolve(role)
        payload = {
            "model": rc.tag,
            "messages": [],
            "stream": False,
            "options": {"num_ctx": rc.num_ctx},
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
        }
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise OllamaClientError(f"preload {rc.tag} failed: {e}") from e
        return TimingRecord.from_response(data if isinstance(data, dict) else {"model": rc.tag})
