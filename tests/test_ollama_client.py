"""ollama_client unit tests — real HTTP against the stub, never a real model."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from mathscrambler import ollama_client, ollama_server
from mathscrambler.config import ProfileRoles, RoleConfig
from mathscrambler.ollama_client import (
    ChatResult,
    OllamaClient,
    StructuredCallError,
    TimingRecord,
    think_param,
)

REASONER = RoleConfig(tag="gpt-oss:test", num_ctx=16384, temperature=0.3, reasoning_effort="high")
VISION = RoleConfig(tag="qwen-test:v", num_ctx=8192, temperature=0.0, think=False)
ROLES = ProfileRoles(vision=VISION, reasoner=REASONER, fast="vision")


class Answer(BaseModel):
    value: int
    unit: str


def _chat_body(content: str, thinking: str | None = None, **counters: int) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    return {
        "model": "stub",
        "message": message,
        "done": True,
        "total_duration": counters.get("total", 2_000_000_000),
        "load_duration": counters.get("load", 500_000_000),
        "prompt_eval_count": counters.get("prompt_tokens", 12),
        "prompt_eval_duration": 100_000_000,
        "eval_count": counters.get("eval_tokens", 40),
        "eval_duration": counters.get("eval_ns", 1_000_000_000),
    }


def _client(http_stub) -> OllamaClient:
    return OllamaClient(f"http://127.0.0.1:{http_stub.port}", ROLES, keep_alive="30m")


# ------------------------------------------------------------------ family adapter


def test_think_param_maps_families():
    assert think_param(REASONER) == "high"  # gpt-oss: effort level string
    assert think_param(VISION) is False  # qwen: boolean
    assert think_param(RoleConfig(tag="plain")) is None  # unset: omit the field


def test_timing_record_tokens_per_s():
    t = TimingRecord.from_response(_chat_body("x", eval_tokens=50, eval_ns=2_000_000_000))
    assert t.eval_tokens == 50
    assert t.tokens_per_s == pytest.approx(25.0)
    assert t.load_s == pytest.approx(0.5)
    zero = TimingRecord.from_response({"model": "m"})
    assert zero.tokens_per_s == 0.0


# ------------------------------------------------------------------ chat payload


async def test_chat_pins_role_options_and_think(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body("hello", thinking="hmm"))
    async with _client(http_stub) as client:
        result = await client.chat("reasoner", [{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResult)
    assert result.content == "hello"
    assert result.thinking == "hmm"
    _, payload = http_stub.posts[0]
    assert payload["model"] == "gpt-oss:test"
    assert payload["options"] == {"num_ctx": 16384, "temperature": 0.3}
    assert payload["think"] == "high"
    assert payload["keep_alive"] == "30m"
    assert payload["stream"] is False


async def test_chat_keep_alive_override_and_alias_role(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body("ok"))
    async with _client(http_stub) as client:
        await client.chat("fast", [{"role": "user", "content": "q"}], keep_alive=0)
    _, payload = http_stub.posts[0]
    assert payload["model"] == "qwen-test:v"  # fast aliases to vision
    assert payload["keep_alive"] == 0
    assert payload["think"] is False


async def test_chat_attaches_images_to_last_user_message(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body("seen"))
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "look at this"},
    ]
    async with _client(http_stub) as client:
        await client.chat("vision", msgs, images=["b64data"])
    _, payload = http_stub.posts[0]
    assert payload["messages"][2]["images"] == ["b64data"]
    assert "images" not in payload["messages"][0]
    assert "images" not in msgs[2], "caller's message list must not be mutated"


async def test_chat_streaming_collects_tokens(tmp_home: Path, http_stub):
    http_stub.post_route(
        "/api/chat",
        [
            {"message": {"content": "he"}, "done": False},
            {"message": {"content": "llo"}, "done": False},
            _chat_body(""),
        ],
    )
    seen: list[str] = []
    async with _client(http_stub) as client:
        result = await client.chat("vision", [{"role": "user", "content": "q"}], on_token=seen.append)
    assert seen == ["he", "llo"]
    assert result.content == "hello"
    assert http_stub.posts[0][1]["stream"] is True


async def test_chat_400_on_think_retries_without_and_remembers(tmp_home: Path, http_stub):
    http_stub.post_route(
        "/api/chat",
        (400, {"error": 'model does not support the "think" option'}),
        _chat_body("worked"),
        _chat_body("again"),
    )
    async with _client(http_stub) as client:
        result = await client.chat("reasoner", [{"role": "user", "content": "q"}])
        assert result.content == "worked"
        assert "think" not in http_stub.posts[1][1]
        assert "gpt-oss:test" in ollama_server.read_state()["think_unsupported"]
        await client.chat("reasoner", [{"role": "user", "content": "q2"}])
    assert "think" not in http_stub.posts[2][1], "remembered: think omitted on later calls"


async def test_chat_400_on_bad_think_value_heals_without_poisoning(tmp_home: Path, http_stub):
    """A bad-VALUE 400 heals this call but must not persist think_unsupported."""
    http_stub.post_route(
        "/api/chat",
        (400, {"error": 'invalid think value: "high"'}),
        _chat_body("healed"),
        (400, {"error": 'invalid think value: "high"'}),
        _chat_body("healed again"),
    )
    async with _client(http_stub) as client:
        result = await client.chat("reasoner", [{"role": "user", "content": "q"}])
        assert result.content == "healed"
        assert "think_unsupported" not in ollama_server.read_state()
        await client.chat("reasoner", [{"role": "user", "content": "q2"}])
    assert http_stub.posts[2][1]["think"] == "high", "not poisoned: think still sent on later calls"


async def test_chat_streaming_400_self_heals(tmp_home: Path, http_stub):
    """A streamed HTTP error must surface through the normal handlers (not
    ResponseNotRead), so the think self-heal works for streaming too."""
    http_stub.post_route(
        "/api/chat",
        (400, {"error": 'model does not support the "think" option'}),
        [
            {"message": {"content": "ok"}, "done": False},
            _chat_body(""),
        ],
    )
    seen: list[str] = []
    async with _client(http_stub) as client:
        result = await client.chat("reasoner", [{"role": "user", "content": "q"}], on_token=seen.append)
    assert result.content == "ok"
    assert seen == ["ok"]
    assert "think" not in http_stub.posts[1][1]
    assert "gpt-oss:test" in ollama_server.read_state()["think_unsupported"]


async def test_chat_stream_without_done_event_raises(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", [{"message": {"content": "partial"}, "done": False}])
    async with _client(http_stub) as client:
        with pytest.raises(ollama_client.OllamaClientError, match="without a done event"):
            await client.chat("vision", [{"role": "user", "content": "q"}], on_token=lambda _: None)


async def test_chat_heal_retry_failure_reports_both_bodies(tmp_home: Path, http_stub):
    http_stub.post_route(
        "/api/chat",
        (400, {"error": 'model does not support the "think" option'}),
        (500, {"error": "model runner has unexpectedly stopped"}),
    )
    async with _client(http_stub) as client:
        with pytest.raises(ollama_client.OllamaClientError) as exc:
            await client.chat("reasoner", [{"role": "user", "content": "q"}])
    msg = str(exc.value)
    assert "500" in msg and "unexpectedly stopped" in msg, "the retry's own failure must be shown"
    assert "original 400" in msg


async def test_chat_unreachable_raises_client_error(tmp_home: Path):
    client = OllamaClient("http://127.0.0.1:9", ROLES)
    with pytest.raises(ollama_client.OllamaClientError, match="unreachable"):
        await client.chat("vision", [{"role": "user", "content": "q"}])
    await client.aclose()


# ------------------------------------------------------------------ structured


async def test_structured_direct_success_caches_direct(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body('{"value": 42, "unit": "kg"}'))
    async with _client(http_stub) as client:
        result = await client.structured("reasoner", [{"role": "user", "content": "q"}], Answer)
    assert result.value == Answer(value=42, unit="kg")
    assert result.mode == "direct"
    assert len(result.timings) == 1
    _, payload = http_stub.posts[0]
    assert payload["format"]["required"] == ["value", "unit"]  # pydantic schema passed through
    assert ollama_server.read_state()["structured_mode"]["gpt-oss:test"] == "direct"


async def test_structured_retry_feeds_error_back(tmp_home: Path, http_stub):
    http_stub.post_route(
        "/api/chat",
        _chat_body("not json at all"),
        _chat_body('{"value": "not-an-int", "unit": "kg"}'),
        _chat_body('{"value": 7, "unit": "m"}'),
    )
    async with _client(http_stub) as client:
        result = await client.structured("reasoner", [{"role": "user", "content": "q"}], Answer)
    assert result.value.value == 7
    assert len(http_stub.posts) == 3
    second = http_stub.posts[1][1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "not json at all"}
    assert "not valid" in second[-1]["content"]
    third = http_stub.posts[2][1]["messages"]
    assert len(third) == len(second) + 2, "each retry appends the bad reply + feedback"


async def test_structured_falls_back_to_two_call_and_caches(tmp_home: Path, http_stub):
    # 3 direct attempts fail; then free-form succeeds; then extraction succeeds.
    http_stub.post_route(
        "/api/chat",
        _chat_body("junk"),
        _chat_body("junk"),
        _chat_body("junk"),
        _chat_body("The answer works out to 42 kilograms."),
        _chat_body('{"value": 42, "unit": "kg"}'),
    )
    async with _client(http_stub) as client:
        result = await client.structured("reasoner", [{"role": "user", "content": "q"}], Answer)
    assert result.mode == "two_call"
    assert result.value.value == 42
    assert len(http_stub.posts) == 5
    free_form = http_stub.posts[3][1]
    assert "format" not in free_form, "reasoning call must be unconstrained"
    extraction = http_stub.posts[4][1]
    assert "42 kilograms" in extraction["messages"][0]["content"]
    assert extraction["think"] == "low", "effort-level family extracts at low effort, never think=false"
    assert ollama_server.read_state()["structured_mode"]["gpt-oss:test"] == "two_call"


async def test_structured_cached_two_call_skips_direct(tmp_home: Path, http_stub):
    ollama_server.write_state({"structured_mode": {"qwen-test:v": "two_call"}})
    http_stub.post_route(
        "/api/chat",
        _chat_body("It is 5 meters."),
        _chat_body('{"value": 5, "unit": "m"}'),
    )
    async with _client(http_stub) as client:
        result = await client.structured("vision", [{"role": "user", "content": "q"}], Answer)
    assert result.mode == "two_call"
    assert "format" not in http_stub.posts[0][1], "cached two_call: first call is free-form"
    assert http_stub.posts[1][1]["think"] is False, "boolean family disables thinking on extraction"


async def test_structured_exhausted_raises_with_all_errors(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body("garbage forever"))
    async with _client(http_stub) as client:
        with pytest.raises(StructuredCallError) as exc:
            await client.structured("reasoner", [{"role": "user", "content": "q"}], Answer, max_retries=2)
    # 2 direct + two-call fallback (1 free-form that "succeeds" + 2 extraction failures)
    assert len(exc.value.attempts) == 4
    assert exc.value.attempts[0].startswith("direct:")
    assert exc.value.attempts[-1].startswith("extract:")
    msg = str(exc.value)
    assert "gpt-oss:test" in msg
    assert "5 model call(s)" in msg
    assert "direct:" in msg, "the root-cause phase must be visible even when errors are elided"


async def test_structured_retries_keep_images(tmp_home: Path, http_stub):
    """A retry that can't see the image would let the model fabricate schema-valid
    values — images must ride along on every attempt."""
    http_stub.post_route(
        "/api/chat",
        _chat_body("not json"),
        _chat_body('{"value": 3, "unit": "cm"}'),
    )
    async with _client(http_stub) as client:
        await client.structured(
            "vision", [{"role": "user", "content": "read the image"}], Answer, images=["b64img"]
        )
    for _, payload in http_stub.posts:
        attached = [m for m in payload["messages"] if m.get("images") == ["b64img"]]
        assert attached, "every attempt's payload must carry the image"


# ------------------------------------------------------------------ preload


async def test_preload_pins_num_ctx_with_empty_messages(tmp_home: Path, http_stub):
    http_stub.post_route("/api/chat", _chat_body(""))
    async with _client(http_stub) as client:
        timing = await client.preload("reasoner", keep_alive="30m")
    _, payload = http_stub.posts[0]
    assert payload["messages"] == []
    assert payload["options"] == {"num_ctx": 16384}
    assert timing.load_s > 0


def test_state_helpers_roundtrip(tmp_home: Path):
    assert ollama_client._cached_structured_mode("x") is None
    ollama_client._cache_structured_mode("x", "two_call")
    assert ollama_client._cached_structured_mode("x") == "two_call"
    # garbage in state.json never crashes the cache readers
    ollama_server.write_state({"structured_mode": "oops", "think_unsupported": 3})
    assert ollama_client._cached_structured_mode("x") is None
    assert ollama_client._think_unsupported("x") is False


def test_stub_ndjson_framing(http_stub):
    """The stub's list responses must really be newline-delimited JSON objects."""
    http_stub.post_route("/p", [{"a": 1}, {"b": 2}])
    resp = httpx.post(f"http://127.0.0.1:{http_stub.port}/p", json={})
    assert resp.headers["content-type"] == "application/x-ndjson"
    lines = resp.text.splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]
    assert resp.text.endswith("\n")
