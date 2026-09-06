# MathScrambler — Build Plan

Spec of record: [SPEC.md](SPEC.md) (**v2** — isolated, self-configuring). This file tracks: done / next / decisions made without you. Updated at every phase gate.

**Status: Phase 1 done and live-verified** (scaffold + sysinfo + ollama_server + doctor + setup + server commands + criterion-2 non-interference proof). Model pulls: gpt-oss:120b ✓, qwen3.6:35b-mlx / qwen3.8:27b-mlx in flight. Next: Phase 2 (`ollama_client` + `models`) on your go.

## Live finding that changes an assumption (Phase 1)

The "one resident model" behavior on this machine was **never a config limit**. `OLLAMA_MAX_LOADED_MODELS=3` demonstrably reaches our private server (`ps eww` shows the scoped env), yet a second large model still got evicted — Ollama 0.30's scheduler logs `system_limited=true` and admits models against **current free system memory** (macOS "free" is cache-depressed; with ComfyUI live it was 13–49 GiB), predicting a load without explicit `num_ctx` at **32K context** (40.4 GiB for a 34 GB model). Consequences, all verified live:
- Two models DO co-reside on the private server when each is loaded with its real `num_ctx` and the second fits current free memory (proven: 14b + 32b, 40.4 GiB total resident on 11435 while the global server stayed empty).
- Changing a model's `num_ctx` forces a full runner reload — the engine must keep per-role `num_ctx` stable across calls, and load the big model first, smaller ones alongside.
- This retroactively explains the global server's "limit-1" observation too; the launchctl variable was always unset and stays untouched.
- Criterion-1 evidence: the scoped env is proven from the process itself (`ps eww <pid>`), plus two-model `/api/ps`; doctor reports the env recorded at spawn.

## Verified environment (2026-09-06)

- Global Ollama: Ollama.app server **0.30.7** on 11434; CLI binary **0.30.10** (the private server spawns from the CLI binary → 0.30.10). ≥ 0.19 requirement satisfied.
- Confirmed live: the global server keeps **one** model resident (second load evicts the first) — exactly the situation Section 1's private server exists for. `launchctl getenv OLLAMA_MAX_LOADED_MODELS` is unset and must stay unset.
- Registry (sizes verified): `qwen3.6:35b-mlx` 23.6 GB · `gpt-oss:120b` 65.4 GB · `qwen3.8:27b-mlx` 18.2 GB · `qwen3.6:27b-mlx` 18.8 GB · `qwen3.5:122b` 81.4 GB · `glm-5.3-flash` **404 — does not exist**; bench renders it as a "not in registry" row, never a crash.
- Disk ~980 GB free · RAM 128 GB · uv 0.11.21 · Chrome installed (PDF export path) · ComfyUI present but not currently running (detection still required).

## Your decisions (logged)

- **Pulls approved:** `gpt-oss:120b` only among >30 GB models. `qwen3.5:122b` declined — bench skips it with its pull size shown. Under-30 GB defaults (`qwen3.6:35b-mlx`, `qwen3.8:27b-mlx`) pre-approved by spec; all three are downloading now via the global server's puller (sequential, sole puller — no collision).
- v2 spec supersedes v1: private server on 11435, zero global side effects. Any future >30 GB pull still requires your explicit OK.

## Decisions made without you (flag if you disagree)

1. **`fast` aliases to `vision` by default.** 23.6+65.4+18.2 = 107 GB of weights exceeds Apple Silicon's Metal wired-memory ceiling (~75–80 % of RAM) and the spec's own 85 % rule — three co-resident models would thrash. vision+reasoner ≈ 89 GB fits. `qwen3.8:27b-mlx` stays pulled, in config as a commented alternative, as the `--lite` reasoner, and as a bench candidate. (Spec sanctions this: "make `fast` reuse the `vision` model — that's why it's a role, not a model.")
2. **Shared-mode probe only when 11434 is idle.** The spec's probe (load two tiny models on the global server) would itself evict a resident model on this limit-1 server, violating "never evict another server's models". Resolution: probe only if `/api/ps` on 11434 shows zero residents; otherwise warn and stay `private`. Default mode is `private` anyway.
3. **Constraints ≠ solver code, different execution tiers.** Constraint expressions are AST-whitelist-validated at blueprint receipt (comparisons/arithmetic/boolean/chained-compare; calls only to a closed table: abs, min, max, gcd, lcm, floor, ceil, round, int; no attributes/subscripts/dunders; Pow bounded) and interpreted in-process — 2000 rejection samples in <100 ms. `solver_code` always runs in the subprocess sandbox: `sandbox-exec` seatbelt profile denying network (no sudo needed), `python -I`, import allowlist (sympy/math/fractions/itertools/functools/decimal), restricted builtins, RLIMIT_CPU + parent RSS watchdog, 5 s per-call wall timeout, one warm process per blueprint (first call = the untrusted validation run). Prose pseudo-constraints ("triangle inequality on p4,p5,p6") fail validation with a named error fed back to the reasoner — normalization happens at blueprint time, in the prompt/schema, not via string hacks.
4. **Template rendering never uses `str.format`** — LaTeX braces would explode it. Regex substitution scoped to `{E\d+}`/`{p\d+}`; blueprint acceptance validates slot sets match in both directions and re-renders with original values as a fidelity check.
5. **Structured outputs + thinking, per-tag capability probe.** Known failure class: JSON-schema-constrained decoding degrades or misroutes thinking-model output. Client uses a family adapter (gpt-oss `reasoning_effort`, qwen `think`) and, where a probe shows format+think conflict, a two-call pattern: free-form reasoning call, then a schema-constrained extraction call. Probe results cached in app-support state; run at setup/bench.
6. **Verification designed for honest ≥90 %:** verifier answers arrive in a schema field demanding plain sympy syntax (no LaTeX parsing); layered canonical comparison (int/Fraction exact, `sympy.simplify(a−b)==0` inside the sandbox timeout, order-insensitive sets); "verifier could not solve" is distinguished from "variant wrong" (one fresh re-verify before burning a resample); method match is per-step alignment + technique-family tag with a deterministic pass rule, calibrated on examples/ then frozen. When a problem has no `given_answer`, one independent fresh-context solve of the *original* must match `solver_code` before the blueprint is accepted — the solver never becomes unchecked ground truth.
7. **PID safety = ownership triple + flock.** PID file stores pid + process create-time + binary + port; every signal requires pid-alive ∧ create-time match ∧ cmdline contains `ollama serve` ∧ `/api/version` answers on the recorded port. Anything else = stale file, cleaned, never signaled. Spawn is `start_new_session=True` (own process group) guarded by `flock` so concurrent `ui`+`run` can't double-spawn.
8. **Pull safety:** `/api/ps` does *not* show in-flight pulls — the reliable signal is `blobs/*-partial*` files stat'ed twice ~3 s apart (growing ⇒ someone is pulling ⇒ refuse). All MathScrambler pulls are explicit (`models --pull`/`setup`), serialized behind the same flock. (Today's initial pulls run through the global server as the sole puller, before any private server exists.)
9. **Engine = typed event queue.** Frozen Pydantic `EngineEvent` union with run_id+seq on an `asyncio.Queue`; rich CLI table, SSE, and the SQLite `events` table are three drains of one stream. Generation runs as a background task decoupled from the SSE connection (reconnect via Last-Event-ID; heartbeats; explicit cancel endpoint) so a tab refresh can't kill a 20-minute run.
10. **Seeds:** per-(seed, problem_id, variant_idx) child PRNGs derived by hash — retries on one variant never shift another's draws. LLM calls stay unseeded (spec's letter; retry-with-feedback needs variation).
11. **Grammar pass is discard-not-retry:** ordered lists of math spans (byte-identical) + bare numbers extracted before/after; any difference discards the grammar output and keeps the pre-pass rendering with a logged warning.
12. **PDF via headless Chrome** (`--headless=new --virtual-time-budget --print-to-pdf`, throwaway `--user-data-dir`): KaTeX is JS, Chrome is installed, and weasyprint would drag in Homebrew C libraries (a global-ish mutation Section 0 forbids). No Chrome ⇒ write the HTML, note "open and Cmd-P", non-fatal.
13. **Memory etiquette uses `psutil.virtual_memory().available`** (free+reclaimable — macOS "free" is deliberately tiny), projected footprint = un-resident configured weights × 1.15, attribution names global-Ollama residents (`/api/ps`), ComfyUI, and "other" on refusal, offering `--lite`.
14. **httpx against Ollama's native API** (not the `ollama` pip package): needs format=schema, both `think` shapes, per-request options/keep_alive, base_url injection, and the server's own eval counters for exact tok/s. Interim note: until `gpt-oss:120b` finishes downloading, integration work can point the reasoner role at `deepseek-r1:70b` (already in the store) purely via config — the default in `config.example.toml` is gpt-oss.
15. **Housekeeping:** git repo initialized at Phase 1 (criterion 2 wants clean-tree evidence); all recorded paths `Path.resolve()`d (case-insensitive FS: folder is `Claude_MathScrambler` on disk); `uv tool install -e . --reinstall` at setup so the tool-venv can't drift from pyproject (doctor flags drift); events/answers keep `raw` + `canonical` forms.

## Build order (spec §8) — gates

- [x] **1. Scaffold + isolation layer** — DONE 2026-09-06. 47 unit tests green in 2.3 s, ruff clean; real `setup` ran end-to-end (private server validated on 11435, pull-collision guard fired live against the in-flight gpt-oss pull, KaTeX vendored, ComfyUI detected live); non-interference proof executed: two models co-resident on the private server while global `ollama ps` stayed empty, `server stop` killed only pid 26929 (global pid 5701 untouched), launchctl still unset, shell profiles unchanged, footprint = repo + app-support + `~/.local/bin/{mathscramble,ms}` shims only.
- [ ] **2. ollama_client + models** — httpx wrapper (structured outputs, retries×3 with error fed back, family adapter, timings), capability probe, `models [--pull]` with pull-collision guard. *Verify:* structured call against a pulled model on the private server; `/api/ps` shows MAX_LOADED_MODELS=3 behavior (two small models resident — the thing the global server verifiably cannot do).
- [ ] **3. Ingest** — text/md/tex/json loaders (`---` splitting, input immutability) + vision image path; examples/ authored incl. 3 rendered PNGs. *Verify:* vision extraction on the 3 example images, load-bearing tokens present.
- [ ] **4. Blueprint + sandbox + sampler** — schema validators (slots, constraints AST check, solver-reproduces-original gate), sandbox (hostile-code test suite), constraint evaluator, seeded rejection sampler, entity swap + guard list + grammar diff-gate. *Verify:* unit suite incl. hostile sandbox cases; real blueprint extraction on examples.
- [ ] **5. Verify (Step D)** — fresh-context solve, canonical comparators, method-match rubric, proof checker; calibrate on examples/, then freeze. *Verify:* real verification round-trips; corrupted-outline fixture returns `valid=false`.
- [ ] **6. `run` end-to-end** — engine A→E, outputs/`<timestamp>-<slug>/`, results.json/md, run.log (4K-truncated prompts, timings, server/port), rich live table. *Verify:* `run examples/ -n 3 --seed 42` twice → same parameter draws; ≥90 % pass; wall-time report.
- [ ] **7. Dashboard** — FastAPI+SSE+SQLite, extraction review, controls, status strip, regenerate/blueprint-edit, exports, history; vendored KaTeX; loopback only. *Verify:* the criterion-5 photo flow.
- [ ] **8. bench** — scorecard incl. `--lite` table, skip/404 rows, recommendation paragraph. *Verify:* criterion 4.
- [ ] **9. README + polish** — ≤3-command install, isolation-in-five-sentences, troubleshooting; final acceptance sweep 1–7 + 10 spot-check variants pasted to you.

## examples/ (authored in Phase 3; known exact answers)

1. Sticker sharing, multi-entity (96 stickers → **16**) · 2. CRT remainders (**24**) · 3. Two red marbles w/o replacement (**5/33**, must stay in (0,1)) · 4. ∫₀⁴(3x²−2x)dx (**48**, sympy in sandbox) · 5. Committee 2-of-6 girls × 2-of-5 boys (**150**) · 6. *(img)* 13-14-15 Heron triangle (**84**, triangle inequality) · 7. *(img)* linear system, aligned LaTeX (**x+y=5**) · 8. *(img)* rectangle P=30, l=w+3 (**54**) · 9. *(proof)* pigeonhole: 7 integers, difference divisible by 6 · 10. *(proof)* induction: 6 | 7ⁿ−1.

## Risk watchlist

- Metal wired-memory ceiling vs. co-residency → fast=vision default; doctor shows projected vs. measured (`/api/ps` size_vram).
- format=json_schema × thinking modes → capability probe + two-call fallback (decision 5).
- ≥90 % bar sunk by comparator false negatives → decision 6; per-outcome accounting in run.log.
- Orphaned/raced private servers → decision 7; doctor detects orphans and names the one fixing command.
- Vision digit-swap on photos → second-read number diff drops confidence below 0.7 (UI flags it); HEIC/EXIF preprocessing.
- `uv tool` venv drift vs project .venv → `--reinstall` at setup; doctor check; shim documented as sanctioned footprint in the criterion-2 proof.
