# MathScrambler — Master Prompt v2 (isolated, self-configuring)
> Paste everything below this line into Claude Code inside the `claude_MathScrambler` folder.
> If you already started from v1: the spec is the same except **Section 0 (constraints)** and the new **Section 1 (isolation & environment)** — everything MathScrambler needs now lives in a dedicated Ollama server on its own port, with env vars scoped to that one process, so nothing global on this Mac is touched. Update `PLAN.md` to match and continue.
---
You are building **MathScrambler**: a fully local tool on my MacBook Pro (Apple M5 Max, 128 GB unified memory, macOS) that takes math problems I give it — typed text, Markdown/LaTeX, or **photos/screenshots of problems** — and produces new problems that are **logically isomorphic**: same solution path, same proof structure, same difficulty, same answer *type*, but with the **nouns / names / objects / scenario swapped** and the **numbers changed** (re-sampled under the original problem's constraints so the new problem is still well-posed and its answer is exactly computable).
Everything runs on-device through **Ollama** (no cloud API calls, ever). I need two ways to use it:
1. **Interactive mode** — a custom local web dashboard I open with one short terminal command, where I upload images or paste problems, pick how many variants I want, watch them generate, review original-vs-variant side by side with rendered LaTeX, and export.
2. **Pipeline mode** — a CLI I can point at a file or folder of problems (text, `.md`, `.tex`, `.json`, or images) that batch-generates variants non-interactively and writes JSON + Markdown outputs. Same engine the dashboard calls; exactly one code path for generation.
This machine also runs **other pipelines** (a ComfyUI image-generation setup, and other projects that use Ollama). **MathScrambler must never change, break, or slow down any of them.** Section 1 tells you exactly how to achieve that. You do the environment setup yourself — I should not have to edit shell profiles, plists, or Ollama settings by hand.
Work in this order: read this whole spec, write a short `PLAN.md`, ask me blocking questions once in a batch, then build incrementally with tests, running things for real against the local Ollama as you go. Do not stub out model calls and call it done.
---
## 0. Hard constraints
- **Local only.** No OpenAI/Anthropic/HF Inference calls. Network is allowed only for `ollama pull` and package installs.
- **Runtime: Ollama** (≥ 0.19 so the MLX backend is active on Apple Silicon). If not installed, give me the exact install step and stop until I confirm. Do not install LM Studio, vLLM, or a separate llama.cpp.
- **Python 3.12+, managed with `uv`**, in a project-local `.venv`. One installable package `mathscrambler` exposing console script **`mathscramble`** (alias **`ms`**). Install with `uv tool install -e .` so the command works from any directory. Nothing installed into the system Python or any other project's venv.
- **The one-command launch** is literally `mathscramble ui` (starts everything it needs, opens the dashboard in my browser). No `cd`, no `source .venv/bin/activate`, no `python -m`.
- **Zero global side effects** (details in Section 1): do not write to `~/.zshrc`, `~/.zprofile`, `~/.bash_profile`, `launchctl setenv`, `/Library/LaunchDaemons`, `~/Library/LaunchAgents`, or the Ollama app's settings. Do not restart, reconfigure, or kill the user's existing Ollama server. Do not change `OLLAMA_MODELS`. The only files you create outside this repo are under `~/Library/Application Support/MathScrambler/` (runs, logs, SQLite) and the `uv tool` shim.
- Models are configured in one file, `config.toml`, never hard-coded.
- Never mutate my input files. Outputs go to `./outputs/<timestamp>-<slug>/` (pipeline) or the app-support runs folder (dashboard).
- macOS only.
---
## 1. Isolation & environment — how MathScrambler coexists with everything else on this Mac
### 1.1 The problem you're solving
This Mac has a system-wide Ollama that other projects depend on, and a prior project set **`OLLAMA_MAX_LOADED_MODELS=1`** (verify with `launchctl getenv OLLAMA_MAX_LOADED_MODELS`, `env | grep OLLAMA`, and `ollama ps` behaviour). MathScrambler needs **≥ 2 models resident** (vision + reasoner) and a long keep-alive. Changing the global setting would alter behaviour for every other Ollama user on this machine. So: **don't.**
### 1.2 The design: a private Ollama server, shared model weights
- MathScrambler runs **its own `ollama serve` process** on a dedicated port, **`127.0.0.1:11435`** (configurable in `config.toml`; if busy, walk up to the next free port and record it in the run log). The user's default server on `11434` is never touched.
- All Ollama-related env vars are passed **only to that child process's environment**, never exported globally:
  - `OLLAMA_HOST=127.0.0.1:<port>`
  - `OLLAMA_MAX_LOADED_MODELS=3`
  - `OLLAMA_NUM_PARALLEL=1`
  - `OLLAMA_KEEP_ALIVE=30m` (overridable per call with `keep_alive`)
  - `OLLAMA_FLASH_ATTENTION=1`
  - `OLLAMA_KV_CACHE_TYPE=q8_0`
  - **Do not set `OLLAMA_MODELS`** — inherit the default so the private server **shares the same model store** as the global one (`~/.ollama/models` or wherever the user's store already is; detect it, don't assume). Pulled weights are downloaded once and visible to both servers. Concurrent *pulls* of the same tag from two servers can corrupt a blob — so `mathscramble models --pull` must refuse to run if the global server is mid-pull (check `ollama ps`/`/api/ps` on 11434 and the `~/.ollama/models/blobs/*-partial*` files).
- **Lifecycle:** `mathscramble ui` and `mathscramble run` start the private server on demand if it isn't already running (write its PID to `~/Library/Application Support/MathScrambler/ollama.pid`, log to `.../logs/ollama.log`), wait until `/api/version` answers, and use it. `mathscramble ui` stops it on Ctrl-C / shutdown (SIGTERM, then wait) unless `--keep-server`. `mathscramble run` leaves it up for `keep_alive` then it idles at ~0 memory (models unload); add `mathscramble server stop|status|start` for manual control. Never send signals to any Ollama process you didn't start — match on the PID file, not on process name.
- **Memory etiquette:** before loading models, read free memory (`vm_stat` / `sysctl hw.memsize` + `memory_pressure`) and check whether the global Ollama has models resident (`/api/ps` on 11434) or ComfyUI is running (look for a `ComfyUI`/`main.py` process or port `8188`). If the resident footprint of the configured MathScrambler models would push the machine past **~85% of physical memory**, refuse to start, tell me exactly what's using the memory, and offer `--lite` (Section 2.2). Never evict another server's models.
- **Ports:** dashboard defaults to **`127.0.0.1:8765`**; if taken, pick the next free port and print it. Bind to loopback only. Nothing MathScrambler listens on is reachable from the network.
- **Filesystem:** repo + `.venv` + `outputs/` inside `claude_MathScrambler/`; state under `~/Library/Application Support/MathScrambler/`. No files anywhere else.
- **Detection of a friendly global server:** if the global server on 11434 is *already* configured with `OLLAMA_MAX_LOADED_MODELS ≥ 2` (probe by loading two tiny models and checking `/api/ps`), `config.toml` may set `ollama.mode = "shared"` to reuse it instead of spawning a private server. Default is `"private"`. Either way the code path is identical: the client just gets a base URL.
### 1.3 `mathscramble doctor` must report, in green/yellow/red:
Ollama binary + version (≥ 0.19); global server status on 11434 and its `MAX_LOADED_MODELS` (informational only); private server status/port; model store path and free disk; each configured role → tag pulled? size? MLX tag? resident?; physical memory, current free memory, projected footprint with the configured roles; ComfyUI/other GPU-heavy processes detected; sandbox self-test; dashboard port availability; `uv tool` shim on `PATH`. Every red item comes with the one command that fixes it.
### 1.4 What `mathscramble setup` does (run once, idempotent, no sudo)
1. Creates `~/Library/Application Support/MathScrambler/{runs,logs}`.
2. Verifies/creates the `uv` venv and installs the package in editable mode; installs the `mathscramble` shim via `uv tool install -e .`; checks the shim dir is on `PATH` and, if not, **prints** the line to add — it does not edit my shell profile.
3. Writes `config.toml` from `config.example.toml` if missing.
4. Starts the private server once to validate, runs `doctor`, lists which model tags are missing with their download sizes, and asks before pulling anything > 30 GB.
5. Vendors KaTeX into `web/static/` if missing.
---
## 2. Model roles and defaults (verify with `ollama show`, benchmark before finalizing)
| Role | Default | Why | Fallback |
|---|---|---|---|
| **`vision`** — read problems out of images into clean Markdown + LaTeX | `qwen3.6:35b-mlx` (~24 GB, MoE 35B/3B-active, image+text, 256K ctx, MLX) | Fast on M5 Max, native vision, good at math notation → LaTeX | `qwen3.6:27b-mlx` |
| **`reasoner`** — blueprint extraction, solving, verification | `gpt-oss:120b` (~65 GB, MXFP4, 128K ctx), `reasoning_effort="high"` | Strongest open-weight math/proof reasoning that fits next to the vision model; exposes chain of thought for the verifier | `qwen3.5:122b` (~81 GB, multimodal; can't co-reside with gpt-oss:120b) |
| **`fast`** — noun/scenario brainstorming, JSON repair, grammar fix, slugs | `qwen3.8:27b-mlx` if it exists, else `qwen3.6:27b-mlx` | Low latency, no deep reasoning needed | reuse `vision` |
### 2.1 Runtime settings
- Default footprint ≈ 24 + 65 (+ ~19 if `fast` is separate) GB ≈ 108 GB max. `num_ctx`: 16K for `reasoner`, 8K for `vision`/`fast` unless a problem needs more. Use `-mlx` tags whenever they exist. Use Ollama **structured outputs** (JSON schema) for every data-returning call; parse with Pydantic; on validation failure retry ≤ 3× feeding the error back. Enable thinking: `think=true` on Qwen; `reasoning_effort=high` on gpt-oss for reasoning calls, `low` for cosmetic ones.
### 2.2 `--lite` profile (for when ComfyUI or another project is hogging memory)
`vision = qwen3.6:35b-mlx`, `reasoner = qwen3.8:27b-mlx` (or `qwen3.6:27b-mlx`), `fast = vision`. ≈ 45 GB total. Selectable via `--lite` on any command or `profile = "lite"` in `config.toml`. Bench it too so I know the fidelity cost.
### 2.3 Benchmark command
Implement `mathscramble bench` so I can drop any Ollama tag into `config.toml` and get a pass-rate/speed scorecard. Out-of-the-box candidates: `gpt-oss:120b`, `qwen3.5:122b`, `qwen3.8:27b-mlx`, `qwen3.6:35b-mlx`, `glm-5.3-flash`, `gemma4` (largest size that fits).
---
## 3. What "scrambled but identical" means — the generation algorithm
Do **not** just ask the model "rewrite this with different numbers" — that yields non-integer answers, impossible triangles, probabilities > 1. Implement the **blueprint approach** (the idea behind the "Computational Blueprints" isomorphic-problem work and GSM-Symbolic-style perturbation):
### Step A — Ingest → canonical problem
- Text/Markdown/LaTeX → one `Problem` record: `{id, source, statement_md, given_answer (optional), tags}`.
- Images → `vision` returns strict JSON `problems: [{statement_md, diagram_description, answer_if_shown, confidence}]`. One image may hold several problems — split them. All math as LaTeX. `confidence < 0.7` is flagged in the UI for me to fix before scrambling; I can always edit the extraction before generation.
### Step B — Blueprint extraction (`reasoner`)
```
{
  "kind": "computational" | "proof" | "mixed",
  "domain": [...],
  "entities": [ {"slot":"E1","original":"Alice","role":"person"}, {"slot":"E2","original":"apples","role":"countable object"} ],
  "parameters": [ {"slot":"p1","original":12,"type":"int","description":"...","constraints":"p1 > 0"} ],
  "constraints": ["p1 % p2 == 0", "triangle inequality on p4,p5,p6"],   // Python expressions over slots
  "template_md": "{E1} has {p1} {E2} ...",
  "solution_outline": ["step 1 ...", "step 2 ..."],                        // logical skeleton that must be preserved
  "solver_code": "def solve(p1, p2, ...): ... return answer",              // pure Python (+sympy); null for pure proofs
  "answer_type": "integer" | "rational" | "expression" | "set" | "proof",
  "invariants": ["answer must be a positive integer", ...]
}
```
- `solver_code` runs in a **sandboxed subprocess** (5 s timeout, no network, restricted builtins, `sympy`/`fractions` allowed). If it fails on the original parameters or doesn't reproduce a known original answer → regenerate the blueprint (≤ 3 tries) with the failure fed back.
- `kind = "proof"`: no solver. Parameters are the structural constants (the 7 and 6 in "among any 7 integers two share a residue mod 6", invariant `p1 > p2`). `solution_outline` is the proof skeleton (pigeonhole, induction, contradiction…) and the variant's reference proof must follow it exactly.
### Step C — Variant sampling (deterministic, not LLM)
- Sample new parameters satisfying **every** constraint, ≠ originals, same magnitude class, answer stays "nice" if the original's was (integer→integer, terminating→terminating). Rejection-sample ≤ 2,000 tries; then relax magnitude rules; then report failure.
- Compute the new answer with `solver_code`; reject degenerate answers (0, 1, equal to an input, sign flip) unless the original was degenerate too.
- `fast` proposes entity replacements preserving role and grammatical number (`Alice→Priya`, `apples→marbles`, `square ABCD→square PQRS`). Mathematical objects keep their type; never touch nouns carrying mathematical meaning ("prime", "median") — keep a guard list.
- Render `template_md`; one `fast` grammar pass allowed to change nothing but articles/pluralization (diff-check that numbers and math are untouched).
### Step D — Independent verification (`reasoner`, fresh context, blueprint hidden)
- Solve the rendered variant from scratch. Answer must match `solver_code` (exact for ints/rationals; `sympy.simplify(a-b)==0` for expressions; set equality for sets). Method must match `solution_outline` (structured `same_method: bool, differences: []` judgment). Pass = answer match **and** method match.
- Proofs: `reasoner` writes the full proof; a second call checks it against `solution_outline` step by step → `valid, missing_steps`.
- Failed variants: resample and retry ≤ 3; persistent failure is reported, never silently dropped. Log every attempt.
### Step E — Output
Per problem: `original`, `blueprint`, `variants[]` with `{statement_md, answer, reference_solution_md, parameter_map, entity_map, verification: {answer_match, method_match, attempts}}`. Write `results.json` and `results.md` (solutions under `<details>`). `--pdf` via Markdown → HTML (KaTeX) → PDF.
---
## 4. Repository layout
```
claude_MathScrambler/
├── pyproject.toml              # uv; console_scripts: mathscramble, ms
├── config.example.toml         # committed; copied to config.toml by `setup`
├── config.toml                 # gitignored; model roles, ollama.mode/port, dashboard port, profile, sampling knobs
├── README.md · PLAN.md
├── src/mathscrambler/
│   ├── cli.py                  # typer: setup, ui, run, bench, models, doctor, server {start,stop,status}
│   ├── config.py
│   ├── ollama_server.py        # private-server lifecycle: spawn with scoped env, PID file, health wait, stop; shared-mode detection
│   ├── ollama_client.py        # async chat/vision/structured-output wrapper with retries + timing, takes base_url
│   ├── sysinfo.py              # memory, ports, running-process detection (ComfyUI, global Ollama)
│   ├── ingest/                 # text/md/tex/json loaders; image → problems via vision
│   ├── blueprint.py · sampler.py · sandbox.py · verify.py · render.py
│   ├── prompts/                # every prompt as a .md/.jinja file, never inline strings
│   ├── engine.py               # orchestrates A→E; the single code path for CLI and dashboard
│   └── web/  app.py (FastAPI + SSE) · static/index.html (vanilla JS + vendored KaTeX, no npm) · db.py (SQLite)
├── tests/                      # pytest; sampler/sandbox/render/sysinfo/server-lifecycle tests need no model; integration tests @pytest.mark.ollama
├── examples/                   # 10 sample problems: 5 text, 3 images, 2 proofs
└── outputs/                    # gitignored
```
---
## 5. CLI spec
```
mathscramble setup                                  # Section 1.4; idempotent
mathscramble ui [--port 8765] [--no-open] [--lite] [--keep-server]
mathscramble run <path...> [-n 3] [--kind auto|computational|proof] [--out DIR] [--pdf] [--seed 42] [--lite]
                          [--models reasoner=...,vision=...]
mathscramble bench [--problems examples/] [--models a,b,c] [--n 2] [--lite]
mathscramble models [--pull]                        # roles, pulled?, sizes, MLX?, resident on which server?
mathscramble doctor                                 # Section 1.3
mathscramble server start|stop|status               # private server control; refuses to touch PIDs it didn't create
```
`--seed` makes Step C reproducible. Every run writes `run.log` with prompts, raw responses (truncated to 4K), timings, and which server/port was used.
---
## 6. Dashboard spec (`mathscramble ui`)
Single page, follows system dark/light, no login, loopback only.
- **Input**: drag-and-drop images (multi, thumbnails), textarea for pasted problems (`---` separated), file picker for `.md/.txt/.json`.
- **Extraction review**: editable Markdown per extracted problem with live KaTeX preview and vision confidence; edit/delete/merge before **Scramble**.
- **Controls**: variants per problem (1–10), kind override, "keep answer nice", "change scenario" vs "numbers only", seed, model role dropdowns from `/api/tags`, profile (default/lite).
- **Status strip**: which Ollama server is in use (private:11435 / shared:11434), models resident, free memory — so I can see at a glance that it isn't touching the global server.
- **Progress**: SSE; per-problem stage (Extracting → Blueprint → Sampling → Verifying ✓/✗, attempt count); current tok/s.
- **Results**: original vs variant side by side, KaTeX; reveal answer/solution; verification badge; "regenerate this variant"; "show blueprint" (editable → re-run).
- **Export**: `results.md`, `results.json`, `results.pdf`; copy-one-variant.
- **History**: sidebar of previous runs from SQLite.
- Works offline (vendored KaTeX).
---
## 7. Acceptance criteria — show me all of these before calling it done
1. `mathscramble setup` then `mathscramble doctor` — all green, and `doctor` explicitly shows the global Ollama on 11434 untouched (its `MAX_LOADED_MODELS` unchanged) while the private server reports `MAX_LOADED_MODELS=3`.
2. **Non-interference proof:** with MathScrambler's server running and two models resident, run `ollama ps` (global) and show it's unaffected; run `mathscramble server stop` and show only the private PID died. Show `git status`-style evidence that nothing was written outside the repo, `~/Library/Application Support/MathScrambler/`, and the `uv tool` shim dir. Show `grep -i ollama ~/.zshrc ~/.zprofile` unchanged and `launchctl getenv OLLAMA_MAX_LOADED_MODELS` unchanged.
3. `mathscramble run examples/ -n 3` — ≥ 90% variants pass verification, zero crashes, avg wall time per variant reported.
4. `mathscramble bench --models gpt-oss:120b,qwen3.5:122b,qwen3.8:27b-mlx` (skip un-pulled ones; tell me pull sizes and wait for OK on anything > 30 GB) — a readable table and a one-paragraph default-`reasoner` recommendation, plus the same table for `--lite`.
5. Dashboard end to end: upload a photo of a textbook page with 2 problems, fix one extraction typo, 3 variants each, export PDF — while you watch the server log.
6. `pytest` green (non-Ollama tests < 10 s; integration behind `-m ollama`).
7. `README.md`: install in ≤ 3 commands, both modes, how to swap models, how the isolation works in five sentences, troubleshooting for: Ollama not installed, model not pulled, out of memory / ComfyUI running, port in use.
Variant quality bar (spot-check 10 and paste them to me): every number changed, every non-math noun changed when "change scenario" is on, identical solution outline, answer computed not guessed, reads like a human wrote it.
---
## 8. How I want you to work
- Ask blocking questions **once, up front, in one batch**. Already decided: name, folder, Ollama with a private server on 11435 sharing the model store, `uv`, FastAPI + vanilla JS, SQLite, model roles, zero global side effects.
- Build order, checking in after each: (1) `sysinfo` + `ollama_server` + `doctor` + `setup` + `server` commands, with the non-interference proof from criterion 2; (2) `ollama_client` + `models`; (3) ingest incl. vision on the 3 example images; (4) blueprint + sandbox + sampler with unit tests; (5) verify; (6) `run` end to end; (7) dashboard; (8) `bench`; (9) README + polish.
- Boring, readable code; type hints; `ruff` clean. Fix model misbehaviour in the **prompt file** or **schema**, not with string hacks.
- If at any point the correct fix seems to require a global change (editing a shell profile, `launchctl setenv`, restarting the user's Ollama, changing `OLLAMA_MODELS`), **stop and ask** — don't do it.
- Keep `PLAN.md` current: done / next / decisions made without me.
