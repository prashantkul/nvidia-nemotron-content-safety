# System Design — Nemotron 3.5 Content Safety Verification Harness & Guardrail Gateway

> Grounded in `docs/00-PROJECT-BRIEF.md`. Where the brief marks a fact `[VERIFY]`, this
> design treats it as runtime-discoverable and isolates it behind a small adapter so the
> coding agent can adapt without touching the rest of the system. This doc is design-only:
> it specifies interfaces and contracts. The coding agent owns `src/`.

This document works at two altitudes:

1. **The Verification Harness** — what we build now. A small Python project that empirically
   verifies the four headline claims (multimodal, multilingual, optional THINK reasoning,
   custom-policy enforcement) against `nvidia/nemotron-3.5-content-safety:free` via OpenRouter.
2. **The Productionized Guardrail Gateway** — the co-build target. A reference architecture
   for an inline guardrail control plane, design-only, mapped onto the latency-budget thesis.

---

# Part 1 — The Verification Harness

## 1.1 Goals & non-goals

| Goal | Non-goal |
|---|---|
| Call the model 4 ways and assert behaviour holds | Re-measuring published benchmarks |
| Turn plain-text verdicts into a typed `SafetyVerdict` | Building a server / API |
| Degrade gracefully with no API key | GPU / local inference (no GPU on dev box) |
| Measure per-call latency | Production-grade observability |
| Survive free-tier rate limits (429) with backoff | High throughput / concurrency tuning |

The four capabilities under test map 1:1 to verification cases:

| Capability | How exercised | Pass signal |
|---|---|---|
| Multimodal | `image_url` block + text prompt | A parsed `User Safety` verdict returns |
| Multilingual | Same unsafe intent in ≥3 languages | Consistent `unsafe` across languages |
| THINK reasoning | `enable_thinking=True` | `think` trace populated, verdict still parses |
| Custom policy | `custom_policy=<text>` flips a benign-by-default prompt | Verdict changes vs. no-policy baseline |

## 1.2 Component breakdown

```
src/
  config.py        # env + settings loader; no-key graceful degradation
  client.py        # OpenRouter (OpenAI-compatible) async client wrapper + retries
  request.py       # typed request builder: messages + extra_body assembly
  parser.py        # plain-text verdict -> SafetyVerdict (incl. THINK extraction)
  models.py        # dataclasses: ModerationRequest, SafetyVerdict, CallResult
  runner.py        # verification/test runner over the 4 capabilities
  cases/           # declarative test cases (text/image/multilingual/policy)
tests/             # unit tests for parser (offline, no network)
```

| Component | Responsibility | Key dependency |
|---|---|---|
| `config` | Load `OPENROUTER_API_KEY`, base URL, model id, timeouts; report key presence | `os.environ`, `.env` via `python-dotenv` |
| `client` | Wrap `openai.AsyncOpenAI(base_url=…)`; add retries/backoff, latency hooks, OR headers | `openai` SDK |
| `request` | Build the messages array + `extra_body` from a `ModerationRequest` | `models` |
| `parser` | Convert raw completion text → `SafetyVerdict`; handle edge cases | none (pure) |
| `runner` | Compose request→client→parser; run capability suite; print/JSON report | all above |

**Design principle:** `parser` and `request` are *pure* (no I/O) so they are unit-testable
offline. Network lives only in `client`. This keeps the `[VERIFY]` risk (exact image block
shape, taxonomy list, language set) contained and testable from captured fixtures.

## 1.3 Core data model

### 1.3.1 Request shape — `ModerationRequest`

```python
@dataclass(frozen=True)
class ModerationRequest:
    prompt: str                                  # the user text to moderate
    response: str | None = None                  # optional assistant response to moderate
    image: str | None = None                     # http(s) URL or data: URI (base64)
    enable_thinking: bool = False                # -> chat_template_kwargs.enable_thinking
    request_categories: bool = True              # -> chat_template_kwargs.request_categories
    custom_policy: str | None = None             # -> chat_template_kwargs.custom_policy
    model: str = "nvidia/nemotron-3.5-content-safety:free"
    temperature: float = 0.0                     # deterministic classification
    max_tokens: int = 512                        # headroom for THINK trace
```

`request_categories` is a bool in our API for ergonomics; the builder translates `True` to
the literal token the model expects (per brief: `"/categories"`). Keeping it a bool means
the `[VERIFY]` of the exact token value lives in one place (`request.py`).

### 1.3.2 The verdict — `SafetyVerdict` (the contract)

This is the central contract every downstream consumer programs against.

```python
class Safety(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"        # line absent or unparseable

@dataclass(frozen=True)
class SafetyVerdict:
    user_safety: Safety                  # always present (UNKNOWN if missing)
    response_safety: Safety | None       # None when no assistant response was moderated
    categories: list[str]                # [] when none requested / none found
    think: str | None                    # reasoning trace, None when THINK disabled/absent
    raw: str                             # the verbatim completion text (audit/debug)
    parse_ok: bool                       # False if required fields were missing/malformed
    parse_notes: list[str]               # human-readable notes on what degraded

    @property
    def is_unsafe(self) -> bool:
        return (self.user_safety is Safety.UNSAFE
                or self.response_safety is Safety.UNSAFE)
```

Rationale for each field:

| Field | Why |
|---|---|
| `user_safety` | The headline classification; never `None` so callers needn't null-check |
| `response_safety` | Distinct `None` vs `SAFE` — absence is meaningful (no response moderated) |
| `categories` | Always a list; empty list is the natural "none" |
| `think` | Audit trail; nullable because it is opt-in |
| `raw` | Forensics + lets the gateway log exactly what the model said |
| `parse_ok` / `parse_notes` | Make degradation explicit instead of silently wrong |

### 1.3.3 Call envelope — `CallResult`

The runner wraps a verdict with operational metadata.

```python
@dataclass(frozen=True)
class CallResult:
    verdict: SafetyVerdict | None        # None if the call itself failed
    latency_ms: float                    # wall-clock for the HTTP round-trip
    model: str
    attempts: int                        # how many tries (retries) were used
    error: str | None = None             # populated on failure / no-key skip
    skipped: bool = False                # True when degraded (no API key)
```

## 1.4 Parsing strategy (plain text → `SafetyVerdict`)

The model returns **plain text, not JSON** (brief §"Output format"). Parsing is line-oriented
and defensive. Algorithm:

```
parse(raw) -> SafetyVerdict:
  notes = []
  think = extract_think(raw)            # 1. pull <think>…</think>, strip from body
  body  = raw with think block removed
  lines = [l.strip() for l in body.splitlines() if l.strip()]
  fields = {}
  for line in lines:
    if ":" in line:
      key, val = line.split(":", 1)
      fields[normalize(key)] = val.strip()   # normalize: lower, spaces->_
  user_safety     = to_safety(fields.get("user_safety"), notes, required=True)
  response_safety = to_safety(fields.get("response_safety"), notes) if present else None
  categories      = split_categories(fields.get("safety_categories"))
  parse_ok        = user_safety is not UNKNOWN
  return SafetyVerdict(user_safety, response_safety, categories, think, raw, parse_ok, notes)
```

### 1.4.1 THINK trace extraction

- Match `<think> … </think>` (DOTALL, non-greedy, case-insensitive tag).
- Strip the whole block from the body **before** line parsing so verdict lines parse cleanly.
- **Unclosed `<think>` (truncated by `max_tokens`):** if an opening tag exists with no close,
  capture everything after it as `think`, add note `"think_unclosed"`, and attempt to parse
  any verdict lines that nonetheless appear. If none do, `parse_ok=False`.
- `enable_thinking=False` ⇒ no tag expected; if one appears anyway, still extract it and note
  `"unexpected_think"`.

### 1.4.2 Category splitting

- Split `Safety Categories:` on commas, trim each, drop empties.
- Tolerate trailing `"None"` / `"N/A"` ⇒ empty list with note `"categories_none_literal"`.
- Do **not** validate against the taxonomy at parse time (taxonomy list is `[VERIFY]`); expose
  raw strings. An optional `validate_categories(verdict, taxonomy)` helper can flag unknowns
  for reporting without failing the parse.

### 1.4.3 Edge cases (explicit contract)

| Input situation | Behaviour |
|---|---|
| Missing `Response Safety` line | `response_safety = None` (expected when no response sent) |
| Missing `Safety Categories` line | `categories = []` |
| Missing `User Safety` line | `user_safety = UNKNOWN`, `parse_ok = False`, note added |
| Case variance (`UNSAFE`, `Unsafe`) | normalized lowercase before enum mapping |
| Unknown safety value (`maybe`) | `UNKNOWN` + note `"unknown_safety_value:maybe"` |
| Extra prose / preamble lines | ignored (only `key: value` lines consumed) |
| Empty / whitespace completion | `parse_ok=False`, note `"empty_completion"` |
| Duplicate keys | last occurrence wins; note `"duplicate_key:<k>"` |

Unit tests (offline) drive these cases from string fixtures — no network needed.

## 1.5 Request builder

```python
def build_messages(req: ModerationRequest) -> list[dict]:
    """user (+optional image) block, plus optional assistant block."""
    content = []
    if req.image:
        content.append({"type": "image_url", "image_url": {"url": req.image}})  # [VERIFY] block shape
    content.append({"type": "text", "text": req.prompt})
    msgs = [{"role": "user", "content": content}]
    if req.response is not None:
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": req.response}]})
    return msgs

def build_extra_body(req: ModerationRequest) -> dict:
    kw = {"enable_thinking": req.enable_thinking}
    if req.request_categories:
        kw["request_categories"] = "/categories"      # [VERIFY] token literal
    if req.custom_policy:
        kw["custom_policy"] = req.custom_policy
    return {"chat_template_kwargs": kw}
```

Image block shape is the highest `[VERIFY]` risk. Isolate it in `build_messages`; if a live
call rejects `image_url`, the only change is here (e.g. fall back to `{"type":"image",…}`).

## 1.6 Client wrapper

```python
class SafetyClient:
    def __init__(self, settings: Settings): ...
    async def moderate(self, req: ModerationRequest) -> CallResult: ...
    async def aclose(self) -> None: ...
```

- Built on `openai.AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)`.
- Sends OpenRouter attribution headers (`HTTP-Referer`, `X-Title`) via `default_headers`.
- Wraps `chat.completions.create(model, messages, extra_body, temperature, max_tokens)`.
- Records `latency_ms` with a `time.perf_counter()` hook around the awaited call.
- Returns a `CallResult` — never raises for expected operational failures (rate limit
  exhaustion, no key); those become `error`/`skipped` so the runner can continue the suite.

### 1.6.1 Retries / backoff (free-tier 429s)

| Condition | Action |
|---|---|
| HTTP 429 / `RateLimitError` | Exponential backoff w/ jitter; honour `Retry-After` if present |
| HTTP 5xx, timeout, conn reset | Retry up to `max_retries` (default 4) |
| HTTP 4xx (≠429) | No retry — surface as `error` (bad request / auth) |
| Backoff schedule | `base=1s`, `delay = min(base * 2**attempt + rand(0,0.5), cap=30s)` |

`attempts` is recorded on `CallResult` for visibility into how hard the free tier pushed back.

### 1.6.2 Graceful degradation (no API key)

- `Settings.has_key` is `False` ⇒ `client.moderate` returns
  `CallResult(verdict=None, skipped=True, error="OPENROUTER_API_KEY not set")` immediately.
- The runner prints a clear banner and runs only offline parser tests, exiting `0` (a missing
  key is a configuration state, not a test failure). CI without a key still validates parsing.

### 1.7 Latency measurement hooks

- Per-call `latency_ms` on every `CallResult`.
- Runner aggregates: p50/p95 over the suite, split by mode (default vs THINK) — directly tests
  the brief's thesis that THINK costs more tokens/latency than default mode.
- Optional `--csv` dump of `(case, mode, latency_ms, attempts, is_unsafe)` for analysis.

## 1.8 End-to-end sequence of one moderation call

```
runner.run_case(case)
  └─ ModerationRequest = case.to_request()
  └─ client.moderate(req):
       1. if not settings.has_key -> return CallResult(skipped=True)
       2. messages   = build_messages(req)            # request.py
       3. extra_body = build_extra_body(req)          # request.py
       4. t0 = perf_counter()
       5. resp = await openai.chat.completions.create(...)   # with retry/backoff loop
       6. latency_ms = (perf_counter()-t0)*1000
       7. raw = resp.choices[0].message.content
       8. verdict = parse(raw)                        # parser.py
       9. return CallResult(verdict, latency_ms, model, attempts)
  └─ runner asserts capability expectation against verdict
  └─ runner records latency + pass/fail; continues suite on per-case error
```

## 1.9 Verification runner & test cases

- Declarative cases in `cases/` (text, image, multilingual triplet, policy-flip pair).
- A **policy-flip** case asserts the *same* prompt yields `safe` with no policy and `unsafe`
  with a `custom_policy` that disallows it — proving programmability, not just classification.
- A **multilingual** case asserts a fixed unsafe intent classifies `unsafe` across the
  consistently-trained languages (English, French, Spanish, German, Chinese, Japanese, Korean,
  Arabic, Hindi, Italian — the disputed pair is excluded per brief `[VERIFY]`).
- Runner output: a table (capability, pass/fail, latency) + machine-readable JSON summary.
- Each capability assertion is *soft* on parse degradation: a `parse_ok=False` is reported as
  an inconclusive result with the raw text attached, not a hard crash.

---

# Part 2 — The Productionized Guardrail Gateway (design-only)

The co-build target: a security startup operates an **inline guardrail gateway / control plane**
between client applications and LLMs. NVIDIA supplies the model + NIM + GPUs; the startup owns
policy, multi-tenancy, observability, and the latency-budget engineering.

## 2.1 The latency-budget thesis (why this architecture exists)

A safety classifier sits on **every** input and **every** output, so it is *additively* in the
end-to-end latency budget twice per turn:

```
client → [INPUT moderation] → LLM generate → [OUTPUT moderation] → client
            (sync, blocking)                      (sync, blocking)
```

Historically: capable safety models were too slow to sit here; fast ones were too dumb. A 4B
multimodal + multilingual + policy-programmable + optionally-reasoning model changes the
tradeoff. The architectural consequence: **run the cheap, fast default-mode classification
synchronously in the hot path; run expensive THINK-mode reasoning asynchronously for audit.**

## 2.2 Reference architecture

```
                         ┌──────────────────────── Control Plane ───────────────────────┐
                         │  Policy registry (policy-as-code, versioned, per-tenant)      │
                         │  Tenant config / keys · Category taxonomy · Rollout flags     │
                         └───────────────▲───────────────────────────┬──────────────────┘
                                         │ pull/push policy            │ audit events
 client app ──HTTP──▶  GUARDRAIL GATEWAY (data plane)                 ▼
                       ├─ AuthN/Z + tenant resolve            ┌──── Async audit lane ────┐
                       ├─ Cache lookup (verdict cache)        │  THINK-mode re-scoring   │
                       ├─ INPUT moderation (sync, default)────┤  Anomaly / drift detect  │
                       │       │ unsafe → block/redact        │  Log to SIEM / warehouse │
                       ├─ forward to LLM (NIM/vLLM/OpenRouter)└──────────────────────────┘
                       ├─ OUTPUT moderation (sync, default)
                       └─ fail-open / fail-closed decision → response
```

### Data plane responsibilities (hot path, latency-critical)
- Terminate client request, resolve tenant, authorize.
- Input moderation (default mode, ~3x lower latency per brief). Block / redact / annotate.
- Forward allowed traffic to the configured LLM backend.
- Output moderation (default mode) before returning to client.
- Apply fail-open vs fail-closed policy on classifier error/timeout.
- Emit an audit event (fire-and-forget) onto the async lane.

### Control plane responsibilities (off the hot path)
- Policy-as-code registry: versioned policies, per-tenant bindings, staged rollout.
- Taxonomy + category management; custom policy text compilation.
- Tenant onboarding, key management, quotas.

### Async audit lane (off the hot path)
- Re-score a sample (or all) traffic with **THINK mode** for explainable audit trails.
- Drift / anomaly detection; feed labelled hard cases back to NVIDIA for fine-tuning (the
  co-build data flywheel).
- Ship structured logs to SIEM / warehouse; retention per tenant compliance needs.

## 2.3 Sync vs async placement (mapping the thesis)

| Work | Lane | Mode | Why |
|---|---|---|---|
| Input moderation | **sync** | default (no THINK) | In hot path; must be fast; ~3x lower latency |
| Output moderation | **sync** | default | Same; gates the user-visible response |
| Verdict caching | sync | — | Skip the classifier entirely on repeat inputs |
| Explainable audit | **async** | THINK on | Reasoning trace too costly for hot path |
| Drift / red-team scan | async | THINK on | Batch; informs policy + model retraining |

The 4B model is small enough to colocate with inference GPUs, making the sync hops cheap; the
expensive reasoning is deferred. This is the core engineering claim the harness empirically
checks (THINK vs default latency split, §1.7).

## 2.4 Policy-as-code

- Policies are versioned artifacts (git-backed), compiled to the model's natural-language
  `custom_policy` block (brief §"Custom policy format") plus gateway-side rules (block/redact/
  allow, severity thresholds, category routing).
- Each tenant binds a policy version; rollouts are staged (canary → percentage → full) with
  instant rollback to a prior version.
- A policy is testable: the same harness cases (§1.9 policy-flip) become CI regression tests
  for a policy change before promotion.

## 2.5 Caching

| Layer | Key | Notes |
|---|---|---|
| Input verdict cache | hash(prompt + image-digest + policy-version + mode) | Big win for repeated/templated prompts |
| Output verdict cache | hash(response + policy-version) | Lower hit rate; still useful for fixed responses |
| Negative cache | known-unsafe fingerprints | Block instantly without a model call |

Cache must be **policy-version-scoped** — a policy change invalidates affected entries, else a
stale verdict could leak disallowed content under a new policy.

## 2.6 Fail-open vs fail-closed

| Tenant posture | Classifier unavailable / times out | Use case |
|---|---|---|
| **Fail-closed** | Block (deny by default) | High-compliance: health, finance, minors |
| **Fail-open** | Allow + flag for async audit | Availability-first: low-risk internal tools |

Per-tenant, per-direction (input vs output) configurable, with a hard global latency budget
(e.g. 150ms) after which the configured posture triggers. Every fail-open bypass is logged and
re-scored on the async lane so nothing escapes audit.

## 2.7 Multi-tenant policy isolation

- Strict tenant scoping on policy, cache keys, logs, and quotas — no cross-tenant leakage.
- Per-tenant rate limits / quotas protect noisy-neighbor latency.
- Optional dedicated model replicas (NIM instances) for tenants needing isolation or a
  fine-tuned policy variant; shared pool for the rest.
- Audit logs are tenant-partitioned with independent retention/residency controls.

## 2.8 Deployment options

| Stage | Backend | Tradeoff |
|---|---|---|
| **Dev / prototype** | OpenRouter `:free` | Zero infra, free; shared rate limits, no latency SLA, data leaves your boundary |
| **Production** | Self-hosted **NIM** or **vLLM** on NVIDIA GPUs | Control latency/SLA, data residency, tenant isolation, autoscale; you run the GPUs (model fits 8GB+ VRAM, 4B params) |

The harness (Part 1) targets OpenRouter and abstracts the backend behind `SafetyClient`. The
*same* `ModerationRequest`/`SafetyVerdict` contract is reused against a NIM/vLLM endpoint in
prod by swapping only `Settings.base_url` and credentials — the request builder, parser, and
verdict schema are backend-agnostic by design. This is the bridge between the two altitudes:
**build the contract once in the harness, productionize it in the gateway.**

## 2.9 What the harness proves for the gateway

| Gateway assumption | Harness evidence (Part 1) |
|---|---|
| Default mode is fast enough for the hot path | §1.7 latency p50/p95, default vs THINK split |
| Policy programmability works | §1.9 policy-flip case |
| Multilingual coverage holds | §1.9 multilingual case |
| Multimodal input is handled | §1.9 image case |
| Plain-text output is reliably parseable | §1.4 parser + offline edge-case tests |
| THINK is viable for async audit | THINK trace populated + latency delta measured |

---

## Appendix A — Contract summary for the coding agent

Implement these signatures exactly; everything else is implementation freedom.

```python
# models.py
class Safety(str, Enum): SAFE="safe"; UNSAFE="unsafe"; UNKNOWN="unknown"

@dataclass(frozen=True)
class ModerationRequest: ...      # §1.3.1
@dataclass(frozen=True)
class SafetyVerdict: ...          # §1.3.2  (the central contract)
@dataclass(frozen=True)
class CallResult: ...             # §1.3.3

# parser.py  (pure, offline-testable)
def parse(raw: str) -> SafetyVerdict: ...

# request.py (pure)
def build_messages(req: ModerationRequest) -> list[dict]: ...
def build_extra_body(req: ModerationRequest) -> dict: ...

# config.py
@dataclass(frozen=True)
class Settings:
    api_key: str | None; base_url: str; model: str
    timeout_s: float; max_retries: int
    @property
    def has_key(self) -> bool: ...
def load_settings() -> Settings: ...

# client.py
class SafetyClient:
    def __init__(self, settings: Settings): ...
    async def moderate(self, req: ModerationRequest) -> CallResult: ...
    async def aclose(self) -> None: ...

# runner.py
async def run_suite() -> int: ...   # exit code; 0 even when no key (offline tests run)
```

## Appendix B — `[VERIFY]` items isolated for live adaptation

| `[VERIFY]` item | Isolated in | Adaptation if wrong |
|---|---|---|
| Image content-block shape (`image_url` vs `image`) | `build_messages` | swap block dict |
| `request_categories` token literal (`/categories`) | `build_extra_body` | change literal |
| Full 13-category taxonomy | `validate_categories` (optional) | update list; parser unaffected |
| Disputed language pair (Thai+Dutch vs Russian+Portuguese) | multilingual case set | exclude/add cases |
