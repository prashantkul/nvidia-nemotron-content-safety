# Design Principles — Nemotron 3.5 Content Safety Verification & Co-Build

These principles govern two things at once: the **engineering of the verification harness**
(a Python project that calls `nvidia/nemotron-3.5-content-safety:free` via OpenRouter) and
the **design of a safety-classifier-based product** built on top of that model. They are
derived from, and must never contradict, `docs/00-PROJECT-BRIEF.md`.

The brief's core thesis frames everything below: a safety classifier sits on **every input
and output**, so it lives directly inside the end-to-end latency budget. Capable-but-slow or
fast-but-dumb are both failure modes. Every principle here is in service of being capable,
fast, and trustworthy at the same time.

---

## Part 1 — Engineering Principles (the harness)

### E1. Model output is untrusted text until proven otherwise
**Statement:** Treat every response from the model as arbitrary text; never assume it is
well-formed, never assume it is JSON.

**Rationale:** This model emits **plain text**, not structured JSON (brief §"Output format").
The shape varies by mode: `Response Safety` only appears when an assistant turn was supplied,
`Safety Categories` only when requested or when violations exist, and THINK mode prepends a
`<think>...</think>` block. Any of these lines can be missing, reordered, truncated at the
context limit, or wrapped in unexpected whitespace. A parser that assumes a fixed layout will
silently misclassify.

**In this project this means:** Parse defensively with tolerant, line-oriented extraction —
match `User Safety:`, `Response Safety:`, `Safety Categories:` case-insensitively and
independently, strip the `<think>` block out before verdict parsing, and treat "field absent"
as a distinct state from "field said safe." When a verdict line cannot be found at all, that
is a parse failure (see E2), not a `safe` result. The parser is the most heavily unit-tested
component in the harness, with fixtures for every documented and malformed shape.

### E2. Fail-closed vs fail-open is an explicit, configured decision
**Statement:** What happens when the classifier errors, times out, or returns unparseable
output must be a deliberate, named, configurable policy — never an accident of control flow.

**Rationale:** A safety classifier that crashes has two options: block the content
(fail-closed) or allow it (fail-open). The right choice is context-dependent. A kids' app
should fail-closed; an internal DevOps assistant where a false block halts an on-call
engineer may justify fail-open with logging. If this is implicit, the system's safety
posture under failure is unknown, which is the same as unsafe.

**In this project this means:** A single configuration knob (e.g. `on_error: fail_closed |
fail_open`) governs behavior for timeouts, HTTP errors, and parse failures, defaulting to
`fail_closed`. The harness records *which* failure mode fired and *why* on every degraded
call. The co-build product exposes this per-surface, never globally hardcoded.

### E3. Latency is a first-class budget item
**Statement:** Treat latency as a tracked resource with a budget, not a number you discover
in production; keep reasoning off the synchronous hot path.

**Rationale:** Because the classifier is on every request, its p50/p95 latency adds directly
to user-perceived response time. The brief notes default mode is ~3x faster than alternative
multimodal safety models and THINK mode uses ~50% fewer tokens than other reasoning models —
but THINK still generates a reasoning trace, which costs time. Real-time blocking decisions
cannot wait on reasoning generation.

**In this project this means:** The harness measures and records wall-clock latency for every
call and reports p50/p95 per mode. Default (non-THINK) mode is the path for synchronous
allow/block decisions. THINK is run asynchronously or out-of-band for audit and review (see
S3), never inserted between the user and their response. The product design treats the
classifier latency as a line item in the SLA, with explicit timeout budgets that trigger E2.

### E4. Verification is deterministic and reproducible
**Statement:** Any claim the harness verifies must be reproducible from recorded inputs and
outputs against a pinned model.

**Rationale:** "We tested it and it worked" is worthless if the next run gives a different
answer with no record of why. Verifying the four headline claims (multimodal, multilingual,
THINK, custom-policy) only has evidentiary value if someone else can re-run it and get the
same result, or diff against a prior run to detect drift.

**In this project this means:** Pin the exact model id `nvidia/nemotron-3.5-content-safety:free`.
Set `temperature=0` (and any other determinism levers the endpoint exposes) wherever the API
allows it. Persist the full request (messages, `extra_body`, model id) and the full raw
response for every verification case to a results artifact. Each verified claim links back to
its recorded prompt/response pair. Items the brief marks `[VERIFY]` (exact image block shape,
full taxonomy, the disputed language pair) are resolved against live calls and the resolution
is recorded, not assumed.

### E5. Secrets hygiene and graceful degradation without keys
**Statement:** No secret ever enters the repository; the harness stays useful and honest when
no key is present.

**Rationale:** The only credential is `OPENROUTER_API_KEY`, and leaking it is a real
incident. Separately, contributors and CI must be able to run the non-live parts of the
project (parsing, policy loading, unit tests) without a key, and must get a clear message
rather than a cryptic stack trace when live calls are skipped.

**In this project this means:** The key is read from the environment only. A `.env.example`
documents the variable name with a placeholder and never contains a real value; `.env` is
git-ignored. When the key is absent, the harness prints a clear, single-line explanation,
skips live calls, and still runs offline tests — it does not crash and does not silently
pretend success. Secrets are never logged, never echoed into results artifacts.

### E6. Calls are idempotent and observable
**Statement:** Classifier calls should be safe to retry and easy to inspect.

**Rationale:** Network calls fail. Retries are only safe if a call has no side effects and a
stable identity. Observability is what turns a mysterious production block into a five-minute
investigation — essential when the decision affects whether a user can speak.

**In this project this means:** Each verification case carries a stable identifier so retries
and re-runs are deduplicated and comparable. Calls emit structured logs/metrics: model id,
mode flags, latency, token usage, parse outcome, and final verdict — never the raw user
content unless policy explicitly permits it. Retries use bounded backoff and respect the
fail-closed/fail-open setting (E2) once exhausted.

---

## Part 2 — Safety & Product Principles (the classifier-based product)

### S1. Policy is code
**Statement:** Safety rules are versioned, reviewed, and testable artifacts — not prompt
strings buried in application code.

**Rationale:** This model accepts a natural-language `custom_policy` at inference time (brief
§"Custom policy format"). That power is dangerous if policies live as inline literals: they
drift, nobody reviews them, and there is no record of what was enforced when an incident
occurred. Treating policy as a deployable artifact makes safety behavior auditable and
changeable without a code release.

**In this project this means:** Custom policies live in their own versioned files (named,
described, with explicit Disallowed/Allowed behaviors per the brief's format), are code-
reviewed like any other change, and each ships with test cases that assert expected verdicts.
A change to a policy is a tracked diff with an author and a reason. The harness loads policies
from these artifacts and can report which policy version produced which decision.

### S2. Defense in depth — the model is one layer, not the control
**Statement:** The classifier is a single layer in a larger control; never the whole control.

**Rationale:** A 4B fine-tuned model is strong but not infallible — benchmarks in the brief
sit in the 85–96% range, meaning a meaningful tail of errors in both directions. A system
that bets everything on one probabilistic component has no margin for that tail, and no
defense against prompt injection aimed at the classifier itself.

**In this project this means:** Pair the model with deterministic layers — allow/deny lists
for known-good and known-bad patterns, rate limits to blunt abuse and probing, and mandatory
human review routing for high-risk categories (e.g. Self-Harm, Criminal Planning). The model
narrows what humans must look at; it does not replace the other layers. No single layer can
unilaterally allow high-risk content.

### S3. Auditability and explainability via THINK, off the blocking path
**Statement:** Use reasoning traces for compliance logs and review, not for real-time
blocking.

**Rationale:** THINK mode produces a concise reasoning trace (brief: ~≤3 sentences, ~50%
fewer tokens than peers). That trace is gold for audits, appeals, and debugging false
positives — but generating it costs latency, so it cannot sit on the synchronous decision
path (E3). Explainability and speed are reconciled by separating *when* each runs.

**In this project this means:** The fast default-mode verdict drives real-time allow/block.
For decisions that are blocked, high-risk, or appealed, run THINK asynchronously to attach a
reasoning trace to the audit record. Traces are stored as compliance evidence and surfaced to
reviewers — never used as the gate the user waits on.

### S4. Context-appropriate strictness — least-restrictive-that-is-safe
**Statement:** The same model, different policies per surface; choose the least restrictive
policy that is still safe for that surface.

**Rationale:** A DevOps assistant must discuss exploits, credentials, and "criminal planning"-
adjacent security topics that a kids' app must hard-block. One global policy is either too
loose for the child or too tight for the engineer. The model's programmable policy makes
per-surface strictness practical; over-blocking is itself a harm (it erodes trust and pushes
users to unguarded tools).

**In this project this means:** Each product surface declares its own policy artifact (S1).
Strictness is tuned per surface to the minimum that keeps that surface safe — not maxed out
by default. The harness demonstrates this explicitly by running the same input through two
contrasting policies and showing divergent, intentional verdicts.

### S5. Measure what matters — cost-asymmetric, per-category, drift-aware
**Statement:** Measure false positives and false negatives separately, set thresholds per
category, and monitor for drift continuously.

**Rationale:** Not all errors cost the same. A false negative on Self-Harm is catastrophic; a
false positive on Spam is a mild annoyance. A single accuracy number hides this asymmetry. And
model behavior, attack patterns, and content distributions all drift over time, so a system
validated once is not validated forever.

**In this project this means:** Evaluation reports false-positive and false-negative rates
**per category**, not just aggregate accuracy, and weights them by category cost. Decision
thresholds are set per category to reflect that asymmetry. The product monitors live verdict
distributions for drift and treats red-teaming as a continuous loop, not a launch gate —
adversarial findings feed back into policy artifacts (S1) and test suites (E1).

### S6. Multilingual and multimodal fairness — no blind spots
**Statement:** Don't let zero-shot languages or image+text interactions become unmonitored
gaps where harmful content slips through.

**Rationale:** The model is explicitly trained on ~12 languages but reaches ~140 zero-shot
via the Gemma 3 base (brief §"Languages"), and it accepts text+image input. Quality is
demonstrably lower outside the trained set and at the image/text boundary (multimodal ~85%).
If we deploy globally and multimodally while only validating English text, we have a fairness
and safety gap that disproportionately affects non-English users and image-borne harms.

**In this project this means:** Verification covers the trained languages *and* samples
zero-shot languages, and explicitly tests image+text cases (including the documented
`image_url` vs `image` block difference, which is `[VERIFY]` per the brief). The disputed
language pair is resolved against the live model and recorded. The product flags lower-
confidence surfaces (zero-shot languages, multimodal) for stricter fallback layers (S2)
rather than treating all inputs as equally well-covered.

### S7. Human-in-the-loop, appeals, and transparency
**Statement:** Keep humans in the loop for consequential decisions, give users a way to
appeal, and tell users when their content is blocked.

**Rationale:** Automated blocking at scale will make mistakes (S5). Silent blocking erodes
trust and gives users no recourse; opaque moderation is both an ethical and a product
failure. Human review and appeals are the safety valve that keeps an automated system
legitimate, and transparency is the minimum users are owed when their speech is restricted.

**In this project this means:** High-risk and contested decisions route to human reviewers
(S2). The product provides an appeals path, and appeal outcomes feed back into test cases and
policy tuning. When content is blocked, the user is told that it was blocked and, where safe
to disclose, the category — backed by the THINK trace (S3) for the reviewer, not necessarily
shown verbatim to the user. Reviewers see the reasoning trace; users see a clear, respectful
explanation.

---

## How the principles interact

These principles reinforce each other rather than standing alone:

- **Speed vs. explainability** is resolved by running fast default-mode on the hot path (E3)
  and THINK off-path for audit (S3).
- **Trust vs. fallibility** is resolved by defense in depth (S2) plus cost-asymmetric, drift-
  aware measurement (S5) plus human appeals (S7).
- **Power vs. governance** of programmable policy is resolved by policy-as-code (S1) and per-
  surface least-restrictive strictness (S4).
- **Correctness vs. messiness of real output** is resolved by defensive parsing (E1),
  explicit failure modes (E2), and reproducible verification (E4).

When two principles appear to conflict, the tie-breaker is the project thesis: stay capable,
fast, and trustworthy simultaneously — and when forced to choose under uncertainty, default
to the configured fail-closed posture (E2) for high-risk surfaces.
