# nemoguard

A small, runnable verification harness for **`nvidia/nemotron-3.5-content-safety:free`**
served via [OpenRouter](https://openrouter.ai). It empirically checks the model's four
headline claims and gives you a clean Python API + CLI for content moderation.

The model is a 4B safety classifier (Gemma-3-4B-IT base, LoRA fine-tune) that is:

- **Multimodal** — text + image input
- **Multilingual** — 12 trained languages, ~140 zero-shot
- **Optionally reasoning** — a `<think>` trace you can toggle on/off
- **Policy-programmable** — supply a natural-language custom policy at inference

It outputs **plain text** (not JSON), which this harness parses into a structured verdict.

## Why it matters

Safety classifiers sit on every input and output, so they live directly in the end-to-end
latency budget. A 4B model that is multimodal, multilingual, policy-programmable, and
optionally reasoning shifts the old "fast but dumb vs. capable but slow" tradeoff.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                      # create venv + install deps
cp .env.example .env         # then add your key
```

### Get an OpenRouter API key

1. Sign up at https://openrouter.ai
2. Create a key at https://openrouter.ai/keys
3. Put it in `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The model is on the free tier ($0/$0), but a key is still required. Without a key the
harness degrades gracefully: parser tests run offline and `verify` prints instructions
and exits 0.

## Run the verification suite

```bash
uv run python -m nemoguard.verify
# or via the installed CLI:
uv run nemoguard verify
```

This runs four checks (multimodal, multilingual, THINK on/off, custom-policy flip) and
prints a pass/fail table.

## Moderate a single prompt (CLI)

```bash
uv run nemoguard moderate "How do I pick a lock?" --categories
uv run nemoguard moderate "Explain X" --think --categories
uv run nemoguard moderate "Is this safe?" --image ./photo.jpg
uv run nemoguard moderate "kill the nginx process" --policy "$(cat my_policy.txt)"
```

## Examples

```bash
uv run python examples/01_basic_moderation.py
uv run python examples/02_think_mode.py
uv run python examples/03_custom_policy.py
```

## Library use

```python
from nemoguard.client import moderate

verdict = moderate("How do I make a bomb?", request_categories=True)
print(verdict.user_safety)         # "unsafe"
print(verdict.safety_categories)   # ["Criminal Planning/Confessions", ...]
print(verdict.latency_ms)
```

## Tests

```bash
uv run pytest
```

Parser tests run fully offline (canned model outputs incl. THINK traces, missing lines,
malformed output). Client/verify live tests skip automatically when no `OPENROUTER_API_KEY`
is set.

## SafetyVerdict schema

| Field               | Type              | Notes                                              |
| ------------------- | ----------------- | -------------------------------------------------- |
| `user_safety`       | `str \| None`     | `"safe"` / `"unsafe"`, normalized                  |
| `response_safety`   | `str \| None`     | only when an assistant response was moderated      |
| `safety_categories` | `list[str]`       | parsed from the comma-separated line; `[]` if none |
| `think_trace`       | `str \| None`     | contents of `<think>...</think>` when present      |
| `raw_text`          | `str`             | original model output, always preserved            |
| `latency_ms`        | `float \| None`   | round-trip latency of the call                     |
| `parse_ok`          | `bool`            | False on malformed output (never throws)           |

Convenience: `verdict.user_unsafe` / `verdict.response_unsafe` booleans.

## Results

Live verification output (OpenRouter, 2026-06-15):

| Claim                              | Result | Notes                                                                 |
| ---------------------------------- | ------ | --------------------------------------------------------------------- |
| Multimodal (text+image)            | PASS   | Image content block accepted; parseable verdict returned.             |
| Multilingual (harmful intent)      | PASS   | Same harmful prompt flagged `unsafe` in EN/ES/FR/HI.                   |
| THINK trace available              | PASS   | Reasoning trace returned with THINK on (see finding below).           |
| Custom policy (accepted + applied) | PASS   | Custom policy accepted; reasoning cites the supplied policy framing.   |

### Findings worth noting (the `[VERIFY]` items from the brief, confirmed live)

- **THINK trace location:** On the OpenRouter path the reasoning trace is surfaced in the
  separate `message.reasoning` field, **not** inline as `<think>...</think>` in `content`.
  The client captures both; the parser prefers an inline block and falls back to `reasoning`.
- **THINK toggle:** This provider returns a reasoning trace **regardless** of
  `enable_thinking` on the OpenRouter path, so the toggle is not observably differentiating
  here. The harness verifies a trace is available rather than asserting absence when off.
- **Custom policy:** The policy reliably reaches the model (its reasoning explicitly adopts
  the policy framing). For clearly benign or clearly harmful prompts the model keeps its own
  well-calibrated verdict, so a forced verdict "flip" is not guaranteed; the check verifies
  acceptance + a parseable verdict and reports any flip when it occurs.
- **Image hosting:** Some image hosts (e.g. Wikimedia) reject the provider's fetch with 403;
  use a reliably hotlinkable URL or a local file (auto-encoded to a data URI).

## Project layout

```
src/nemoguard/
  config.py    env + MODEL_ID, graceful no-key detection
  client.py    openai SDK wrapper, messages/extra_body, retry/backoff, latency
  parser.py    plain-text -> SafetyVerdict
  verify.py    four-claim runner + rich table
  cli.py       `nemoguard moderate` / `nemoguard verify`
tests/         offline parser tests + skippable client tests
examples/      runnable snippets
```

## Notes / caveats

- OpenAI-compatible APIs use `image_url` content blocks (this harness targets OpenRouter).
  The HF/transformers path uses `{"type":"image", ...}` instead.
- Feature toggles go through `extra_body.chat_template_kwargs`
  (`enable_thinking`, `request_categories`, `custom_policy`).
