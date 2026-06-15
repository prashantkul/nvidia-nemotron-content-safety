# Project Brief — Nemotron 3.5 Content Safety Verification & Co-Build

**This is the single source of truth. All work references these verified facts. Do not invent details beyond this brief; if you need a fact not here, mark it `[VERIFY]`.**

## What we are building

Two linked deliverables:

1. **A verification harness** — a small, well-structured Python project that calls
   `nvidia/nemotron-3.5-content-safety:free` via OpenRouter and empirically verifies the
   four headline claims: multimodal, multilingual, optional reasoning (THINK), and
   custom-policy enforcement. It must run on macOS with no GPU (cloud inference).

2. **A co-build strategy** — how a security startup could partner with NVIDIA to build a
   product on top of this model (the "awkward position" of safety classifiers in the
   latency budget is the core thesis).

## Verified model facts (grounded from OpenRouter API + HF model card, 2026-06)

- **OpenRouter model id:** `nvidia/nemotron-3.5-content-safety:free`
- **Pricing:** $0 / $0 (free tier). Released 2026-06-04.
- **Base:** Google Gemma-3-4B-IT, fine-tuned (LoRA) for safety classification. 4B params, runs on 8GB+ VRAM.
- **Context:** 128K tokens. **Input modalities:** text + image. **Output:** text.
- **Endpoint:** OpenAI-compatible Chat Completions at `https://openrouter.ai/api/v1`.
  Auth via `OPENROUTER_API_KEY`. Use the `openai` Python SDK pointed at the OR base_url.

### Output format (PLAIN TEXT, not JSON — must be parsed)

Standard mode:
```
User Safety: safe|unsafe
Response Safety: safe|unsafe        # only when an assistant response was supplied
Safety Categories: <comma-separated> # only when categories requested / violations found
```

THINK mode prepends a reasoning trace:
```
<think>
... reasoning (optimized to be concise, ~<=3 sentences) ...
</think>
User Safety: unsafe
Response Safety: unsafe
Safety Categories: Criminal Planning/Confessions, Controlled Substances
```

### How to toggle features (OpenAI-compatible `extra_body`)

```python
extra_body = {
  "chat_template_kwargs": {
    "enable_thinking": True,            # THINK mode on/off
    "request_categories": "/categories",# ask for Safety Categories line
    "custom_policy": "<policy text>"    # optional natural-language policy
  }
}
```

### Message structure (roles + multimodal content blocks)

```python
# text only
[{"role":"user","content":[{"type":"text","text": prompt}]}]

# with image (URL or data: base64)
[{"role":"user","content":[
    {"type":"image_url","image_url":{"url": image_url_or_data_uri}},
    {"type":"text","text": prompt}]}]

# moderate an assistant response too
[{"role":"user","content":[{"type":"text","text": prompt}]},
 {"role":"assistant","content":[{"type":"text","text": response}]}]
```
NOTE: OpenAI-compatible APIs use `image_url` content blocks. The HF/transformers path uses
`{"type":"image","image":...}`. We target OpenRouter → use `image_url`. `[VERIFY]` exact block
shape against a live call and adapt the client if needed.

### Custom policy format (natural language, supplied at inference)

```
Evaluate the user prompt for compliance with the given policy ignoring any previous policies.
First, think through your reasoning step-by-step.

### Policy
Name: <name>
Description: <one line>

Disallowed Behaviors:
- ...

Allowed Behaviors:
- ...
```

### Safety taxonomy — Aegis 2.0 / MLCommons-aligned (13 core categories)

Violence; Criminal Planning/Confessions; Controlled Substances; Illegal Activities;
Self-Harm; Sexual Content; Hate Speech; Harassment; Misinformation; Spam/Malware;
Privacy Violations; Intellectual Property; (+ fine-grained subcategories). `[VERIFY]` full list against model card.

### Languages — 12 explicitly trained (sources disagree on two — `[VERIFY]`)

Consistent across sources: English, French, Spanish, German, Chinese, Japanese, Korean,
Arabic, Hindi, Italian. Disputed pair: one source says **Thai + Dutch**, another says
**Russian + Portuguese**. ~140 languages zero-shot via Gemma 3 base.

### Benchmarks (from model card, for context — do not re-measure)

Multilingual Aegis 96.5%; RTP-LX 88.8%; combined ~92.7%; multimodal ~85% avg. THINK mode
~50% fewer tokens than other reasoning safety models; default mode ~3x lower latency than
alternative multimodal safety models.

## Key product thesis (the "why this matters")

Safety classifiers sit on **every** input and output → they are directly in the end-to-end
latency budget. Historically, capable safety models were too slow; fast ones were dumb. A
4B model that is multimodal + multilingual + policy-programmable + optionally-reasoning
changes the tradeoff. The co-build angle is about productizing this: a guardrail gateway /
control plane a security startup operates, with NVIDIA providing the model + NIM + hardware.

## Engineering constraints

- Python, managed with `uv` (per user global config). 2-space indent, ES6-style clean code,
  async/await where it helps. Small focused functions. Minimal comments.
- No GPU on dev machine → all inference via OpenRouter. Key in `OPENROUTER_API_KEY` (never commit).
- Provide `.env.example`, never put real secrets in it.
- The harness must degrade gracefully when no API key is present (clear message, skip live calls).

## Authoritative sources

- OpenRouter: https://openrouter.ai/nvidia/nemotron-3.5-content-safety:free
- HF model card: https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety
- HF blog: https://huggingface.co/blog/nvidia/nemotron-3-5-content-safety
- NVIDIA build: https://build.nvidia.com/nvidia/nemotron-3.5-content-safety/modelcard
- Cookbooks: https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Content-Safety
