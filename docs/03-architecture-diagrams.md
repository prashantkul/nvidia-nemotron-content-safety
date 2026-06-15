# Architecture Diagrams — Nemotron 3.5 Content Safety

Diagrams are grounded in `docs/00-PROJECT-BRIEF.md`. The verification harness calls
`nvidia/nemotron-3.5-content-safety:free` via OpenRouter (OpenAI-compatible Chat Completions).
The co-build product is a guardrail gateway operated by a security startup on top of NVIDIA's
model + NIM + GPU.

---

## 1. Context diagram

How the pieces talk to each other end to end. The harness owns the orchestration: it builds
OpenAI-compatible requests (with `extra_body.chat_template_kwargs` toggles), sends them over
HTTPS to OpenRouter, and OpenRouter routes inference to the Nemotron safety model. The model
returns **plain text** (not JSON), which the harness parses into a structured verdict before
handing it back to the client app.

```mermaid
flowchart LR
    Client["Client app / CLI"]
    Harness["Verification harness<br/>(Python, uv)"]
    OR["OpenRouter<br/>api/v1 (OpenAI-compatible)"]
    Model["nvidia/nemotron-3.5-<br/>content-safety:free<br/>(Gemma-3-4B, LoRA)"]

    Client -->|"prompt / image / policy"| Harness
    Harness -->|"Chat Completions + extra_body<br/>(Bearer OPENROUTER_API_KEY)"| OR
    OR -->|"routes inference"| Model
    Model -->|"plain-text safety verdict"| OR
    OR -->|"text response"| Harness
    Harness -->|"parsed Verdict object"| Client
```

---

## 2. Component diagram — the harness

Internal structure of the verification harness. The **config loader** reads `OPENROUTER_API_KEY`
(degrading gracefully when absent), the **request builder** assembles messages + `extra_body`,
the **OR client** wraps the `openai` SDK pointed at the OpenRouter base URL, the **plain-text
parser** converts the response into a **verdict model**, and the **test runner** drives the four
capability checks. The parser is the load-bearing component because the model emits text, not JSON.

```mermaid
flowchart TD
    subgraph Harness["Verification harness"]
        Config["Config loader<br/>(.env, OPENROUTER_API_KEY)"]
        Builder["Request builder<br/>(messages + extra_body)"]
        ORClient["OpenRouter client<br/>(openai SDK, base_url)"]
        Parser["Plain-text parser<br/>(regex -> fields)"]
        Verdict["Verdict model<br/>(user_safety, response_safety,<br/>categories, think_trace)"]
        Runner["Test runner<br/>(4 capability checks)"]
    end

    Config --> ORClient
    Runner --> Builder
    Builder --> ORClient
    ORClient -->|"text"| Parser
    Parser --> Verdict
    Verdict --> Runner
    Config -.->|"no key: skip live calls"| Runner
```

---

## 3. Sequence diagram — one moderation call WITH THINK mode

A single moderation call with reasoning enabled. The builder sets
`chat_template_kwargs.enable_thinking = true` (plus `request_categories`), the model returns a
`<think>...</think>` block followed by the verdict lines, and the parser splits the reasoning
trace from the structured fields. Key insight: THINK adds a reasoning preamble to the **same**
plain-text response — there is no separate channel, so parsing must strip the `<think>` block.

```mermaid
sequenceDiagram
    participant C as Client
    participant B as Request builder
    participant OR as OpenRouter
    participant M as Nemotron model
    participant P as Parser

    C->>B: moderate(prompt, think=true, categories=true)
    B->>OR: POST /chat/completions<br/>messages + extra_body.chat_template_kwargs<br/>{enable_thinking:true, request_categories:"/categories"}
    OR->>M: routed inference
    M-->>OR: "&lt;think&gt;...reasoning...&lt;/think&gt;<br/>User Safety: unsafe<br/>Response Safety: unsafe<br/>Safety Categories: Controlled Substances"
    OR-->>P: plain-text response
    P->>P: split think trace from verdict lines
    P-->>C: Verdict{user_safety:unsafe,<br/>categories:[...], think_trace:"..."}
```

---

## 4. The four-capabilities map

The four headline claims the harness verifies, each with the toggle that activates it and the
observable output that confirms it. Multimodal and multilingual are driven by **input shape**
(content blocks / language of text), while THINK and custom-policy are driven by **`extra_body`
toggles**. Each row is independently testable.

```mermaid
flowchart LR
    Root["Nemotron 3.5<br/>Content Safety"]

    Root --> MM["Multimodal"]
    Root --> ML["Multilingual"]
    Root --> TH["THINK (reasoning)"]
    Root --> CP["Custom policy"]

    MM -->|"toggle: image_url content block"| MMo["output: verdict on image+text<br/>(~85% multimodal avg)"]
    ML -->|"toggle: non-English prompt text"| MLo["output: verdict in 12 trained langs<br/>(~140 zero-shot)"]
    TH -->|"toggle: enable_thinking=true"| THo["output: &lt;think&gt;...&lt;/think&gt; + verdict<br/>(~50% fewer tokens)"]
    CP -->|"toggle: custom_policy=text"| CPo["output: verdict vs supplied policy<br/>(allowed/disallowed behaviors)"]
```

---

## 5. Productionized guardrail gateway (co-build product)

The product a security startup operates. Every client call passes through the gateway twice:
an **input check** before the LLM and an **output check** after. The gateway reads a **policy
store** (policy-as-code), uses a **cache** to skip repeat checks, and applies a **fail-open vs
fail-closed** decision when the classifier is unavailable. THINK runs **asynchronously** into an
audit log so reasoning never blocks the response. Inference can hit a **self-hosted NIM on NVIDIA
GPU** or fall back to **OpenRouter**.

```mermaid
flowchart TD
    Client["Client app"]
    GW["Guardrail gateway<br/>(control plane)"]
    LLM["Application LLM"]
    Policy["Policy store<br/>(policy-as-code)"]
    Cache["Verdict cache"]
    Audit["Async THINK audit log"]
    NIM["Self-hosted NIM<br/>(NVIDIA GPU)"]
    ORfb["OpenRouter (fallback)"]

    Client -->|"1. request"| GW
    GW -->|"2. input check (sync, binary)"| Safety{"Safety classifier"}
    Policy --> GW
    Cache <--> GW
    Safety --> NIM
    Safety -.->|"fallback"| ORfb
    GW -->|"3. allowed -> forward"| LLM
    LLM -->|"4. model output"| GW
    GW -->|"5. output check (sync, binary)"| Safety
    GW -->|"6. response"| Client
    Safety -.->|"THINK trace (async)"| Audit
    GW -->|"classifier down:<br/>fail-open or fail-closed"| Decision["Policy-driven decision"]
```

---

## 6. Latency-budget diagram

Why a 4B reasoning-capable safety model changes the tradeoff. The **sync hot path** returns a
**binary verdict** with low latency (default mode ~3x lower latency than other multimodal safety
models). The **async path** fires the **THINK reasoning trace** off the critical path into the
audit log, so explainability is captured without adding latency to the user-visible request.

```mermaid
flowchart LR
    In["Request in"]

    subgraph Hot["Sync hot path (in latency budget)"]
        direction LR
        IC["Input check<br/>(binary verdict)"]
        L["LLM call"]
        OC["Output check<br/>(binary verdict)"]
        IC --> L --> OC
    end

    In --> IC
    OC --> Out["Response out (low latency)"]

    subgraph Cold["Async path (off hot path)"]
        TH["THINK reasoning trace"]
        AL["Audit / compliance log"]
        TH --> AL
    end

    IC -.->|"enqueue"| TH
    OC -.->|"enqueue"| TH
```

---

## 7. Co-build value split

Who builds what. NVIDIA supplies the foundational ML assets — model, NIM packaging, GPU
hardware, and the Aegis 2.0 taxonomy/dataset. The startup builds the operational product around
it — the gateway, policy-as-code, audit/compliance tooling, and customer integrations. The split
lets each side do what it does best: NVIDIA owns the model and silicon, the startup owns the
control plane and the customer relationship.

```mermaid
flowchart LR
    subgraph NVIDIA["NVIDIA provides"]
        M["Safety model<br/>(Nemotron 3.5, Gemma-3-4B)"]
        N["NIM (deployable microservice)"]
        G["GPU hardware"]
        T["Aegis 2.0 taxonomy + dataset"]
    end

    subgraph Startup["Startup builds"]
        GW["Guardrail gateway / control plane"]
        PaC["Policy-as-code engine"]
        AC["Audit + compliance layer"]
        INT["Customer integrations / SDKs"]
    end

    M --> GW
    N --> GW
    G --> N
    T --> PaC
    GW --> AC
    GW --> INT
```
