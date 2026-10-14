# iOS on-device email agent — viability POC: results & verdict

**TL;DR:** The architecture is viable on-device — Apple Foundation Models and
MLX both run the agentic email loop in-process and private. On memory: **3B-4bit
fits once MLX's GPU buffer cache is capped; 7B-4bit also fits on an 8 GB device
with the increased-memory-limit entitlement** (our earlier "7B doesn't fit"
call was wrong — the first crash was the default per-app cap, not a physical-RAM
wall), **but 7B runs ~4× slower (~7 tok/s, ~2 min per answer).** On reliability,
the headline holds and sharpens: **no model we tested reliably drives the loop.**
The ~3B models (Apple 2-bit, MLX 4-bit) fabricate facts, mis-select evidence,
invent search constraints, and ignore guardrails — consistently. The 7B is
clearly better at *reasoning*: it finds the hard adversarial email and drops the
crude mistakes — **yet it still fabricates**, now by fusing two unrelated emails
into one coherent-sounding order. And bigger is **not** monotonically better:
the largest model tested (Qwen 14B) was among the *worst*, because it kept
**narrating tool calls instead of making them** — proving tool-use and
loop-following are *training* properties, not *scale* properties. Across every
model the failures fall into four buckets — can't plan the query, can't confine
the answer, won't drive the loop, or emits unparseable tool calls. The path forward is the same regardless:
move decision-making and retrieval out of the model into deterministic code, and
constrain/verify even the final answer-synthesis step. The model should format
verified evidence, not decide or act.

## What the POC tested

A throwaway SwiftUI app running the same agent design as the Python POC
(bootstrap → structured search → inspect → answer) against a **fake in-memory
mailbox** (to isolate *model* viability from mail plumbing). Tools:
`resolve_relative_date`, `search_email_structured`, `search_email`,
`get_message`. Pluggable model behind a `ModelProvider` protocol:

- **Apple** — Foundation Models (built-in ~3B, 2-bit), in-process tool calling.
- **MLX** — on-device models via MLX Swift, real tool-calling loop. A model
  dropdown selects any of ~13 curated `mlx-community` models (0.5B → 14B) or a
  custom HuggingFace id, downloaded on first use.
- **Mock** — scripted, for plumbing checks.

Runs on both a physical **iPhone** (the viability target) and a native **macOS**
build (for iterating on reliability and running larger models — 8B/14B — that a
phone can't hold). Mac results speak to model *reliability* (device-independent),
not phone *viability*.

The fake mailbox includes an adversarial case: a real wine-club shipment whose
subject says "Slurpers shipment" (no keyword "wine"), a wine *promotion* that is
not an order, a clear clothing order, and an out-of-window wine order.

## Architecture result: VIABLE

- Both Apple Foundation Models and MLX run the full agent loop **on-device,
  in-process, with no network** — matching the privacy/E2EE goal.
- MLX's tool API (`Tool<Input,Output>`, `UserInput(tools:)`, streamed
  `.toolCall`) drives the multi-step loop. Foundation Models drives it via
  `LanguageModelSession`.
- No architectural blocker. The desktop Python POC already proved the end-to-end
  system with a capable model; iOS proves it *runs* on the phone.

## Performance result (MLX 4-bit 3B, on device)

Measured on-device (resident memory, headroom-before-jetsam, throughput):

| Run | Load | Total | Throughput | Resident | Headroom |
| --- | --- | --- | --- | --- | --- |
| gen-only (1 pass)              | 6.9 s | 14.5 s | 23.5 tok/s | 1.84 GB | 937 MB |
| tool loop (3 turns)            | 9.5 s | 36.6 s | 27.2 tok/s | ~0.7 GB | **111 MB** |
| tool loop (2 turns)            | 9.3 s | 25.4 s | 27.4 tok/s | ~0.4 GB | 249 MB |
| tool loop (cache cap, 2 turns) | 10.0 s | 21.8 s | 26.6 tok/s | 0.30 GB | **1.52 GB** |

- **~27 tokens/sec** decode — usable.
- A **multi-turn agent loop costs 22–37 s** (each turn re-runs generation);
  agentic latency is multiplicative, not additive.

### The memory story (important — a correction)

Early 3B tool-loop runs showed headroom collapsing to ~110–250 MB, and one
*unchanged* repeat run was OS-killed. That looked like a hard 3B ceiling. It was
not — it was **MLX's GPU buffer cache accumulating across runs.** MLX keeps
freed Metal buffers for reuse; without a ceiling the cache grows unbounded and
eventually trips jetsam even though the model itself didn't change.

**Fix:** `MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)` at load (the same guard
LLMEval uses). After this, the same 3B run holds at **~300 MB resident with
~1.5 GB headroom** — comfortably within budget. 3B-4bit is genuinely viable
on-device.

### 7B-4bit: DOES fit — with the increased-memory-limit entitlement

First attempt: with the GPU cache fixed but no memory entitlement, we tried
`Qwen2.5-7B-Instruct-4bit` (~4.3 GB weights). It loaded, then **crashed
mid-generation with no exception and no log** — a memory kill, not a code fault.
The jetsam report said **`vm-compressor-thrashing`**, and we initially concluded
this was a hard physical-RAM wall that no entitlement could fix.

**That conclusion was wrong.** Adding the **Increased (Debugging) Memory Limit**
entitlement (`com.apple.developer.kernel.increased-debugging-memory-limit`,
added via Signing & Capabilities, applies to development/debugger runs) let the
7B **run to completion**. On the test device (iPhone 17, 8 GB RAM) the 7B then
reported **~2.3–2.5 GB headroom** throughout — no jetsam, no thrashing.

So the earlier kill was the app hitting iOS's **default per-app cap**, which is
set conservatively below physical RAM. The entitlement raises that cap toward
the device's real limit; it does not create RAM, but on an 8 GB device there was
headroom to be unlocked. (On a lower-RAM device the result could differ.)

The cost is **throughput**: the 7B ran at **~7.2 tok/s and ~127 s total** for a
two-turn answer — roughly **4× slower** than the 3B's ~27 tok/s. (Resident
memory as sampled read *lower* than the 3B — ~40–165 MB — because MLX mmaps the
weights and file-backed pages are not counted the same way; headroom is the
meaningful figure here, not resident.)

| Model | Fits? | Entitlement needed | Throughput | Headroom |
| --- | --- | --- | --- | --- |
| 3B-4bit | yes | no (just GPU cache cap) | ~27 tok/s | ~1.5 GB |
| 7B-4bit | yes | **yes** (increased memory limit) | ~7 tok/s | ~2.3–2.5 GB |

**Corrected conclusion:** both 3B and 7B are *runnable* on an 8 GB device. 7B
needs the memory entitlement and pays a ~4× latency penalty (a two-minute answer
is a real UX problem), but it is not blocked by memory as we first thought. The
remaining question is reliability, below — where 7B is better but still not
correct.

## Reliability result: FAILS (the headline)

The agent task is: pick the right evidence and reason correctly ("shipping is
not delivery"; a promo is not an order; apply/omit date windows as asked). Every
~3B configuration failed this, each differently — and the 7B, which *does* run
on-device with the memory entitlement, improved but still failed:

1. **Apple 2-bit 3B** — invented a date window the question didn't ask for
   (`after=2026-10-05`), set `merchant=wine` (product-as-merchant), ignored the
   "no window" guardrail; **inconsistent across identical questions**.
2. **MLX 4-bit 3B, generation-only** — looked correct, but only because we
   *removed* the tool-decision step and handed it the mailbox inline. Still
   answered about the wrong (clothing) order, not the wine question.
3. **MLX 4-bit 3B, tools, run A** — called the **wrong tool** (`search_email`
   instead of structured) and **fabricated an operator syntax**
   (`intent:purchase merchant:wine brand:example after:...`), inventing both a
   merchant and a date window.
4. **MLX 4-bit 3B, tools, run B** — fixed tool *mechanics* (right tool, correct
   3-month window), then **fabricated a shipment**: returned the Banana Republic
   *clothing* order and asserted "your order shipped on 2026-09-26" — conflating
   the order-placed date with shipping, about the wrong product, with full
   confidence. The explicit "shipping is not delivery" instruction was ignored.
5. **MLX 4-bit 3B, tools, run C** (after the cache-limit fix, clean memory) —
   called the right tool but with a **useless query**: searched `search_email`
   for the literal keywords `"order ship"`, which match nothing in the mailbox,
   then concluded "no shipping email found, please provide an order number." A
   graceful failure rather than a hallucination this time — but still the wrong
   search strategy (it should have filtered by merchant/product or searched
   "wine"), so it never found the relevant orders at all.
6. **MLX 4-bit 3B, tools, run D** (head-to-head, same question as Apple below) —
   structured search with the computed 3-month window (good mechanics), but
   `query="order ship"` and no product filter, so it grabbed the **wrong order
   entirely**: Banana Republic *clothing* (503). It then asserted the order
   "shipped on 2026-09-26" AND "placed on 2026-09-26" — the same date for two
   different events, because 503 only states a *placed* date (Sept 26, est.
   delivery Oct 1) and never mentions shipping. Confidently wrong product,
   fabricated ship event, "shipping ≠ delivery" guardrail ignored.
7. **Apple 2-bit 3B, tools, run B** (head-to-head, same question as MLX run D) —
   *found the right evidence*: bootstrapped, correctly selected the no-keyword
   wine order (501, the hard adversarial case), read the body, and **answered
   the core question correctly ("shipped: yes")**. Then embellished with
   fabrication: invented an "order date" from the email's *send* date,
   manufactured a "you were notified on Oct 4" event, and dropped the stated
   six-bottle count while listing only three items. Right conclusion, confident
   fiction in the supporting detail.
8. **Apple 2-bit 3B, tools, run C** (question reworded to "Did my wine orders
   get sent?") — the most incoherent run of all. Searched `merchant=wine`
   (product-as-merchant again), then returned two "wine orders," one of which it
   *itself labelled as not wine*: "Order 2... but this is not a wine order. It's
   a burger order from Shake Shack" (506) — yet it still counted that order and
   concluded "both orders have been sent." It flagged its own error and ignored
   the flag. It also surfaced the deliberately out-of-window Fortnum order (504,
   May 2) and **missed the one genuine recent wine shipment entirely** (501,
   Purple Pachyderm — the `merchant=wine` filter cannot match it). "Both orders
   have been sent" is fabricated; neither 504 nor 506 mentions shipping.
9. **MLX 4-bit 7B, tools** (with the memory entitlement; "Did my wine order get
   sent?") — the **best run of the project, but still a fail.** For the first
   time a model **found email 501** — the hard no-keyword Purple Pachyderm wine
   shipment that every 3B run missed — and did *not* use the bogus
   `merchant=wine` filter (it searched `intent=purchase` with no merchant,
   the sensible move). But it then **welded two unrelated emails into one
   order**: it reported "your wine order from Fortnum & Mason was confirmed May
   2... a shipment notification was sent October 4," fusing 504 (Fortnum,
   out-of-window English sparkling) with 501 (Purple Pachyderm club shipment) as
   if the Fortnum order shipped on Oct 4. It also surfaced the out-of-window 504
   as relevant. Better retrieval, subtler failure: instead of inventing a fact
   outright it fabricated a *relationship* between real emails — still wrong,
   still confident.
10. **MLX 4-bit Mistral 7B v0.3, tools** ("How many wine orders in the last 3
    years?") — a *new* failure mode: it **didn't call any tool at all.** Instead
    of invoking `search_email_structured`, it wrote the user a prose tutorial on
    *how to* call it, including a JSON snippet with `merchant: "wine"`
    (product-as-merchant again) and nonsense dates (`after 2018-01-01, before
    2021-01-01` — 5–8 years ago, ignoring the pre-computed windows). One turn,
    `final`, zero tool calls. The lesson: **tool-calling is model-specific, not
    just size-specific.** Mistral 7B v0.3 is a capable model but is not tuned for
    the structured function-calling convention, so it treats the tools as
    documentation to describe rather than functions to invoke — whereas the
    similarly-sized Qwen 7B (run 9) called them fine. (This is why Mistral's
    function-calling-tuned **Ministral 8B** is the one worth testing next.)
11. **MLX 4-bit Qwen2.5 14B, tools** (the *largest* model tested, run on the Mac
    target) — size did **not** buy tool-loop reliability; it was among the worst.
    Observed across **three distinct questions**, same failure class every time —
    the model *narrates* tool calls instead of making them (consistent, not a
    fluke):
    - "Did I get a receipt from Shake Shack recently?" → one turn, `final`, **zero
      tool calls**, 14 tokens: "I will check for any receipts from Shake Shack in
      the last week." It announced intent and stopped.
    - "Has my Banana Republic order been delivered, and when was it placed?" →
      again one turn, `final`, **zero tool calls**, 28 tokens: "I will check your
      Banana Republic order status… First, I'll search for emails…" — narrated and
      stopped.
    - "How many wine orders in the last three months?" → it *did* call
      `resolve_relative_date` correctly (turn 1), got the bounds, then on turn 2
      said "Now that I have the correct date range, I will use
      search_email_structured…" and **stopped** — narrating the next step instead
      of taking it. It proved it *can* call a tool but **cannot sustain the
      loop**: it bails to prose at the point the second call should happen.

    The agent loop treats "no tool call this turn" as "final answer," so the
    narration gets captured as the answer and the loop ends one step short. The
    14B being the worst at this — despite being the largest — is the clearest
    evidence that **tool-calling and loop-following are training properties, not
    scale properties.**
12. **MLX 4-bit Llama 3.2 1B, tools** (the *smallest* model tested — the floor)
    — a *fourth* failure mode, and partly a harness bug: the model **emitted raw
    chat-template control tokens as answer text.** Its output was a stream of
    `<|python_tag|>{…tool JSON…}<|eom_id|><|start_header_id|>assistant…`,
    repeated for four malformed tool-call blocks, 364 tokens until it hit the
    limit. It is *trying* to call tools, but (a) it loses control of the chat
    template so the delimiters leak as text, and (b) our loop's tool-call parser
    doesn't recognize Llama's `<|python_tag|>` format, so it detected **zero**
    tool calls (`turn 1 final`) and captured the template garbage as the
    "answer." The call *contents* are nonsense too: `intent` set to the literal
    enum list `"any|purchase|promotion|reservation|updates"`, a hallucinated
    `sender_domain: "banana republic.com"`, and `message_id: "order_id"` (a
    placeholder, not a real id). This is pattern-matching the *shape* of tool
    calls with zero understanding — the hard floor.

Across runs, the failures split into four buckets: **can't plan the query**
(wrong tool/args), **can't confine the answer** (fabricates facts or
relationships), **won't drive the loop** (describes tool calls in prose — never
calling, or calling once then narrating), and **emits unparseable calls** (raw
template tokens / malformed JSON that the loop can't read, so they leak into the
answer). The usable band is narrow and model-specific: the 1B floor emits
garbage, the 14B ceiling won't emit calls at all, and only the mid-range
(3B/7B, tool-tuned) produces *parseable* calls — which are then wrong or
fabricated. Model *size* moves the planning bucket somewhat; model *tool-tuning*
and chat-template fidelity govern the rest.

> **Harness note:** the Llama 1B run also exposed that our tool-call parsing is
> tied to the format MLX surfaces for Qwen-style calls and does not handle
> Llama's `<|python_tag|>` convention. A production harness would need
> format-aware parsing per model family — another argument for *not* making the
> model responsible for the loop (see recommendation 1).

Four 3B tool-enabled runs, four *different* query-planning / reasoning failures
(fabricated operators → fabricated ship date → ineffective literal keywords →
wrong product with conflated dates). Same root cause: the 3B model cannot
reliably translate the question into the right retrieval strategy *or* confine
its answer to what the evidence says.

### Head-to-head (runs D & B, identical question "Did my wine order ship?")

| Model | Found right evidence? | Answered core question right? | Fabricated detail? |
| --- | --- | --- | --- |
| **MLX 4-bit 3B** | ✗ (grabbed clothing order 503) | ✗ ("shipped" — but wrong order, no ship event exists) | ✓ (invented ship date from placed date) |
| **Apple 2-bit 3B** | ✓ (found no-keyword wine order 501) | ✓ ("shipped: yes") | ✓ (invented order date + notification event, dropped bottle count) |

The split is instructive: **Apple got the evidence right and lied about the
details; MLX got the evidence wrong and lied about the headline.** Neither is
trustworthy, and critically — *even the model that reasoned correctly to the
right answer still fabricated supporting facts with full confidence.* Better
retrieval would not have saved the Apple run; the fabrication is downstream of
evidence selection, in the answer-synthesis step itself.

### And the 7B? Better, but still not trustworthy

The 7B (run 9) is the strongest result here: it finally **finds the hard
adversarial email (501)** that every 3B run missed, and drops the crude
product-as-merchant mistake. But it still fabricates — now by **conflating two
unrelated emails into one coherent-sounding order** (fusing out-of-window
Fortnum 504 with the Purple Pachyderm shipment 501). The failure mode gets
*subtler* as the model grows: a 3B invents a fact you can spot; the 7B invents a
plausible *relationship* between real facts, which is harder to catch. Scaling
the model up improved retrieval and reduced crude errors but did not make the
answer trustworthy — and it cost ~4× in latency.

The trend is unambiguous and consistent across two vendors and two frameworks:
**fixing the model's mechanical mistakes in code (dates, tool dispatch, operator
stripping) just moves the failure one layer deeper into reasoning.** That is a
capability ceiling, not a prompt-tuning gap.

### What did NOT fix it

- Prompt guardrails (pre-computed date windows; "omit dates if none asked";
  "merchant is not a product"; "shipping ≠ delivery"; "promotions aren't
  evidence"; anti-copy-paste placeholders). Each reduced a specific error; the
  model substituted a new one.
- Moving from Apple's 2-bit 3B to MLX's higher-precision 4-bit 3B.
- Making tools forgiving of malformed input.

### What *did* help (and the lesson)

Everything that worked moved a decision **out of the model into code**:
- date math (pre-computed windows injected as literal strings),
- deterministic tool dispatch,
- stripping fabricated query operators.

The lesson, consistent across the whole project: **the smaller the model, the
more "intelligence" must live in deterministic code around it.** A ~3B model can
format and summarize, but cannot be trusted to *decide* (which tool, which
evidence, which constraints) for this task.

## Recommended path forward

Pick based on how hard the on-device/privacy constraint is:

1. **Minimize the model's job (most promising for on-device).** Have *code*,
   not the model, parse the question and build the search (merchant, intent,
   date window), run retrieval deterministically, and use the model only to
   phrase a final answer from the retrieved evidence. Caveat: even final
   synthesis fabricated a ship date in run B, so constrain/verify that step too
   (e.g. structured output, explicit "state only what the evidence says").
2. **Use a stronger model.** A 7B+ handled this on desktop. On-device, 7B-4bit
   *does* run on an 8 GB device with the increased-memory-limit entitlement —
   and it was the best run here (found the hard adversarial email, dropped the
   crude mistakes). But it still fabricated (fused two unrelated emails into one
   order) and it is **~4× slower** (~7 tok/s, ~2 min per answer), which is a real
   UX problem for an interactive agent. So a bigger on-device model helps but
   does not by itself make the loop trustworthy. Options: pair 7B with option 1
   (constrained synthesis), wait for faster/better small models, or Apple's
   Private Cloud Compute (stronger model, but leaves the device — breaks the
   pure-privacy premise).
3. **Adapter fine-tuning (Apple's sanctioned fallback).** Train an adapter to
   enforce tool/format behavior on the built-in model. Steep, recurring cost
   (32 GB Mac, dataset, re-train every OS/model update — the adapter toolkit's
   own deprecation notice shows this treadmill). Only if committed to the
   built-in model and options 1–2 are insufficient.

## Fine-tuning and the model-shipping trade-off

A natural question: if prompt instructions aren't obeyed, can we *fine-tune* the
behavior in — and if so, do we have to ship a model with the app? There are two
paths, and they answer the shipping question differently.

**Why fine-tuning (not more prompting) is the lever.** These models are
stateless: the system prompt is re-sent every turn and is always present in
context — the model isn't *forgetting* the rules ("merchant is not a product",
"shipping ≠ delivery", "never fabricate"), it is *disobeying* them. The only way
to change that tendency in the weights (rather than in the prompt) is
fine-tuning: an adapter/LoRA that bakes the pattern in as learned behavior.

| | Apple built-in + adapter | Open model (MLX) + LoRA |
| --- | --- | --- |
| Ship a model in the app? | **No** — base model ships with iOS; you bundle only a small **adapter** (tens of MB) | **Yes** — you own the weights and must bundle (~1.7 GB 3B / ~4.3 GB 7B) or download on first run |
| Base model | Apple's ~3B 2-bit (weakest in our tests); not yours to redistribute | Your choice — including the stronger 7B |
| Capability ceiling | Fixed by Apple | You pick the model |
| Maintenance | Adapter is version-locked to Apple's base model; **re-train + ship an app update whenever Apple updates the OS model** | Stable until *you* change the model |
| Hosting / size cost | Negligible (adapter only) | App bloat (bundle) or CDN hosting + multi-GB first-run download |
| Licensing | Apple's terms | Yours to honor (e.g. Qwen license) |

**The irony for a privacy-first product.** On-device means the weights must
physically be on the device *somehow*. The only way to avoid shipping/hosting
gigabytes yourself is to ride Apple's pre-installed model — but that ties you to
the **weakest** model we tested *and* a re-training treadmill on every OS update.
The path that lets you use the stronger 7B (which did best here) is exactly the
one where you must ship or download the model yourself. There is no option that
is simultaneously (a) strong, (b) no-model-to-ship, and (c) low-maintenance.

**Caveat regardless of path:** fine-tuning reduces specific, repeated errors but
does not lift the underlying capability ceiling — it makes a pattern *more
likely*, not guaranteed. Combine any fine-tune with option 1 (keep decisions in
code; constrain/verify the final synthesis).

## Status

- POC complete; question answered with on-device evidence.
- Not built: real mail (deliberately faked), `llama.cpp` path (MLX covered the
  "bigger on-device model" question), typed-tool polish beyond what proved the
  point.
- The Python POC remains the complete reference implementation (real mail +
  capable model); this POC's job was device viability, now resolved.
