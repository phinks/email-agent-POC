# email-agent-POC

A small iOS POC that answers one question before committing to a full Swift port
of the private-email-agent:

> Can an **on-device** model (Apple Foundation Models, or a larger model via MLX)
> reliably run a multi-step email-retrieval agent loop — on the phone, with no
> network?

**Short answer: the architecture works on-device, but no model we tested
reliably drives the loop. The ~3B models fabricate and mis-select evidence;
the 7B (which runs on an 8 GB device with the memory entitlement, ~4× slower)
is better but still fabricates.** See [**VERDICT.md**](VERDICT.md) for the full
results, perf numbers, and the model-by-model breakdown.

Everything here is deliberately minimal. The mailbox is **faked in memory** — no
IMAP, no OAuth, no Keychain, no networking, no real mail. The POC tests the
**model + tools + loop**, not the plumbing (which we already know ports from the
Python reference implementation).

## What it exercises

- A `ModelProvider` abstraction with pluggable backends:
  - `AppleFoundationModel` — Apple's on-device model (iOS 26 Foundation Models).
  - `MLXModel` — a larger on-device model via MLX Swift (Qwen2.5 3B / 7B, 4-bit),
    with real tool-calling and on-device perf measurement.
  - `MockModel` — scripted; runs the full loop with no real model (plumbing check).
  - `LlamaCppModel` — an unused placeholder that documents the bundled-GGUF swap
    path. Not wired into the UI.
- Three email tools (`search_email`, `search_email_structured`, `get_message`)
  over a fake mailbox that includes **adversarial** messages (a no-keyword wine
  order, a wine promo that is not an order, an out-of-window order, and noise).
- A system prompt with code-computed relative-date bounds, the
  promotions-aren't-evidence framing, and anti-copy-paste placeholder docstrings.
- A minimal agent loop and a SwiftUI screen that shows every tool call, its
  arguments, and the final answer as an on-screen trace.

## What it measures

Run the sample questions and watch the on-screen trace:

1. **Multi-step?** Does it do bootstrap → structured search → get_message, or
   quit early?
2. **Dates correct?** Does it use the pre-computed `after`/`before`, or guess?
3. **Avoids the traps?** Doesn't copy the docstring shapes; doesn't treat promo
   mail as evidence.
4. **The hard case:** does it find the no-keyword wine order (`Purple Pachyderm`
   analogue) that a subject-only keyword search misses?
5. **Resource behavior:** memory footprint, per-turn latency, throughput, and
   whether the model survives (⏱ perf lines show resident MB vs. headroom MB).

## Requirements

- A Mac with **Xcode 26+**.
- A physical **Apple-Intelligence-capable iPhone** for the Apple path (the
  Simulator may not expose the on-device model). The MLX path needs a device
  with enough RAM to hold the chosen model.
- The MLX Swift packages (see MLX setup below).

## Layout

```
email-agent-POC/
  README.md
  VERDICT.md                     # results + verdict (read this)
  email-agent/                   # the Xcode project
    email-agent.xcodeproj
    email-agent/                 # app sources (what Xcode builds)
      EmailAgentApp.swift        # @main entry point
      ContentView.swift          # SwiftUI harness: model picker + tool-call trace
      ModelProvider.swift        # protocol + Apple / Mock / llama.cpp(placeholder)
      MLXModel.swift             # MLX on-device model + agent loop
      MLXEmailTools.swift        # typed MLX tool definitions
      FMTools.swift              # FoundationModels Tool adapters (@Generable args)
      MockModel.swift            # scripted provider — runs the loop, no real model
      FakeMailbox.swift          # in-memory emails (incl. adversarial) + data model
      EmailTools.swift           # search_email / search_email_structured / get_message
      Prompt.swift               # system prompt + relative-date resolver
      Agent.swift                # the multi-step loop + trace events
      Perf.swift                 # on-device perf sampling (memory, headroom, tok/s)
      SampleQuestions.swift      # the questions that probe the 5 measures
    email-agentTests/            # unit test target
    email-agentUITests/          # UI test target
```

## Model picker (segmented control in the UI)

- **Apple** — Apple Foundation Models (built-in ~3B, 2-bit), on-device. Needs an
  Apple-Intelligence-capable device; the badge shows availability / why-not.
- **MLX 3B** — `mlx-community/Qwen2.5-3B-Instruct-4bit` via MLX Swift. Downloads
  on first run; emits on-device perf (load time, resident memory, headroom,
  tokens/sec) in the trace.
- **MLX 7B** — `mlx-community/Qwen2.5-7B-Instruct-4bit`. See the memory note below.
- **Mock** — scripted; runs the full tools + trace + loop with no real model. Use
  it in the Simulator to confirm plumbing. Does not reason.

### MLX setup

1. Add the MLX Swift packages to the target (Xcode → Add Package Dependencies):
   - `https://github.com/ml-explore/mlx-swift-examples` (products: `MLXLLM`,
     `MLXLMCommon`)
   - its transitive dependency `https://github.com/ml-explore/mlx-swift`
2. First run downloads the model (~1.7 GB for 3B-4bit, ~4.3 GB for 7B-4bit) to
   the app container.
3. MLX caps its GPU buffer cache at load (`MLX.GPU.set(cacheLimit:)`) — without
   this, the cache grows across runs and can trip a memory kill even on 3B.

### Memory notes (from the POC)

- **3B-4bit fits** once the GPU cache is capped: ~300 MB resident, ~1.5 GB
  headroom, ~27 tok/s.
- **7B-4bit also fits on an 8 GB device**, but only with the **Increased
  (Debugging) Memory Limit** capability added (Signing & Capabilities →
  `com.apple.developer.kernel.increased-debugging-memory-limit`). Without it the
  app hits iOS's default per-app cap and is killed mid-generation
  (`vm-compressor-thrashing`); with it, the 7B runs with ~2.3–2.5 GB headroom.
  The trade-off is speed: ~7 tok/s (~4× slower than 3B), ~2 min per answer. On a
  lower-RAM device it may still not fit. See VERDICT.md.

### Context-window caveat (Apple Foundation Models)

Apple's on-device model has a small **~4,096-token context** (instructions +
prompt + tool results + output, combined). The search tools deliberately return
metadata/snippets only; `get_message` returns a full body on demand. A request
that fails mid-loop is most likely a large tool result overflowing this window.

## Status

Complete. The POC answered the viability question (see VERDICT.md). It is
deliberately not production code — no real mail, no Keychain, no UI polish. The
Python private-email-agent remains the full reference implementation.
