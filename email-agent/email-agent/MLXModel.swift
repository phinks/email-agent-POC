import Foundation

// MLXModel — Option A: full search-driven agent loop with typed MLX tools.
//
// Fair comparison to the Apple built-in run: the model must DECIDE search
// parameters and call tools (resolve_relative_date, search_email_structured,
// search_email, get_message) — no inline mailbox shortcut. This tests whether a
// 4-bit 3B via MLX handles the agentic reasoning the 2-bit built-in model
// fumbled (invented date windows / merchants).
//
// API matched to mlx-swift-examples 2.29.1 (LLMEval reference):
//   LLMModelFactory.shared.loadContainer(configuration:)
//   UserInput(chat:tools:) + MLXLMCommon.generate(input:parameters:context:)
//   stream items: .chunk (text), .info (tokensPerSecond), .toolCall
//   tools defined in MLXEmailTools (Tool<Input,Output>, .schema, execute(with:))

#if canImport(MLXLLM)
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
#endif

final class MLXModel: ModelProvider {

    static let defaultModelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    let modelID: String
    init(modelID: String = MLXModel.defaultModelID) { self.modelID = modelID }

    var label: String { "MLX on-device (\(modelID.split(separator: "/").last.map(String.init) ?? modelID))" }

    var isAvailable: Bool {
        #if canImport(MLXLLM)
        return true
        #else
        return false
        #endif
    }

    func answer(
        systemPrompt: String,
        question: String,
        tools: [ModelTool],
        maxSteps: Int,
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        #if canImport(MLXLLM)
        trace(.prompt(system: systemPrompt, question: question))
        let wallStart = Perf.now()

        // --- 1. Load / download the model -----------------------------------
        // Cap MLX's GPU buffer cache so it doesn't accumulate across runs and
        // push the app into jetsam (observed: a repeat 3B run was OS-killed for
        // memory even though nothing changed — headroom was already ~110MB).
        // LLMEval uses the same guard.
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)   // 20 MB cache ceiling

        let loadStart = Perf.now()
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: modelID)
        ) { progress in
            trace(.perf("downloading \(Int(progress.fractionCompleted * 100))%"))
        }
        let loadMs = (Perf.now() - loadStart) * 1000
        trace(.perf("loaded: \(Int(loadMs))ms · footprint \(Int(Perf.physFootprintMB()))MB · "
                    + "headroom \(Int(Perf.availableMemoryMB()))MB"))

        // --- 2. Search-driven tool loop -------------------------------------
        // Build the tool schemas inline (type inferred — see MLXEmailTools note).
        let toolSchemas = [
            MLXEmailTools.resolveDate().schema,
            MLXEmailTools.searchStructured().schema,
            MLXEmailTools.searchRaw().schema,
            MLXEmailTools.getMessage().schema,
        ]
        var chat: [Chat.Message] = [.system(systemPrompt), .user(question)]
        var finalText = ""
        var outputTokens = 0
        var lastTokPerSec = 0.0
        let genStart = Perf.now()

        // Loop-progress guard: remember the previous tool call's signature so we
        // can detect a model that re-issues the IDENTICAL call without reacting
        // to the result (observed: Mistral repeated the same search 4×; Ministral
        // resolved the same date twice). On a repeat we inject a corrective nudge
        // instead of silently spinning.
        var lastCallSignature: String? = nil
        func signature(_ name: String, _ args: [String: Any]) -> String {
            let flat = args.keys.sorted().map { "\($0)=\(args[$0].map { "\($0)" } ?? "")" }
            return name + "(" + flat.joined(separator: ",") + ")"
        }

        for step in 1...maxSteps {
            let input = UserInput(chat: chat, tools: toolSchemas)

            // The generation runs on the container's executor. Return results
            // from the closure (don't mutate captured vars — that's a data race
            // under Swift concurrency).
            let turn: (text: String, tool: ToolCall?, tokens: Int, tps: Double) =
                try await container.perform { (context: ModelContext) in
                    var text = ""
                    var tool: ToolCall?
                    var tokens = 0
                    var tps = 0.0
                    let lmInput = try await context.processor.prepare(input: input)
                    let stream = try MLXLMCommon.generate(
                        input: lmInput,
                        parameters: GenerateParameters(maxTokens: 512, temperature: 0),
                        context: context
                    )
                    for await item in stream {
                        if let chunk = item.chunk { text += chunk; tokens += 1 }
                        if let info = item.info { tps = info.tokensPerSecond }
                        if let tc = item.toolCall { tool = tc; break }
                    }
                    return (text, tool, tokens, tps)
                }

            outputTokens += turn.tokens
            lastTokPerSec = turn.tps

            // Native tool call (the format MLX recognizes, e.g. Qwen)?
            if let tc = turn.tool {
                let args: [String: Any] = tc.function.arguments.reduce(into: [:]) {
                    $0[$1.key] = "\($1.value)"
                }
                lastCallSignature = signature(tc.function.name, args)
                trace(.modelTurn(step: step, madeToolCall: true))
                let result = await MLXEmailTools.dispatch(tc, trace: trace)
                chat.append(.assistant(turn.text))
                chat.append(.tool(result))
                if step == maxSteps { finalText = turn.text }
                continue
            }

            // Fallback: MLX didn't surface a tool call, but the model may have
            // emitted one in its own format (Mistral [TOOL_CALLS], Llama
            // <|python_tag|>, or bare JSON) that would otherwise leak into the
            // answer as text. Parse it so non-Qwen models get a fair shot.
            if let parsed = MLXEmailTools.parseToolCall(from: turn.text) {
                let sig = signature(parsed.name, parsed.args)
                // Loop-progress guard: identical call to last turn => don't spin.
                if sig == lastCallSignature {
                    trace(.modelTurn(step: step, madeToolCall: true))
                    trace(.perf("repeat call suppressed: \(parsed.name)"))
                    // Append BOTH the model's (repeated) turn as assistant AND the
                    // nudge as user, so roles keep alternating — Mistral/Ministral's
                    // template rejects two consecutive same-role messages.
                    chat.append(.assistant("(repeated \(parsed.name) — suppressed)"))
                    chat.append(.user(
                        "You already called \(parsed.name) with the same arguments and "
                        + "have its result above. Do NOT call it again. Use that result: "
                        + "if the question needs more data, call a DIFFERENT tool "
                        + "(e.g. search_email_structured with the resolved dates); "
                        + "otherwise answer the question now from what you have. "
                        + "Do not output a tool result as your final answer."))
                    if step == maxSteps { finalText = turn.text }
                    continue
                }
                lastCallSignature = sig
                trace(.modelTurn(step: step, madeToolCall: true))
                let result = await MLXEmailTools.dispatchParsed(parsed, trace: trace)
                // Feed the turn back WITHOUT re-emitting a structured tool call.
                // Mistral/Ministral's chat template validates that any tool call
                // in history carries a 9-char alphanumeric id; our parsed calls
                // have none, so re-rendering `.assistant(rawToolCallText)` +
                // `.tool(...)` throws TemplateException. Instead, record the call
                // and its result as plain assistant/user turns — the model still
                // sees what it called and what came back, with no id to validate.
                chat.append(.assistant("Called \(parsed.name). Result:\n\(result)"))
                chat.append(.user("Continue using that result. If you have enough "
                    + "information, answer the question now; otherwise call a "
                    + "DIFFERENT tool. Do not repeat the same call, and do not "
                    + "output a tool result verbatim as your answer."))
                if step == maxSteps { finalText = turn.text }
                continue
            }

            // Genuinely no tool call in any known format => final answer.
            trace(.modelTurn(step: step, madeToolCall: false))
            finalText = turn.text
            break
        }

        let genMs = (Perf.now() - genStart) * 1000

        // --- 3. Perf summary -------------------------------------------------
        var perf = PerfSample()
        perf.modelLoadMs = loadMs
        perf.totalMs = (Perf.now() - wallStart) * 1000
        perf.outputTokens = outputTokens
        perf.decodeTokPerSec = lastTokPerSec > 0
            ? lastTokPerSec
            : (genMs > 0 ? Double(outputTokens) / (genMs / 1000) : 0)
        perf.footprintMB = Perf.physFootprintMB()
        perf.residentMB = Perf.residentMemoryMB()
        perf.availableMB = Perf.availableMemoryMB()
        trace(.perf(perf.summary))

        let answer = finalText.isEmpty ? "No answer produced." : finalText
        trace(.finalAnswer(answer))
        return answer
        #else
        throw ModelProviderError.unavailable(
            "MLXLLM/MLXLMCommon not linked. Add them from the mlx-libraries package to the target."
        )
        #endif
    }
}
