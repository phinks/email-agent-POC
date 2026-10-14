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
        trace(.perf("loaded: \(Int(loadMs))ms · res \(Int(Perf.residentMemoryMB()))MB · "
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
                trace(.modelTurn(step: step, madeToolCall: true))
                let result = await MLXEmailTools.dispatchParsed(parsed, trace: trace)
                chat.append(.assistant(turn.text))
                chat.append(.tool(result))
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
