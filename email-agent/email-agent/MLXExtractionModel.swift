import Foundation

// MLXExtractionModel — the "model understands, code acts" mode on MLX.
//
// ONE tool-less generation: question -> JSON query. Then ExtractionPipeline
// (pure code) resolves dates, runs the search, and composes the answer. No tool
// loop, so none of the tool-loop failure modes apply: no multi-step chaining to
// stall on, no per-family tool-call format to parse, no template id/alternation
// crashes, and no fabrication (code composes the final answer).

#if canImport(MLXLLM)
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers
#endif

final class MLXExtractionModel: ModelProvider {

    let modelID: String
    init(modelID: String = MLXModel.defaultModelID) { self.modelID = modelID }

    var label: String {
        "MLX extraction (\(modelID.split(separator: "/").last.map(String.init) ?? modelID))"
    }

    var isAvailable: Bool {
        #if canImport(MLXLLM)
        return true
        #else
        return false
        #endif
    }

    func answer(
        systemPrompt: String,          // unused: this mode uses its own extraction prompt
        question: String,
        tools: [ModelTool],            // unused: code drives the search
        maxSteps: Int,                 // unused: single pass
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        #if canImport(MLXLLM)
        trace(.prompt(system: "extraction mode (code-driven search)", question: question))
        let wallStart = Perf.now()

        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

        let loadStart = Perf.now()
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: modelID)
        ) { progress in
            trace(.perf("downloading \(Int(progress.fractionCompleted * 100))%"))
        }
        let loadMs = (Perf.now() - loadStart) * 1000
        trace(.perf("loaded: \(Int(loadMs))ms · footprint \(Int(Perf.physFootprintMB()))MB · "
                    + "headroom \(Int(Perf.availableMemoryMB()))MB"))

        // --- ONE generation: question -> JSON query -------------------------
        let prompt = ExtractionPipeline.extractionPrompt(for: question)
        let genStart = Perf.now()
        let turn: (text: String, tokens: Int, tps: Double) =
            try await container.perform { (context: ModelContext) in
                var text = ""
                var tokens = 0
                var tps = 0.0
                let input = UserInput(chat: [.user(prompt)])   // no tools, no system
                let lmInput = try await context.processor.prepare(input: input)
                let stream = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: GenerateParameters(maxTokens: 256, temperature: 0),
                    context: context
                )
                for await item in stream {
                    if let chunk = item.chunk { text += chunk; tokens += 1 }
                    if let info = item.info { tps = info.tokensPerSecond }
                }
                return (text, tokens, tps)
            }
        let genMs = (Perf.now() - genStart) * 1000

        // --- parse the extracted query (in code) ----------------------------
        guard let query = Self.parseQuery(from: turn.text) else {
            trace(.perf("extraction parse failed; raw: \(turn.text.prefix(120))"))
            trace(.finalAnswer("Could not understand the question well enough to search."))
            return "Could not understand the question well enough to search."
        }
        trace(.perf("extracted: topic=\(query.topic ?? "-") sender=\(query.sender ?? "-") "
                    + "intent=\(query.intent ?? "-") timeframe=\(query.timeframe ?? "-") "
                    + "op=\(query.operation ?? "lookup")"))

        // --- CODE runs the search + composes the answer ---------------------
        let answer = await ExtractionPipeline.run(query, question: question, trace: trace)

        var perf = PerfSample()
        perf.modelLoadMs = loadMs
        perf.totalMs = (Perf.now() - wallStart) * 1000
        perf.outputTokens = turn.tokens
        perf.decodeTokPerSec = turn.tps > 0
            ? turn.tps
            : (genMs > 0 ? Double(turn.tokens) / (genMs / 1000) : 0)
        perf.footprintMB = Perf.physFootprintMB()
        perf.residentMB = Perf.residentMemoryMB()
        perf.availableMB = Perf.availableMemoryMB()
        trace(.perf(perf.summary))

        trace(.finalAnswer(answer))
        return answer
        #else
        throw ModelProviderError.unavailable("MLXLLM/MLXLMCommon not linked.")
        #endif
    }

    /// Extract the first JSON object from the model's output and decode it into
    /// an ExtractedQuery. Reuses the lenient object-finder behavior (handles
    /// prose around the JSON, and single-quote styles).
    static func parseQuery(from raw: String) -> ExtractedQuery? {
        guard let start = raw.firstIndex(of: "{") else { return nil }
        // find matching close brace
        var depth = 0, inStr = false, esc = false
        var end: String.Index? = nil
        var i = start
        while i < raw.endIndex {
            let c = raw[i]
            if inStr {
                if esc { esc = false } else if c == "\\" { esc = true } else if c == "\"" { inStr = false }
            } else if c == "\"" { inStr = true }
            else if c == "{" { depth += 1 }
            else if c == "}" { depth -= 1; if depth == 0 { end = raw.index(after: i); break } }
            i = raw.index(after: i)
        }
        guard let e = end else { return nil }
        let objStr = String(raw[start..<e])
        func decode(_ s: String) -> ExtractedQuery? {
            guard let d = s.data(using: .utf8) else { return nil }
            return try? JSONDecoder().decode(ExtractedQuery.self, from: d)
        }
        return decode(objStr) ?? decode(objStr.replacingOccurrences(of: "'", with: "\""))
    }
}
