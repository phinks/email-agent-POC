import Foundation

// AppleExtractionModel — the "model understands, code acts" architecture on
// Apple Foundation Models, using @Generable for CONSTRAINED generation.
//
// This is the constrained-decoding "real fix" for free on the Apple path: by
// asking the session to generate a @Generable type, Foundation Models masks the
// decode so the output is GUARANTEED to match the schema — no invalid JSON, no
// enum-list echo, no missing fields. The model that was WORST at the agentic
// tool loop (fabrication, inconsistency) may be the BEST at extraction, because
// it structurally cannot emit a malformed query.
//
// Flow: one constrained respond() → map to the plain ExtractedQuery → hand to
// the shared ExtractionPipeline (same code-driven search + answer the MLX path
// uses). The model never searches, counts, or phrases a fact.

#if canImport(FoundationModels)
import FoundationModels

/// The query the Apple model fills, as a @Generable so Foundation Models
/// constrains generation to this exact shape. Mirrors `ExtractedQuery`.
/// `@Guide` descriptions carry the same rules the MLX few-shot prompt teaches.
@Generable
struct GenerableQuery {
    @Guide(description: "What the email is ABOUT — a product/subject like \"wine\" or \"travel\". null if the question names no subject. NEVER \"order\"/\"receipt\" (those are intent, not topic), and NEVER a value the user did not say.")
    var topic: String?

    @Guide(description: "WHO the email is from — a specific store, brand, person, or service named in the question (e.g. \"Banana Republic\"). null if none named. Never a generic word like \"store\".")
    var sender: String?

    @Guide(description: "one of: any, purchase, promotion, reservation, updates. Orders/receipts/shipping/deliveries are \"purchase\".")
    var intent: String?

    @Guide(description: "The raw time phrase from the question (\"last 3 months\", \"this year\", \"recently\"), or null if none.")
    var timeframe: String?

    @Guide(description: "\"count\" for how-many; \"list\" for what/which/show-me; \"lookup\" for yes/no.")
    var operation: String?

    /// Map to the backend-neutral query the pipeline consumes.
    var plain: ExtractedQuery {
        ExtractedQuery(topic: topic, sender: sender, intent: intent,
                       timeframe: timeframe, operation: operation)
    }
}
#endif

final class AppleExtractionModel: ModelProvider {

    let label = "Apple extraction (@Generable, constrained)"

    var isAvailable: Bool {
        #if canImport(FoundationModels)
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
        #else
        return false
        #endif
    }

    func answer(
        systemPrompt: String,          // unused: extraction uses its own prompt
        question: String,
        tools: [ModelTool],            // unused: code drives the search
        maxSteps: Int,                 // unused: single pass
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        #if canImport(FoundationModels)
        guard isAvailable else { throw ModelProviderError.unavailable("Apple Intelligence unavailable") }
        trace(.prompt(system: "Apple extraction (@Generable, constrained)", question: question))
        let wallStart = Perf.now()

        // One CONSTRAINED generation: the response is guaranteed to be a
        // GenerableQuery (schema-valid) — no parsing, no malformed JSON.
        let session = LanguageModelSession(
            instructions: "Convert the user's email question into a query. Fill only "
                + "fields the question actually mentions; leave the rest null. Never "
                + "invent a sender or topic the user did not say."
        )
        let genStart = Perf.now()
        let response = try await session.respond(to: question, generating: GenerableQuery.self)
        let genMs = (Perf.now() - genStart) * 1000

        let query = response.content.plain
        trace(.perf("extracted: topic=\(query.topic ?? "-") sender=\(query.sender ?? "-") "
                    + "intent=\(query.intent ?? "-") timeframe=\(query.timeframe ?? "-") "
                    + "op=\(query.operation ?? "lookup")"))

        // CODE runs the search + composes the answer (shared with the MLX path).
        let answer = await ExtractionPipeline.run(query, question: question, trace: trace)

        var perf = PerfSample()
        perf.totalMs = (Perf.now() - wallStart) * 1000
        perf.decodeTokPerSec = 0        // Foundation Models doesn't expose tok/s
        perf.footprintMB = Perf.physFootprintMB()
        perf.availableMB = Perf.availableMemoryMB()
        _ = genMs
        trace(.perf(perf.summary))

        trace(.finalAnswer(answer))
        return answer
        #else
        throw ModelProviderError.unavailable("FoundationModels not importable on this target")
        #endif
    }
}
