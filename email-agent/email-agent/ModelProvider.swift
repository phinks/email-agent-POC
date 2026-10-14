import Foundation

// NOTE: Foundation Models API names are written to the WWDC26 shape and are
// TO BE VERIFIED against current Apple docs. Adjust imports/types to compile.
// The point of this file is the ABSTRACTION: the agent loop depends only on
// `ModelProvider`, so swapping the built-in model for llama.cpp is a one-line
// change, not a rewrite.

#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - Abstraction

/// A tool the model may call. Backend-neutral: the same definitions are given
/// to Apple Foundation Models or to a llama.cpp runtime.
///
/// `@unchecked Sendable`: FoundationModels' `Tool` protocol is `Sendable`, and
/// these are held by the tool structs. ModelTool is immutable after init and its
/// `[String: Any]` parameters are only read, so this is safe for the POC. The
/// closures below are `@Sendable` so the compiler can verify the hot path.
struct ModelTool: @unchecked Sendable {
    let name: String
    let description: String
    /// JSON Schema for the arguments (kept as a dictionary so it is portable
    /// across whatever each backend expects).
    let parameters: [String: Any]
    /// Executes the tool. Receives decoded args, returns a result payload the
    /// model will see as the tool result (or a TOOL_ERROR-marked string).
    let call: @Sendable ([String: Any]) async -> String
}

/// One step the loop can observe for the on-screen trace.
/// `@unchecked Sendable`: carries a read-only `[String: Any]` in `.toolCall`;
/// only ever constructed and handed one-way to the UI.
enum TraceEvent: @unchecked Sendable {
    case prompt(system: String, question: String)
    case toolCall(name: String, arguments: [String: Any], resultChars: Int, isError: Bool)
    case modelTurn(step: Int, madeToolCall: Bool)
    case finalAnswer(String)
    case aborted(reason: String)
    case perf(String)   // on-device performance summary (load/mem/tok-per-sec)
}

/// The seam. Everything above this line is backend-neutral.
protocol ModelProvider {
    /// Human-readable name for the trace/UI.
    var label: String { get }

    /// Whether this provider is usable on the current device (e.g. Apple
    /// Intelligence availability). The POC shows this in the UI.
    var isAvailable: Bool { get }

    /// Run the full agentic loop: given a system prompt, the user's question,
    /// and the available tools, iterate (call tools, feed results) until the
    /// model produces a final answer. `trace` is called for each observable
    /// step so the harness can display it.
    func answer(
        systemPrompt: String,
        question: String,
        tools: [ModelTool],
        maxSteps: Int,
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String
}

enum ModelProviderError: Error, CustomStringConvertible {
    case unavailable(String)
    case toolFailure(String)
    case exceededSteps(Int)

    var description: String {
        switch self {
        case .unavailable(let m): return "model unavailable: \(m)"
        case .toolFailure(let m): return "tool failure: \(m)"
        case .exceededSteps(let n): return "exceeded \(n) steps without an answer"
        }
    }
}

// MARK: - Apple Foundation Models (the POC target)

/// Drives the loop with Apple's on-device model. The exact FoundationModels
/// session/tool API is TO BE VERIFIED; the implementation is structured so only
/// the body of `answer(...)` changes when the real API is confirmed.
final class AppleFoundationModel: ModelProvider {
    let label = "Apple Foundation Models (on-device)"

    var isAvailable: Bool {
        #if canImport(FoundationModels)
        if case .available = SystemLanguageModel.default.availability { return true }
        return false
        #else
        return false
        #endif
    }

    /// Human-readable availability detail for the UI/trace.
    var availabilityDetail: String {
        #if canImport(FoundationModels)
        switch SystemLanguageModel.default.availability {
        case .available: return "available"
        case .unavailable(.deviceNotEligible): return "device not eligible for Apple Intelligence"
        case .unavailable(.appleIntelligenceNotEnabled): return "enable Apple Intelligence in Settings"
        case .unavailable(.modelNotReady): return "model downloading / not ready"
        case .unavailable(let other): return "unavailable: \(other)"
        }
        #else
        return "FoundationModels not available on this target"
        #endif
    }

    func answer(
        systemPrompt: String,
        question: String,
        tools: [ModelTool],
        maxSteps: Int,
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        #if canImport(FoundationModels)
        guard isAvailable else {
            throw ModelProviderError.unavailable(availabilityDetail)
        }
        trace(.prompt(system: systemPrompt, question: question))

        // Wrap our backend-neutral ModelTools as FoundationModels `Tool`s. The
        // framework drives the full multi-step loop: it decides which tool to
        // call, invokes `call`, feeds the result back, and iterates until it
        // produces a final answer. We trace each invocation here.
        //
        // NOTE: FoundationModels resolves `@Generable` arguments into typed
        // structs; our ModelTool uses a [String: Any] dictionary, so the glue
        // tool (`BridgedTool`, in FMTools.swift) adapts between the two. See
        // that file for the per-tool Generable argument types.
        let fmTools: [any Tool] = FMTools.make(from: tools, trace: trace)

        let session = LanguageModelSession(
            tools: fmTools,
            instructions: systemPrompt
        )

        do {
            let response = try await session.respond(to: question)
            // NOTE: Foundation Models' 4,096-token context (instructions +
            // prompt + tool results + output) is small — if a tool returns a
            // large result the request can fail here. Our search tools return
            // metadata/snippets only to stay within it.
            trace(.finalAnswer(response.content))
            return response.content
        } catch let error as LanguageModelSession.ToolCallError {
            // A tool threw (our TOOL_ERROR path surfaces here). Fail loud —
            // never let the model fabricate from a failed lookup.
            trace(.aborted(reason: "tool \(error.tool.name) failed: \(error)"))
            throw ModelProviderError.toolFailure("\(error.tool.name): \(error)")
        }
        #else
        throw ModelProviderError.unavailable("FoundationModels not importable on this target")
        #endif
    }
}

// MARK: - llama.cpp (the fallback / swap target)

/// Placeholder for a bundled-model path. Only needed if the built-in model is
/// too weak for the multi-step loop (desktop evidence: 7B works, 3B struggles).
/// Implementing this means: load a GGUF via llama.cpp's C API, drive an
/// OpenAI-style tool-calling loop in code, and manage unload-on-background.
/// Deliberately unimplemented — its existence documents that "built-in vs
/// bundled" is a swap behind this one protocol, not a rewrite.
final class LlamaCppModel: ModelProvider {
    let label = "llama.cpp (bundled GGUF) — not implemented in POC"
    var isAvailable: Bool { false }

    func answer(
        systemPrompt: String,
        question: String,
        tools: [ModelTool],
        maxSteps: Int,
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        throw ModelProviderError.unavailable(
            "LlamaCppModel is a placeholder. Implement only if the built-in "
            + "model fails the viability POC; the agent loop stays unchanged."
        )
    }
}
