import Foundation

/// Thin orchestrator. Assembles the system prompt + tools and hands the loop to
/// whichever ModelProvider is selected. All the multi-step/tool-calling logic
/// lives in the provider (for Apple Foundation Models, the framework drives it);
/// this type just wires the pieces and relays trace events.
final class Agent {
    let model: ModelProvider
    let backend: String
    let maxSteps: Int

    init(model: ModelProvider, backend: String = "fastmail", maxSteps: Int = 8) {
        self.model = model
        self.backend = backend
        self.maxSteps = maxSteps
    }

    func answer(_ question: String, trace: @escaping @Sendable (TraceEvent) -> Void) async -> String {
        guard model.isAvailable else {
            let msg = "Model \"\(model.label)\" is not available on this device."
            trace(.aborted(reason: msg))
            return msg
        }
        let systemPrompt = Prompt.system(backend: backend)
        do {
            return try await model.answer(
                systemPrompt: systemPrompt,
                question: question,
                tools: EmailTools.all(),
                maxSteps: maxSteps,
                trace: trace
            )
        } catch {
            let msg = "Failed: \(error)"
            trace(.aborted(reason: msg))
            return msg
        }
    }
}
