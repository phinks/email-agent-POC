import Foundation

/// A fake ModelProvider that runs a scripted multi-step sequence against the
/// real tools, with NO on-device model. Purpose: exercise the whole harness —
/// tools, trace, fail-loud, final answer — in the Simulator or on any device,
/// independent of whether Apple's model is available.
///
/// It is deliberately dumb: it does not "reason". It plays a fixed script of
/// tool calls (chosen to look like what a competent agent would do for the wine
/// question), then emits a canned final answer built from what the tools
/// actually returned. This verifies the PLUMBING, not model quality.
final class MockModel: ModelProvider {
    let label = "Mock (scripted, no real model)"
    var isAvailable: Bool { true }

    func answer(
        systemPrompt: String,
        question: String,
        tools: [ModelTool],
        maxSteps: Int,
        trace: @escaping @Sendable (TraceEvent) -> Void
    ) async throws -> String {
        trace(.prompt(system: systemPrompt, question: question))

        func tool(_ name: String) -> ModelTool? { tools.first { $0.name == name } }

        // Fail loud if a scripted tool result is a TOOL_ERROR — mirrors the
        // real providers' contract.
        func callAndTrace(_ name: String, _ args: [String: Any], step: Int) async throws -> String {
            guard let t = tool(name) else {
                throw ModelProviderError.toolFailure("mock: no tool \(name)")
            }
            let result = await t.call(args)
            let isError = result.hasPrefix("TOOL_ERROR")
            trace(.toolCall(name: name, arguments: args,
                            resultChars: result.count, isError: isError))
            if isError {
                trace(.aborted(reason: "tool \(name) failed"))
                throw ModelProviderError.toolFailure(name)
            }
            return result
        }

        // Scripted "reasoning": a plausible competent sequence for the wine
        // question — resolve the window, structured purchase search, then
        // inspect the one promising hit's body.
        trace(.modelTurn(step: 1, madeToolCall: true))
        _ = try await callAndTrace("resolve_relative_date",
                                   ["expression": "last 3 months"], step: 1)

        trace(.modelTurn(step: 2, madeToolCall: true))
        let searchJSON = try await callAndTrace(
            "search_email_structured",
            ["merchant": "", "intent": "purchase",
             "after": "2026-07-12", "before": "2026-10-12", "limit": 10],
            step: 2
        )

        // Pull the first message id from the search result and inspect it,
        // to show the get_message (body-read) step working.
        trace(.modelTurn(step: 3, madeToolCall: true))
        if let firstID = firstMessageID(in: searchJSON) {
            _ = try await callAndTrace("get_message", ["message_id": firstID], step: 3)
        }

        trace(.modelTurn(step: 4, madeToolCall: false))
        let answer = """
        [MOCK] Plumbing check only — this is not model reasoning. The scripted \
        sequence ran: resolve_relative_date -> search_email_structured \
        (purchase, last 3 months) -> get_message on the first hit. Inspect the \
        TRACE above to confirm tools, args, result sizes, and fail-loud all work. \
        Switch off Mock and use Apple Foundation Models for the real viability test.
        """
        trace(.finalAnswer(answer))
        return answer
    }

    /// Minimal JSON scrape for the first message id in a search result.
    private func firstMessageID(in json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let messages = obj["messages"] as? [[String: Any]],
              let first = messages.first,
              let id = first["id"] as? String else {
            return nil
        }
        return id
    }
}
