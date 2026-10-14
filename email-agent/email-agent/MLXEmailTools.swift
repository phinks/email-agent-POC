import Foundation

// Typed MLX tools for the email agent, matching the LLMEval 2.29.1 pattern:
//   Tool<Input: Codable, Output: Codable>(name:description:parameters:) { input in ... }
//   pass [tool.schema] into UserInput(tools:)
//   dispatch a model-emitted ToolCall via toolCall.execute(with:)
//
// These delegate to the SAME fake-mailbox logic used elsewhere (via the
// FakeMailbox data + EmailTools helpers), so behavior matches the other
// providers. Output is JSON text (what the model sees as the tool result),
// mirroring EmailTools' string results.
//
// NOTE: Tool / ToolCall / .schema / execute(with:) and the `.parameters`
// ToolParameter DSL names follow the LLMEval reference. If a name differs in
// your exact 2.29.x patch, adjust — the structure is the stable part.

#if canImport(MLXLLM)
import MLXLMCommon

enum MLXEmailTools {

    // MARK: Codable I/O structs (one per tool)

    struct ResolveDateInput: Codable { let expression: String }
    struct TextOutput: Codable { let result: String }

    struct SearchStructuredInput: Codable {
        let merchant: String?
        let sender_domain: String?
        let intent: String?
        let after: String?
        let before: String?
        let limit: Int?
    }
    struct SearchInput: Codable { let query: String; let limit: Int? }
    struct GetMessageInput: Codable { let message_id: String }

    // MARK: Tool definitions

    static func resolveDate() -> Tool<ResolveDateInput, TextOutput> {
        Tool(
            name: "resolve_relative_date",
            description: "Convert a relative time expression (e.g. 'last 3 months', 'last 3 years') "
                + "into exact after/before ISO dates. Use this instead of computing dates yourself.",
            parameters: [
                .required("expression", type: .string,
                          description: "A relative time expression, e.g. 'last 3 months'")
            ]
        ) { input in
            guard let r = Prompt.resolveRelativeDate(input.expression) else {
                return TextOutput(result: "TOOL_ERROR: could not parse '\(input.expression)'")
            }
            return TextOutput(result: #"{"after":"\#(r.after)","before":"\#(r.before)"}"#)
        }
    }

    static func searchStructured() -> Tool<SearchStructuredInput, TextOutput> {
        Tool(
            name: "search_email_structured",
            description: "REQUIRED for ordinary email questions. Filter by merchant/intent/date. "
                + "If the question names no time period, OMIT after/before. 'merchant' is a store/brand, "
                + "never a product like 'wine'. Returns metadata/snippets.",
            parameters: [
                .optional("merchant", type: .string, description: "Store/brand named in the question"),
                .optional("sender_domain", type: .string, description: "Only a domain seen in evidence"),
                .optional("intent", type: .string,
                          description: "any|purchase|promotion|reservation|updates"),
                .optional("after", type: .string, description: "Inclusive ISO start (YYYY-MM-DD)"),
                .optional("before", type: .string, description: "Exclusive ISO end (YYYY-MM-DD)"),
                .optional("limit", type: .int, description: "Max results"),
            ]
        ) { input in
            var args: [String: Any] = [:]
            if let v = input.merchant { args["merchant"] = v }
            if let v = input.sender_domain { args["sender_domain"] = v }
            if let v = input.intent { args["intent"] = v }
            if let v = input.after { args["after"] = v }
            if let v = input.before { args["before"] = v }
            args["limit"] = input.limit ?? 10
            let result = await callEmailTool("search_email_structured", args)
            return TextOutput(result: result)
        }
    }

    static func searchRaw() -> Tool<SearchInput, TextOutput> {
        Tool(
            name: "search_email",
            description: "Advanced raw keyword search. Prefer search_email_structured. "
                + "Returns metadata/snippets, not full bodies.",
            parameters: [
                .required("query", type: .string, description: "Keyword query"),
                .optional("limit", type: .int, description: "Max results"),
            ]
        ) { input in
            let result = await callEmailTool("search_email",
                                             ["query": input.query, "limit": input.limit ?? 10])
            return TextOutput(result: result)
        }
    }

    static func getMessage() -> Tool<GetMessageInput, TextOutput> {
        Tool(
            name: "get_message",
            description: "Retrieve one email INCLUDING its full body, by message id.",
            parameters: [
                .required("message_id", type: .string, description: "The message id")
            ]
        ) { input in
            let result = await callEmailTool("get_message", ["message_id": input.message_id])
            return TextOutput(result: result)
        }
    }

    // NOTE: the schema array is built inline at the UserInput(tools:) call site
    // in MLXModel (as LLMEval does) so we never have to name the schema type —
    // its concrete type varies between MLX versions and Swift infers it there.

    // MARK: dispatch a model toolCall by name -> result string + trace

    static func dispatch(_ call: ToolCall, trace: @escaping @Sendable (TraceEvent) -> Void) async -> String {
        let name = call.function.name
        // call.function.arguments is already a parsed [String: JSONValue]; we
        // only need a displayable form for the trace (execute(with:) decodes
        // the real typed args itself). Unwrap JSONValue to a plain scalar so the
        // trace shows `after=2026-09-12`, not `after=string("2026-09-12")`.
        let args: [String: Any] = call.function.arguments.reduce(into: [:]) { acc, kv in
            acc[kv.key] = Self.plain(kv.value)
        }
        let out: String
        switch name {
        case "resolve_relative_date":
            out = (try? await call.execute(with: resolveDate()).toolResult) ?? "TOOL_ERROR: resolve failed"
        case "search_email_structured":
            out = (try? await call.execute(with: searchStructured()).toolResult) ?? "TOOL_ERROR"
        case "search_email":
            out = (try? await call.execute(with: searchRaw()).toolResult) ?? "TOOL_ERROR"
        case "get_message":
            out = (try? await call.execute(with: getMessage()).toolResult) ?? "TOOL_ERROR"
        default:
            out = "TOOL_ERROR: unknown tool \(name)"
        }
        trace(.toolCall(name: name, arguments: args,
                        resultChars: out.count, isError: out.hasPrefix("TOOL_ERROR")))
        return out
    }

    /// Unwrap a JSONValue to a plain display scalar without depending on the
    /// enum's exact case names (which vary between MLX versions). We render it
    /// and strip a leading `case(` ... `)` wrapper, e.g. `string("x")` -> `x`,
    /// `int(10)` -> `10`.
    private static func plain(_ value: Any) -> String {
        var s = "\(value)"
        if let open = s.firstIndex(of: "("), s.hasSuffix(")") {
            s = String(s[s.index(after: open)..<s.index(before: s.endIndex)])
        }
        if s.hasPrefix("\"") && s.hasSuffix("\"") && s.count >= 2 {
            s = String(s.dropFirst().dropLast())
        }
        return s
    }

    // MARK: fallback — parse tool calls MLX's own parser did not recognize
    //
    // MLX surfaces a structured `.toolCall` only for the format it knows (the
    // Qwen-style path). Other families emit valid-but-different conventions that
    // otherwise leak into the answer as raw text:
    //   Mistral:  [TOOL_CALLS][{"name": "...", "arguments": {...}}]
    //   Llama 3:  <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>
    //   bare:     {"name": "...", "arguments": {...}}
    // We detect and parse these so non-Qwen models get a fair shot at the loop.

    struct ParsedCall { let name: String; let args: [String: Any] }

    /// Try to extract the FIRST tool call from a model's raw output text,
    /// across the Mistral / Llama / bare-JSON formats. Returns nil if none.
    static func parseToolCall(from raw: String) -> ParsedCall? {
        var text = raw

        // Strip Llama special tokens so the JSON object underneath is clean.
        for token in ["<|python_tag|>", "<|eom_id|>", "<|eot_id|>",
                      "<|start_header_id|>", "<|end_header_id|>", "assistant"] {
            text = text.replacingOccurrences(of: token, with: " ")
        }
        // Strip Mistral's [TOOL_CALLS] prefix (keeps the following array/object).
        text = text.replacingOccurrences(of: "[TOOL_CALLS]", with: " ")

        // Find the first balanced {...} object.
        guard let objRange = firstJSONObjectRange(in: text) else { return nil }
        let objString = String(text[objRange])

        // Decode leniently. Mistral (via this template) emits Python-repr style
        // with SINGLE quotes — {'name': 'x', 'arguments': {...}} — which is not
        // valid JSON. Try strict JSON first; if that fails, retry with single
        // quotes normalized to double quotes (safe here: tool-call keys/values
        // are short identifiers with no apostrophes).
        func decode(_ str: String) -> [String: Any]? {
            guard let data = str.data(using: .utf8) else { return nil }
            return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        }
        guard let obj = decode(objString)
                ?? decode(objString.replacingOccurrences(of: "'", with: "\""))
        else { return nil }

        // Accept {name, arguments} or {name, parameters}.
        guard let name = obj["name"] as? String else { return nil }
        let argsAny = (obj["arguments"] as? [String: Any])
            ?? (obj["parameters"] as? [String: Any])
            ?? [:]
        // Normalize string-encoded numbers (models often quote limit, etc.).
        return ParsedCall(name: name, args: argsAny)
    }

    /// Index range of the first balanced top-level JSON object in `s`.
    private static func firstJSONObjectRange(in s: String) -> Range<String.Index>? {
        guard let start = s.firstIndex(of: "{") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var i = start
        while i < s.endIndex {
            let c = s[i]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else {
                if c == "\"" { inString = true }
                else if c == "{" { depth += 1 }
                else if c == "}" {
                    depth -= 1
                    if depth == 0 { return start..<s.index(after: i) }
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    /// Dispatch a tool call parsed from raw text (bypasses MLX's ToolCall type,
    /// which we cannot synthesize). Mirrors `dispatch`'s behavior and tracing.
    static func dispatchParsed(_ call: ParsedCall,
                               trace: @escaping @Sendable (TraceEvent) -> Void) async -> String {
        let out = await callEmailTool(normalizedName(call.name), coerce(call.args))
        trace(.toolCall(name: call.name, arguments: call.args,
                        resultChars: out.count, isError: out.hasPrefix("TOOL_ERROR")))
        return out
    }

    /// Some models prefix/namespace the tool name; keep only a known tool.
    private static func normalizedName(_ name: String) -> String {
        let known = ["resolve_relative_date", "search_email_structured",
                     "search_email", "get_message"]
        return known.first { name.hasSuffix($0) } ?? name
    }

    /// Coerce string-encoded scalars (e.g. "limit": "10") into usable types,
    /// since the downstream EmailTools closures expect Int for limit.
    private static func coerce(_ args: [String: Any]) -> [String: Any] {
        var out = args
        if let l = out["limit"] as? String, let n = Int(l) { out["limit"] = n }
        return out
    }

    // MARK: shared bridge to the EmailTools closures (single source of logic)

    private static func callEmailTool(_ name: String, _ args: [String: Any]) async -> String {
        guard let t = EmailTools.all().first(where: { $0.name == name }) else {
            return "TOOL_ERROR: unknown tool \(name)"
        }
        return await t.call(args)
    }
}

#endif
