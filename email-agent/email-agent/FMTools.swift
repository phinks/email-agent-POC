import Foundation

#if canImport(FoundationModels)
import FoundationModels

// Foundation Models requires each tool to be a concrete type conforming to
// `Tool` with a `@Generable` Arguments struct (resolved at compile time). Our
// backend-neutral `ModelTool` is dynamic ([String: Any] + closure), so we can't
// synthesize Generable types from it at runtime. Instead we define one concrete
// Tool per email tool here, each delegating to the matching ModelTool.call so
// the actual logic (filtering the fake mailbox) stays in EmailTools.swift.
//
// NOTE: FoundationModels type/method names below follow the iOS 26 API
// (Tool, @Generable, @Guide). Tool.call returns String (ToolOutput was removed
// in Xcode 26 beta 5). Verify against the SDK you build with.

enum FMTools {

    /// Build the concrete FoundationModels tools, matching by name to the
    /// ModelTool closures (so EmailTools remains the single source of logic).
    /// `trace` is invoked for each call so the harness can display it.
    static func make(from tools: [ModelTool], trace: @escaping @Sendable (TraceEvent) -> Void) -> [any Tool] {
        func lookup(_ name: String) -> ModelTool {
            guard let t = tools.first(where: { $0.name == name }) else {
                fatalError("missing ModelTool \(name)")
            }
            return t
        }
        return [
            ResolveRelativeDateTool(impl: lookup("resolve_relative_date"), trace: trace),
            SearchEmailTool(impl: lookup("search_email"), trace: trace),
            SearchEmailStructuredTool(impl: lookup("search_email_structured"), trace: trace),
            GetMessageTool(impl: lookup("get_message"), trace: trace),
        ]
    }

    /// Shared helper: run a ModelTool.call, emit a trace, return the tool's
    /// textual output. (Tool.call returns a String directly in current SDKs;
    /// `ToolOutput` was removed in Xcode 26 beta 5 in favour of String /
    /// GeneratedContent.)
    fileprivate static func run(
        _ impl: ModelTool, _ args: [String: Any],
        _ trace: @escaping @Sendable (TraceEvent) -> Void
    ) async -> String {
        let result = await impl.call(args)
        let isError = result.hasPrefix("TOOL_ERROR")
        trace(.toolCall(name: impl.name, arguments: args,
                        resultChars: result.count, isError: isError))
        return result
    }
}

// MARK: - resolve_relative_date

struct ResolveRelativeDateTool: Tool {
    let impl: ModelTool
    let trace: @Sendable (TraceEvent) -> Void

    let name = "resolve_relative_date"
    var description: String { impl.description }

    @Generable
    struct Arguments {
        @Guide(description: "A relative time expression, e.g. 'last 3 years', 'past 18 months'")
        var expression: String
    }

    func call(arguments: Arguments) async throws -> String {
        await FMTools.run(impl, ["expression": arguments.expression], trace)
    }
}

// MARK: - search_email

struct SearchEmailTool: Tool {
    let impl: ModelTool
    let trace: @Sendable (TraceEvent) -> Void

    let name = "search_email"
    var description: String { impl.description }

    @Generable
    struct Arguments {
        @Guide(description: "Raw provider query string")
        var query: String
        @Guide(description: "Max results", .range(1...50))
        var limit: Int
    }

    func call(arguments: Arguments) async throws -> String {
        await FMTools.run(impl, ["query": arguments.query, "limit": arguments.limit], trace)
    }
}

// MARK: - search_email_structured

struct SearchEmailStructuredTool: Tool {
    let impl: ModelTool
    let trace: @Sendable (TraceEvent) -> Void

    let name = "search_email_structured"
    var description: String { impl.description }

    @Generable
    struct Arguments {
        @Guide(description: "Store/organization named in the question")
        var merchant: String?
        @Guide(description: "Only a domain actually seen in evidence; never guess")
        var senderDomain: String?
        @Guide(description: "one of: any, purchase, promotion, reservation, updates")
        var intent: String?
        @Guide(description: "Inclusive start date, YYYY-MM-DD (use pre-computed bounds)")
        var after: String?
        @Guide(description: "Exclusive end date, YYYY-MM-DD (use pre-computed bounds)")
        var before: String?
        @Guide(description: "Max results", .range(1...50))
        var limit: Int?
    }

    func call(arguments: Arguments) async throws -> String {
        var args: [String: Any] = [:]
        if let v = arguments.merchant { args["merchant"] = v }
        if let v = arguments.senderDomain { args["sender_domain"] = v }
        if let v = arguments.intent { args["intent"] = v }
        if let v = arguments.after { args["after"] = v }
        if let v = arguments.before { args["before"] = v }
        args["limit"] = arguments.limit ?? 10
        return await FMTools.run(impl, args, trace)
    }
}

// MARK: - get_message

struct GetMessageTool: Tool {
    let impl: ModelTool
    let trace: @Sendable (TraceEvent) -> Void

    let name = "get_message"
    var description: String { impl.description }

    @Generable
    struct Arguments {
        @Guide(description: "The message id to retrieve in full")
        var messageId: String
    }

    func call(arguments: Arguments) async throws -> String {
        await FMTools.run(impl, ["message_id": arguments.messageId], trace)
    }
}

#endif
