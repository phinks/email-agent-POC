import Foundation

// ExtractionAgent — the "model understands, code acts" architecture.
//
// Every failure in the tool-loop runs was at a DECISION or a HANDOFF: which
// tool, what args, resolve-then-fail-to-search, narrate-instead-of-call. This
// path removes all of that from the model. The model makes exactly ONE call:
// turn the question into a structured query (JSON). Then CODE does the rest —
// resolve dates, run the search, count/collect, and compose the answer. The
// model never drives a loop, never picks a tool, never counts, and (in this
// no-synthesis version) never phrases the final answer, so it cannot fabricate
// an email, a date, or a shipment.
//
// This is recommendation #1 from VERDICT.md made concrete, with the model also
// doing the INPUT parsing (question -> query), not just output phrasing.

// MARK: - the structured query the model fills in

/// What the model extracts from the question. All fields optional so the model
/// only fills what the question actually specifies.
struct ExtractedQuery: Codable {
    var product: String?        // "wine", "shoes" — a PRODUCT, not a store
    var merchant: String?       // "Banana Republic" — a store/brand
    var intent: String?         // any|purchase|promotion|reservation|updates
    var timeframe: String?      // raw relative phrase, e.g. "last 3 months"
    var operation: String?      // lookup | count | list  (what the user wants back)

    /// The JSON shape we ask the model to produce (kept in sync with the prompt).
    static let schemaHint = """
    {"product": string|null, "merchant": string|null, \
    "intent": "any|purchase|promotion|reservation|updates"|null, \
    "timeframe": string|null, "operation": "lookup|count|list"|null}
    """
}

// MARK: - deterministic pipeline (NO model involved past extraction)

/// Runs the extracted query against the mailbox entirely in code and composes a
/// factual answer. Reuses the SAME search logic as the tool path (EmailTools /
/// Prompt), so behavior is consistent — only the DECIDER changes (code, not model).
nonisolated enum ExtractionPipeline {

    /// Correct an obvious extraction mistake in code: if the model put a product
    /// word in `merchant`, move it to `product`. This is the one place the
    /// product-vs-merchant confusion gets fixed deterministically.
    private static let productWords: Set<String> = [
        "wine", "coffee", "shoes", "clothing", "clothes", "burger", "food", "beer",
    ]

    static func run(_ raw: ExtractedQuery,
                    trace: @escaping @Sendable (TraceEvent) -> Void) async -> String {
        var q = raw

        // Fix product-as-merchant: "merchant: wine" -> "product: wine".
        if let m = q.merchant?.lowercased(), productWords.contains(m) {
            q.product = q.product ?? m
            q.merchant = nil
            trace(.perf("corrected merchant→product: \(m)"))
        }

        // Resolve a relative timeframe in code (never ask the model for dates).
        var after: String? = nil
        var before: String? = nil
        if let tf = q.timeframe, !tf.isEmpty {
            if let r = Prompt.resolveRelativeDate(tf) {
                after = r.after; before = r.before
                trace(.perf("resolved '\(tf)' → \(r.after)..\(r.before)"))
            } else {
                trace(.perf("could not resolve timeframe '\(tf)' — searching full history"))
            }
        }

        // Validate the extracted intent: models sometimes echo the enum option
        // list ("reservation|updates") as the value. Keep only a single valid enum.
        let validIntents: Set<String> = ["any", "purchase", "promotion", "reservation", "updates"]
        let intent: String? = q.intent.flatMap { validIntents.contains($0) ? $0 : nil }
        if q.intent != nil && intent == nil {
            trace(.perf("dropped malformed intent '\(q.intent!)'"))
        }

        // Out-of-schema guard. If the question extracted NO usable constraint
        // (no product, merchant, timeframe, and intent is any/none), the schema
        // could not express what was asked (e.g. "unread", "urgent", "from my
        // wife" — concepts this schema/mailbox doesn't model). Rather than run a
        // blanket search and confidently miscount ALL emails, say so honestly.
        let hasConstraint =
            (q.product?.isEmpty == false) ||
            (q.merchant?.isEmpty == false) ||
            (q.timeframe?.isEmpty == false) ||
            (intent != nil && intent != "any")
        if !hasConstraint {
            trace(.perf("no usable constraint extracted — out of schema"))
            let msg = "I can't answer that from the available fields. I can filter by "
                + "merchant, product, intent (purchase/promotion/etc.), and date range — "
                + "but not by things like read/unread, urgency, or sender relationship."
            trace(.finalAnswer(msg))
            return msg
        }

        // Build args for the SAME structured search the tool path uses.
        // IMPORTANT: a PRODUCT ("wine") is NOT a merchant. We expand it to a set
        // of body keywords (code-side category map) so semantic product queries
        // can match emails that never contain the literal product word — e.g.
        // the wine order (501) says "Pinot Noir / Cabernet / bottles / cellar"
        // but never "wine". Keyword-only matching WOULD miss it; this is where a
        // real system needs embeddings. The category map is the POC stand-in.
        var args: [String: Any] = [:]
        if let m = q.merchant, !m.isEmpty { args["merchant"] = m }
        if let i = intent, !i.isEmpty { args["intent"] = i }
        if let a = after { args["after"] = a }
        if let b = before { args["before"] = b }
        args["limit"] = 50

        let resultJSON: String
        if let p = q.product?.lowercased(), !p.isEmpty, q.merchant == nil {
            // Product query: match ANY of the category keywords in the body, then
            // apply the SAME date/intent filtering as structured search, in code.
            resultJSON = await productSearch(product: p, intent: intent,
                                             after: after, before: before)
            trace(.toolCall(name: "product_search (code-driven, category keywords)",
                            arguments: ["product": p, "intent": intent ?? "any"],
                            resultChars: resultJSON.count,
                            isError: resultJSON.hasPrefix("TOOL_ERROR")))
        } else {
            guard let tool = EmailTools.all().first(where: { $0.name == "search_email_structured" })
            else { return "Internal error: search tool unavailable." }
            resultJSON = await tool.call(args)
            trace(.toolCall(name: "search_email_structured (code-driven)", arguments: args,
                            resultChars: resultJSON.count, isError: resultJSON.hasPrefix("TOOL_ERROR")))
        }

        // Parse the result and compose the answer IN CODE — no model, no fabrication.
        let messages = decodeMessages(resultJSON)
        return compose(operation: q.operation ?? "lookup",
                       product: q.product, merchant: q.merchant,
                       timeframe: q.timeframe, messages: messages)
    }

    /// Compose a factual answer from the retrieved messages. States only what the
    /// evidence supports; never asserts delivery/shipping that isn't present.
    private static func compose(operation: String,
                                product: String?, merchant: String?,
                                timeframe: String?, messages: [(id: String, subject: String, date: String)]) -> String {
        let subject = merchant ?? product ?? ""
        let window = timeframe.map { " in the \($0)" } ?? ""
        // Noun: "order(s)" only when the query is actually about orders
        // (a product or merchant is named); otherwise plain "email(s)".
        let isOrderQuery = (product?.isEmpty == false) || (merchant?.isEmpty == false)
        func noun(_ n: Int) -> String {
            let base = isOrderQuery ? "order" : "email"
            return base + (n == 1 ? "" : "s")
        }
        let qualifier = subject.isEmpty ? "" : "\(subject) "

        if messages.isEmpty {
            return "No \(qualifier)\(noun(0)) found\(window)."
        }

        switch operation {
        case "count":
            let n = messages.count
            return "You have \(n) \(qualifier)\(noun(n))\(window)."
        case "list":
            let n = messages.count
            let lines = messages.map { "• \($0.subject) (\($0.date), id \($0.id))" }
            return "Found \(n) \(qualifier)\(noun(n))\(window):\n" + lines.joined(separator: "\n")
        default: // lookup
            let m = messages[0]
            return "Found: \(m.subject) (dated \(m.date), id \(m.id))."
                + (messages.count > 1 ? " (+\(messages.count - 1) more)" : "")
        }
    }

    /// Code-side category map: a product term expands to body keywords that
    /// imply it. This is the POC stand-in for semantic search — it lets "wine"
    /// match the adversarial order (501) that never says "wine" but lists
    /// Pinot Noir / Cabernet / Chardonnay / bottles / cellar. A production
    /// system would use embeddings here; the point is that IMPROVING RETRIEVAL
    /// IS A CODE CHANGE, not a model-reliability gamble.
    private static let categoryKeywords: [String: [String]] = [
        "wine": ["wine", "bottle", "bottles", "pinot", "cabernet", "chardonnay",
                 "merlot", "sauvignon", "sparkling", "vineyard", "cellar", "burgundy"],
        "coffee": ["coffee", "espresso", "beans", "roast"],
        "shoes": ["shoe", "shoes", "sneaker", "boot", "loafer"],
    ]

    /// Deterministic product search: match ANY category keyword in subject/body,
    /// then apply the same date + purchase/promotion intent filtering as the
    /// structured tool. Returns the SAME JSON shape so compose()/decode work.
    private static func productSearch(product: String, intent: String?,
                                      after: String?, before: String?) async -> String {
        let keywords = categoryKeywords[product] ?? [product]
        let a = isoDate(after); let b = isoDate(before)
        let hits = FakeMailbox.emails.filter { e in
            let hay = "\(e.subject) \(e.body)".lowercased()
            guard keywords.contains(where: { hay.contains($0) }) else { return false }
            if let a, e.date < a { return false }
            if let b, e.date >= b { return false }
            switch intent {
            case "purchase":  return EmailTools.looksLikePurchase(e)
            case "promotion": return EmailTools.looksLikePromotion(e)
            default:          return true
            }
        }
        let items: [[String: Any]] = hits.map {
            ["id": $0.id, "from": $0.from, "subject": $0.subject,
             "date": isoString($0.date), "snippet": $0.snippet]
        }
        return (try? String(data: JSONSerialization.data(
            withJSONObject: ["query": "product:\(product)", "count": hits.count,
                             "messages": items], options: [.sortedKeys]),
            encoding: .utf8)) ?? #"{"count":0,"messages":[]}"#
    }

    private static func isoDate(_ s: String?) -> Date? {
        guard let s, !s.isEmpty else { return nil }
        let f = DateFormatter(); f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"; return f.date(from: s)
    }
    private static func isoString(_ d: Date) -> String {
        let f = DateFormatter(); f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"; return f.string(from: d)
    }

    private static func decodeMessages(_ json: String)
        -> [(id: String, subject: String, date: String)] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = obj["messages"] as? [[String: Any]] else { return [] }
        return items.map {
            (id: ($0["id"] as? String) ?? "?",
             subject: ($0["subject"] as? String) ?? "",
             date: ($0["date"] as? String) ?? "")
        }
    }

    /// The extraction prompt: ask the model for ONLY the JSON query, nothing else.
    static func extractionPrompt(for question: String) -> String {
        """
        You convert a user's email question into a JSON query. Output ONLY the \
        JSON object, no prose, no explanation.

        Rules:
        - "merchant" is a STORE or BRAND (e.g. "Banana Republic"). NEVER a product.
        - "product" is a product type (e.g. "wine", "shoes"). NEVER a store.
        - "timeframe" is the raw relative phrase from the question (e.g. "last 3 \
        months"), or null if none is mentioned.
        - "operation" is "count" if the user asks "how many", "list" if they want \
        all matches, otherwise "lookup".

        Schema: \(ExtractedQuery.schemaHint)

        Question: \(question)
        JSON:
        """
    }
}
