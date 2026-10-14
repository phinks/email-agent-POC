import Foundation

/// Builds the three email tools over the fake mailbox. Mirrors the Python
/// agent: search returns metadata + snippet only (NOT full bodies, so a result
/// set can't blow the context and the model must `get_message` to read a body);
/// get_message returns the full body.
nonisolated enum EmailTools {

    static func all() -> [ModelTool] {
        [resolveRelativeDate, searchEmail, searchEmailStructured, getMessage]
    }

    // MARK: resolve_relative_date (date math in code, for ANY "last N units")

    static let resolveRelativeDate = ModelTool(
        name: "resolve_relative_date",
        description: """
        Convert a relative time expression ("last 3 years", "past 18 months", \
        "last 2 weeks") into exact after/before ISO dates. ALWAYS use this instead \
        of computing dates yourself when the phrase is not in the system prompt's \
        pre-computed list. Returns {after, before}; before is exclusive.
        """,
        parameters: [
            "type": "object",
            "properties": ["expression": ["type": "string"]],
            "required": ["expression"],
        ]
    ) { args in
        let expr = (args["expression"] as? String) ?? ""
        guard let r = Prompt.resolveRelativeDate(expr) else {
            return "TOOL_ERROR: could not parse relative date \"\(expr)\"; ask the user to clarify."
        }
        return jsonString(["after": r.after, "before": r.before])
    }

    // MARK: search_email (raw keyword escape hatch)

    static let searchEmail = ModelTool(
        name: "search_email",
        description: """
        Advanced raw keyword search. Prefer search_email_structured for ordinary \
        questions. Returns metadata/snippets, not full bodies.
        """,
        parameters: [
            "type": "object",
            "properties": [
                "query": ["type": "string"],
                "limit": ["type": "integer"],
            ],
            "required": ["query"],
        ]
    ) { args in
        let query = (args["query"] as? String) ?? ""
        let limit = (args["limit"] as? Int) ?? 10
        // Keyword/substring match over subject+from+body — deliberately "dumb",
        // like IMAP TEXT, so the no-keyword wine order (501) is NOT found by
        // searching "wine" unless the body is matched... note 501's BODY does
        // contain "wine"/"bottles", so a body-inclusive keyword search CAN find
        // it; a subject-only query ("Slurpers") would miss it. This lets the
        // POC show the difference.
        // Small models often fabricate operator syntax and stuff it into the raw
        // query, e.g. "intent:purchase merchant:wine brand:example after:2026-09-12".
        // Strip any `operator:value` tokens AND bare operator keywords so only
        // real search words survive (here, "wine"). This keeps the raw tool
        // robust to that misuse instead of silently matching nothing.
        let cleaned = Self.stripFakeOperators(query)
        let terms = cleaned.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 1 }
        let hits = FakeMailbox.emails.filter { e in
            let hay = "\(e.subject) \(e.from) \(e.body)".lowercased()
            return terms.isEmpty ? false : terms.allSatisfy { hay.contains($0) }
        }
        return encodeResults(query: cleaned, messages: Array(hits.prefix(limit)))
    }

    // MARK: search_email_structured (the REQUIRED semantic filter)

    static let searchEmailStructured = ModelTool(
        name: "search_email_structured",
        description: """
        REQUIRED search for ordinary email questions. Call before answering when \
        the user mentions a merchant, order, receipt, shipment, delivery, \
        reservation, or date range. A broad search may contain only promotions; \
        that is NOT evidence the requested email is missing — filter again here. \
        Fill arguments FROM THE QUESTION (do not copy example shapes). For relative \
        periods use the pre-computed after/before from the system prompt; do not \
        compute dates yourself. before is exclusive. Returns metadata/snippets.
        """,
        parameters: [
            "type": "object",
            "properties": [
                "merchant": ["type": "string"],
                "sender_domain": ["type": "string"],
                "intent": ["type": "string",
                           "enum": ["any", "purchase", "promotion", "reservation", "updates"]],
                "after": ["type": "string"],
                "before": ["type": "string"],
                "limit": ["type": "integer"],
            ],
        ]
    ) { args in
        let merchant = (args["merchant"] as? String)?.lowercased()
        let domain = (args["sender_domain"] as? String)?.lowercased()
        let intent = (args["intent"] as? String) ?? "any"
        let after = parseISO(args["after"] as? String)
        let before = parseISO(args["before"] as? String)   // exclusive
        let limit = (args["limit"] as? Int) ?? 10

        let hits = FakeMailbox.emails.filter { e in
            if let m = merchant, !m.isEmpty,
               !("\(e.subject) \(e.from) \(e.body)".lowercased().contains(m)) { return false }
            if let d = domain, !d.isEmpty, !e.from.lowercased().contains(d) { return false }
            if let a = after, e.date < a { return false }
            if let b = before, e.date >= b { return false }     // exclusive upper
            if intent == "purchase", !looksLikePurchase(e) { return false }
            if intent == "promotion", !looksLikePromotion(e) { return false }
            return true
        }
        return encodeResults(query: "structured(merchant=\(merchant ?? "-"),intent=\(intent))",
                             messages: Array(hits.prefix(limit)))
    }

    // MARK: get_message (full body — the inspect step)

    static let getMessage = ModelTool(
        name: "get_message",
        description: "Retrieve one email INCLUDING its full decoded body, by message id.",
        parameters: [
            "type": "object",
            "properties": ["message_id": ["type": "string"]],
            "required": ["message_id"],
        ]
    ) { args in
        let id = (args["message_id"] as? String) ?? ""
        guard let e = FakeMailbox.emails.first(where: { $0.id == id }) else {
            return "TOOL_ERROR: no message with id \(id); it returned NO email data."
        }
        return jsonString([
            "id": e.id, "from": e.from, "subject": e.subject,
            "date": isoString(e.date), "body": e.body,
        ])
    }

    // MARK: helpers

    /// Remove fabricated `operator:value` tokens (and leftover operator words)
    /// the model sometimes injects into a raw query, leaving only real words.
    private static func stripFakeOperators(_ query: String) -> String {
        let fakeKeys: Set<String> = [
            "intent", "merchant", "brand", "after", "before", "from",
            "subject", "category", "to", "label", "in",
        ]
        let kept = query.split(separator: " ").filter { token in
            if let colon = token.firstIndex(of: ":") {
                let key = token[token.startIndex..<colon].lowercased()
                // drop known operator:value tokens entirely
                if fakeKeys.contains(key) { return false }
            }
            // drop bare operator keywords
            return !fakeKeys.contains(token.lowercased())
        }
        return kept.joined(separator: " ")
    }

    private static func looksLikePurchase(_ e: FakeEmail) -> Bool {
        let s = "\(e.subject) \(e.body)".lowercased()
        return ["order", "confirmation", "receipt", "shipped", "shipment", "shipping",
                "delivery", "club selection", "your case"].contains { s.contains($0) }
            && !looksLikePromotion(e)
    }

    private static func looksLikePromotion(_ e: FakeEmail) -> Bool {
        let s = "\(e.subject) \(e.body)".lowercased()
        return ["% off", "sale", "promotional", "unsubscribe", "offer", "deal"]
            .contains { s.contains($0) }
    }

    /// Search results: metadata + snippet only, NEVER full body.
    private static func encodeResults(query: String, messages: [FakeEmail]) -> String {
        let items: [[String: Any]] = messages.map {
            ["id": $0.id, "from": $0.from, "subject": $0.subject,
             "date": isoString($0.date), "snippet": $0.snippet]
        }
        return jsonString(["query": query, "count": messages.count, "messages": items])
    }

    private static func parseISO(_ s: String?) -> Date? {
        guard let s, !s.isEmpty else { return nil }
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"
        return f.date(from: s)
    }

    private static func isoString(_ d: Date) -> String {
        let f = DateFormatter()
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: d)
    }

    private static func jsonString(_ obj: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: obj,
                                                     options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else {
            return "TOOL_ERROR: failed to encode result"
        }
        return s
    }
}
