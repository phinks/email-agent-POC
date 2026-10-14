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
    var topic: String?          // what it's ABOUT: "wine", "travel", "invoice"
    var sender: String?         // who it's FROM: "Banana Republic", a person, a service
    var intent: String?         // any|purchase|promotion|reservation|updates
    var timeframe: String?      // raw relative phrase, e.g. "last 3 months"
    var operation: String?      // lookup | count | list  (what the user wants back)

    /// The JSON shape we ask the model to produce (kept in sync with the prompt).
    static let schemaHint = """
    {"topic": string|null, "sender": string|null, \
    "intent": "any|purchase|promotion|reservation|updates"|null, \
    "timeframe": string|null, "operation": "lookup|count|list"|null}
    """
}

// MARK: - deterministic pipeline (NO model involved past extraction)

/// Runs the extracted query against the mailbox entirely in code and composes a
/// factual answer. Reuses the SAME search logic as the tool path (EmailTools /
/// Prompt), so behavior is consistent — only the DECIDER changes (code, not model).
nonisolated enum ExtractionPipeline {

    /// Correct an obvious extraction mistake in code: if the model put a topic
    /// word in `sender`, move it to `topic`. This is the one place the
    /// topic-vs-sender confusion gets fixed deterministically.
    private static let topicWords: Set<String> = [
        "wine", "coffee", "shoes", "clothing", "clothes", "burger", "food", "beer",
    ]

    static func run(_ raw: ExtractedQuery, question: String,
                    trace: @escaping @Sendable (TraceEvent) -> Void) async -> String {
        var q = raw

        // Fix topic-as-sender: "sender: wine" -> "topic: wine".
        if let s = q.sender?.lowercased(), topicWords.contains(s) {
            q.topic = q.topic ?? s
            q.sender = nil
            trace(.perf("corrected sender→topic: \(s)"))
        }

        // Drop a bogus `topic` that is really an order/intent word, not subject
        // matter. "What have I ORDERED from Banana Republic" makes the model set
        // topic="orders" — but "orders" is the thing being counted (the intent),
        // not a topic to keyword-match against the body. Leaving it in filters
        // for the literal word and finds nothing. The sender + intent already
        // capture the question; a non-topic topic only hurts.
        let nonTopicWords: Set<String> = [
            "order", "orders", "ordered", "purchase", "purchases", "receipt",
            "receipts", "email", "emails", "message", "messages", "mail",
        ]
        if let t = q.topic?.lowercased(), nonTopicWords.contains(t) {
            trace(.perf("dropped non-topic '\(t)' (it's an order/intent word)"))
            q.topic = nil
        }

        // Drop a bogus `sender` that is a GENERIC placeholder, not a named
        // sender. "How many wine orders" names no sender, but the model reaches
        // for a generic "store"/"shop"/"retailer". Filtering `from` for "store"
        // matches nothing (and leaks "store" into the answer wording). If the
        // question names a real sender it will be a specific brand/person, not
        // one of these catch-all nouns.
        let nonSenderWords: Set<String> = [
            "store", "stores", "shop", "shops", "merchant", "merchants",
            "retailer", "retailers", "sender", "company", "business", "vendor",
        ]
        if let s = q.sender?.lowercased(), nonSenderWords.contains(s) {
            trace(.perf("dropped generic sender '\(s)' (not a named sender)"))
            q.sender = nil
        }

        // GROUNDING guard: a topic/sender the model invented that the user never
        // said is a hallucination (observed: topic="wine" on a Banana Republic
        // question — bleed-through from prior wine runs). Require at least one
        // significant word of the value to actually appear in the question text;
        // otherwise drop it. This is a principled check (the model may not claim
        // a filter the user didn't mention), not a keyword blocklist.
        let questionWords = Set(
            question.lowercased()
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count > 2 }        // ignore tiny words
        )
        func grounded(_ value: String) -> Bool {
            let valueWords = value.lowercased()
                .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
                .map(String.init)
                .filter { $0.count > 2 }
            // Grounded if ANY significant word of the value appears in the question.
            return valueWords.contains { questionWords.contains($0) }
        }
        if let t = q.topic, !t.isEmpty, !grounded(t) {
            trace(.perf("dropped ungrounded topic '\(t)' (not in the question)"))
            q.topic = nil
        }
        if let s = q.sender, !s.isEmpty, !grounded(s) {
            trace(.perf("dropped ungrounded sender '\(s)' (not in the question)"))
            q.sender = nil
        }

        // Resolve a relative timeframe in code (never ask the model for dates).
        // Handles both "last N units" and vague words ("recently", "lately").
        var after: String? = nil
        var before: String? = nil
        var timeframeResolved = false
        if let tf = q.timeframe, !tf.isEmpty, tf.lowercased() != "last" {
            if let r = resolveTimeframe(tf) {
                after = r.after; before = r.before
                timeframeResolved = true
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
            (q.topic?.isEmpty == false) ||
            (q.sender?.isEmpty == false) ||
            (q.timeframe?.isEmpty == false) ||
            (intent != nil && intent != "any")
        if !hasConstraint {
            trace(.perf("no usable constraint extracted — out of schema"))
            let msg = "I can't answer that from the available fields. I can filter by "
                + "sender, topic, intent (purchase/promotion/etc.), and date range — "
                + "but not by things like read/unread or urgency."
            trace(.finalAnswer(msg))
            return msg
        }

        // Run the search in code. A TOPIC ("wine") is matched against the body
        // via the category map (POC stand-in for embeddings — the wine order 501
        // never says "wine" but lists Pinot/Cabernet/bottles/cellar). A SENDER
        // ("Banana Republic") is matched against the email's `from` field.
        let resultJSON = await search(topic: q.topic, sender: q.sender, intent: intent,
                                      after: after, before: before)
        trace(.toolCall(name: "search (code-driven)",
                        arguments: ["sender": q.sender ?? "-", "topic": q.topic ?? "-",
                                    "intent": intent ?? "any"],
                        resultChars: resultJSON.count,
                        isError: resultJSON.hasPrefix("TOOL_ERROR")))

        // Parse the result and compose the answer IN CODE — no model, no fabrication.
        let messages = decodeMessages(resultJSON)
        return compose(operation: q.operation ?? "lookup",
                       topic: q.topic, sender: q.sender,
                       timeframe: q.timeframe, timeframeResolved: timeframeResolved,
                       messages: messages)
    }

    /// Deterministic search over the mailbox. Applies, in code: sender match
    /// (against `from`), topic match (category keywords in subject+body, negation
    /// -aware), date window, and purchase/promotion intent. Returns the standard
    /// results JSON. One unified path for sender and/or topic queries.
    private static func search(topic: String?, sender: String?, intent: String?,
                               after: String?, before: String?) async -> String {
        let a = isoDate(after); let b = isoDate(before)
        let topicKeywords = topic.map { categoryKeywords[$0.lowercased()] ?? [$0.lowercased()] }
        let senderLower = sender?.lowercased()

        let hits = FakeMailbox.emails.filter { e in
            // Sender → match the FROM field only (precise, not body mentions).
            if let s = senderLower, !s.isEmpty, !e.from.lowercased().contains(s) { return false }
            // Topic → category keyword in subject+body, ignoring negated mentions.
            if let kws = topicKeywords {
                let hay = "\(e.subject) \(e.body)".lowercased()
                guard kws.contains(where: { containsNonNegated($0, in: hay) }) else { return false }
            }
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
            withJSONObject: ["count": hits.count, "messages": items],
            options: [.sortedKeys]), encoding: .utf8)) ?? #"{"count":0,"messages":[]}"#
    }

    /// Compose a factual answer from the retrieved messages. States only what the
    /// evidence supports; never asserts delivery/shipping that isn't present.
    /// `timeframeResolved` tells us whether the timeframe became a real date
    /// filter — if not (e.g. an unparseable "last", or we chose to report
    /// without filtering), we don't claim a window in the wording.
    private static func compose(operation: String,
                                topic: String?, sender: String?,
                                timeframe: String?, timeframeResolved: Bool,
                                messages: [(id: String, subject: String, date: String)]) -> String {
        let subject = sender ?? topic ?? ""
        // Only state a window if we actually applied one as a date filter.
        let window = timeframeResolved ? windowPhrase(timeframe) : ""
        // Noun: "order(s)" only when the query is actually about orders
        // (a topic or sender is named); otherwise plain "email(s)".
        let isOrderQuery = (topic?.isEmpty == false) || (sender?.isEmpty == false)
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
            // Content answer: for each match, surface the body's key facts
            // (items / total) read in code — not free generation, just echoing
            // what the body states. Falls back to subject if nothing extractable.
            let lines = messages.map { m -> String in
                let detail = orderContents(forMessageID: m.id)
                return detail.map { "• \($0) (\(m.date))" }
                    ?? "• \(m.subject) (\(m.date))"
            }
            return "Found \(n) \(qualifier)\(noun(n))\(window):\n" + lines.joined(separator: "\n")
        default: // lookup — usually a yes/no question ("did I get …?")
            let m = messages[0]
            var out = "Yes — \(qualifier.isEmpty ? "" : "\(qualifier)")\(noun(1)): "
                + "\"\(m.subject)\", dated \(m.date)."
            // Answer the delivery/shipping half of the question (if asked) by
            // reading the matched email's BODY and reporting ONLY what it states.
            // Critical: distinguish ESTIMATED delivery from CONFIRMED delivery,
            // and SHIPPED from DELIVERED — this is exactly where the tool loop
            // fabricated ("delivered Oct 1" when the body only said "estimated
            // delivery Oct 1"). Code states facts; it never upgrades an estimate
            // to a confirmation.
            if let status = deliveryStatus(forMessageID: m.id) {
                out += " " + status
            }
            // If the user asked "recently" but we couldn't bound it, be honest
            // that the match may not actually be recent (code states the date;
            // it does not pretend to judge recency it couldn't compute).
            if let tf = timeframe, !timeframeResolved,
               ["recently", "lately", "recent"].contains(tf.lowercased()) {
                out += " (note: no exact \"\(tf)\" window could be applied, so this "
                    + "is the most recent match regardless of age — check the date)."
            }
            if messages.count > 1 { out += " (+\(messages.count - 1) more)" }
            return out
        }
    }

    /// Phrase a resolved timeframe for the answer. "last 3 months" → " in the
    /// last 3 months"; vague words like "recently" → " recently" (no "in the").
    private static func windowPhrase(_ timeframe: String?) -> String {
        guard let tf = timeframe?.trimmingCharacters(in: .whitespaces), !tf.isEmpty else { return "" }
        let lower = tf.lowercased()
        // "this year/month/week", "last year/month": read naturally without
        // "in the" ("…this year", not "…in the this year").
        if lower.hasPrefix("this ") || lower == "last year" || lower == "last month"
            || lower == "last week" {
            return " \(tf)"
        }
        // "last 3 months", "past 18 months": "…in the last 3 months".
        if lower.hasPrefix("last") || lower.hasPrefix("past") {
            return " in the \(tf)"
        }
        return " \(tf)"   // "recently", etc. — avoids "in the recently"
    }

    /// Replace spelled-out number words with digits ("three" → "3") so the
    /// relative-date parser accepts either form. Finite, deterministic map.
    private static func normalizeNumberWords(_ s: String) -> String {
        let map: [String: String] = [
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "a": "1", "an": "1",
        ]
        let out = s.lowercased()
            .split(separator: " ")
            .map { map[String($0)] ?? String($0) }
            .joined(separator: " ")
        return out
    }

    /// Resolve a timeframe to a date window. Handles "last N units" (via Prompt),
    /// calendar-anchored periods ("this year", "last year", "this month"), and a
    /// few vague words with a documented default. All relative to Prompt.today.
    private static func resolveTimeframe(_ tfRaw: String) -> (after: String, before: String)? {
        // Normalize spelled-out numbers to digits so "last three months" parses
        // the same as "last 3 months" (the model sometimes writes either). This
        // is a finite, well-defined mapping — not open-ended model variance.
        let tf = normalizeNumberWords(tfRaw)
        if let r = Prompt.resolveRelativeDate(tf) { return r }
        let now = Prompt.today
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let comps = cal.dateComponents([.year, .month], from: now)
        let year = comps.year ?? 2026

        func ymd(_ y: Int, _ m: Int, _ d: Int) -> String {
            Prompt.iso(FakeMailbox.makeDate(y, m, d))
        }

        switch tf.lowercased() {
        case "recently", "lately", "recent":
            return Prompt.resolveRelativeDate("last 2 weeks")
        case "this week":
            return Prompt.resolveRelativeDate("last 7 days")
        case "this year":
            // Jan 1 of the current year → today (before is exclusive).
            return (ymd(year, 1, 1), Prompt.iso(now))
        case "last year":
            // Jan 1 → Dec 31 (exclusive end = Jan 1 of this year).
            return (ymd(year - 1, 1, 1), ymd(year, 1, 1))
        case "this month":
            return (ymd(year, comps.month ?? 1, 1), Prompt.iso(now))
        default:
            return nil
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

    /// Surface a one-line CONTENT summary from a message body (items and/or
    /// total) for "what did I order"-style questions. Echoes facts present in
    /// the body; it does not generate or infer. Returns nil if nothing itemlike
    /// is found (caller falls back to the subject).
    private static func orderContents(forMessageID id: String) -> String? {
        guard let e = FakeMailbox.emails.first(where: { $0.id == id }) else { return nil }
        let body = e.body
        var parts: [String] = []

        // Items: capture "Nx item" or "N item" style phrases (e.g. "1x Merino
        // sweater, 1x chino trousers", "1 case of English sparkling wine").
        if let re = try? NSRegularExpression(
            pattern: #"\b\d+\s*x?\s+[A-Za-z][A-Za-z ]{2,40}"#) {
            let ns = body as NSString
            let items = re.matches(in: body, range: NSRange(location: 0, length: ns.length))
                .map { ns.substring(with: $0.range).trimmingCharacters(in: .whitespaces) }
                .prefix(5)
            if !items.isEmpty { parts.append(items.joined(separator: ", ")) }
        }
        // Total: capture a "$NN.NN" amount if present.
        if let re = try? NSRegularExpression(pattern: #"\$\s?\d+(?:\.\d{2})?"#) {
            let ns = body as NSString
            if let m = re.firstMatch(in: body, range: NSRange(location: 0, length: ns.length)) {
                parts.append("total " + ns.substring(with: m.range).replacingOccurrences(of: " ", with: ""))
            }
        }
        guard !parts.isEmpty else { return nil }
        return "\(e.subject): \(parts.joined(separator: ", "))"
    }

    /// Report delivery/shipping status for a message by reading its BODY and
    /// stating ONLY what the text supports. Honest about the estimate↔confirmed
    /// and shipped↔delivered distinctions (where the tool loop fabricated).
    /// Returns nil if the body says nothing about shipping/delivery.
    private static func deliveryStatus(forMessageID id: String) -> String? {
        guard let e = FakeMailbox.emails.first(where: { $0.id == id }) else { return nil }
        let b = e.body.lowercased()

        // Confirmed delivery requires explicit past-tense delivery language.
        if b.contains("was delivered") || b.contains("has been delivered")
            || b.contains("delivered on") || b.contains("your package was delivered") {
            return "It has been delivered (per the email)."
        }
        // Estimated delivery — an estimate, NOT a confirmation. Surface the date
        // if present, but do not claim it was delivered.
        if let range = b.range(of: "estimated delivery") ?? b.range(of: "estimated to arrive") {
            let tail = b[range.upperBound...].prefix(40)
            let when = tail.trimmingCharacters(in: CharacterSet(charactersIn: " .:")).prefix(20)
            return "No delivery confirmation on file — the email only gives an "
                + (when.isEmpty ? "estimated delivery date." : "estimated delivery around \(when).")
        }
        // Shipped / on its way — in transit, not delivered.
        if b.contains("shipped") || b.contains("has shipped") || b.contains("on its way")
            || b.contains("on the way") || b.contains("shipment") {
            return "It has shipped (in transit); no delivery confirmation in the email."
        }
        // Order placed/confirmed only — nothing about fulfilment.
        if b.contains("order confirmed") || b.contains("order confirmation")
            || b.contains("thank you for your order") || b.contains("placed") {
            return "The email is an order confirmation; it does not report shipping or delivery."
        }
        return nil
    }

    /// True if `needle` appears in `hay` at least once NOT immediately preceded
    /// by a negation word ("not", "no", "non"). Narrow guard against false
    /// positives like "(Not wine — a burger order.)". Keyword search can't do
    /// real negation handling — this covers the obvious case only.
    private static func containsNonNegated(_ needle: String, in hay: String) -> Bool {
        var searchStart = hay.startIndex
        while let r = hay.range(of: needle, range: searchStart..<hay.endIndex) {
            // Look at the ~5 chars before the match for a negation word.
            let preStart = hay.index(r.lowerBound, offsetBy: -5, limitedBy: hay.startIndex)
                ?? hay.startIndex
            let preceding = hay[preStart..<r.lowerBound]
            if !preceding.contains("not") && !preceding.contains("no ")
                && !preceding.hasSuffix("non") {
                return true          // a clean, non-negated occurrence
            }
            searchStart = r.upperBound
        }
        return false
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
        - "sender" is WHO the email is from — a store, brand, person, or service \
        (e.g. "Banana Republic"). It is who sent it, NOT what it is about.
        - "topic" is WHAT the email is about (e.g. "wine", "travel", "invoice"). \
        It is the subject matter, NOT who sent it.
        - "timeframe" is the raw relative phrase from the question (e.g. "last 3 \
        months"), or null if none is mentioned.
        - "operation": "count" if the user asks "how many"; "list" if they ask \
        "what", "which", or "show me" (they want the contents/items, not a tally); \
        otherwise "lookup" (a yes/no "did I get…?").

        Schema: \(ExtractedQuery.schemaHint)

        Question: \(question)
        JSON:
        """
    }
}
