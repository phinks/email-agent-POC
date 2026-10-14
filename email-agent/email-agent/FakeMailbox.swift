import Foundation

/// A normalized email, mirroring the Python agent's message shape
/// (id, from, subject, date, body). In-memory only — no IMAP, no network.
struct FakeEmail: Identifiable {
    let id: String          // stands in for a UID / message id
    let from: String
    let subject: String
    let date: Date
    let body: String

    var snippet: String {
        let flat = body.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return String(flat.prefix(200))
    }
}

/// Deterministic in-memory mailbox. Dates are expressed relative to a fixed
/// "today" so the POC's relative-date behavior is reproducible regardless of
/// when it's run. `FakeMailbox.today` is the reference date the Prompt layer
/// also uses to compute windows.
nonisolated enum FakeMailbox {
    /// Fixed reference date for the POC (keep in sync with Prompt.today).
    static let today = makeDate(2026, 10, 12)

    static func makeDate(_ y: Int, _ m: Int, _ d: Int) -> Date {
        var c = DateComponents()
        c.year = y; c.month = m; c.day = d
        c.timeZone = TimeZone(identifier: "UTC")
        return Calendar(identifier: .gregorian).date(from: c)!
    }

    /// The messages. Deliberately includes adversarial cases that probe the
    /// 5 viability measures (see README).
    static let emails: [FakeEmail] = [

        // --- ADVERSARIAL #1: a real wine order with NO keyword in subject/sender.
        // The Purple Pachyderm analogue. Keyword search for "wine" MUST miss this;
        // only reading the body (or semantic search) reveals it is wine.
        FakeEmail(
            id: "501",
            from: "Purple Pachyderm <info@purplepachyderm.com>",
            subject: "Your League of Extraordinary Slurpers shipment is on its way",
            date: makeDate(2026, 10, 4),
            body: """
            Hello member, your October club selection has shipped. This month's \
            case includes six bottles: a 2022 Russian River Pinot Noir, a 2021 \
            Napa Cabernet Sauvignon, and two bottles of our estate Chardonnay. \
            Tasting notes and food pairings are enclosed. Cheers from the cellar!
            """
        ),

        // --- ADVERSARIAL #2: promotional wine mail that is NOT an order.
        // Must NOT be counted as a "wine order". Tests the promotions-aren't-
        // evidence rule and the order-vs-marketing discrimination.
        FakeEmail(
            id: "502",
            from: "K&L Wine Merchants <offers@klwines.com>",
            subject: "48 hours only: 20% off all Burgundy",
            date: makeDate(2026, 9, 28),
            body: """
            Our biggest Burgundy sale of the year ends Sunday. Shop hundreds of \
            wines and save 20% at checkout. This is a promotional email; no \
            purchase has been made. Unsubscribe anytime.
            """
        ),

        // --- ADVERSARIAL #3: a clear, obvious order confirmation (control case).
        // A capable model should find this easily; a weak one may still stop
        // at the bootstrap. Tests the baseline multi-step behavior.
        FakeEmail(
            id: "503",
            from: "Banana Republic <orders@email.bananarepublic.com>",
            subject: "Order Confirmation #1S43XF4",
            date: makeDate(2026, 9, 26),
            body: """
            Thank you for your order #1S43XF4 placed on September 26, 2026. \
            1x Merino sweater, 1x chino trousers. Estimated delivery October 1.
            """
        ),

        // --- OUT-OF-WINDOW wine order: real wine order but ~5 months ago.
        // Tests date filtering: must be EXCLUDED from "last three months".
        FakeEmail(
            id: "504",
            from: "Fortnum & Mason <orders@fortnumandmason.com>",
            subject: "Your order of English Sparkling Wine is confirmed",
            date: makeDate(2026, 5, 2),
            body: """
            Order confirmed: 1 case of English sparkling wine. Placed May 2, 2026. \
            Thank you for shopping with us.
            """
        ),

        // --- NOISE: unrelated recent mail, to pad the window and tempt
        // false positives (finance newsletter, like the real mailbox had).
        FakeEmail(
            id: "505",
            from: "Agora Financial <news@agorafinancial.com>",
            subject: "5 Min. Forecast: markets and the week ahead",
            date: makeDate(2026, 10, 8),
            body: "This week in markets... (unrelated newsletter content)."
        ),
        FakeEmail(
            id: "506",
            from: "Shake Shack <no-reply@shakeshack.com>",
            subject: "Your Shake Shack order receipt",
            date: makeDate(2026, 7, 2),
            body: "Thanks for your order. Total $30.54. (Not wine — a burger order.)"
        ),
    ]
}
