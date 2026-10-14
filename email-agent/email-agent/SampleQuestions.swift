import Foundation

/// Questions chosen to probe the 5 viability measures (see README). Each note
/// says what a CORRECT run looks like against the fake mailbox.
enum SampleQuestions {
    static let all: [(question: String, expected: String)] = [

        // Measure 1 & 3: multi-step + don't treat promo as the answer.
        // Correct: finds the Banana Republic order confirmation (503); inspects it.
        ("Has my Banana Republic order been delivered, and when was it placed?",
         "Should find order #1S43XF4 (id 503), placed Sep 26; note it only has an "
         + "estimated delivery (Oct 1), not a delivery confirmation."),

        // Measure 2 & 4: relative dates + the no-keyword wine case.
        // Correct: window = last 3 months (after 2026-07-12). The Fortnum order
        // (504, May) is OUT of window; the K&L mail (502) is a PROMO not an order;
        // the real in-window wine order is Purple Pachyderm (501) whose subject
        // says "Slurpers shipment", not "wine" -> model must read the body.
        // Ideal answer: 1 wine order (Purple Pachyderm, Oct 4).
        ("How many wine orders have I placed in the last three months?",
         "Ideal: 1 — the Purple Pachyderm club shipment (id 501, Oct 4). Must EXCLUDE "
         + "Fortnum (May, out of window) and NOT count K&L promo as an order. Finding "
         + "501 requires reading its body or semantic search (subject has no 'wine')."),

        // Measure 2 (generalized dates): the 'last 3 years' gap we discussed.
        // Correct: uses resolve_relative_date -> after 2023-10-12. Fortnum (May 2026)
        // and all others in range; model should count wine ORDERS not promos.
        ("How many wine orders in the last 3 years?",
         "Window via resolve_relative_date -> after 2023-10-12. Wine ORDERS: Purple "
         + "Pachyderm (501) + Fortnum (504) = 2. K&L (502) is a promo, not an order."),

        // Measure 3: anti-copy-paste. A different merchant than the docstring/examples.
        ("Did I get a receipt from Shake Shack recently?",
         "Correct: finds the Shake Shack receipt (506, Jul 2). Must use 'Shake Shack', "
         + "NOT echo any example merchant."),
    ]
}
