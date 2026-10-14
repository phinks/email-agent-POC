import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Copy text to the system clipboard. Cross-platform: UIKit on iOS, AppKit on
/// macOS.
private func copyToClipboard(_ text: String) {
    #if canImport(UIKit)
    UIPasteboard.general.string = text
    #elseif canImport(AppKit)
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
    #endif
}

/// Minimal harness: pick a model provider, ask a question (or tap a sample),
/// run the agent, and watch the tool-call trace + final answer on screen.
/// This is the whole UI — deliberately bare; the POC is about the trace.
struct ContentView: View {
    enum ProviderChoice: String, CaseIterable, Identifiable {
        case apple = "Apple"
        case mlx = "MLX"
        case mlxExtract = "MLX·Extract"
        case mock = "Mock"
        var id: String { rawValue }
    }

    /// Curated MLX model ids (all under the `mlx-community` HuggingFace org, so
    /// they download on first use). The list is just a convenience — any valid
    /// MLX-compatible id works via the "Custom…" option. Verified to resolve on
    /// HuggingFace at time of writing; a bad/offline id fails at download time.
    struct MLXCatalog {
        static let models: [(label: String, id: String)] = [
            // Small end — fast, tiny, likely weaker (the "where's the floor" curve).
            ("Qwen2.5 0.5B (4-bit)", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
            ("Llama 3.2 1B (4-bit)", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
            ("Qwen2.5 1.5B (4-bit)", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
            // ~2–3B — the main phone-viable comparison class.
            ("Gemma 2 2B (4-bit)",   "mlx-community/gemma-2-2b-it-4bit"),
            ("Qwen2.5 3B (4-bit)",   "mlx-community/Qwen2.5-3B-Instruct-4bit"),
            ("Llama 3.2 3B (4-bit)", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
            ("Phi-3.5 mini (4-bit)", "mlx-community/Phi-3.5-mini-instruct-4bit"),
            // ~7–8B — phone-marginal (needs the memory entitlement).
            ("Qwen2.5 7B (4-bit)",   "mlx-community/Qwen2.5-7B-Instruct-4bit"),
            ("Mistral 7B v0.3 (4-bit)", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
            ("Ministral 8B · tool-tuned (4-bit)", "mlx-community/Ministral-8B-Instruct-2410-4bit"),
            ("Llama 3.1 8B (4-bit)", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"),
            // Large — Mac only (won't fit a phone even with the entitlement).
            ("Qwen2.5 14B · Mac only (4-bit)", "mlx-community/Qwen2.5-14B-Instruct-4bit"),
        ]
        static let customLabel = "Custom…"
    }

    @State private var question = ""
    @State private var trace: [TraceLine] = []
    @State private var answer = ""
    @State private var running = false
    /// Label of the model used for the most recent run (shown with the answer
    /// and included in the "Copy all" text, so pasted traces are unambiguous).
    @State private var modelLabel = ""

    // --- persisted selections (restored on next launch) --------------------
    // @AppStorage writes to UserDefaults so the last-chosen provider + MLX model
    // are the defaults next time the app opens. ProviderChoice persists via its
    // String rawValue (RawRepresentable).
    @AppStorage("provider") private var choice: ProviderChoice = .apple
    /// Selected MLX model id (defaults to the Qwen2.5 3B baseline).
    @AppStorage("mlxModelID") private var mlxModelID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    /// True when the user picked "Custom…" and is typing an arbitrary id.
    @AppStorage("mlxCustom") private var mlxCustom = false
    /// The free-text custom id (used only when `mlxCustom` is true).
    @AppStorage("mlxCustomID") private var mlxCustomID = ""

    /// The effective MLX id, honoring the custom field when active.
    private var effectiveMLXID: String {
        let custom = mlxCustomID.trimmingCharacters(in: .whitespacesAndNewlines)
        return (mlxCustom && !custom.isEmpty) ? custom : mlxModelID
    }

    private var provider: ModelProvider {
        switch choice {
        case .apple:       return AppleFoundationModel()
        case .mlx:         return MLXModel(modelID: effectiveMLXID)
        case .mlxExtract:  return MLXExtractionModel(modelID: effectiveMLXID)
        case .mock:        return MockModel()
        }
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                providerRow
                questionField
                sampleButtons
                Divider()
                traceView
            }
            .padding()
            .navigationTitle("Email Agent POC")
        }
    }

    private var providerRow: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Model", selection: $choice) {
                ForEach(ProviderChoice.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            if choice == .mlx || choice == .mlxExtract { mlxModelPicker }

            Label(availabilityText,
                  systemImage: provider.isAvailable ? "checkmark.circle" : "xmark.circle")
                .font(.caption)
                .foregroundStyle(provider.isAvailable ? .green : .red)
        }
    }

    /// Dropdown of curated MLX models plus a "Custom…" free-text option.
    /// The chosen id is downloaded by MLX on first run (see the trace's
    /// "downloading" line).
    private var mlxModelPicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("MLX model").font(.caption).foregroundStyle(.secondary)
                Menu {
                    ForEach(MLXCatalog.models, id: \.id) { model in
                        Button(model.label) {
                            mlxModelID = model.id
                            mlxCustom = false
                        }
                    }
                    Divider()
                    Button(MLXCatalog.customLabel) { mlxCustom = true }
                } label: {
                    HStack(spacing: 4) {
                        Text(mlxCustom
                             ? MLXCatalog.customLabel
                             : (MLXCatalog.models.first { $0.id == mlxModelID }?.label ?? mlxModelID))
                        Image(systemName: "chevron.up.chevron.down").font(.caption2)
                    }
                }
                .disabled(running)
            }
            if mlxCustom {
                TextField("mlx-community/Your-Model-4bit", text: $mlxCustomID)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.caption, design: .monospaced))
                    .disabled(running)
                    #if os(iOS)
                    .autocorrectionDisabled(true)
                    .textInputAutocapitalization(.never)
                    #endif
            }
            Text(effectiveMLXID)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var availabilityText: String {
        #if canImport(FoundationModels)
        if let apple = provider as? AppleFoundationModel { return apple.availabilityDetail }
        #endif
        return provider.isAvailable ? "available" : "unavailable"
    }

    private var questionField: some View {
        HStack {
            TextField("Ask about the (fake) mailbox…", text: $question, axis: .vertical)
                .textFieldStyle(.roundedBorder)
            Button(running ? "…" : "Ask") { Task { await run(question) } }
                .buttonStyle(.borderedProminent)
                .disabled(running || question.isEmpty)
        }
    }

    private var sampleButtons: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack {
                ForEach(Array(SampleQuestions.all.enumerated()), id: \.offset) { _, item in
                    Button(item.question) {
                        question = item.question
                        Task { await run(item.question) }
                    }
                    .font(.caption2)
                    .buttonStyle(.bordered)
                    .disabled(running)
                }
            }
        }
    }

    private var traceView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                if !answer.isEmpty {
                    HStack {
                        Text("ANSWER").font(.caption).bold()
                        Spacer()
                        Button("Copy all") { copyToClipboard(plainText) }
                            .font(.caption2).buttonStyle(.bordered)
                    }
                    Text(answer).padding(8)
                        .background(.green.opacity(0.12)).cornerRadius(8)
                        .textSelection(.enabled)
                }
                Text("TRACE").font(.caption).bold()
                ForEach(trace) { line in
                    Text(line.text)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(line.color)
                        .textSelection(.enabled)
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// The whole run (answer + trace) as plain text, for the Copy button.
    private var plainText: String {
        var out = ""
        if !modelLabel.isEmpty { out += "MODEL: \(modelLabel)\n\n" }
        out += "ANSWER:\n\(answer)\n\nTRACE:\n"
        out += trace.map(\.text).joined(separator: "\n")
        return out
    }

    @MainActor
    private func run(_ q: String) async {
        running = true; trace = []; answer = ""
        let selected = provider                 // capture once (stable label)
        modelLabel = selected.label
        trace.append(TraceLine(model: selected.label))   // self-identify the run
        let agent = Agent(model: selected)
        let result = await agent.answer(q) { event in
            Task { @MainActor in appendTrace(TraceLine(event)) }
        }
        answer = result
        running = false
    }

    /// Append a trace line — but collapse consecutive download-progress updates
    /// onto a single line (so "downloading 1%..2%..100%" doesn't flood the log).
    @MainActor
    private func appendTrace(_ line: TraceLine) {
        if line.isDownloadProgress, let last = trace.last, last.isDownloadProgress {
            trace[trace.count - 1] = line     // overwrite the previous % in place
        } else {
            trace.append(line)
        }
    }
}

/// A renderable trace line derived from a TraceEvent.
struct TraceLine: Identifiable {
    let id = UUID()
    let text: String
    let color: Color
    /// True for the "downloading X%" progress ticks, so the UI can collapse a
    /// run of them onto one updating line instead of one line per percent.
    let isDownloadProgress: Bool

    /// A plain header line naming the model used for the run.
    init(model label: String) {
        text = "🧠 model: \(label)"
        color = .indigo
        isDownloadProgress = false
    }

    init(_ event: TraceEvent) {
        var isDownload = false
        switch event {
        case .prompt(_, let question):
            text = "▶ question: \(question)"; color = .primary
        case .toolCall(let name, let args, let chars, let isError):
            let a = args.map { "\($0)=\($1)" }.joined(separator: ", ")
            text = "\(isError ? "✗" : "•") \(name)(\(a)) -> \(chars) chars"
            color = isError ? .red : .blue
        case .modelTurn(let step, let made):
            text = "  model turn \(step)\(made ? " (tool call)" : " (final)")"; color = .secondary
        case .finalAnswer:
            text = "✓ final answer produced"; color = .green
        case .aborted(let reason):
            text = "⚠ aborted: \(reason)"; color = .orange
        case .perf(let summary):
            text = "⏱ \(summary)"; color = .purple
            isDownload = summary.hasPrefix("downloading")
        }
        isDownloadProgress = isDownload
    }
}
