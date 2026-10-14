import SwiftUI
import UIKit

/// Minimal harness: pick a model provider, ask a question (or tap a sample),
/// run the agent, and watch the tool-call trace + final answer on screen.
/// This is the whole UI — deliberately bare; the POC is about the trace.
struct ContentView: View {
    enum ProviderChoice: String, CaseIterable, Identifiable {
        case apple = "Apple"
        case mlx = "MLX 3B"
        case mlx7b = "MLX 7B"
        case mock = "Mock"
        var id: String { rawValue }
    }

    @State private var question = ""
    @State private var trace: [TraceLine] = []
    @State private var answer = ""
    @State private var running = false
    @State private var choice: ProviderChoice = .apple

    private var provider: ModelProvider {
        switch choice {
        case .apple:  return AppleFoundationModel()
        case .mlx:    return MLXModel()
        case .mlx7b:  return MLXModel(modelID: "mlx-community/Qwen2.5-7B-Instruct-4bit")
        case .mock:   return MockModel()
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
            Label(availabilityText,
                  systemImage: provider.isAvailable ? "checkmark.circle" : "xmark.circle")
                .font(.caption)
                .foregroundStyle(provider.isAvailable ? .green : .red)
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
                        Button("Copy all") { UIPasteboard.general.string = plainText }
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
        var out = "ANSWER:\n\(answer)\n\nTRACE:\n"
        out += trace.map(\.text).joined(separator: "\n")
        return out
    }

    @MainActor
    private func run(_ q: String) async {
        running = true; trace = []; answer = ""
        let agent = Agent(model: provider)
        let result = await agent.answer(q) { event in
            Task { @MainActor in trace.append(TraceLine(event)) }
        }
        answer = result
        running = false
    }
}

/// A renderable trace line derived from a TraceEvent.
struct TraceLine: Identifiable {
    let id = UUID()
    let text: String
    let color: Color

    init(_ event: TraceEvent) {
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
        }
    }
}
