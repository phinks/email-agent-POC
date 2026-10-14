import Foundation

/// Lightweight on-device performance sampling for the POC. All read-only;
/// surfaced in the trace so you can judge whether a given model is viable on
/// the phone (load time, resident memory, throughput).
enum Perf {

    /// Resident memory of this process, in MB. Uses mach task_info.
    static func residentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        return Double(info.resident_size) / (1024 * 1024)
    }

    /// Memory iOS will still grant this app before jetsam, in MB (iOS 13+).
    /// A small/negative headroom while a model is resident is the warning sign
    /// that a larger model (e.g. 7B) will be killed.
    static func availableMemoryMB() -> Double {
        Double(os_proc_available_memory()) / (1024 * 1024)
    }

    /// Monotonic seconds, for timing load/generation.
    static func now() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }
}

/// A snapshot of one run's performance, formatted for the trace/UI.
struct PerfSample {
    var modelLoadMs: Double = 0        // time to load weights into memory
    var totalMs: Double = 0            // wall-clock for the whole answer
    var promptTokens: Int = 0
    var outputTokens: Int = 0
    var decodeTokPerSec: Double = 0    // output tokens / generation time
    var residentMB: Double = 0         // peak-ish resident while answering
    var availableMB: Double = 0        // headroom before jetsam at that point

    var summary: String {
        String(
            format: "load %.0fms · total %.0fms · %d→%d tok · %.1f tok/s · "
                  + "res %.0fMB · headroom %.0fMB",
            modelLoadMs, totalMs, promptTokens, outputTokens,
            decodeTokPerSec, residentMB, availableMB
        )
    }
}
