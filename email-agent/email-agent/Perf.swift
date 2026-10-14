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

    /// Physical memory footprint of this process, in MB — the `phys_footprint`
    /// field from `task_vm_info`. This is the number Xcode's memory gauge and
    /// Instruments report, and it is the one jetsam/App-Store limits key off of.
    /// Unlike `resident_size`, it is STABLE run-to-run and counts dirty +
    /// compressed memory (not just whatever mmap'd pages happen to be faulted in
    /// at sample time), so it is the honest figure for "how much is this using".
    static func physFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    /// Memory still available to this app before it is killed, in MB.
    ///
    /// iOS: the headroom iOS will grant before jetsam (`os_proc_available_memory`,
    /// iOS 13+). A small/negative value while a model is resident is the warning
    /// sign that a larger model (e.g. 7B) will be killed.
    ///
    /// macOS: there is no per-app jetsam limit, so "headroom before jetsam" is
    /// not a meaningful number. We report free physical RAM instead as a rough
    /// analog (the Mac is not memory-constrained the way the phone is).
    static func availableMemoryMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / (1024 * 1024)
        #else
        // macOS: approximate "available" as free + inactive physical pages.
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return -1 }
        let pageSize = Double(vm_kernel_page_size)
        let freeBytes = (Double(stats.free_count) + Double(stats.inactive_count)) * pageSize
        return freeBytes / (1024 * 1024)
        #endif
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
    var footprintMB: Double = 0        // phys_footprint (Xcode-gauge number)
    var residentMB: Double = 0         // raw resident_size (noisy; mmap-dependent)
    var availableMB: Double = 0        // headroom before jetsam at that point

    var summary: String {
        String(
            format: "load %.0fms · total %.0fms · %d→%d tok · %.1f tok/s · "
                  + "footprint %.0fMB · headroom %.0fMB",
            modelLoadMs, totalMs, promptTokens, outputTokens,
            decodeTokPerSec, footprintMB, availableMB
        )
    }
}
