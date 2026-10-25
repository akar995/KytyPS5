#include "common/perfCounters.h"

#include <cstdio>
#include <intrin.h>
#include <thread>

namespace Common::Perf {

static const char* const g_names[Count] = {
    "presents",  "draws",       "dispatches", "submits",      "gpu_waits",    "gpu_wait_ms",
    "sync_dl",   "sync_dl_KB",  "sync_dl_ms", "img_readback", "img_rb_ms",    "cpu_wr_faults",
    "prep_ms",   "exec_ms",     "disp_ms",    "flipwait_ms",  "gputhr_idle_ms", "zpass_dumps", "predications", "pred_draws", "skipped_draws",
    "la_posted", "la_adopted", "la_missing", "la_unprep", "la_stale", "la_dropped", // PERF-INSTR
    "ph0_ms", "ph1_ms", "ph2_ms", "ph3_ms", "ph4_ms", "ph5_ms", "ph6_ms", "ph7_ms", "ph8_ms", "ph9_ms", // PERF-INSTR
    "c0", "c1_KB", "c2", "c3", // PERF-INSTR
};

static const char* const g_vk_names[VkGroupCount] = {
    "dyn", "pushdesc", "pushconst", "bindpipe", "draw", "beginrender", "begininner",
    "barrier", "end4barrier", "globalbarrier", "bindvb", "dispatch", "commitall",
};

uint64_t ReadTsc() {
	return __rdtsc();
}

static bool IsTime(uint32_t i) {
	return i == GpuWaitNs || i == SyncDownloadNs || i == ImageReadbackNs || i == PrepareDrawNs ||
	       i == ExecuteDrawNs || i == DispatchNs || i == FlipWaitNs || i == GpuThreadIdleNs || // PERF-INSTR
	       (i >= Phase0Ns && i <= Phase9Ns);
}

void StartReporter() {
	static bool started = false;
	if (started || !g_enabled) {
		return;
	}
	started = true;
	std::thread([] {
		uint64_t last[Count] {};
		uint64_t last_vk_cycles[VkGroupCount] {};
		uint64_t last_vk_calls[VkGroupCount] {};
		auto     last_time = NowNs();
		auto     last_tsc  = __rdtsc();
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(5));
			const auto   now    = NowNs();
			const auto   tsc    = __rdtsc();
			const double secs   = static_cast<double>(now - last_time) / 1e9;
			const double tsc_per_us = static_cast<double>(tsc - last_tsc) / (secs * 1e6);
			last_time           = now;
			last_tsc            = tsc;
			const double frames = static_cast<double>(g_counters[Presents].load() - last[Presents]);
			std::fprintf(stderr, "PERF fps=%.1f", frames / secs);
			for (uint32_t i = 0; i < Count; i++) {
				const auto   value = g_counters[i].load();
				const double delta = static_cast<double>(value - last[i]);
				last[i]            = value;
				if (i == Presents) continue;
				const double per_frame = frames > 0 ? delta / frames : 0.0;
				if (IsTime(i)) {
					std::fprintf(stderr, " %s/f=%.2f", g_names[i], per_frame / 1e6);
				} else if (i == SyncDownloadBytes) {
					std::fprintf(stderr, " %s/f=%.1f", g_names[i], per_frame / 1024.0);
				} else {
					std::fprintf(stderr, " %s/f=%.1f", g_names[i], per_frame);
				}
			}
			std::fprintf(stderr, "\n");
			if (frames > 0) {
				std::fprintf(stderr, "VKTIME");
				for (uint32_t i = 0; i < VkGroupCount; i++) {
					const auto cycles = g_vk_cycles[i];
					const auto calls  = g_vk_calls[i];
					const double us   = static_cast<double>(cycles - last_vk_cycles[i]) / tsc_per_us / frames;
					const double n    = static_cast<double>(calls - last_vk_calls[i]) / frames;
					last_vk_cycles[i] = cycles;
					last_vk_calls[i]  = calls;
					std::fprintf(stderr, " %s=%.0fus/%.0f", g_vk_names[i], us, n);
				}
				std::fprintf(stderr, "\n");
			}
			std::fflush(stderr);
		}
	}).detach();
}

} // namespace Common::Perf
