#ifndef KYTY_COMMON_PERFCOUNTERS_H_
#define KYTY_COMMON_PERFCOUNTERS_H_

// TEMPORARY performance instrumentation (KYTY_PERF=1 prints per-frame rates to stderr).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace Common::Perf {

// The compute shader whose resources the GPU thread is materializing (readback attribution).
inline thread_local uint64_t t_cs_hash = 0;

enum Counter : uint32_t {
	Presents,
	Draws,
	Dispatches,
	Submits,
	GpuWaits,
	GpuWaitNs,
	SyncDownloads,
	SyncDownloadBytes,
	SyncDownloadNs,
	ImageReadbacks,
	ImageReadbackNs,
	CpuWriteFaults,
	PrepareDrawNs,
	ExecuteDrawNs,
	DispatchNs,
	FlipWaitNs,
	GpuThreadIdleNs,
	ZpassDumps,
	Predications,
	PredicatedDraws,
	SkippedDraws,
	LookaheadPosted, // PERF-INSTR
	LookaheadAdopted, // PERF-INSTR
	LookaheadMissing, // PERF-INSTR
	LookaheadUnprepared, // PERF-INSTR
	LookaheadStale, // PERF-INSTR
	LookaheadDropped, // PERF-INSTR
	Phase0Ns, // PERF-INSTR
	Phase1Ns, // PERF-INSTR
	Phase2Ns, // PERF-INSTR
	Phase3Ns, // PERF-INSTR
	Phase4Ns, // PERF-INSTR
	Phase5Ns, // PERF-INSTR
	Phase6Ns, // PERF-INSTR
	Phase7Ns, // PERF-INSTR
	Phase8Ns, // PERF-INSTR
	Phase9Ns, // PERF-INSTR
	Count0, // PERF-INSTR
	Count1, // PERF-INSTR
	Count2, // PERF-INSTR
	Count3, // PERF-INSTR
	Count
};

inline std::atomic<uint64_t> g_counters[Count];
inline const bool            g_enabled = std::getenv("KYTY_PERF") != nullptr;

inline void Add(Counter c, uint64_t value = 1) {
	if (g_enabled) {
		g_counters[c].fetch_add(value, std::memory_order_relaxed);
	}
}

inline uint64_t NowNs() {
	if (!g_enabled) {
		return 0;
	}
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct ScopedTimer {
	Counter  counter;
	uint64_t start = NowNs();
	explicit ScopedTimer(Counter c): counter(c) {}
	~ScopedTimer() {
		if (g_enabled) {
			Add(counter, NowNs() - start);
		}
	}
};

void StartReporter();

// rdtsc cycles and calls per Vulkan command group (GPU thread; unsynchronized).
enum VkGroup : uint32_t {
	VkDynamic,
	VkPushDesc,
	VkPushConst,
	VkBindPipe,
	VkDraw,
	VkBeginRender,
	VkBeginInner,
	VkBarrier,
	VkEndForBarrier,
	VkGlobalBarrier,
	VkBindVb,
	VkDispatch,
	VkCommitAll,
	VkGroupCount
};
inline uint64_t g_vk_cycles[VkGroupCount];
inline uint64_t g_vk_calls[VkGroupCount];
uint64_t        ReadTsc();
inline void     VkAdd(VkGroup group, uint64_t start) {
    g_vk_cycles[group] += ReadTsc() - start;
    g_vk_calls[group]++;
}

} // namespace Common::Perf

#endif
