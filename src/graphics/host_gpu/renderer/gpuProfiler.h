#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_

// TEMPORARY instrumentation (KYTY_GPU_PROF=1).

#include "common/abi.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// What was recorded since the last profiler mark; the next Before mark is keyed by it.
enum GpuProfWork : uint32_t {
	GpuProfTransit   = 1u << 0u,
	GpuProfBeginPass = 1u << 1u,
	GpuProfEndPass   = 1u << 2u,
	GpuProfBarrier   = 1u << 3u,
	GpuProfUpload    = 1u << 4u,
	GpuProfCopy      = 1u << 5u,
	GpuProfClear     = 1u << 6u,
	GpuProfFill      = 1u << 7u,
	GpuProfPrepass   = 1u << 8u,
	GpuProfImageOp   = 1u << 9u,
};
inline uint32_t g_gpu_prof_work = 0;
inline void     GpuProfNote(uint32_t work) {
    g_gpu_prof_work |= work;
}
// Shape of the last pass ended and begun: width << 16 | height.
inline uint64_t g_gpu_prof_ended_pass = 0;
inline uint64_t g_gpu_prof_begun_pass = 0;
inline uint32_t g_gpu_prof_pass_flags = 0;

class GpuProfiler {
public:
	enum class Kind : uint32_t { Before, Draw, MeshDraw, Dispatch };

	GpuProfiler(GraphicContext& graphics, CommandScheduler& scheduler);
	~GpuProfiler();
	KYTY_CLASS_NO_COPY(GpuProfiler);

	[[nodiscard]] static bool Enabled();

	void Mark(vk::CommandBuffer command, Kind kind, uint64_t key0 = 0, uint64_t key1 = 0,
	          uint32_t extra = 0);
	void Frame();

private:
	struct Event {
		Kind     kind  = Kind::Before;
		uint64_t key0  = 0;
		uint64_t key1  = 0;
		uint32_t extra = 0;
		uint64_t tick  = 0;
	};
	struct Total {
		double   ms    = 0;
		uint64_t count = 0;
		uint32_t extra = 0;
	};
	struct KeyHash {
		size_t operator()(const std::pair<uint32_t, std::pair<uint64_t, uint64_t>>& key) const {
			return static_cast<size_t>(key.first * 0x9e3779b97f4a7c15ull ^ key.second.first ^
			                           (key.second.second << 1u));
		}
	};

	static constexpr uint32_t Capacity = 1u << 16u;

	void Collect();

	GraphicContext&    m_graphics;
	CommandScheduler&  m_scheduler;
	vk::QueryPool      m_pool      = nullptr;
	double             m_period_ms = 0;
	std::vector<Event> m_events;
	uint64_t           m_written   = 0;
	uint64_t           m_read      = 0;
	uint64_t           m_last_time = 0;
	uint64_t           m_last_tick = 0;
	bool               m_have_last = false;
	uint64_t           m_frames    = 0;
	double             m_total_ms  = 0;
	double             m_before_ms = 0;
	double             m_boundary_ms = 0;
	uint64_t           m_boundaries  = 0;
	std::unordered_map<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>, Total, KeyHash> m_totals;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_
