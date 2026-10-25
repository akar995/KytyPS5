#include "graphics/host_gpu/renderer/gpuProfiler.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

bool GpuProfiler::Enabled() {
	static const bool enabled = std::getenv("KYTY_GPU_PROF") != nullptr;
	return enabled;
}

GpuProfiler::GpuProfiler(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler), m_events(Capacity) {
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = Capacity;
	RequireVulkanSuccess(graphics.device.createQueryPool(&info, nullptr, &m_pool),
	                     "create GPU profiler query pool");
	graphics.device.resetQueryPool(m_pool, 0, Capacity);
	m_period_ms = graphics.physical_device_properties.limits.timestampPeriod / 1e6;
}

GpuProfiler::~GpuProfiler() {
	m_graphics.device.destroyQueryPool(m_pool, nullptr);
}

void GpuProfiler::Mark(vk::CommandBuffer command, Kind kind, uint64_t key0, uint64_t key1,
                       uint32_t extra) {
	if (m_written - m_read >= Capacity) {
		return;
	}
	const auto index = static_cast<uint32_t>(m_written % Capacity);
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_pool, index);
	if (kind == Kind::Before) {
		key0  = g_gpu_prof_work;
		key1  = (g_gpu_prof_work & (GpuProfEndPass | GpuProfBeginPass)) != 0
		            ? (g_gpu_prof_ended_pass << 32u) | g_gpu_prof_begun_pass
		            : 0;
		extra = (g_gpu_prof_work & GpuProfBeginPass) != 0 ? g_gpu_prof_pass_flags : 0;
	}
	g_gpu_prof_work = 0;
	m_events[index] = {kind, key0, key1, extra, m_scheduler.CurrentTick()};
	m_written++;
}

void GpuProfiler::Collect() {
	std::vector<uint64_t> times;
	while (m_read < m_written) {
		const auto first = static_cast<uint32_t>(m_read % Capacity);
		uint32_t   count = 0;
		while (m_read + count < m_written && first + count < Capacity &&
		       m_scheduler.IsFree(m_events[first + count].tick)) {
			count++;
		}
		if (count == 0) {
			return;
		}
		times.resize(count);
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_pool, first, count, count * sizeof(uint64_t), times.data(), sizeof(uint64_t),
		    vk::QueryResultFlagBits::e64);
		if (result != vk::Result::eSuccess) {
			return;
		}
		for (uint32_t i = 0; i < count; i++) {
			const auto& event = m_events[first + i];
			if (m_have_last && times[i] >= m_last_time) {
				const double ms = static_cast<double>(times[i] - m_last_time) * m_period_ms;
				if (event.tick != m_last_tick) {
					// Includes the GPU waiting for this submission and anything recorded in it
					// ahead of its first mark.
					m_boundary_ms += ms;
					m_boundaries++;
				} else {
					m_total_ms += ms;
					if (event.kind == Kind::Before) {
						m_before_ms += ms;
					}
					auto& total = m_totals[{static_cast<uint32_t>(event.kind), {event.key0, event.key1}}];
					total.ms += ms;
					total.count++;
					total.extra = event.extra;
				}
			}
			m_last_time = times[i];
			m_last_tick = event.tick;
			m_have_last = true;
		}
		m_graphics.device.resetQueryPool(m_pool, first, count);
		m_read += count;
	}
}

void GpuProfiler::Frame() {
	Collect();
	if (++m_frames < 300) {
		return;
	}
	std::vector<std::pair<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>, Total>> rows(
	    m_totals.begin(), m_totals.end());
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
	const auto frames = static_cast<double>(m_frames);
	std::fprintf(stderr,
	             "GPUPROF frames=%" PRIu64
	             " busy=%.2fms/f before_work=%.2fms/f boundary=%.2fms/f (%.1f/f)\n",
	             m_frames, m_total_ms / frames, m_before_ms / frames, m_boundary_ms / frames,
	             static_cast<double>(m_boundaries) / frames);
	static constexpr const char* names[] = {"before", "draw", "mesh", "dispatch"};
	for (size_t i = 0; i < std::min<size_t>(rows.size(), 40); i++) {
		const auto& [key, total] = rows[i];
		std::fprintf(stderr,
		             "GPUPROF %8.3fms/f %7.1f/f %-8s %016" PRIx64 " %016" PRIx64 " extra=%08x\n",
		             total.ms / frames, static_cast<double>(total.count) / frames,
		             names[key.first], key.second.first, key.second.second, total.extra);
	}
	std::fflush(stderr);
	m_totals.clear();
	m_frames      = 0;
	m_total_ms    = 0;
	m_before_ms   = 0;
	m_boundary_ms = 0;
	m_boundaries  = 0;
}

} // namespace Libs::Graphics
