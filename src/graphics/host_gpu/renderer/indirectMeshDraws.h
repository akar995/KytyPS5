#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTMESHDRAWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTMESHDRAWS_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;

// Mesh draws whose indirect record the GPU wrote. A one-thread compute pass derives the task
// launch and the mesh shader's draw parameters on the GPU, so the CPU never reads the record
// and never waits for the work that produced it.
class IndirectMeshDraws {
public:
	// Matches the push constants of indirect_mesh_prepass.comp.
	struct Params {
		uint32_t args_offset          = 0;
		uint32_t indexed              = 0;
		uint32_t element_size         = 0;
		uint32_t index_base_lo        = 0;
		uint32_t index_base_hi        = 0;
		uint32_t index_buffer_size    = 0;
		uint32_t primitive_size       = 0;
		uint32_t primitive_step       = 0;
		uint32_t primitives_per_group = 0;
		uint32_t max_groups_x         = 0;
		uint32_t max_groups_y         = 0;
		uint32_t max_groups_total     = 0;
	};

	struct Prepared {
		vk::Buffer        command_buffer = nullptr;
		uint64_t          command_offset = 0;
		vk::DeviceAddress parameters     = 0;
	};

	IndirectMeshDraws(GraphicContext& graphics, CommandScheduler& scheduler);
	~IndirectMeshDraws();
	KYTY_CLASS_NO_COPY(IndirectMeshDraws);

	// Whether Prepare has a record to use: the next one is not still in the open submission.
	[[nodiscard]] bool CanPrepare() const;
	// Records the derivation pass; it must be called outside a render pass. args_offset is
	// the record's dword-aligned offset within args.
	[[nodiscard]] Prepared Prepare(vk::CommandBuffer command, vk::Buffer args,
	                               uint64_t args_offset, uint64_t args_size, Params params);

private:
	static constexpr uint32_t RecordCount   = 1024;
	static constexpr uint32_t CommandOffset = 32;

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	uint64_t                m_record_stride = 0;
	Buffer                  m_records;
	std::vector<uint64_t>   m_record_ticks;
	uint32_t                m_next_record   = 0;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTMESHDRAWS_H_
