#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 116);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	// Shared with the compile worker that sets it; a copy takes the current value.
	struct ReadyFlag: std::atomic_bool {
		ReadyFlag(): std::atomic_bool(true) {}
		ReadyFlag(const ReadyFlag& other): std::atomic_bool(other.load()) {}
		ReadyFlag& operator=(const ReadyFlag& other) {
			store(other.load());
			return *this;
		}
	};

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// False while a background worker is still compiling `pipeline`.
		ReadyFlag               ready;
	};

	// The registers a lookahead record is prepared from. The context is a snapshot shared with
	// the neighbouring records while no context packet changed it.
	struct DrawRecordRegisters {
		HW::Shader         shaders;
		HW::UserConfig     user_config;
		const HW::Context* context      = nullptr;
		uint32_t           context_slot = UINT32_MAX;
		uint32_t           slot         = 0;
	};
	// What a record is for: the draw or dispatch packet, and the dispatch's launch.
	struct DrawRecordInfo {
		const void*             buffer       = nullptr;
		uint32_t                offset       = 0;
		uint64_t                memory_epoch = 0;
		bool                    compute      = false;
		bool                    indirect     = false;
		uint32_t                mode         = 0;
		std::array<uint32_t, 3> groups {};
	};
	struct DrawRecord;

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	// memory_epoch: RenderContext::MemoryEpoch, to match against a speculation's.
	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, uint64_t memory_epoch = 0,
	                    DrawRecord* record = nullptr);
	// Starts materializing, on the helper thread, the resources of the programs draw `target`
	// will use with these registers, while the GPU thread records the current draw. That draw's
	// GetGraphicsPrograms uses the result when it selects the same programs with the same user
	// data and memory_epoch has not changed; otherwise it materializes as usual.
	void SpeculateGraphicsPrograms(
	    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, uint64_t memory_epoch, uint64_t target);
	// Called as draw `draw` starts. A speculation meant for another draw is dropped: its draw
	// ended before using it, and the packets since then may have changed what it read.
	void DropSpeculationUnlessFor(uint64_t draw);

	// Draw lookahead. The command processor records the registers of the draws ahead of the one
	// executing, worker threads prepare each recorded draw's programs and resources, and
	// GetGraphicsPrograms adopts that preparation when the draw arrives with those registers.
	[[nodiscard]] bool DrawLookaheadEnabled() const { return m_lookahead != nullptr; }
	// A free record's registers to fill, or nullptr while every record is in use.
	[[nodiscard]] DrawRecordRegisters* AcquireDrawRecord();
	// Hands a filled record to the workers. context is snapshotted when context_changed, or
	// when no snapshot is current; otherwise the record shares the current snapshot.
	void PostDrawRecord(DrawRecordRegisters* registers, const DrawRecordInfo& info,
	                    const HW::Context& context, bool context_changed);
	// The record posted for the draw packet at buffer/offset, or nullptr. Records of earlier
	// packets of that buffer are dropped: their draws did not execute.
	[[nodiscard]] DrawRecord* TakeDrawRecord(const void* buffer, uint32_t offset);
	// Returns a taken record to the ring after its draw.
	void ReleaseDrawRecord(DrawRecord* record);
	void DropDrawRecords();
	// Makes the records posted since the last call visible to the workers.
	void PublishDrawRecords();
	// input_info carries the launch (dispatch_thread_dimensions, workgroup_counts) in. A record
	// prepared from these registers and launch is adopted when memory_epoch matches its own.
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                uint64_t memory_epoch = 0, DrawRecord* record = nullptr);

	// With `allow_async`, a pipeline that is not built yet is compiled on a worker thread and
	// nullptr is returned until it is ready, so the caller can skip the draw instead of stalling.
	// Without it the result is never null.
	Pipeline* GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs, bool allow_async = false);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

private:
	struct ProgramCache;
	class CompileWorkers;
	class MaterializeHelper;
	class LookaheadWorkers;
	struct GraphicsInputs;
	struct Speculation;

	// The program parameters and stage inputs of a draw with these registers.
	GraphicsInputs PrepareGraphicsInputs(
	    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	    ShaderPixelInputInfo& pixel_info);
	// Waits for a posted speculation, whose job uses program cache entries, and returns whether
	// it completed without needing the GPU thread.
	bool FinishSpeculation();
	// This draw's programs from the speculation posted for it, when it saw the same registers.
	std::optional<GraphicsPrograms> AdoptSpeculation(
	    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	    const HW::ShaderRegisters& sh,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	    ShaderPixelInputInfo& pixel_info, uint64_t memory_epoch);

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	std::unique_ptr<CompileWorkers>                         m_workers;
	std::unique_ptr<MaterializeHelper>                      m_materialize_helper;
	std::unique_ptr<Speculation>                            m_speculation;
	std::unique_ptr<LookaheadWorkers>                       m_lookahead;

	// Prepares a record's programs and resources; runs on a lookahead worker.
	void PrepareDrawRecord(DrawRecord& record);
	void PrepareComputeRecord(DrawRecord& record);
	// This draw's programs from its record, when the record was prepared from these registers.
	std::optional<GraphicsPrograms> AdoptDrawRecord(
	    DrawRecord& record, const HW::VertexShaderInfo& vertex_regs,
	    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	    ShaderPixelInputInfo& pixel_info, uint64_t memory_epoch);

	void InitializeDriverCache();
	void CheckpointDriverCache();
};

// Owns every create-info structure of one graphics pipeline so the driver compile can run on
// another thread after the renderer state it was built from has changed.
struct GraphicsPipelineBuild;
struct GraphicsPipelineBuildDeleter {
	void operator()(GraphicsPipelineBuild* build) const;
};
using GraphicsPipelineBuildPtr = std::unique_ptr<GraphicsPipelineBuild, GraphicsPipelineBuildDeleter>;

// Creates the layouts on `pipeline` and captures the rest; cheap, runs on the renderer thread.
GraphicsPipelineBuildPtr PrepareGraphicsPipeline(GraphicContext& graphics,
                                                 PipelineCache::Pipeline& pipeline,
                                                 const PipelineRenderingState&          rendering,
                                                 const PipelineVertexInputState&        vertex_input,
                                                 std::span<const ShaderVertexInputInfo> vertex_info,
                                                 const ShaderPixelInputInfo*            ps_input_info,
                                                 const PipelineCache::GraphicsPrograms& programs,
                                                 const PipelineStaticParameters& static_params);
// The driver compile; safe to call from any thread.
vk::Pipeline FinishGraphicsPipeline(GraphicsPipelineBuild& build, vk::PipelineCache driver_cache);

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
