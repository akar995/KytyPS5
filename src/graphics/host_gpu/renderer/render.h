#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Nonzero: the host GPU reads the guest's indirect record here itself. index_addr and
	// index_count then describe the whole index buffer, and instance_count is a placeholder.
	uint64_t         indirect_args_addr         = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Nonzero: the host GPU reads the guest's indirect record here; the counts are placeholders.
	uint64_t         indirect_args_addr         = 0;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

// The graphics state last recorded into a command buffer. Every draw pipeline declares the same
// dynamic states, so dynamic state and vertex/index bindings persist across draws, pipeline binds
// and render passes; a draw records only what differs. Values are compared bit for bit.
struct GraphicsStateCache {
	static constexpr uint32_t MaxViewports     = 16;
	static constexpr uint32_t MaxVertexBuffers = 32;

	struct Viewports {
		uint32_t                               count = 0;
		std::array<vk::Viewport, MaxViewports> viewports {};
		std::array<vk::Rect2D, MaxViewports>   scissors {};
	};
	struct Depth {
		vk::Bool32    test_enable  = VK_FALSE;
		vk::Bool32    write_enable = VK_FALSE;
		vk::CompareOp compare_op   = vk::CompareOp::eNever;
		vk::Bool32    bounds_test  = VK_FALSE;
		float         bounds_min   = 0.0f;
		float         bounds_max   = 0.0f;
	};
	struct DepthBias {
		vk::Bool32 enable   = VK_FALSE;
		float      constant = 0.0f;
		float      clamp    = 0.0f;
		float      slope    = 0.0f;
	};
	struct Stencil {
		vk::Bool32         enable = VK_FALSE;
		vk::StencilOpState front;
		vk::StencilOpState back;
	};
	struct ColorWrite {
		uint32_t                                                count = 0;
		std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> enable {};
	};
	struct VertexBuffers {
		uint32_t                                   count = 0;
		std::array<vk::Buffer, MaxVertexBuffers>     buffers {};
		std::array<vk::DeviceSize, MaxVertexBuffers> offsets {};
		std::array<vk::DeviceSize, MaxVertexBuffers> sizes {};
	};
	struct IndexBuffer {
		vk::Buffer     buffer = nullptr;
		vk::DeviceSize offset = 0;
		vk::IndexType  type   = vk::IndexType::eUint16;
	};

	// Returns whether `value` differs from the cached one (or the cache is unset), and stores it.
	template <typename T>
	static bool Update(std::optional<T>& cached, const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (cached && std::memcmp(&*cached, &value, sizeof(T)) == 0) {
			return false;
		}
		cached = value;
		return true;
	}

	std::optional<vk::Pipeline>         pipeline;
	std::optional<Viewports>            viewports;
	std::optional<float>                line_width;
	std::optional<std::array<float, 4>> blend_constants;
	std::optional<Depth>                depth;
	std::optional<DepthBias>            depth_bias;
	std::optional<Stencil>              stencil;
	std::optional<ColorWrite>           color_write;
	std::optional<vk::ImageAspectFlags> feedback_loop;
	std::optional<VertexBuffers>        vertex_buffers;
	std::optional<IndexBuffer>          index_buffer;
};

// The registers the next draw of a command buffer will see, found by looking ahead in it.
struct NextDrawRegisters {
	const HW::Shader*     shaders     = nullptr;
	const HW::Context*    context     = nullptr;
	const HW::UserConfig* user_config = nullptr;
};

// Register-only queries of a draw's state; they run on the lookahead workers too.
[[nodiscard]] uint32_t DrawColorOutputMask(const HW::Context& ctx);
[[nodiscard]] bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs);
[[nodiscard]] std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
TargetExportMapping(const HW::Context& ctx, uint32_t color_output_mask);

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void ClearInsidePass(const RenderState& state) const;
	// Ends the rendering instance, then records the barrier of any deferred shader writes.
	void EndRendering() const;
	// A draw whose shaders wrote buffers leaves the barrier that makes the writes visible pending
	// while rendering continues. Every command other than a draw ends rendering before it records,
	// which records the barrier; a later draw that may read the writes ends rendering first.
	void DeferShaderWriteBarrier(vk::PipelineStageFlags source_stages) const {
		m_pending_shader_writes |= source_stages;
	}
	[[nodiscard]] bool HasPendingShaderWrites() const noexcept {
		return static_cast<bool>(m_pending_shader_writes);
	}

	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Every command is recorded through a Handle() call, so an unchanged count since the last
	// global barrier means nothing was recorded after it and another would order nothing new.
	[[nodiscard]] bool FollowsGlobalBarrier() const noexcept {
		return m_barrier_handle_uses == m_handle_uses;
	}
	void MarkGlobalBarrier() const noexcept { m_barrier_handle_uses = m_handle_uses; }
	[[nodiscard]] GraphicsStateCache& StateCache() const noexcept { return m_state_cache; }
	// For recording paths that bind their own graphics pipeline or set state outside the cache.
	void InvalidateGraphicsState() const noexcept { m_state_cache = {}; }
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable GraphicsStateCache m_state_cache;
	mutable bool        m_rendering   = false;
	mutable uint64_t    m_handle_uses = 0;
	mutable uint64_t    m_barrier_handle_uses = UINT64_MAX;
	mutable vk::PipelineStageFlags m_pending_shader_writes;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// The next draw's registers, when only register packets come before it. The draw about to
	// execute then has the resource helper prepare that draw meanwhile.
	void SetNextDraw(const NextDrawRegisters& next) { m_next_draw = next; }
	// The lookahead record of the draw about to execute, if one was posted for it.
	void SetDrawRecord(PipelineCache::DrawRecord* record) { m_draw_record = record; }

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// Whether an indirect draw record is current only in GPU memory, so reading it on the CPU
	// would wait for the GPU, and the host GPU can consume it directly instead.
	[[nodiscard]] bool CanDrawIndirectOnGpu(CommandBuffer& buffer, uint64_t args_addr,
	                                        uint64_t args_size);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	// Keeps a binding ResolveTexture made for the same resource and descriptor while the image
	// set is unchanged, doing only the per-use work of a resolution; false if it must resolve.
	[[nodiscard]] bool KeepTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                               const ShaderRecompiler::IR::DescriptorValue& value,
	                               TextureBinding&                              binding);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	void                     UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
	                                                 const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool ReadsPendingWrites(std::span<PreparedBindings* const> stages,
	                                      const ShaderVertexInputInfo&       vertex_input,
	                                      const DrawIndexBufferSource&       index_source,
	                                      const DrawCallInfo&                draw) const;
	void RecordPendingWrites(std::span<PreparedBindings* const> stages);

	// Buffer ranges that draws wrote while their barrier is still pending (see
	// CommandBuffer::DeferShaderWriteBarrier), and whether only atomics wrote each one.
	struct PendingWrite {
		uint64_t begin       = 0;
		uint64_t end         = 0;
		bool     atomic_only = false;
	};

	RenderContext&                        m_context;
	NextDrawRegisters                     m_next_draw;
	PipelineCache::DrawRecord*            m_draw_record = nullptr;
	// Counts the draws started, to tell which one a speculation was posted for.
	uint64_t                              m_draw_sequence = 0;
	std::vector<PendingWrite>             m_pending_writes;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	struct WrittenRange {
		GuestRange range;
		bool       image = false;
	};
	std::vector<WrittenRange>             m_written_ranges;
	std::vector<std::pair<uint64_t, uint64_t>> m_read_runs;
	// Texture descriptions computed from descriptor words and static resource attributes.
	std::unordered_map<TextureDescKey, ResolvedTextureDesc, TextureDescKeyHash> m_texture_descs;
	// The last color target resolved per slot. It holds while the slot's registers and the call's
	// arguments repeat and no image overlapping the target is registered or unregistered.
	struct ColorTargetMemo {
		std::array<uint32_t, 64>         registers {};
		uint32_t                         slot_mask          = 0;
		uint32_t                         slice_offset       = 0;
		bool                             ignore_target_mask = false;
		bool                             exact_format       = false;
		uint64_t                         epoch              = 0;
		bool                             valid              = false;
		RenderColorInfo                  result;
	};
	std::array<ColorTargetMemo, RENDER_COLOR_ATTACHMENTS_MAX> m_color_target_memos;
	struct DepthTargetMemo {
		std::array<uint32_t, 64> registers {};
		uint64_t                 epoch = 0;
		bool                     valid = false;
		TextureCache::ImageDesc  desc;
		ImageId                  image_id;
	};
	DepthTargetMemo m_depth_target_memo;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
