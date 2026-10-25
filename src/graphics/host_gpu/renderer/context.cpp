#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h" // PERF-INSTR
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <utility>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	++m_handle_uses;
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || m_pending_shader_writes || IsInvalid());
	m_state_cache = {};
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

// Whether two passes bind the same attachments the same way; they may differ only in clears.
static bool SameAttachments(const RenderState& a, const RenderState& b) {
	if (a.width != b.width || a.height != b.height || a.num_layers != b.num_layers ||
	    a.num_color_attachments != b.num_color_attachments) {
		return false;
	}
	for (uint32_t i = 0; i < a.num_color_attachments; i++) {
		const auto& x = a.color_attachments[i];
		const auto& y = b.color_attachments[i];
		if (x.image_view != y.image_view || x.image_layout != y.image_layout) {
			return false;
		}
	}
	const auto& x = a.depth_stencil_attachment;
	const auto& y = b.depth_stencil_attachment;
	return x.image_view == y.image_view && x.image_layout == y.image_layout &&
	       x.has_depth == y.has_depth && x.has_stencil == y.has_stencil;
}

// The load-op clears a pass beginning with state would perform, inside the current pass.
void CommandBuffer::ClearInsidePass(const RenderState& state) const {
	std::array<vk::ClearAttachment, RENDER_COLOR_ATTACHMENTS_MAX + 1> clears {};
	uint32_t                                                         count = 0;
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		if (attachment.is_clear) {
			clears[count].aspectMask                  = vk::ImageAspectFlagBits::eColor;
			clears[count].colorAttachment             = i;
			clears[count++].clearValue.color.uint32 = attachment.clear_value;
		}
	}
	const auto&          depth_stencil = state.depth_stencil_attachment;
	vk::ImageAspectFlags aspects {};
	if (depth_stencil.has_depth && depth_stencil.depth_clear) {
		aspects |= vk::ImageAspectFlagBits::eDepth;
	}
	if (depth_stencil.has_stencil && depth_stencil.stencil_clear) {
		aspects |= vk::ImageAspectFlagBits::eStencil;
	}
	if (aspects) {
		clears[count].aspectMask = aspects;
		clears[count].clearValue.depthStencil.depth =
		    std::bit_cast<float>(depth_stencil.clear_value[0]);
		clears[count++].clearValue.depthStencil.stencil = depth_stencil.clear_value[1];
	}
	if (count == 0) {
		return;
	}
	const vk::ClearRect rect {{{0, 0}, {state.width, state.height}}, 0, state.num_layers};
	Handle().clearAttachments(count, clears.data(), 1, &rect);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && SameAttachments(m_render_state, state)) {
		// Only the clears differ. Restarting the pass would store and reload every attachment;
		// clearing inside it touches only the cleared ones.
		ClearInsidePass(state);
		m_render_state = state;
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	GpuProfNote(GpuProfBeginPass); // PERF-INSTR
	g_gpu_prof_begun_pass = (static_cast<uint64_t>(state.width) << 16u) | state.height; // PERF-INSTR
	g_gpu_prof_pass_flags = state.num_color_attachments | (depth_stencil.has_depth ? 0x100u : 0u); // PERF-INSTR
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (m_rendering) {
		Handle().endRendering();
		GpuProfNote(GpuProfEndPass); // PERF-INSTR
		g_gpu_prof_ended_pass = (static_cast<uint64_t>(m_render_state.width) << 16u) | m_render_state.height; // PERF-INSTR
		m_rendering    = false;
		m_render_state = {};
	}
	if (m_pending_shader_writes) {
		ShaderWriteBarrier(Handle(), std::exchange(m_pending_shader_writes, {}));
	}
}

} // namespace Libs::Graphics
