#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static bool logged_113 = false;
		if (!logged_113) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
			logged_113 = true;
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static bool logged_121 = false;
		if (!logged_121) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
			logged_121 = true;
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u\n", raw_format);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

static void CreateDescriptorLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                   std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint64_t storage_buffers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		if (storage) storage_buffers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (binding.descriptorType != vk::DescriptorType::eSampler)
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	pipeline.uses_push_descriptors = descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &create, nullptr, &pipeline.descriptor_set_layout) != vk::Result::eSuccess);
}

// Every create-info structure a graphics pipeline needs, owned in one place so that the driver
// compile can run on a worker thread after the renderer state it was built from has changed.
// The object is never moved after PrepareGraphicsPipeline() links its internal pointers.
struct GraphicsPipelineBuild {
	vk::Device                                    device;
	vk::ShaderModule                              tess_control_shader_module = nullptr;
	vk::ShaderModule                              tess_eval_shader_module    = nullptr;
	vk::PipelineShaderStageCreateInfo             shader_stages[4] {};
	vk::VertexInputAttributeDescription           input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription             input_desc[ShaderVertexInputInfo::RES_MAX] {};
	vk::PipelineVertexInputStateCreateInfo        vertex_input_info {};
	vk::PipelineInputAssemblyStateCreateInfo      input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	vk::PipelineViewportStateCreateInfo           viewport_state {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	vk::PipelineRasterizationStateCreateInfo      rasterizer {};
	vk::PipelineMultisampleStateCreateInfo        multisampling {};
	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] {};
	vk::Bool32                                    color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] {};
	vk::PipelineColorWriteCreateInfoEXT           color_write {};
	vk::PipelineColorBlendStateCreateInfo         color_blending {};
	vk::PipelineDepthStencilStateCreateInfo       depth_stencil_info {};
	std::vector<vk::DynamicState>                 dynamic_states;
	vk::PipelineDynamicStateCreateInfo            dynamic_state {};
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::PipelineRenderingCreateInfo               rendering_info {};
	vk::PipelineTessellationStateCreateInfo       tessellation_state {};
	vk::GraphicsPipelineCreateInfo                pipeline_info {};
	uint64_t                                      vertex_program_id = 0;
	uint64_t                                      pixel_program_id  = 0;

	GraphicsPipelineBuild() = default;
	GraphicsPipelineBuild(const GraphicsPipelineBuild&)            = delete;
	GraphicsPipelineBuild& operator=(const GraphicsPipelineBuild&) = delete;

	~GraphicsPipelineBuild() {
		if (tess_control_shader_module != nullptr) {
			device.destroyShaderModule(tess_control_shader_module, nullptr);
		}
		if (tess_eval_shader_module != nullptr) {
			device.destroyShaderModule(tess_eval_shader_module, nullptr);
		}
	}
};

void GraphicsPipelineBuildDeleter::operator()(GraphicsPipelineBuild* build) const {
	delete build;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
GraphicsPipelineBuildPtr PrepareGraphicsPipeline(GraphicContext& graphics,
                                                 PipelineCache::Pipeline& pipeline,
                                                 const PipelineRenderingState&          rendering,
                                                 const PipelineVertexInputState&        vertex_input,
                                                 std::span<const ShaderVertexInputInfo> vertex_info,
                                                 const ShaderPixelInputInfo*            ps_input_info,
                                                 const PipelineCache::GraphicsPrograms& programs,
                                                 const PipelineStaticParameters& static_params) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	GraphicsPipelineBuildPtr build_ptr(new GraphicsPipelineBuild);
	auto&                    b = *build_ptr;
	b.device                   = graphics.device;
	b.vertex_program_id        = vertex_program.id;
	b.pixel_program_id         = ps_active ? pixel_program.id : 0;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		b.tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(b.tess_control_shader_module));
		}

		b.tess_eval_shader_module = CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(b.tess_eval_shader_module));
		}
	}

	EXIT_NOT_IMPLEMENTED(rect_list && (b.tess_control_shader_module == nullptr ||
	                                   b.tess_eval_shader_module == nullptr));

	uint32_t shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		b.shader_stages[shader_stage_count++] = {.stage =
		                                             NativeShaderStage(vertex_info[i].logical_stage),
		                                         .module = programs.vertex[i].module,
		                                         .pName  = "main"};
	}
	if (rect_list) {
		b.shader_stages[shader_stage_count++] = {.stage =
		                                             vk::ShaderStageFlagBits::eTessellationControl,
		                                         .module = b.tess_control_shader_module,
		                                         .pName  = "main"};
		b.shader_stages[shader_stage_count++] = {.stage =
		                                             vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                         .module = b.tess_eval_shader_module,
		                                         .pName  = "main"};
	}
	if (ps_active) {
		b.shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                         .module = pixel_program.module,
		                                         .pName  = "main"};
	}

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		b.input_desc[binding].binding   = binding;
		b.input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		b.input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                      ? vk::VertexInputRate::eInstance
		                                      : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		b.input_attr[index].binding  = vertex_input.attributes[index].binding;
		b.input_attr[index].location = index;
		b.input_attr[index].offset   = vertex_input.attributes[index].offset;

		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], b.input_attr[index].format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, b.input_attr[index].binding, b.input_attr[index].offset,
				     b.input_desc[b.input_attr[index].binding].stride,
				     static_cast<int>(b.input_attr[index].format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static bool logged = false;
			if (!logged) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
				logged = true;
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	b.vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	b.vertex_input_info.pVertexBindingDescriptions      = b.input_desc;
	b.vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	b.vertex_input_info.pVertexAttributeDescriptions    = b.input_attr;

	b.input_assembly.topology = static_params.topology;
	b.input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	b.depth_clip_control.negativeOneToOne =
	    (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);
	b.viewport_state.pNext = &b.depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	b.clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	b.rasterizer.depthClampEnable = VK_TRUE;
	b.rasterizer.pNext = &b.clip_ext;
#endif
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		b.provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		b.provoking_vertex.pNext = b.rasterizer.pNext;
		b.rasterizer.pNext = &b.provoking_vertex;
	}
	b.rasterizer.cullMode    = cull_mode;
	b.rasterizer.frontFace   = front_face;
	b.rasterizer.polygonMode = static_params.polygon_mode;
	b.rasterizer.lineWidth   = 1.0f;

	b.multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	b.multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	b.multisampling.minSampleShading     = 1.0f;

	for (uint32_t i = 0; i < rendering.color_count; i++) {
		auto& attachment = b.color_blend_attachment[i];
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		attachment.colorWriteMask = vk::ColorComponentFlags {static_params.color_mask[i]};
		attachment.blendEnable    = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		attachment.srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		attachment.dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		attachment.colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		attachment.srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : attachment.srcColorBlendFactor);
		attachment.dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : attachment.dstColorBlendFactor);
		attachment.alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : attachment.colorBlendOp);
	}

	for (uint32_t i = 0; i < rendering.color_count; i++) {
		b.color_write_enable[i] = VK_TRUE;
	}

	b.color_write.attachmentCount    = rendering.color_count;
	b.color_write.pColorWriteEnables = b.color_write_enable;

	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	b.color_blending.pNext = &b.color_write;
#endif
	b.color_blending.logicOp         = vk::LogicOp::eCopy;
	b.color_blending.attachmentCount = rendering.color_count;
	b.color_blending.pAttachments    = b.color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		graphics_stages |= native_stage;
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
	}
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {graphics_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &pipeline.descriptor_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout begin VS=%" PRIu64 " PS=%" PRIu64
		     " set_layouts=1 push_constants=%" PRIu32 "\n",
		     b.vertex_program_id, b.pixel_program_id, 1u);
	}
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                   &pipeline.pipeline_layout);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreatePipelineLayout done result=%s layout=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	b.dynamic_states = {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	b.dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	b.dynamic_states.push_back(vk::DynamicState::eDepthBounds);
	if (rendering.color_count != 0) {
		b.dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		b.dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	b.dynamic_state.dynamicStateCount = static_cast<uint32_t>(b.dynamic_states.size());
	b.dynamic_state.pDynamicStates    = b.dynamic_states.data();

	b.color_formats                          = rendering.color_formats;
	b.rendering_info.colorAttachmentCount    = rendering.color_count;
	b.rendering_info.pColorAttachmentFormats = b.color_formats.data();
	b.rendering_info.depthAttachmentFormat   = rendering.depth_format;
	b.rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	b.tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;

	auto& pipeline_info               = b.pipeline_info;
	pipeline_info.pNext               = &b.rendering_info;
	pipeline_info.stageCount          = shader_stage_count;
	pipeline_info.pStages             = b.shader_stages;
	pipeline_info.pVertexInputState   = mesh ? nullptr : &b.vertex_input_info;
	pipeline_info.pInputAssemblyState = mesh ? nullptr : &b.input_assembly;
	pipeline_info.pTessellationState  = (rect_list || tessellation) ? &b.tessellation_state : nullptr;
	pipeline_info.pViewportState      = &b.viewport_state;
	pipeline_info.pRasterizationState = &b.rasterizer;
	pipeline_info.pMultisampleState   = &b.multisampling;
	pipeline_info.pDepthStencilState  = (with_depth ? &b.depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState    = &b.color_blending;
	pipeline_info.pDynamicState       = &b.dynamic_state;
	pipeline_info.layout              = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex   = -1;

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines prepared VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     b.vertex_program_id, b.pixel_program_id,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     b.dynamic_state.dynamicStateCount);
	}
	return build_ptr;
}

vk::Pipeline FinishGraphicsPipeline(GraphicsPipelineBuild& build, vk::PipelineCache driver_cache) {
	vk::Pipeline pipeline = nullptr;
	const auto   result =
	    build.device.createGraphicsPipelines(driver_cache, 1, &build.pipeline_info, nullptr, &pipeline);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done VS=%" PRIu64 " PS=%" PRIu64
		     " result=%s pipeline=%p\n",
		     build.vertex_program_id, build.pixel_program_id, vk::to_string(result).c_str(),
		     static_cast<void*>(pipeline));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(pipeline == nullptr);
	return pipeline;
}

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache) {
	EXIT_IF(pipeline.pipeline != nullptr);
	const auto build = PrepareGraphicsPipeline(graphics, pipeline, rendering, vertex_input,
	                                           vertex_info, ps_input_info, programs, static_params);
	pipeline.pipeline = FinishGraphicsPipeline(*build, driver_cache);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	CreateDescriptorLayout(graphics, pipeline, descriptor_bindings);
	const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &pipeline.descriptor_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_IF(pipeline.pipeline_layout != nullptr);

	LOGF("PipelineTrace: vkCreatePipelineLayout CS begin set_layouts=1 push_constants=%u\n",
	     1u);
	auto result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                  &pipeline.pipeline_layout);
	LOGF("PipelineTrace: vkCreatePipelineLayout CS done result=%s layout=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline_layout));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);

	vk::ComputePipelineCreateInfo info {};
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
