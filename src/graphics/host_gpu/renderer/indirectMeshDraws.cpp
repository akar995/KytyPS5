#include "graphics/host_gpu/renderer/indirectMeshDraws.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/indirect_mesh_prepass_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

static uint64_t RecordStride(const GraphicContext& graphics) {
	// Records hold eleven dwords and are bound one at a time as storage buffers.
	return std::max<uint64_t>(
	    64, graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment);
}

IndirectMeshDraws::IndirectMeshDraws(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler), m_record_stride(RecordStride(graphics)),
      m_records(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress,
                m_record_stride * RecordCount),
      m_record_ticks(RecordCount, 0) {
	SetVulkanObjectNameF(m_graphics.device, m_records.Handle(), "Indirect Mesh Draw Records");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_set_layout),
	    "create indirect mesh draw descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Params)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create indirect mesh draw pipeline layout");

	const auto module = CompileSPV(INDIRECT_MESH_PREPASS_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create indirect mesh draw pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Indirect Mesh Draw Prepass");
}

IndirectMeshDraws::~IndirectMeshDraws() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

bool IndirectMeshDraws::CanPrepare() const {
	const auto tick = m_record_ticks[m_next_record];
	return tick == 0 || tick < m_scheduler.CurrentTick();
}

IndirectMeshDraws::Prepared IndirectMeshDraws::Prepare(vk::CommandBuffer command, vk::Buffer args,
                                                       uint64_t args_offset, uint64_t args_size,
                                                       Params params) {
	EXIT_IF(command == nullptr || args == nullptr || (args_offset & 3u) != 0 || args_size == 0);

	// A record is reused only once the GPU has finished the draw that last read it.
	EXIT_IF(!CanPrepare());
	const auto slot = m_next_record;
	m_next_record   = (m_next_record + 1) % RecordCount;
	auto& tick      = m_record_ticks[slot];
	if (tick != 0 && !m_scheduler.IsFree(tick)) {
		m_scheduler.Wait(tick);
	}
	tick = m_scheduler.CurrentTick();

	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto bound_offset = args_offset / alignment * alignment;
	params.args_offset      = static_cast<uint32_t>((args_offset - bound_offset) / 4u);
	const auto record       = static_cast<uint64_t>(slot) * m_record_stride;

	const vk::DescriptorBufferInfo infos[] {
	    {args, bound_offset, args_offset - bound_offset + args_size},
	    {m_records.Handle(), record, m_record_stride},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                        0, nullptr);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
	                      &params);
	command.dispatch(1, 1, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eMeshShaderEXT,
	                        {}, 1, &after, 0, nullptr, 0, nullptr);

	return {m_records.Handle(), record + CommandOffset, m_records.BufferDeviceAddress() + record};
}

} // namespace Libs::Graphics
