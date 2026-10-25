#include "common/perfCounters.h" // PERF-INSTR
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// The resource walker evaluates every SRT read the shader might perform, including ones behind
// a null pointer that the shader only follows on paths the game never takes. Unmapped memory
// reads as zero (a null descriptor) rather than failing or faulting on the host.
bool ReadUnmappedShaderMemory(uint64_t address, std::span<uint32_t> values) {
	if (Libs::LibKernel::Memory::IsGpuMapped(address, values.size_bytes())) {
		return false;
	}
	std::ranges::fill(values, 0u);
	return true;
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(),
	                                                      values.size_bytes()) ||
	        ReadUnmappedShaderMemory(address, values));
}

bool ReadShaderOrdinaryMemory(void*, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (ReadUnmappedShaderMemory(address, values)) {
		return true;
	}
	// A page that also holds GPU-written data faults into a full GPU readback on a plain load;
	// read those bytes exactly instead.
	const auto bytes = values.size_bytes();
	if (Libs::LibKernel::Memory::GpuReadableEnd(address) < address + bytes &&
	    Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), bytes)) {
		return true;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), bytes);
	return true;
}

// The resource helper's readers. Where the GPU thread is needed they read zeros instead of
// failing or loading a GPU-written page, and the blocked evaluation is discarded.
bool ReadShaderGuestMemoryOffThread(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (ReadShaderGuestMemory(userdata, address, values)) {
		return true;
	}
	if (!Libs::LibKernel::Memory::GpuResourceHelperBlocked()) {
		return false;
	}
	std::ranges::fill(values, 0u);
	return true;
}

bool ReadShaderOrdinaryMemoryOffThread(void*, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (ReadUnmappedShaderMemory(address, values)) {
		return true;
	}
	const auto bytes = values.size_bytes();
	if (Libs::LibKernel::Memory::GpuReadableEnd(address) < address + bytes) {
		if (Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), bytes)) {
			return true;
		}
		if (Libs::LibKernel::Memory::GpuResourceHelperBlocked()) {
			std::ranges::fill(values, 0u);
			return true;
		}
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), bytes);
	return true;
}

// The ordinary reader of a speculative evaluation, which overlaps the GPU thread's recording: a
// page may lose read access at any time, so every load goes through the backing store, which
// never faults. The GPU thread discards the result if any page lost read access meanwhile.
bool ReadShaderOrdinaryMemorySpeculative(void*, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (ReadUnmappedShaderMemory(address, values)) {
		return true;
	}
	const auto bytes = values.size_bytes();
	if (Libs::LibKernel::Memory::GpuReadableEnd(address) < address + bytes ||
	    !Libs::LibKernel::Memory::TryReadBacking(address, values.data(), bytes)) {
		Libs::LibKernel::Memory::BlockGpuResourceHelper();
		std::ranges::fill(values, 0u);
	}
	return true;
}

// The strict reader of a speculative evaluation: the helper's, except that what it cannot read
// blocks the evaluation, where the walker would otherwise fall back to a plain load.
bool ReadShaderGuestMemorySpeculative(void* userdata, uint64_t address,
                                      std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (!ReadShaderGuestMemoryOffThread(userdata, address, values)) {
		Libs::LibKernel::Memory::BlockGpuResourceHelper();
		std::ranges::fill(values, 0u);
	}
	return true;
}

// The walker's inline loads in a speculative evaluation.
void LoadShaderMemorySpeculative(uint64_t address, void* data, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadBacking(address, data, size)) {
		Libs::LibKernel::Memory::BlockGpuResourceHelper();
		std::memset(data, 0, size);
	}
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// Runs driver pipeline compiles off the renderer thread. Vulkan allows concurrent pipeline
// creation against one VkPipelineCache, which the driver synchronizes internally.
class PipelineCache::CompileWorkers {
public:
	CompileWorkers() {
		const auto hardware = std::max(1u, std::thread::hardware_concurrency());
		const auto count    = std::clamp(hardware / 4u, 1u, 6u);
		m_threads.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~CompileWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(CompileWorkers);

	void Submit(std::function<void()> job) {
		{
			std::lock_guard lock(m_mutex);
			m_jobs.push_back(std::move(job));
			m_pending++;
		}
		m_wake.notify_one();
	}

	void WaitIdle() {
		std::unique_lock lock(m_mutex);
		m_idle.wait(lock, [this] { return m_pending == 0; });
	}

private:
	void Run() {
		for (;;) {
			std::function<void()> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
				if (m_jobs.empty()) {
					return;
				}
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			job();
			{
				std::lock_guard lock(m_mutex);
				m_pending--;
			}
			m_idle.notify_all();
		}
	}

	std::mutex                        m_mutex;
	std::condition_variable           m_wake;
	std::condition_variable           m_idle;
	std::deque<std::function<void()>> m_jobs;
	uint32_t                          m_pending = 0;
	bool                              m_stop    = false;
	std::vector<std::thread>          m_threads;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		template <bool HasCalls>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor) {
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    resource_plan, runtime, resources, specialization));
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// The permutation Get returned last; consecutive draws usually need it again.
		size_t                                      last_permutation = 0;
	};

	// MaterializeResources for an entry. Walks of one plan keep per-thread scratch, so threads
	// materialize the same program at once. An entry with a call source is left to Get, which
	// refreshes the source first and may replace the plan: helpers fail on it, and the GPU thread
	// skips it here.
	static bool Materialize(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                        ShaderRecompiler::IR::ResourceSnapshot&       snapshot,
	                        ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		if (entry.call_source) {
			return !Libs::LibKernel::Memory::IsGpuResourceHelper();
		}
		return ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime, snapshot,
		                                                  specialization);
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		if (const char* wanted = std::getenv("KYTY_DUMP_HASHES"); wanted != nullptr) { // PERF-INSTR
			char key[32]; // PERF-INSTR
			std::snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(options.shader_hash)); // PERF-INSTR
			if (std::strstr(wanted, key) != nullptr) { // PERF-INSTR
				spvtools::SpirvTools dis(SPV_ENV_VULKAN_1_3); // PERF-INSTR
				std::string          text; // PERF-INSTR
				if (dis.Disassemble(result.spirv, &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) { // PERF-INSTR
					static std::atomic_int dump_id {0}; // PERF-INSTR
					char path[512]; // PERF-INSTR
					std::snprintf(path, sizeof(path), // PERF-INSTR
					              "C:\\Users\\akari\\AppData\\Local\\Temp\\claude\\d--ps5-emulator-code-KytyPS5\\d8beb6e6-00f1-41ce-ad41-50571e7f41b9\\scratchpad\\dump_%s_%s_%d.spvasm", // PERF-INSTR
					              key, stage_name, dump_id++); // PERF-INSTR
					if (FILE* f = std::fopen(path, "w")) { // PERF-INSTR
						std::fwrite(text.data(), 1, text.size(), f); // PERF-INSTR
						std::fclose(f); // PERF-INSTR
					} // PERF-INSTR
				} // PERF-INSTR
			} // PERF-INSTR
		} // PERF-INSTR
		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	static ShaderType StageOf(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			return ShaderType::Compute;
		}
	}

	template <typename InputInfo>
	static ShaderRecompiler::IR::SrtRuntime Runtime(const ShaderParams& params,
	                                                const InputInfo&    input_info) {
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderOrdinaryMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .find_mapped_range          = Libs::LibKernel::Memory::FindGpuMappedRange,
		    .readable_end               = Libs::LibKernel::Memory::GpuReadableEnd,
		    .read_backing               = Libs::LibKernel::Memory::TryReadBacking,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		return runtime;
	}

	// Runtime for the resource helper thread.
	template <typename InputInfo>
	static ShaderRecompiler::IR::SrtRuntime OffThreadRuntime(const ShaderParams& params,
	                                                         const InputInfo&    input_info) {
		auto runtime                       = Runtime(params, input_info);
		runtime.read_memory                = ReadShaderOrdinaryMemoryOffThread;
		runtime.read_specialization_memory = ReadShaderGuestMemoryOffThread;
		return runtime;
	}

	// Runtime for the helper while the GPU thread keeps recording: every load goes through the
	// backing store, which never faults on a page the GPU thread protects meanwhile.
	template <typename InputInfo>
	static ShaderRecompiler::IR::SrtRuntime SpeculativeRuntime(const ShaderParams& params,
	                                                           const InputInfo&    input_info) {
		auto runtime                       = OffThreadRuntime(params, input_info);
		runtime.read_memory                = ReadShaderOrdinaryMemorySpeculative;
		runtime.read_specialization_memory = ReadShaderGuestMemorySpeculative;
		runtime.load                       = LoadShaderMemorySpeculative;
		return runtime;
	}

	// The cached source program for these parameters, if compiled before.
	template <typename InputInfo>
	SourceEntry* Find(const ShaderParams& params, const InputInfo& input_info) {
		return Find(params, input_info, lookup_key);
	}

	// key: scratch for the lookup. Threads other than the GPU thread bring their own, and the
	// lookup takes the shared lock that insertions take exclusively.
	template <typename InputInfo>
	SourceEntry* Find(const ShaderParams& params, const InputInfo& input_info, ProgramKey& key) {
		key.stage           = StageOf(input_info);
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
		std::shared_lock lock(programs_mutex);
		const auto       entry = programs.find(key);
		return entry == programs.end() ? nullptr : &entry->second;
	}

	// materialized: an entry whose resources were already materialized for these parameters.
	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                  SourceEntry* materialized = nullptr) {
		const ShaderType stage     = StageOf(input_info);
		const auto       user_data = std::span(params.user_data).first(params.user_data_count);
		const auto       runtime   = Runtime(params, input_info);
		SourceEntry*     entry     = materialized != nullptr ? materialized : Find(params, input_info);
		bool             source_changed = false;
		if (entry != nullptr && entry->call_source) {
			source_changed = entry->RefreshCallSource(runtime);
			if (!source_changed) {
				if (auto* permutation = entry->FindPermutation<true>(runtime, push_data_cursor)) {
					input_info.stage = {.program = &permutation->program, .resources = &entry->resources};
					permutation->program.bindings.AdvancePushData(push_data_cursor);
					return permutation->handle;
				}
			}
		} else if (entry != nullptr) {
			if (materialized == nullptr) {
				EXIT_IF(!ProgramCache::Materialize(*entry, runtime, entry->resources, entry->specialization));
			}
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
				                                                    layout.ShaderDataDwords()) &&
				       candidate.specialization == entry->specialization;
			};
			auto permutation = entry->permutations.end();
			if (entry->last_permutation < entry->permutations.size() &&
			    matches(entry->permutations[entry->last_permutation])) {
				permutation = entry->permutations.begin() +
				              static_cast<std::ptrdiff_t>(entry->last_permutation);
			} else {
				permutation = std::ranges::find_if(entry->permutations, matches);
			}
			if (permutation != entry->permutations.end()) {
				entry->last_permutation =
				    static_cast<size_t>(permutation - entry->permutations.begin());
				input_info.stage = {.program = &permutation->program, .resources = &entry->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == nullptr) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == nullptr ? &source : entry->call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (current->call) {
				if (entry == nullptr) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == nullptr) {
			// Find set lookup_key for this program and nothing has looked up another since.
			auto retained = source.call ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			auto             plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			std::unique_lock lock(programs_mutex);
			entry = &programs.try_emplace(lookup_key, std::move(plan), std::move(retained)).first->second;
			lock.unlock();
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(entry->resource_plan, runtime,
			                                                    entry->resources, entry->specialization));
		} else if (source_changed) {
			entry->resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (const auto* permutation = entry->FindPermutation<true>(runtime, push_data_cursor)) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		entry->permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->specialization, push_data_cursor,
		    entry->FunctionCode()));
		const auto& permutation = entry->permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(GraphicContext& graphics): graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	uint64_t                                                    next_shader_id = 0;
	// Taken exclusively around insertions; lookups from the lookahead workers share it.
	std::shared_mutex                                           programs_mutex;
};

// Materializes one stage's resources beside the GPU thread, which materializes the draw's other
// stage meanwhile and then spins until the helper is done. While draws keep coming the helper
// spins too, so a handoff costs no scheduler wakeup; after a quiet period it sleeps.
class PipelineCache::MaterializeHelper {
public:
	using Job = void (*)(void* context);

	MaterializeHelper(): m_thread([this] {
		  Common::Thread::DedicateCore(1);
		  Run();
	  }) {}
	~MaterializeHelper() {
		m_state.store(Stop);
		m_state.notify_one();
		m_thread.join();
	}
	KYTY_CLASS_NO_COPY(MaterializeHelper);

	void Start(Job job, void* context) {
		m_job     = job;
		m_context = context;
		m_state.store(Posted);
		if (m_sleeping.load()) {
			m_state.notify_one();
		}
	}

	// False when the helper had not started the job, which then never runs: a helper that is
	// asleep or preempted would only make the GPU thread wait, so the caller does the work.
	bool Join() {
		auto expected = static_cast<uint32_t>(Posted);
		if (m_state.compare_exchange_strong(expected, Idle)) {
			return false;
		}
		while (m_state.load(std::memory_order_acquire) != Done) {
			Pause();
		}
		m_state.store(Idle, std::memory_order_relaxed);
		return true;
	}

private:
	enum State : uint32_t { Idle, Posted, Running, Done, Stop };
	static constexpr auto SpinPeriod = std::chrono::milliseconds(4);

	static void Pause() {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}

	void Run() {
		Libs::LibKernel::Memory::SetGpuResourceHelper(true);
		auto     last_job = std::chrono::steady_clock::now();
		uint32_t spins    = 0;
		for (;;) {
			const auto state = m_state.load(std::memory_order_acquire);
			if (state == Stop) {
				return;
			}
			if (state == Posted) {
				auto expected = static_cast<uint32_t>(Posted);
				if (m_state.compare_exchange_strong(expected, Running)) {
					m_job(m_context);
					m_state.store(Done, std::memory_order_release);
				}
				last_job = std::chrono::steady_clock::now();
				continue;
			}
			// The clock is read once per many pauses; spinning should stay light on its core.
			if (state != Idle || (++spins % 256u) != 0u ||
			    std::chrono::steady_clock::now() - last_job < SpinPeriod) {
				Pause();
				continue;
			}
			// Start notifies only a sleeper: publish the sleep before waiting for a new state.
			m_sleeping.store(true);
			m_state.wait(Idle);
			m_sleeping.store(false);
		}
	}

	std::atomic<uint32_t> m_state {Idle};
	std::atomic_bool      m_sleeping {false};
	Job                   m_job     = nullptr;
	void*                 m_context = nullptr;
	std::thread           m_thread;
};

struct PipelineCache::GraphicsInputs {
	std::array<ShaderParams, 3> vertex_params;
	ShaderParams                pixel_params;
	bool                        tess_active      = false;
	uint32_t                    push_data_cursor = 0;
};

// The next draw's resources, which the helper materializes while the GPU thread records the
// current draw. They stay out of the program cache entries, which the current draw still uses,
// until the next draw adopts them.
struct PipelineCache::Speculation {
	// The registers the speculation was prepared from, and what preparing them gave.
	HW::VertexShaderInfo                           vertex_regs;
	HW::PixelShaderInfo                            pixel_regs;
	HW::ShaderRegisters                            sh;
	std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
	bool                                           pixel_active = false;
	GraphicsInputs                                 inputs;
	ShaderVertexInputInfo                          vertex_info;
	ShaderPixelInputInfo                           pixel_info;
	ProgramCache::SourceEntry*                   vertex_entry = nullptr;
	ProgramCache::SourceEntry*                   pixel_entry  = nullptr;
	ShaderParams                                 vertex_params;
	ShaderParams                                 pixel_params;
	ShaderRecompiler::IR::SrtRuntime             vertex_runtime;
	ShaderRecompiler::IR::SrtRuntime             pixel_runtime;
	ShaderRecompiler::IR::ResourceSnapshot       vertex_resources;
	ShaderRecompiler::IR::ResourceSnapshot       pixel_resources;
	ShaderRecompiler::IR::ResourceSpecialization vertex_specialization;
	ShaderRecompiler::IR::ResourceSpecialization pixel_specialization;
	uint64_t                                     memory_epoch = 0;
	uint64_t                                     target       = 0;
	bool                                         posted       = false;
	// Written by the helper; read after joining it.
	bool                                         done = false;
};

// A draw recorded ahead of its execution: its registers, and the programs and resources a
// worker prepared from them, which the GPU thread adopts when the draw arrives.
struct PipelineCache::DrawRecord {
	enum State : uint32_t { Free, Posted, Claimed, Done, Taken, Abandoned };

	DrawRecordRegisters   registers;
	DrawRecordInfo        info;
	uint64_t              sequence = 0;
	std::atomic<uint32_t> state {Free};
	// Written by the worker; read by the GPU thread once the state is Done. A compute record
	// keeps its program in the vertex fields.
	bool                                           prepared     = false;
	ShaderParams                                   compute_params;
	ShaderComputeInputInfo                         compute_info;
	bool                                           pixel_active = false;
	std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
	GraphicsInputs                                 inputs;
	ShaderVertexInputInfo                          vertex_info;
	ShaderPixelInputInfo                           pixel_info;
	ProgramCache::SourceEntry*                     vertex_entry = nullptr;
	ProgramCache::SourceEntry*                     pixel_entry  = nullptr;
	ShaderRecompiler::IR::ResourceSnapshot         vertex_resources;
	ShaderRecompiler::IR::ResourceSnapshot         pixel_resources;
	ShaderRecompiler::IR::ResourceSpecialization   vertex_specialization;
	ShaderRecompiler::IR::ResourceSpecialization   pixel_specialization;
};

// The lookahead ring and its workers. The GPU thread posts records in draw order and takes them
// in that order; workers claim posted records in order and prepare them. The GPU thread never
// waits for a worker: a record still being prepared when its draw arrives is abandoned, and
// the draw prepares itself. Like the resource helper, idle workers spin for a while, then sleep.
class PipelineCache::LookaheadWorkers {
public:
	LookaheadWorkers(PipelineCache& cache, uint32_t depth, uint32_t workers,
	                 std::chrono::microseconds spin_period)
	    : m_cache(cache), m_spin_period(spin_period) {
		m_records.reserve(depth);
		for (uint32_t i = 0; i < depth; i++) {
			m_records.push_back(std::make_unique<DrawRecord>());
			m_records.back()->registers.slot = i;
		}
		// Every record may hold one snapshot, plus the current one and a spare.
		m_contexts     = std::vector<HW::Context>(depth + 2);
		m_context_uses = std::vector<std::atomic<uint32_t>>(depth + 2);
		m_threads.reserve(workers);
		for (uint32_t i = 0; i < workers; i++) {
			m_threads.emplace_back([this, i] {
				Common::Thread::DedicateCore(2 + i);
				Run();
			});
		}
	}
	~LookaheadWorkers() {
		m_stop.store(true, std::memory_order_release);
		m_posted.fetch_add(1, std::memory_order_release);
		m_posted.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(LookaheadWorkers);

	DrawRecordRegisters* Acquire() {
		const auto posted = m_posted.load(std::memory_order_relaxed) + m_unpublished;
		if (posted - m_taken >= m_records.size()) {
			return nullptr;
		}
		auto& record = *m_records[posted % m_records.size()];
		if (record.state.load(std::memory_order_acquire) != DrawRecord::Free) {
			return nullptr;
		}
		return &record.registers;
	}

	void Post(DrawRecordRegisters* registers, const DrawRecordInfo& info,
	          const HW::Context& context, bool context_changed) {
		auto& record = *m_records[registers->slot];
		EXIT_IF(&record.registers != registers ||
		        record.state.load(std::memory_order_relaxed) != DrawRecord::Free);
		if (context_changed || m_current_context == UINT32_MAX) {
			// The scan's reference moves to a fresh snapshot.
			if (m_current_context != UINT32_MAX) {
				m_context_uses[m_current_context].fetch_sub(1, std::memory_order_acq_rel);
			}
			m_current_context = UINT32_MAX;
			for (size_t i = 0; i < m_contexts.size(); i++) {
				const auto slot = (m_context_cursor + i) % m_contexts.size();
				if (m_context_uses[slot].load(std::memory_order_acquire) == 0) {
					m_current_context = static_cast<uint32_t>(slot);
					m_context_cursor  = slot + 1;
					break;
				}
			}
			EXIT_IF(m_current_context == UINT32_MAX);
			m_contexts[m_current_context] = context;
			m_context_uses[m_current_context].store(1, std::memory_order_release);
		}
		m_context_uses[m_current_context].fetch_add(1, std::memory_order_acq_rel);
		registers->context      = &m_contexts[m_current_context];
		registers->context_slot = m_current_context;
		record.info             = info;
		record.prepared         = false;
		record.sequence         = m_posted.load(std::memory_order_relaxed) + m_unpublished;
		record.state.store(DrawRecord::Posted, std::memory_order_release);
		// Publish publishes the scan's records together: one update of the shared counter.
		m_unpublished++;
	}

	void Publish() {
		if (m_unpublished == 0) {
			return;
		}
		m_posted.fetch_add(m_unpublished, std::memory_order_release);
		m_unpublished = 0;
		if (m_sleepers.load(std::memory_order_acquire) != 0) {
			m_posted.notify_all();
		}
	}

	DrawRecord* Take(const void* buffer, uint32_t offset) {
		Publish();
		const auto posted = m_posted.load(std::memory_order_relaxed);
		while (m_taken < posted) {
			auto& record = *m_records[m_taken % m_records.size()];
			if (record.info.buffer != buffer || record.info.offset > offset) {
				// Draws of another buffer, or after this one: keep them.
				return nullptr;
			}
			m_taken++;
			const bool match = record.info.offset == offset;
			if (!match) {
				Common::Perf::Add(Common::Perf::LookaheadDropped); // PERF-INSTR
				static uint32_t drop_log = 0; // PERF-INSTR
				if (drop_log++ < 40) { // PERF-INSTR
					const auto* cmds = static_cast<const uint32_t*>(buffer); // PERF-INSTR
					std::fprintf(stderr, "LADROP rec_off=%u op=%02x len=%u cur_off=%u cur_op=%02x\n", record.info.offset, (cmds[record.info.offset] >> 8u) & 0xffu, cmds[record.info.offset] >> 16u & 0x3fffu, offset, (cmds[offset] >> 8u) & 0xffu); // PERF-INSTR
				} // PERF-INSTR
			}
			auto state = record.state.load(std::memory_order_acquire);
			// A worker in the middle of this draw's record finishes within microseconds; its
			// draw waits that long rather than preparing itself. The wait is bounded, as the
			// worker may be waiting for the GPU thread itself (a page it reads needs a download).
			if (match && state == DrawRecord::Claimed) {
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(30);
				for (uint32_t spins = 0; state == DrawRecord::Claimed; spins++) {
					Pause();
					if ((spins % 64u) == 0u && std::chrono::steady_clock::now() > deadline) {
						break;
					}
					state = record.state.load(std::memory_order_acquire);
				}
			}
			for (;;) {
				if (state == DrawRecord::Posted || state == DrawRecord::Done) {
					if (record.state.compare_exchange_strong(state, DrawRecord::Taken,
					                                         std::memory_order_acq_rel)) {
						if (match) {
							return &record;
						}
						Free(record);
						break;
					}
					continue;
				}
				// A worker is on it; it frees the record when done.
				if (record.state.compare_exchange_strong(state, DrawRecord::Abandoned,
				                                         std::memory_order_acq_rel)) {
					if (match) {
						return nullptr;
					}
					break;
				}
			}
		}
		return nullptr;
	}

	void Release(DrawRecord* record) {
		EXIT_IF(record->state.load(std::memory_order_relaxed) != DrawRecord::Taken);
		Free(*record);
	}

	void Drop() {
		Publish();
		const auto posted = m_posted.load(std::memory_order_relaxed);
		while (m_taken < posted) {
			auto& record = *m_records[m_taken % m_records.size()];
			m_taken++;
			auto state = record.state.load(std::memory_order_acquire);
			for (;;) {
				if (state == DrawRecord::Claimed) {
					if (record.state.compare_exchange_strong(state, DrawRecord::Abandoned,
					                                         std::memory_order_acq_rel)) {
						break;
					}
					continue;
				}
				if (record.state.compare_exchange_strong(state, DrawRecord::Taken,
				                                         std::memory_order_acq_rel)) {
					Free(record);
					break;
				}
			}
		}
	}

private:
	static void Pause() {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}

	// Returns a record (taken, or abandoned by the GPU thread) and its context reference.
	void Free(DrawRecord& record) {
		if (record.registers.context_slot != UINT32_MAX) {
			m_context_uses[record.registers.context_slot].fetch_sub(1, std::memory_order_acq_rel);
			record.registers.context_slot = UINT32_MAX;
		}
		record.state.store(DrawRecord::Free, std::memory_order_release);
	}

	// Claims and prepares the next posted record; false when none was posted.
	bool Work() {
		auto claim = m_claim.load(std::memory_order_relaxed);
		for (;;) {
			const auto posted = m_posted.load(std::memory_order_acquire);
			if (claim >= posted) {
				return false;
			}
			if (!m_claim.compare_exchange_weak(claim, claim + 1, std::memory_order_acq_rel)) {
				continue;
			}
			auto& record = *m_records[claim % m_records.size()];
			auto  state  = record.state.load(std::memory_order_acquire);
			if (state != DrawRecord::Posted || record.sequence != claim ||
			    !record.state.compare_exchange_strong(state, DrawRecord::Claimed,
			                                          std::memory_order_acq_rel)) {
				// Taken or dropped by the GPU thread meanwhile.
				claim++;
				continue;
			}
			if (record.info.compute) {
				m_cache.PrepareComputeRecord(record);
			} else {
				m_cache.PrepareDrawRecord(record);
			}
			state = DrawRecord::Claimed;
			if (!record.state.compare_exchange_strong(state, DrawRecord::Done,
			                                          std::memory_order_acq_rel)) {
				// Abandoned: its draw went on without it.
				Free(record);
			}
			return true;
		}
	}

	void Run() {
		Libs::LibKernel::Memory::SetGpuResourceHelper(true);
		auto     last_job = std::chrono::steady_clock::now();
		uint32_t spins    = 0;
		while (!m_stop.load(std::memory_order_acquire)) {
			const auto posted = m_posted.load(std::memory_order_acquire);
			if (Work()) {
				last_job = std::chrono::steady_clock::now();
				continue;
			}
			if ((++spins % 256u) != 0u ||
			    std::chrono::steady_clock::now() - last_job < m_spin_period) {
				Pause();
				continue;
			}
			m_sleepers.fetch_add(1, std::memory_order_acq_rel);
			m_posted.wait(posted, std::memory_order_acquire);
			m_sleepers.fetch_sub(1, std::memory_order_acq_rel);
		}
	}

	PipelineCache&                           m_cache;
	// How long an idle worker spins before sleeping; a sleeper wakes on the next post.
	const std::chrono::microseconds          m_spin_period;
	std::vector<std::unique_ptr<DrawRecord>> m_records;
	// Context snapshots and how many references (records, plus the scan's) each holds.
	std::vector<HW::Context>                 m_contexts;
	std::vector<std::atomic<uint32_t>>       m_context_uses;
	uint32_t                                 m_current_context = UINT32_MAX;
	size_t                                   m_context_cursor  = 0;
	std::atomic<uint64_t>                    m_posted {0};
	std::atomic<uint64_t>                    m_claim {0};
	uint64_t                                 m_taken       = 0;
	// Records posted since the last Publish, known to the GPU thread only.
	uint64_t                                 m_unpublished = 0;
	std::atomic<uint32_t>                    m_sleepers {0};
	std::atomic<bool>                        m_stop {false};
	std::vector<std::thread>                 m_threads;
};

namespace {

// Whether materializing a program with these parameters reads the same values: the walk
// depends on the user data and the shader's base address, beyond the program itself.
bool SameWalkInputs(const ShaderParams& a, const ShaderParams& b) {
	return a.Base() == b.Base() && a.user_data_count == b.user_data_count &&
	       std::equal(a.user_data.begin(), a.user_data.begin() + a.user_data_count,
	                  b.user_data.begin());
}

} // namespace

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval),
      m_workers(std::make_unique<CompileWorkers>()),
      m_materialize_helper(std::make_unique<MaterializeHelper>()),
      m_speculation(std::make_unique<Speculation>()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	{
		const auto setting = [](const char* name, uint32_t fallback) {
			const char* value = std::getenv(name);
			return value != nullptr ? static_cast<uint32_t>(std::atoi(value)) : fallback;
		};
		const auto workers = setting("KYTY_LOOKAHEAD_WORKERS", Common::Thread::LookaheadWorkerCount);
		const auto depth   = setting("KYTY_LOOKAHEAD_DEPTH", 16);
		const auto spin_us = setting("KYTY_LOOKAHEAD_SPIN_US", 4000);
		// With few cores the guest threads need every one of them.
		if (setting("KYTY_LOOKAHEAD", 1) != 0 && workers != 0 && depth >= 2 &&
		    std::thread::hardware_concurrency() >= 16) {
			m_lookahead = std::make_unique<LookaheadWorkers>(*this, depth, workers,
			                                                 std::chrono::microseconds(spin_us));
		}
	}
	ShaderRecompiler::SetHardwareStorageBufferBounds(graphics.robust_buffer_access2_enabled);
	ShaderRecompiler::SetSharedMemoryLimits(
	    graphics.physical_device_properties.limits.maxComputeSharedMemorySize,
	    graphics.mesh_shader_enabled ? graphics.mesh_shader_properties.maxMeshSharedMemorySize : 0u);
	if (std::getenv("KYTY_NO_SHADER_BOUNDS_OPT") != nullptr) { ShaderRecompiler::SetHardwareStorageBufferBounds(false); ShaderRecompiler::SetSharedMemoryLimits(0, 0); } // PERF-INSTR
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	(void)FinishSpeculation();
	m_lookahead.reset();
	m_materialize_helper.reset();
	m_workers.reset();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() >= m_next_driver_cache_save) {
		Save();
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr || !m_driver_cache_dirty) {
		return;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	m_driver_cache_dirty = false;
}

PipelineCache::GraphicsInputs PipelineCache::PrepareGraphicsInputs(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	GraphicsInputs inputs;
	inputs.tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	const bool tess_active = inputs.tess_active;
	if (tess_active) {
		inputs.vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		inputs.vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	if (pixel_active) {
		inputs.pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	inputs.push_data_cursor =
	    ShaderRecompiler::IR::PushData::GraphicsDrawDwords(mesh_active, object_primitive_id);
	return inputs;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    uint64_t memory_epoch, DrawRecord* record) {
	if (record != nullptr) {
		if (auto adopted = AdoptDrawRecord(*record, vertex_regs, pixel_regs, sh,
		                                   target_export_mapping, pixel_active, vertex_info,
		                                   pixel_info, memory_epoch)) {
			Common::Perf::Add(Common::Perf::LookaheadAdopted); // PERF-INSTR
			return *adopted;
		}
	} else {
		Common::Perf::Add(Common::Perf::LookaheadMissing); // PERF-INSTR
	}
	if (auto adopted = AdoptSpeculation(vertex_regs, pixel_regs, sh, target_export_mapping,
	                                    pixel_active, vertex_info, pixel_info, memory_epoch)) {
		return *adopted;
	}
	auto inputs = PrepareGraphicsInputs(vertex_regs, pixel_regs, sh, context, user_config,
	                                    target_export_mapping, pixel_active, vertex_info, pixel_info);
	// Preparing inputs reads no program state, which a posted speculation may be walking; the
	// rest does.
	const bool speculated = FinishSpeculation();
	const auto& vertex_params    = inputs.vertex_params;
	const auto& pixel_params     = inputs.pixel_params;
	const bool  tess_active      = inputs.tess_active;
	auto&       push_data_cursor = inputs.push_data_cursor;
	// Takes the previous draw's speculation when it materialized these very programs with the
	// same inputs, while memory stayed unchanged.
	const auto adopt = [&](ProgramCache::SourceEntry* vertex_entry,
	                       ProgramCache::SourceEntry* pixel_entry) {
		auto& speculation = *m_speculation;
		if (speculation.memory_epoch != memory_epoch || speculation.vertex_entry != vertex_entry ||
		    speculation.pixel_entry != pixel_entry ||
		    !SameWalkInputs(speculation.vertex_params, vertex_params[0]) ||
		    (pixel_entry != nullptr && !SameWalkInputs(speculation.pixel_params, pixel_params))) {
			return false;
		}
		std::swap(vertex_entry->resources, speculation.vertex_resources);
		std::swap(vertex_entry->specialization, speculation.vertex_specialization);
		if (pixel_entry != nullptr) {
			std::swap(pixel_entry->resources, speculation.pixel_resources);
			std::swap(pixel_entry->specialization, speculation.pixel_specialization);
		}
		return true;
	};
	GraphicsPrograms result;
	if (pixel_active && !tess_active && m_materialize_helper != nullptr) {
		auto* pixel_entry  = m_program_cache->Find(pixel_params, pixel_info);
		auto* vertex_entry = pixel_entry != nullptr
		                         ? m_program_cache->Find(vertex_params[0], vertex_info[0])
		                         : nullptr;
		if (vertex_entry != nullptr) {
			if (speculated && adopt(vertex_entry, pixel_entry)) {
				// The helper materialized these during the previous draw.
			} else {
				// With both programs cached, the helper materializes the vertex stage meanwhile.
				struct Job {
					ProgramCache::SourceEntry*       entry;
					ShaderRecompiler::IR::SrtRuntime runtime;
					bool                             done = false;
				} job {vertex_entry, ProgramCache::OffThreadRuntime(vertex_params[0], vertex_info[0])};
				m_materialize_helper->Start(
				    [](void* context) {
					    auto&      job          = *static_cast<Job*>(context);
					    const bool materialized = ProgramCache::Materialize(*job.entry, job.runtime, job.entry->resources,
					        job.entry->specialization);
					    const bool blocked = Libs::LibKernel::Memory::TakeGpuResourceHelperBlocked();
					    job.done           = materialized && !blocked;
				    },
				    &job);
				EXIT_IF(!ProgramCache::Materialize(*pixel_entry, ProgramCache::Runtime(pixel_params, pixel_info),
				    pixel_entry->resources, pixel_entry->specialization));
				if (!m_materialize_helper->Join() || !job.done) {
					// The helper had not started, needed the GPU thread, or failed (which repeating it
					// here reports).
					EXIT_IF(!ProgramCache::Materialize(*vertex_entry, ProgramCache::Runtime(vertex_params[0], vertex_info[0]),
					    vertex_entry->resources, vertex_entry->specialization));
				}
			}
			result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, pixel_entry);
			result.vertex[0] =
			    m_program_cache->Get(vertex_params[0], vertex_info[0], push_data_cursor, vertex_entry);
			return result;
		}
	}
	if (!pixel_active && !tess_active && speculated) {
		// A draw without a pixel stage, such as a depth-only pass.
		auto* vertex_entry = m_program_cache->Find(vertex_params[0], vertex_info[0]);
		if (vertex_entry != nullptr && adopt(vertex_entry, nullptr)) {
			result.vertex[0] =
			    m_program_cache->Get(vertex_params[0], vertex_info[0], push_data_cursor, vertex_entry);
			return result;
		}
	}
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

void PipelineCache::SpeculateGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    uint64_t memory_epoch, uint64_t target) {
	auto& speculation = *m_speculation;
	EXIT_IF(speculation.posted);
	if (m_materialize_helper == nullptr) {
		return;
	}
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
	const auto inputs = PrepareGraphicsInputs(vertex_regs, pixel_regs, sh, context, user_config,
	                                          target_export_mapping, pixel_active, vertex_info,
	                                          pixel_info);
	if (inputs.tess_active) {
		return;
	}
	ProgramCache::SourceEntry* pixel_entry = nullptr;
	if (pixel_active) {
		pixel_entry = m_program_cache->Find(inputs.pixel_params, pixel_info);
		if (pixel_entry == nullptr) {
			return;
		}
	}
	auto* vertex_entry = m_program_cache->Find(inputs.vertex_params[0], vertex_info[0]);
	if (vertex_entry == nullptr) {
		return;
	}
	speculation.vertex_regs  = vertex_regs;
	speculation.pixel_regs   = pixel_regs;
	speculation.sh           = sh;
	std::ranges::copy(target_export_mapping, speculation.export_mapping.begin());
	speculation.pixel_active  = pixel_active;
	speculation.inputs        = inputs;
	speculation.vertex_info   = vertex_info[0];
	speculation.pixel_info    = pixel_info;
	speculation.vertex_entry  = vertex_entry;
	speculation.pixel_entry   = pixel_entry;
	speculation.vertex_params = inputs.vertex_params[0];
	speculation.pixel_params  = inputs.pixel_params;
	// The runtimes view the user data of these copies.
	speculation.vertex_runtime =
	    ProgramCache::SpeculativeRuntime(speculation.vertex_params, vertex_info[0]);
	speculation.pixel_runtime = ProgramCache::SpeculativeRuntime(speculation.pixel_params, pixel_info);
	speculation.memory_epoch  = memory_epoch;
	speculation.target        = target;
	speculation.done          = false;
	speculation.posted        = true;
	m_materialize_helper->Start(
	    [](void* context) {
		    auto&      job          = *static_cast<Speculation*>(context);
		    const bool materialized =
		        ProgramCache::Materialize(*job.vertex_entry,
		                                                   job.vertex_runtime, job.vertex_resources,
		                                                   job.vertex_specialization) &&
		        (job.pixel_entry == nullptr ||
		         ProgramCache::Materialize(*job.pixel_entry,
		                                                    job.pixel_runtime, job.pixel_resources,
		                                                    job.pixel_specialization));
		    const bool blocked = Libs::LibKernel::Memory::TakeGpuResourceHelperBlocked();
		    job.done           = materialized && !blocked;
	    },
	    &speculation);
}

std::optional<PipelineCache::GraphicsPrograms> PipelineCache::AdoptSpeculation(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    uint64_t memory_epoch) {
	auto&      speculation = *m_speculation;
	const auto same        = [](const auto& a, const auto& b) {
		return std::memcmp(&a, &b, sizeof(a)) == 0;
	};
	// A speculation still posted was posted for this draw: stale ones are dropped as draws
	// start. When it was prepared from these very registers, its inputs and programs are this
	// draw's; the context and user configuration came from the same packets.
	if (!speculation.posted || speculation.pixel_active != pixel_active ||
	    !same(speculation.vertex_regs, vertex_regs) || !same(speculation.pixel_regs, pixel_regs) ||
	    !same(speculation.sh, sh) ||
	    !std::ranges::equal(speculation.export_mapping, target_export_mapping)) {
		return std::nullopt;
	}
	const bool done      = FinishSpeculation();
	vertex_info[0]       = speculation.vertex_info;
	pixel_info           = speculation.pixel_info;
	auto* vertex_entry   = speculation.vertex_entry;
	auto* pixel_entry    = speculation.pixel_entry;
	auto& vertex_params  = speculation.inputs.vertex_params[0];
	auto& pixel_params   = speculation.inputs.pixel_params;
	auto  push_data_cursor = speculation.inputs.push_data_cursor;
	if (done && speculation.memory_epoch == memory_epoch) {
		std::swap(vertex_entry->resources, speculation.vertex_resources);
		std::swap(vertex_entry->specialization, speculation.vertex_specialization);
		if (pixel_entry != nullptr) {
			std::swap(pixel_entry->resources, speculation.pixel_resources);
			std::swap(pixel_entry->specialization, speculation.pixel_specialization);
		}
	} else {
		// The helper needed the GPU thread, or memory changed meanwhile.
		EXIT_IF(!ProgramCache::Materialize(*vertex_entry, ProgramCache::Runtime(vertex_params, vertex_info[0]),
		    vertex_entry->resources, vertex_entry->specialization));
		if (pixel_entry != nullptr) {
			EXIT_IF(!ProgramCache::Materialize(*pixel_entry, ProgramCache::Runtime(pixel_params, pixel_info),
			    pixel_entry->resources, pixel_entry->specialization));
		}
	}
	GraphicsPrograms result;
	if (pixel_entry != nullptr) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, pixel_entry);
	}
	result.vertex[0] =
	    m_program_cache->Get(vertex_params, vertex_info[0], push_data_cursor, vertex_entry);
	return result;
}

void PipelineCache::PrepareComputeRecord(DrawRecord& record) {
	record.prepared     = false;
	const auto& regs    = record.registers.shaders.GetCs();
	const auto& sh      = record.registers.context->GetShaderRegisters();
	auto&       info    = record.compute_info;
	info                = {};
	if (!record.info.indirect) {
		// As RenderExecutor::DispatchDirect derives the launch from the packet.
		info.dispatch_thread_dimensions =
		    (record.info.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			info.workgroup_counts[axis] = record.info.groups[axis];
		}
		if (info.dispatch_thread_dimensions) {
			const uint32_t group_sizes[] = {regs.cs_regs.num_thread_x, regs.cs_regs.num_thread_y,
			                                regs.cs_regs.num_thread_z};
			for (uint32_t axis = 0; axis < 3u; ++axis) {
				const auto size    = std::max(group_sizes[axis], 1u);
				const auto threads = info.workgroup_counts[axis];
				info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
			}
		}
	}
	info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	record.compute_params   = PrepareProgram(regs, sh, info);
	info.lds_storage        = info.lds_size_dwords * 4u >
	                   m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	thread_local ProgramCache::ProgramKey key;
	record.vertex_entry = m_program_cache->Find(record.compute_params, info, key);
	if (record.vertex_entry == nullptr) {
		return;
	}
	const auto runtime      = ProgramCache::SpeculativeRuntime(record.compute_params, info);
	const bool materialized = ProgramCache::Materialize(*record.vertex_entry, runtime,
	                                                    record.vertex_resources,
	                                                    record.vertex_specialization);
	const bool blocked      = Libs::LibKernel::Memory::TakeGpuResourceHelperBlocked();
	record.prepared         = materialized && !blocked;
}

void PipelineCache::PrepareDrawRecord(DrawRecord& record) {
	record.prepared          = false;
	const auto& context      = *record.registers.context;
	const auto& sh           = context.GetShaderRegisters();
	const auto& vertex_regs  = record.registers.shaders.GetVs();
	const auto& pixel_regs   = record.registers.shaders.GetPs();
	const auto  color_output_mask = DrawColorOutputMask(context);
	record.pixel_active = pixel_regs.ps_regs.data_addr != 0 &&
	                      (color_output_mask != 0 || PixelShaderHasDepthOrCoverageSideEffects(sh));
	record.export_mapping = TargetExportMapping(context, color_output_mask);
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
	record.inputs = PrepareGraphicsInputs(vertex_regs, pixel_regs, sh, context,
	                                      record.registers.user_config, record.export_mapping,
	                                      record.pixel_active, vertex_info, pixel_info);
	if (record.inputs.tess_active) {
		return;
	}
	record.vertex_info = vertex_info[0];
	record.pixel_info  = pixel_info;
	thread_local ProgramCache::ProgramKey key;
	record.pixel_entry =
	    record.pixel_active ? m_program_cache->Find(record.inputs.pixel_params, record.pixel_info, key)
	                        : nullptr;
	if (record.pixel_active && record.pixel_entry == nullptr) {
		return;
	}
	record.vertex_entry = m_program_cache->Find(record.inputs.vertex_params[0], record.vertex_info, key);
	if (record.vertex_entry == nullptr) {
		return;
	}
	// The runtimes view the user data of the record's copies.
	const auto vertex_runtime =
	    ProgramCache::SpeculativeRuntime(record.inputs.vertex_params[0], record.vertex_info);
	const auto pixel_runtime =
	    ProgramCache::SpeculativeRuntime(record.inputs.pixel_params, record.pixel_info);
	const bool materialized =
	    ProgramCache::Materialize(*record.vertex_entry,
	                                               vertex_runtime, record.vertex_resources,
	                                               record.vertex_specialization) &&
	    (record.pixel_entry == nullptr ||
	     ProgramCache::Materialize(*record.pixel_entry,
	                                                pixel_runtime, record.pixel_resources,
	                                                record.pixel_specialization));
	const bool blocked = Libs::LibKernel::Memory::TakeGpuResourceHelperBlocked();
	record.prepared    = materialized && !blocked;
}

std::optional<PipelineCache::GraphicsPrograms> PipelineCache::AdoptDrawRecord(
    DrawRecord& record, const HW::VertexShaderInfo& vertex_regs,
    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    uint64_t memory_epoch) {
	// A record taken while posted was never prepared; one taken when done was.
	if (!record.prepared) {
		Common::Perf::Add(Common::Perf::LookaheadUnprepared); // PERF-INSTR
		return std::nullopt;
	}
	const auto same = [](const auto& a, const auto& b) {
		return std::memcmp(&a, &b, sizeof(a)) == 0;
	};
	// Register packets the scan applied the same way the draw did give these very registers;
	// otherwise the record was prepared for another state, and memory_epoch tells whether
	// what the worker read could have changed since.
	if (record.info.compute || record.info.memory_epoch != memory_epoch ||
	    record.pixel_active != pixel_active ||
	    !same(record.registers.shaders.GetVs(), vertex_regs) ||
	    !same(record.registers.shaders.GetPs(), pixel_regs) ||
	    !same(record.registers.context->GetShaderRegisters(), sh) ||
	    !std::ranges::equal(record.export_mapping, target_export_mapping)) {
		Common::Perf::Add(Common::Perf::LookaheadStale); // PERF-INSTR
		return std::nullopt;
	}
	// The resource helper may still be walking the entries' current resources.
	(void)FinishSpeculation();
	vertex_info[0] = record.vertex_info;
	pixel_info     = record.pixel_info;
	auto* vertex_entry = record.vertex_entry;
	auto* pixel_entry  = record.pixel_entry;
	std::swap(vertex_entry->resources, record.vertex_resources);
	std::swap(vertex_entry->specialization, record.vertex_specialization);
	if (pixel_entry != nullptr) {
		std::swap(pixel_entry->resources, record.pixel_resources);
		std::swap(pixel_entry->specialization, record.pixel_specialization);
	}
	auto             push_data_cursor = record.inputs.push_data_cursor;
	GraphicsPrograms result;
	if (pixel_entry != nullptr) {
		result.pixel = m_program_cache->Get(record.inputs.pixel_params, pixel_info,
		                                    push_data_cursor, pixel_entry);
	}
	result.vertex[0] = m_program_cache->Get(record.inputs.vertex_params[0], vertex_info[0],
	                                        push_data_cursor, vertex_entry);
	return result;
}

PipelineCache::DrawRecordRegisters* PipelineCache::AcquireDrawRecord() {
	return m_lookahead != nullptr ? m_lookahead->Acquire() : nullptr;
}

void PipelineCache::PostDrawRecord(DrawRecordRegisters* registers, const DrawRecordInfo& info,
                                   const HW::Context& context, bool context_changed) {
	Common::Perf::Add(Common::Perf::LookaheadPosted); // PERF-INSTR
	m_lookahead->Post(registers, info, context, context_changed);
}

PipelineCache::DrawRecord* PipelineCache::TakeDrawRecord(const void* buffer, uint32_t offset) {
	return m_lookahead != nullptr ? m_lookahead->Take(buffer, offset) : nullptr;
}

void PipelineCache::ReleaseDrawRecord(DrawRecord* record) {
	m_lookahead->Release(record);
}

void PipelineCache::PublishDrawRecords() {
	if (m_lookahead != nullptr) {
		m_lookahead->Publish();
	}
}

void PipelineCache::DropDrawRecords() {
	if (m_lookahead != nullptr) {
		m_lookahead->Drop();
	}
}

void PipelineCache::DropSpeculationUnlessFor(uint64_t draw) {
	if (m_speculation->posted && m_speculation->target != draw) {
		(void)FinishSpeculation();
	}
}

bool PipelineCache::FinishSpeculation() {
	auto& speculation = *m_speculation;
	if (!speculation.posted) {
		return false;
	}
	speculation.posted = false;
	return m_materialize_helper->Join() && speculation.done;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               uint64_t memory_epoch, DrawRecord* record) {
	// Speculation stops at dispatches, so one is rarely posted here; materializing below must
	// not overlap it either way.
	(void)FinishSpeculation();
	if (record != nullptr && record->info.compute) {
		const auto same = [](const auto& a, const auto& b) {
			return std::memcmp(&a, &b, sizeof(a)) == 0;
		};
		const auto& prepared = record->compute_info;
		if (record->prepared && record->info.memory_epoch == memory_epoch &&
		    prepared.dispatch_thread_dimensions == input_info.dispatch_thread_dimensions &&
		    std::equal(std::begin(prepared.workgroup_counts), std::end(prepared.workgroup_counts),
		               std::begin(input_info.workgroup_counts)) &&
		    same(record->registers.shaders.GetCs(), regs) &&
		    same(record->registers.context->GetShaderRegisters(), sh)) {
			Common::Perf::Add(Common::Perf::LookaheadAdopted); // PERF-INSTR
			auto* entry = record->vertex_entry;
			std::swap(entry->resources, record->vertex_resources);
			std::swap(entry->specialization, record->vertex_specialization);
			input_info                = prepared;
			uint32_t push_data_cursor = 0;
			return m_program_cache->Get(record->compute_params, input_info, push_data_cursor, entry);
		}
		Common::Perf::Add(record->prepared ? Common::Perf::LookaheadStale : Common::Perf::LookaheadUnprepared); // PERF-INSTR
	} else {
		Common::Perf::Add(Common::Perf::LookaheadMissing); // PERF-INSTR
	}
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::Perf::t_cs_hash = params.hash; // PERF-INSTR
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool allow_async) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (!found.ready.load(std::memory_order_acquire)) {
			if (allow_async) {
				return nullptr;
			}
			found.ready.wait(false, std::memory_order_acquire);
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, programs, static_params);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	auto& pipeline = *iter->second;
	CheckpointDriverCache();

	if (allow_async && Config::AsyncPipelinesEnabled()) {
		pipeline.ready.store(false, std::memory_order_relaxed);
		// The map owns `pipeline` through a unique_ptr, so its address is stable, and the
		// destructor joins the workers before destroying it.
		m_workers->Submit([&pipeline, driver_cache = m_driver_cache,
		                   build = std::shared_ptr<GraphicsPipelineBuild>(std::move(build))] {
			pipeline.pipeline = FinishGraphicsPipeline(*build, driver_cache);
			pipeline.ready.store(true, std::memory_order_release);
			pipeline.ready.notify_all();
		});
		return nullptr;
	}

	pipeline.pipeline = FinishGraphicsPipeline(*build, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
	return &pipeline;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
