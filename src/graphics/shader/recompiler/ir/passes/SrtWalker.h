#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;
struct FastSrtTrace;
struct FastSrtState;
struct WalkScratch;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	// Optional inline form of read_memory: when set, read_memory loads guest memory inside the
	// GPU-mapped range this reports and reads zero outside it.
	bool (*find_mapped_range)(uint64_t address, uint64_t size, uint64_t& begin,
	                          uint64_t& end)                     = nullptr;
	// With find_mapped_range: the end of the range from an address that plain loads can read
	// without faulting on GPU-written pages. Elsewhere read_specialization_memory is used, which
	// waits for the GPU only when the bytes themselves are GPU-written.
	uint64_t (*readable_end)(uint64_t address) = nullptr;
	// With readable_end: what read_specialization_memory reads where no byte is GPU-written.
	bool (*read_backing)(uint64_t address, void* data, uint64_t size) = nullptr;
	// With find_mapped_range: replaces the inline form's plain loads, for walks that must not
	// fault (it reads zero where it cannot load).
	void (*load)(uint64_t address, void* data, uint64_t size) = nullptr;
	// Receives read_memory's ranges while specialization reads are captured.
	std::vector<std::pair<uint64_t, uint64_t>>* captured_reads = nullptr;
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

// Retained reads no longer need the guest instruction PC. A clean read evaluates
// its address and bounds through the strict reader as well as the final DWORD.
struct SrtReadFlags {
	uint32_t index = 0;
	uint32_t clean = 0;
};

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
void CompileSrtWalker(ResourcePlan& program);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          SrtWalker* clean_evaluator = nullptr);
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

	// This thread's scratch for the plan: the descriptor sources the last refresh found active,
	// and the key scratch of indirect descriptor materialization.
	[[nodiscard]] static std::vector<uint8_t>&  ActiveSources(const ResourcePlan& program);
	[[nodiscard]] static std::vector<uint32_t>& MaterialKeys(const ResourcePlan& program);
	// This thread's resolved selector keys, one per ResourcePlan::selector_keys group.
	[[nodiscard]] static std::vector<ResourcePlan::SelectorKeys>& SelectorKeys(
	    const ResourcePlan& program);
	// The blocks this thread's last RefreshFlatBuffer of the program visited.
	[[nodiscard]] static std::vector<uint8_t>&  VisitedBlocks(const ResourcePlan& program);

private:
	SrtWalker(SrtWalker& parent, Value active_mask, bool clean = false,
	          SrtWalker* clean_evaluator = nullptr);
	ResourcePlan::EvaluationContext& AcquireContext();
	void ResetUnknowns();
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);
	bool RefreshFlatBufferFast(std::vector<uint32_t>& flat, bool& handled);
	bool RefreshFlatBufferSlow(std::vector<uint32_t>& flat);
	// Leaves the live control-flow blocks in the scratch's visited_blocks.
	void SettleReachability();
	// The descriptor as computed by the last fast refresh: Failed where that refresh stopped at an
	// unreadable word, Unknown where it did not reach every word.
	enum class FastDescriptor { Unknown, Ready, Failed };
	FastDescriptor TryFastDescriptor(uint32_t source, DescriptorValue& result);
	bool DemandFast(FastSrtPlan& plan, uint32_t index);
	bool DemandFastNode(FastSrtPlan& plan, uint32_t index);
	// A node's value from its already evaluated args (leaves, reads and operations).
	bool ComputeFastNode(const FastSrtPlan& plan, uint32_t index, uint64_t& out);
	// Where a replay stopped: steps replayed, all of them when the trace completed. A step that
	// differed only in validity was evaluated; any other stop leaves its node to the walk.
	struct ReplayStop {
		size_t   steps     = 0;
		bool     evaluated = false;
		uint32_t node      = 0;
		uint8_t  valid     = 0;
		uint8_t  decision  = 0;
	};
	// Evaluates a trace's steps from first on, stopping before anything the demand walk would
	// not evaluate.
	ReplayStop ReplayFastTrace(FastSrtPlan& plan, const FastSrtTrace& trace, size_t first);
	// Fills the flat buffer and active sources as the completed trace's walk did.
	void FinishFastTrace(const FastSrtPlan& plan, const FastSrtTrace& trace,
	                     std::vector<uint32_t>& flat);
	bool WalkFastPlan(FastSrtPlan& plan, std::vector<uint32_t>& flat);
	void SettleFastReachability(FastSrtPlan& plan);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	const std::vector<uint8_t>*      m_reachable_blocks = nullptr;
	const SrtWalker*                m_session;
	uint8_t                         m_read_policy;
	WalkScratch&                    m_scratch;
	ResourcePlan::EvaluationContext& m_context;
	// This thread's fast-walk state of the plan, once a fast refresh selected it.
	FastSrtState*                   m_fast            = nullptr;
	uint32_t                        m_fast_generation = 0;
	// Set once the memoized IR walk has run; a fast refresh would not share its reads.
	bool                            m_evaluated       = false;
	// The trace the demand walk appends each evaluation to, if any, and the number of block
	// conditions it already holds from a replay the walk continues.
	FastSrtTrace*                   m_recording       = nullptr;
	size_t                          m_skip_conditions = 0;
	// GPU-mapped range of the last inline ordinary read; mappings cannot change during a refresh.
	uint64_t                        m_mapped_begin    = 0;
	uint64_t                        m_mapped_end      = 0;
	// Part of the mapped range known free of GPU-written pages during this refresh.
	uint64_t                        m_readable_begin  = 0;
	uint64_t                        m_readable_end    = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
