#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <mutex>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	runtime.find_mapped_range = nullptr;
	runtime.readable_end      = nullptr;
	runtime.read_backing      = nullptr;
	runtime.load              = nullptr;
	runtime.captured_reads    = nullptr;
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};


} // namespace

void CompileSrtWalker(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.capture_specialization_reads || !program.control_flow.empty() ||
	    program.srt_reads.empty()) return;

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    inst->NumArgs() < 2 || !IsRawRead(program, *inst) ||
		    read.flat_offset >= program.srt_reads.size()) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return;
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return;
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return;
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return;
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0))) return;
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return;
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return;
	}
	if (visited.size() != pointer_uses.size()) return;

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		if (first < 0) {
			code.mov(code.r10, -static_cast<int64_t>(first));
			code.cmp(code.rdi, code.r10);
			code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}


float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

namespace {
// Control Phi evaluations (and hits on retryable unknowns) on this thread: an evaluation that
// fails without moving it does not depend on block reachability.
thread_local uint64_t t_reachability_reads = 0;
// The value of a completed-unknown memo entry that no reachability change can resolve.
constexpr uint64_t PermanentUnknown = UINT64_MAX;
} // namespace

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns that read reachability (a control Phi, directly or
	// through a retryable unknown) are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 2u) &&
	    m_context.values[index].value == PermanentUnknown) {
		return false;
	}
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    (m_context.values[index].generation == (m_context.generation | 2u) &&
	     m_context.values[index].value == m_context.unknown_epoch)) {
		++t_reachability_reads;
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t   out       = 0;
	const auto reads     = t_reachability_reads;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		memo.value      = t_reachability_reads != reads ? m_context.unknown_epoch : PermanentUnknown;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		++t_reachability_reads;
		bool found = false;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming) || (found && incoming != result)) return false;
			result = incoming;
			found = true;
		}
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (inst.GetOpcode() == ValueOpcode::LoadBufferU32 &&
				      m_clean_evaluator->m_active_mask == m_active_mask))) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
			if (binary()) {
				result = inst.GetOpcode() == ValueOpcode::UMin32
				    ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
				    : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!Arg(inst, 0, a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 1, b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				if (active.EvaluateWide(inst.Arg(1), b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	if (const auto fast = TryFastDescriptor(source, result); fast != FastDescriptor::Unknown) {
		return fast == FastDescriptor::Ready;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

// The evaluation order of a complete fast refresh. Draws of one shader nearly always take the
// same short circuits and block edges, so a later refresh evaluates this list in order, without
// the demand recursion or the block walk. A step that decides differently, or whose validity
// differs, ends the replay before anything the demand walk would not evaluate.
struct FastSrtTrace {
	struct Step {
		uint32_t node      = 0;
		uint8_t  condition = 0; // a block condition the walk branched on, not an evaluation
		uint8_t  valid     = 0;
		uint8_t  decision  = 0; // Select: chosen arg; And/Or, RawVector: 1 short circuit, 2 full

		bool operator==(const Step&) const = default;
	};
	std::vector<Step>                          steps;
	std::vector<std::pair<uint32_t, uint32_t>> flat; // (node, flat offset) in refresh order
	std::vector<uint8_t>                       active;
	bool                                       ready  = false;
	bool                                       strict = false; // walk had the strict reader
};

// Compact, dependency-ordered form of a plan's SRT expressions. MaterializeResources walks the
// same expressions through two memoized walkers: the ordinary one and, when the plan needs strict
// specialization reads, a clean one that reads through the specialization reader. Each node here
// belongs to one of those modes, so a refresh computes every needed value at most once, in the
// same block order as the IR walk, and then seeds both walkers' memos.
struct FastSrtPlan {
	enum class Kind : uint8_t {
		Imm,
		UserData,
		ShaderBase,
		Alias,
		ExtractU64,
		AddCarry,
		RawAddress,
		RawConstBuffer,
		RawVector,
		Op,
		Select,
		LogicalAnd,
		LogicalOr,
		Undef,      // never evaluates, as in the IR walk
		ControlPhi, // incoming values of possibly reachable predecessors (imm: control_phis row)
	};
	struct Node {
		Kind        kind   = Kind::Imm;
		ValueOpcode op     = ValueOpcode::Phi;
		bool        flush  = false;
		bool        clean  = false; // read through the clean walker's runtime
		uint8_t     used   = 0;     // bit mask of meaningful args
		uint32_t    args[6] {};
		uint64_t    imm    = 0;
		uint32_t    memory = 0;
		uint32_t    memo   = UINT32_MAX; // walker memo index of the instruction
	};
	struct BlockPlan {
		uint32_t condition = UINT32_MAX;
	};

	// A node's result in the current refresh, packed so one demand touches one cache line.
	struct NodeState {
		uint64_t value = 0;
		uint32_t stamp = 0;
		uint32_t valid = 0;
	};

	std::vector<Node>                  nodes;
	std::vector<uint32_t>              slot_nodes;
	// Memo entries of instructions that only forward another node's value (bitcasts, ReadConst,
	// resolved phis); seeded together with that node.
	struct AliasMemo {
		uint32_t memo;
		bool     clean;
	};
	std::vector<std::vector<AliasMemo>> alias_memos;
	std::vector<BlockPlan>             blocks;
	std::vector<std::vector<uint32_t>> source_nodes; // dwords evaluated by the ordinary walker
	// Node states before any refresh: immediates valid, everything else unstamped.
	std::vector<NodeState>             initial_state;
	// Clean-walker memo entries a refresh seeds from the nodes it computed: (node, memo).
	std::vector<std::pair<uint32_t, uint32_t>> clean_seeds;
	// Stamp of nodes valid in every refresh (immediates); generations stay below it.
	static constexpr uint32_t          AlwaysValid  = UINT32_MAX;
	// Descriptor sources live before the control-flow walk: those outside every block.
	std::vector<uint8_t>               initial_active;
	bool                               control_flow = false;
	bool                               has_clean    = false;
	// Every block condition compiled: the walk never consults the IR walker.
	bool                        replayable = false;
	// Control Phis in branch predicates: reachability is settled by a fixed point, over the
	// compiled conditions when every one compiled (fast_reachability), else by the IR walk.
	bool                        ir_reachability   = false;
	bool                        fast_reachability = false;
	// Predecessor blocks of each ControlPhi's incoming values.
	std::vector<std::vector<uint32_t>> control_phis;
};

// A thread's state of fast walks of one plan.
struct FastSrtState {
	std::vector<FastSrtPlan::NodeState> state;
	uint32_t                            generation = 0;
	// Most recently used first: draws of one shader often alternate between two setups.
	std::array<FastSrtTrace, 2> traces;
};

// A thread's scratch for walks of one plan: nested clean/EXEC memos, activity, material keys
// and the fast-walk state. Threads walk the same plan at once, each with its own scratch.
struct WalkScratch {
	std::deque<ResourcePlan::EvaluationContext> evaluation_contexts;
	std::vector<uint8_t>                        active_sources;
	std::vector<uint8_t>                        visited_blocks;
	std::vector<uint8_t>                        reachable_blocks;
	std::vector<uint32_t>                       pending_blocks;
	std::vector<uint32_t>                       material_keys;
	std::vector<ResourcePlan::SelectorKeys>     selector_keys;
	const FastSrtPlan*                          fast_plan = nullptr;
	FastSrtState                                fast;
};

namespace {

WalkScratch& ScratchFor(const ResourcePlan& program) {
	thread_local std::unordered_map<uint64_t, std::unique_ptr<WalkScratch>> scratch;
	auto& entry = scratch[program.scratch_id];
	if (entry == nullptr) {
		entry = std::make_unique<WalkScratch>();
	}
	return *entry;
}

} // namespace

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this), m_read_policy(clean_evaluator != nullptr ? 2u : 0u),
      m_scratch(ScratchFor(program)), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program),
      m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)),
      m_scratch(parent.m_scratch), m_context(AcquireContext()) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	for (auto& context: m_scratch.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

std::vector<uint8_t>& SrtWalker::ActiveSources(const ResourcePlan& program) {
	return ScratchFor(program).active_sources;
}

std::vector<uint32_t>& SrtWalker::MaterialKeys(const ResourcePlan& program) {
	return ScratchFor(program).material_keys;
}

std::vector<ResourcePlan::SelectorKeys>& SrtWalker::SelectorKeys(const ResourcePlan& program) {
	return ScratchFor(program).selector_keys;
}

std::vector<uint8_t>& SrtWalker::VisitedBlocks(const ResourcePlan& program) {
	return ScratchFor(program).visited_blocks;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	m_evaluated   = true;
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

// A context per session, EXEC mask and read policy, in this thread's scratch for the plan.
ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_scratch.evaluation_contexts) {
		if (context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) return context;
		if (context.session == nullptr && unused == nullptr) unused = &context;
	}
	auto& context = unused != nullptr ? *unused : m_scratch.evaluation_contexts.emplace_back();
	context.session     = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	return context;
}

void SrtWalker::ResetUnknowns() {
	for (auto& context: m_scratch.evaluation_contexts) {
		if (context.session != m_session) continue;
		++context.unknown_epoch;
	}
}

namespace {


class FastSrtCompiler {
public:
	explicit FastSrtCompiler(const ResourcePlan& program)
	    : m_program(program),
	      m_has_clean(program.capture_specialization_reads ||
	                  program.requires_specialization_memory) {}

	std::shared_ptr<FastSrtPlan> Build() {
		const auto& program = m_program;
		if (!program.srt_plan_complete) {
			return nullptr;
		}
		m_plan                  = std::make_shared<FastSrtPlan>();
		m_plan->has_clean       = m_has_clean;
		m_plan->ir_reachability = !program.control_phi_predecessors.empty();
		for (uint32_t slot = 0; slot < program.srt_reads.size(); ++slot) {
			const auto* read       = program.srt_reads[slot].value.ResolveInstruction();
			const bool  clean_slot = read != nullptr && read->Flags<SrtReadFlags>().clean != 0u;
			if (clean_slot && !m_has_clean) {
				return nullptr;
			}
			// Refresh reads a clean slot through the clean walker.
			const auto node = Compile(program.srt_reads[slot].value, clean_slot);
			if (node == Fail) return nullptr;
			m_plan->slot_nodes.push_back(node);
		}
		// Mirror MaterializeResources: indirect descriptors are evaluated elsewhere, captured
		// written descriptors through the clean walker, everything else through the ordinary one.
		std::vector<std::vector<uint32_t>> source_roots(program.descriptor_sources.size());
		const auto add_source = [&](uint32_t source, bool clean) {
			if (source >= program.descriptor_sources.size()) return false;
			const auto& descriptor = program.descriptor_sources[source];
			if (descriptor.indirect_descriptor.has_value() || clean) return true;
			if (!source_roots[source].empty()) return true;
			for (uint32_t word = 0; word < descriptor.dword_count; ++word) {
				const auto node = Compile(descriptor.dwords[word], false);
				if (node == Fail) return false;
				source_roots[source].push_back(node);
			}
			return true;
		};
		const bool capture = program.capture_specialization_reads;
		for (const auto& buffer: program.info.buffers) {
			if (!add_source(buffer.source, capture && buffer.written)) return nullptr;
		}
		for (const auto& image: program.info.images) {
			if (!add_source(image.source, capture && image.written)) return nullptr;
		}
		for (const auto& sampler: program.info.samplers) {
			if (!add_source(sampler.source, false)) return nullptr;
		}
		if (!program.control_flow.empty()) {
			m_plan->control_flow = true;
			m_plan->blocks.resize(program.control_flow.size());
			for (size_t index = 0; index < program.control_flow.size(); ++index) {
				const auto& block = program.control_flow[index];
				auto&       out   = m_plan->blocks[index];
				for (const auto slot: block.srt_reads) {
					if (slot >= m_plan->slot_nodes.size()) {
						return nullptr;
					}
				}
				for (const auto successor: block.successors) {
					if (successor >= program.control_flow.size()) {
						return nullptr;
					}
				}
				if (!block.condition.IsEmpty() && block.successors.size() < 2u) {
					return nullptr;
				}
				if (!block.condition.IsEmpty()) {
					// The predicate walker is the clean one when the plan has it.
					const auto node = Compile(block.condition, m_has_clean);
					// The IR predicate walker decides what the plan cannot: EXEC and lane rules
					// over Undef planning values, and anything it does not compile.
					if (node != Fail && m_undefined[node] == 0u) out.condition = node;
				}
			}
		}
		m_plan->source_nodes = std::move(source_roots);
		m_plan->initial_state.resize(m_plan->nodes.size());
		m_plan->alias_memos.resize(m_plan->nodes.size());
		for (uint32_t index = 0; index < m_plan->nodes.size(); ++index) {
			const auto& node = m_plan->nodes[index];
			if (node.kind == FastSrtPlan::Kind::Imm) {
				m_plan->initial_state[index] = {node.imm, FastSrtPlan::AlwaysValid, 1u};
				continue;
			}
			if (node.memo != UINT32_MAX && node.clean) {
				m_plan->clean_seeds.emplace_back(index, node.memo);
			}
			for (const auto& alias: m_plan->alias_memos[index]) {
				if (alias.clean) m_plan->clean_seeds.emplace_back(index, alias.memo);
			}
		}
		if (m_plan->control_flow) {
			auto& active = m_plan->initial_active;
			active.assign(m_program.descriptor_sources.size(), 1u);
			for (const auto& block: m_program.control_flow) {
				for (const auto source: block.sources) active.at(source) = 0u;
			}
		}
		m_plan->replayable        = !m_plan->ir_reachability;
		m_plan->fast_reachability = m_plan->ir_reachability;
		for (size_t index = 0; index < m_plan->blocks.size(); ++index) {
			if (!program.control_flow[index].condition.IsEmpty() &&
			    m_plan->blocks[index].condition == UINT32_MAX) {
				m_plan->replayable        = false;
				m_plan->fast_reachability = false;
			}
		}
		return std::move(m_plan);
	}

private:
	static constexpr uint32_t Fail = UINT32_MAX;

	std::vector<uint8_t> m_undefined; // per node: depends on an Undef node

	struct Key {
		const Inst* inst;
		bool        clean;
		bool operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const {
			return std::hash<const void*>()(key.inst) ^ static_cast<size_t>(key.clean);
		}
	};

	uint32_t Push(FastSrtPlan::Node node, const Inst* inst, bool clean) {
		node.clean       = clean;
		const auto index = static_cast<uint32_t>(m_plan->nodes.size());
		bool undefined   = node.kind == FastSrtPlan::Kind::Undef;
		for (uint32_t arg = 0; arg < std::size(node.args); ++arg) {
			if ((node.used & (1u << arg)) != 0u && m_undefined[node.args[arg]] != 0u) undefined = true;
		}
		m_undefined.push_back(undefined ? 1u : 0u);
		m_plan->nodes.push_back(node);
		if (inst != nullptr) {
			m_plan->nodes.back().memo = inst->EvaluationIndex(m_program.evaluation_value_count);
			m_nodes.emplace(Key {inst, clean}, index);
		}
		return index;
	}

	uint32_t Immediate(Value value) {
		FastSrtPlan::Node node;
		switch (value.GetType()) {
			case Type::U1: node.imm = value.U1(); break;
			case Type::U8: node.imm = value.U8(); break;
			case Type::U16: node.imm = value.U16(); break;
			case Type::U32: node.imm = value.U32(); break;
			case Type::U64: node.imm = value.U64(); break;
			case Type::F32: node.imm = std::bit_cast<uint32_t>(value.F32Value()); break;
			default: return Fail;
		}
		if (const auto found = m_immediates.find(node.imm); found != m_immediates.end()) {
			return found->second;
		}
		const auto index = Push(node, nullptr, false);
		m_immediates.emplace(node.imm, index);
		return index;
	}

	uint32_t Alias(const Inst* inst, uint32_t target, bool clean) {
		if (target == Fail) return Fail;
		if (m_plan->alias_memos.size() <= target) m_plan->alias_memos.resize(target + 1u);
		m_plan->alias_memos[target].push_back(
		    {inst->EvaluationIndex(m_program.evaluation_value_count), clean});
		m_nodes.emplace(Key {inst, clean}, target);
		return target;
	}

	uint32_t Compile(Value value, bool clean) {
		value = value.Resolve();
		if (value.IsImmediate()) return Immediate(value);
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return Fail;
		if (const auto found = m_nodes.find(Key {inst, clean}); found != m_nodes.end())
			return found->second;
		if (!m_visiting.insert(Key {inst, clean}).second || m_visiting.size() > 8192) return Fail;
		const auto result = CompileInst(*inst, clean);
		m_visiting.erase(Key {inst, clean});
		return result;
	}

	uint32_t CompileArgs(const Inst& inst, FastSrtPlan::Node& node, size_t count, bool clean) {
		for (size_t index = 0; index < count; ++index) {
			if (index >= inst.NumArgs()) return Fail;
			node.args[index] = Compile(inst.Arg(index), clean);
			if (node.args[index] == Fail) return Fail;
			node.used = static_cast<uint8_t>(node.used | (1u << index));
		}
		return 0;
	}

	uint32_t CompileInst(const Inst& inst, bool clean) {
		FastSrtPlan::Node node;
		node.op = inst.GetOpcode();
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) return Fail;
				node.kind = FastSrtPlan::Kind::UserData;
				node.imm  = reg - m_program.user_data_base;
				return Push(node, &inst, clean);
			}
			case ValueOpcode::GetShaderBase:
				node.kind = FastSrtPlan::Kind::ShaderBase;
				return Push(node, &inst, clean);
			case ValueOpcode::Phi: {
				if (const auto control = m_program.control_phi_predecessors.find(&inst);
				    control != m_program.control_phi_predecessors.end()) {
					if (inst.NumArgs() > std::size(node.args) ||
					    control->second.size() != inst.NumArgs()) {
						return Fail;
					}
					node.kind = FastSrtPlan::Kind::ControlPhi;
					if (CompileArgs(inst, node, inst.NumArgs(), clean) == Fail) return Fail;
					node.imm = m_plan->control_phis.size();
					m_plan->control_phis.push_back(control->second);
					return Push(node, &inst, clean);
				}
				const auto resolved =
				    ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (resolved.IsEmpty()) {
					return Fail;
				}
				return Alias(&inst, Compile(resolved, clean), clean);
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef: return Alias(&inst, Compile(inst.Arg(0), clean), clean);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				const auto index = inst.Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u)
					return Fail;
				const auto component = index.U32();
				if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
					node.kind    = FastSrtPlan::Kind::ExtractU64;
					node.imm     = component;
					node.args[0] = Compile(inst.Arg(0), clean);
					if (node.args[0] == Fail) return Fail;
					node.used = 1u;
					return Push(node, &inst, clean);
				}
				const auto* source = inst.Arg(0).ResolveInstruction();
				if (source == nullptr) return Fail;
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					return Alias(&inst, Compile(source->Arg(component), clean), clean);
				}
				if (source->GetOpcode() != ValueOpcode::IAddCarry32) return Fail;
				node.kind = FastSrtPlan::Kind::AddCarry;
				node.imm  = component;
				if (CompileArgs(*source, node, 2, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			}
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size())
					return Fail;
				return Alias(&inst, Compile(m_program.srt_reads[slot.U32()].value, clean), clean);
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
			case ValueOpcode::LoadBufferU32: {
				if (!IsRawRead(m_program, inst)) {
					return Fail;
				}
				const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
				if ((vector || inst.Flags<SrtReadFlags>().clean != 0u) && !clean && m_has_clean) {
					// The ordinary walker delegates clean and vector reads to its clean walker.
					return Alias(&inst, Compile(Value(const_cast<Inst*>(&inst)), true), false);
				}
				const auto* handle = inst.Arg(0).ResolveInstruction();
				if (handle == nullptr) return Fail;
				const bool four = inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector;
				if (four ? handle->NumArgs() != 4u : handle->NumArgs() < 2u) return Fail;
				node.kind   = vector ? FastSrtPlan::Kind::RawVector
				              : four ? FastSrtPlan::Kind::RawConstBuffer
				                     : FastSrtPlan::Kind::RawAddress;
				node.memory = inst.Flags<SrtReadFlags>().index;
				if (node.memory >= m_program.memory_info.size()) return Fail;
				// The read's immediate offset, kept with the node.
				node.imm = m_program.memory_info[node.memory].offset;
				for (uint32_t word = 0; word < (four ? 4u : 2u); ++word) {
					node.args[word] = Compile(handle->Arg(word), clean);
					if (node.args[word] == Fail) return Fail;
					node.used = static_cast<uint8_t>(node.used | (1u << word));
				}
				node.args[4] = Compile(inst.Arg(1), clean);
				if (node.args[4] == Fail) return Fail;
				node.used = static_cast<uint8_t>(node.used | (1u << 4));
				if (vector) {
					// The guard is evaluated first and a disabled read skips the rest.
					if (inst.NumArgs() < 5u) return Fail;
					node.args[5] = Compile(inst.Arg(4), clean);
					if (node.args[5] == Fail) return Fail;
					node.used = static_cast<uint8_t>(node.used | (1u << 5));
				}
				return Push(node, &inst, clean);
			}
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64:
				node.kind = FastSrtPlan::Kind::Undef;
				return Push(node, &inst, clean);
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				node.kind    = FastSrtPlan::Kind::Select;
				node.args[0] = Compile(inst.Arg(0), clean || m_has_clean);
				if (node.args[0] == Fail) return Fail;
				node.args[1] = Compile(inst.Arg(1), clean);
				if (node.args[1] == Fail) return Fail;
				node.args[2] = Compile(inst.Arg(2), clean);
				if (node.args[2] == Fail) return Fail;
				node.used = 7u;
				return Push(node, &inst, clean);
			}
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
				node.kind = inst.GetOpcode() == ValueOpcode::LogicalAnd ? FastSrtPlan::Kind::LogicalAnd
				                                                          : FastSrtPlan::Kind::LogicalOr;
				if (CompileArgs(inst, node, 2, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalXor:
				node.kind = FastSrtPlan::Kind::Op;
				if (CompileArgs(inst, node, 2, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				node.kind  = FastSrtPlan::Kind::Op;
				node.flush = inst.Flags<FPCompareFlags>().flush_input_denorms;
				if (CompileArgs(inst, node, 2, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::LogicalNot:
				node.kind = FastSrtPlan::Kind::Op;
				if (CompileArgs(inst, node, 1, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
				node.kind = FastSrtPlan::Kind::Op;
				if (CompileArgs(inst, node, 3, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			case ValueOpcode::BitFieldInsert:
				node.kind = FastSrtPlan::Kind::Op;
				if (CompileArgs(inst, node, 4, clean) == Fail) return Fail;
				return Push(node, &inst, clean);
			default: return Fail;
		}
	}

	const ResourcePlan&                           m_program;
	const bool                                    m_has_clean;
	std::shared_ptr<FastSrtPlan>                  m_plan;
	std::unordered_map<Key, uint32_t, KeyHash>    m_nodes;
	std::unordered_set<Key, KeyHash>              m_visiting;
	std::unordered_map<uint64_t, uint32_t>        m_immediates;
};

bool EvaluateFastOp(const FastSrtPlan::Node& node, const FastSrtPlan::NodeState* v,
                    uint64_t& result) {
	const auto a = v[node.args[0]].value;
	const auto b = v[node.args[1]].value;
	const auto c = v[node.args[2]].value;
	const auto f32 = [](uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); };
	switch (node.op) {
		case ValueOpcode::CompositeConstructU64:
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::IMul64: result = a * b; return true;
		case ValueOpcode::UMulHi:
			result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			const auto value = f32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX)
				return false;
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32: result = std::bit_cast<uint32_t>(f32(a) * f32(b)); return true;
		case ValueOpcode::FPTrunc32: result = std::bit_cast<uint32_t>(std::trunc(f32(a))); return true;
		case ValueOpcode::FPRecipIFlag32: {
			const auto exponent = (a >> 23u) & 0xffu;
			if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case ValueOpcode::FPIsNan32: result = std::isnan(f32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32: {
			const auto operand = [&](uint64_t bits) {
				if (node.flush && (bits & 0x7fffffffu) < 0x00800000u) bits &= 0x80000000u;
				return f32(bits);
			};
			result = node.op == ValueOpcode::FPOrdLessThanEqual32 ? operand(a) <= operand(b)
			                                                      : operand(a) >= operand(b);
			return true;
		}
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::ShiftLeftLogical32: result = static_cast<uint32_t>(a) << (b & 31u); return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32: result = static_cast<uint32_t>(a) >> (b & 31u); return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) return false;
			const auto mask = width == 32u ? UINT32_MAX : width == 0u ? 0u : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) return false;
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) bits |= ~mask;
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			const auto d      = v[node.args[3]].value;
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) return false;
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32: result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b); return true;
		case ValueOpcode::INotEqual32: result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b); return true;
		case ValueOpcode::ULessThan32: result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b); return true;
		case ValueOpcode::ULessThanEqual32:
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32: result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b); return true;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		default: return false;
	}
}

} // namespace

// Evaluates a node on demand, in the walker's operand order and with its short circuits, so the
// same reads happen as in the IR walk. Each node is computed at most once per refresh.
// Nodes this refresh already reached, and immediates, answer without entering DemandFastNode.
inline bool SrtWalker::DemandFast(FastSrtPlan& plan, uint32_t index) {
	const auto& state = m_fast->state[index];
	if (state.stamp == m_fast->generation || state.stamp == FastSrtPlan::AlwaysValid) {
		return state.valid != 0;
	}
	return DemandFastNode(plan, index);
}

bool SrtWalker::DemandFastNode(FastSrtPlan& plan, uint32_t index) {
	auto* state      = m_fast->state.data();
	const auto& node = plan.nodes[index];
	state[index].stamp = m_fast->generation;
	uint8_t decision   = 0;
	const auto finish  = [&](bool ok, uint64_t value) {
		state[index].value = ok ? value : 0u;
		state[index].valid = ok ? 1u : 0u;
		if (m_recording != nullptr) {
			m_recording->steps.push_back({index, 0u, static_cast<uint8_t>(ok ? 1u : 0u), decision});
		}
		return ok;
	};
	switch (node.kind) {
		case FastSrtPlan::Kind::Undef: return finish(false, 0);
		case FastSrtPlan::Kind::ControlPhi: {
			const auto& predecessors = plan.control_phis[node.imm];
			bool        found        = false;
			uint64_t    merged       = 0;
			for (size_t arg = 0; arg < predecessors.size(); ++arg) {
				if (m_reachable_blocks != nullptr && !m_reachable_blocks->at(predecessors[arg])) continue;
				if (!DemandFast(plan, node.args[arg])) return finish(false, 0);
				const auto incoming = state[node.args[arg]].value;
				if (found && incoming != merged) return finish(false, 0);
				merged = incoming;
				found  = true;
			}
			return finish(found, merged);
		}
		case FastSrtPlan::Kind::Select: {
			if (!DemandFast(plan, node.args[0])) return finish(false, 0);
			decision          = state[node.args[0]].value != 0u ? 1u : 2u;
			const auto chosen = node.args[decision];
			if (!DemandFast(plan, chosen)) return finish(false, 0);
			return finish(true, state[chosen].value);
		}
		case FastSrtPlan::Kind::LogicalAnd:
		case FastSrtPlan::Kind::LogicalOr: {
			const bool is_and = node.kind == FastSrtPlan::Kind::LogicalAnd;
			const bool left   = DemandFast(plan, node.args[0]);
			const auto a      = state[node.args[0]].value;
			decision          = 1u;
			if (left && (is_and ? a == 0u : a != 0u)) return finish(true, is_and ? 0u : 1u);
			decision = 2u;
			if (!DemandFast(plan, node.args[1])) return finish(false, 0);
			const auto b = state[node.args[1]].value;
			if (is_and ? (b != 0u && !left) : (b == 0u && !left)) return finish(false, 0);
			return finish(true, b != 0u);
		}
		case FastSrtPlan::Kind::RawVector:
			// A disabled lane reads zero before the descriptor is looked at.
			if (!DemandFast(plan, node.args[5])) return finish(false, 0);
			decision = 1u;
			if (state[node.args[5]].value == 0u) return finish(true, 0u);
			decision = 2u;
			break;
		default: break;
	}
	for (uint32_t arg = 0; arg < 5u; ++arg) {
		if ((node.used & (1u << arg)) != 0u && !DemandFast(plan, node.args[arg])) {
			return finish(false, 0);
		}
	}
	uint64_t   out = 0;
	const bool ok  = ComputeFastNode(plan, index, out);
	return finish(ok, out);
}

bool SrtWalker::ComputeFastNode(const FastSrtPlan& plan, uint32_t index, uint64_t& out) {
	const auto* state   = m_fast->state.data();
	const auto& node    = plan.nodes[index];
	const auto& runtime = node.clean ? m_clean_evaluator->m_runtime : m_runtime;
	out                 = 0;
	switch (node.kind) {
		case FastSrtPlan::Kind::Imm: out = node.imm; return true;
		case FastSrtPlan::Kind::UserData:
			if (node.imm >= runtime.user_data.size()) return false;
			out = runtime.user_data[node.imm];
			return true;
		case FastSrtPlan::Kind::ShaderBase: out = runtime.shader_base; return true;
		case FastSrtPlan::Kind::Alias: out = state[node.args[0]].value; return true;
		case FastSrtPlan::Kind::ExtractU64:
			out = static_cast<uint32_t>(state[node.args[0]].value >> (node.imm * 32u));
			return true;
		case FastSrtPlan::Kind::AddCarry: {
			const auto sum = static_cast<uint64_t>(static_cast<uint32_t>(state[node.args[0]].value)) +
			                 static_cast<uint32_t>(state[node.args[1]].value);
			out = node.imm == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case FastSrtPlan::Kind::RawAddress:
		case FastSrtPlan::Kind::RawConstBuffer:
		case FastSrtPlan::Kind::RawVector: {
			const bool  vector    = node.kind == FastSrtPlan::Kind::RawVector;
			const auto  low       = state[node.args[0]].value;
			const auto  high      = state[node.args[1]].value;
			const auto  offset    = state[node.args[4]].value;
			const auto  base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
			const auto  immediate = static_cast<int64_t>(static_cast<int32_t>(node.imm));
			uint64_t    address   = 0;
			if (node.kind != FastSrtPlan::Kind::RawAddress) {
				const auto records     = state[node.args[2]].value;
				const auto word3       = state[node.args[3]].value;
				const auto byte_offset = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) +
				                         (static_cast<uint32_t>(offset) & ~3u);
				const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
				if (immediate < 0) return false;
				if (vector) {
					if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
						return false;
					if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) return true;
				}
				const auto size =
				    stride == 0u ? static_cast<uint64_t>(static_cast<uint32_t>(records))
				                 : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
				// Out-of-range DWORDs read zero (out is zero).
				if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) return true;
				address = (base & ~uint64_t {3}) + byte_offset;
			} else {
				const auto relative = (immediate & ~int64_t {3}) +
				                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
				if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) return false;
			}
			uint32_t word = 0;
			// Inline forms of both readers, without the indirect calls. Ordinary and strict
			// reads share the ordinary runtime's mapping queries and capture list.
			if (!vector && m_runtime.find_mapped_range != nullptr) {
				if (address < m_mapped_begin || address + sizeof(uint32_t) > m_mapped_end) {
					uint64_t begin = 0;
					uint64_t end   = 0;
					if (!m_runtime.find_mapped_range(address, sizeof(uint32_t), begin, end)) {
						begin = end = 0;
					}
					m_mapped_begin = begin;
					m_mapped_end   = end;
				}
				const bool mapped =
				    address >= m_mapped_begin && address + sizeof(uint32_t) <= m_mapped_end;
				// A page without GPU-written data loads directly: a plain load cannot fault into
				// a full GPU readback there, and the strict reader would not wait for the GPU.
				bool readable = false;
				if (mapped && m_runtime.readable_end != nullptr) {
					if (address < m_readable_begin ||
					    address + sizeof(uint32_t) > m_readable_end) {
						const auto end = std::min(m_runtime.readable_end(address), m_mapped_end);
						const bool any = end >= address + sizeof(uint32_t);
						m_readable_begin = any ? address & ~uint64_t {0xfff} : 0;
						m_readable_end   = any ? end : 0;
					}
					readable = address >= m_readable_begin &&
					           address + sizeof(uint32_t) <= m_readable_end;
				}
				if (!node.clean) {
					// read_memory: mapped memory loads, the rest reads zero.
					const auto load = [&] {
						if (m_runtime.load != nullptr) {
							m_runtime.load(address, &word, sizeof(word));
						} else {
							std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
						}
					};
					if (mapped && (readable || m_runtime.readable_end == nullptr)) {
						load();
					} else if (mapped &&
					           (m_runtime.read_specialization_memory == nullptr ||
					            !m_runtime.read_specialization_memory(m_runtime.userdata, address,
					                                                  {&word, 1}))) {
						load();
					}
					if (m_runtime.captured_reads != nullptr) {
						m_runtime.captured_reads->emplace_back(address, sizeof(uint32_t));
					}
					out = word;
					return true;
				}
				// read_specialization_memory: without GPU-written bytes it reads the backing.
				if (readable && m_runtime.read_backing != nullptr &&
				    m_runtime.read_backing(address, &word, sizeof(word))) {
					if (m_runtime.captured_reads != nullptr) {
						m_runtime.captured_reads->emplace_back(address, sizeof(uint32_t));
					}
					out = word;
					return true;
				}
			}
			const auto reader = vector ? runtime.read_specialization_memory : runtime.read_memory;
			bool       ok     = true;
			if (reader != nullptr) {
				ok = reader(runtime.userdata, address, {&word, 1});
			} else if (vector) {
				ok = false;
			} else {
				std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
			}
			out = word;
			return ok;
		}
		case FastSrtPlan::Kind::Op: return EvaluateFastOp(node, state, out);
		default: return false;
	}
}

SrtWalker::ReplayStop SrtWalker::ReplayFastTrace(FastSrtPlan& plan, const FastSrtTrace& trace,
                                                 size_t first) {
	auto*      state      = m_fast->state.data();
	const auto generation = m_fast->generation;
	// Every arg the walk would look at was evaluated earlier in this refresh; anything else
	// means the trace no longer describes it.
	const auto current = [&](uint32_t index) {
		return state[index].stamp == generation || state[index].stamp == FastSrtPlan::AlwaysValid;
	};
	for (size_t index = first; index < trace.steps.size(); ++index) {
		const auto& step = trace.steps[index];
		if (step.condition != 0u) {
			const auto& condition = state[step.node];
			if (!current(step.node) || condition.valid != step.valid ||
			    (condition.valid != 0u && (condition.value != 0u ? 1u : 2u) != step.decision)) {
				return {index};
			}
			continue;
		}
		const auto& node     = plan.nodes[step.node];
		uint64_t    out      = 0;
		bool        ok       = false;
		bool        compute  = true;
		uint8_t     decision = 0;
		switch (node.kind) {
			case FastSrtPlan::Kind::Undef: compute = false; break;
			case FastSrtPlan::Kind::Select: {
				compute = false;
				if (!current(node.args[0])) return {index};
				const auto& condition = state[node.args[0]];
				if (condition.valid != 0u) {
					decision = condition.value != 0u ? 1u : 2u;
					if (decision != step.decision || !current(node.args[decision])) return {index};
					ok  = state[node.args[decision]].valid != 0u;
					out = state[node.args[decision]].value;
				}
				break;
			}
			case FastSrtPlan::Kind::LogicalAnd:
			case FastSrtPlan::Kind::LogicalOr: {
				compute           = false;
				const bool is_and = node.kind == FastSrtPlan::Kind::LogicalAnd;
				if (!current(node.args[0])) return {index};
				const bool left = state[node.args[0]].valid != 0u;
				const auto a    = state[node.args[0]].value;
				if (left && (is_and ? a == 0u : a != 0u)) {
					decision = 1u;
					ok       = true;
					out      = is_and ? 0u : 1u;
				} else {
					decision = 2u;
					if (decision != step.decision || !current(node.args[1])) return {index};
					const auto& right = state[node.args[1]];
					const auto  b     = right.value;
					ok  = right.valid != 0u && !(is_and ? (b != 0u && !left) : (b == 0u && !left));
					out = b != 0u;
				}
				break;
			}
			case FastSrtPlan::Kind::RawVector: {
				if (!current(node.args[5])) return {index};
				const auto& guard = state[node.args[5]];
				if (guard.valid == 0u) {
					compute = false;
				} else if (guard.value == 0u) {
					decision = 1u;
					compute  = false;
					ok       = true;
				} else {
					decision = 2u;
				}
				break;
			}
			default: break;
		}
		// The node stays unevaluated, so the demand walk evaluates it its own way.
		if (decision != step.decision) return {index};
		if (compute) {
			for (uint32_t arg = 0; arg < 5u && compute; ++arg) {
				if ((node.used & (1u << arg)) == 0u) continue;
				if (!current(node.args[arg])) return {index};
				compute = state[node.args[arg]].valid != 0u;
			}
			// The walk reads only after every arg evaluated.
			ok = compute && ComputeFastNode(plan, step.node, out);
		}
		auto& result = state[step.node];
		result.stamp = generation;
		result.value = ok ? out : 0u;
		result.valid = ok ? 1u : 0u;
		if (result.valid != step.valid) {
			return {index, true, step.node, static_cast<uint8_t>(result.valid), decision};
		}
	}
	return {trace.steps.size()};
}

void SrtWalker::FinishFastTrace(const FastSrtPlan& plan, const FastSrtTrace& trace,
                                std::vector<uint32_t>& flat) {
	auto& active = m_scratch.active_sources;
	if (!plan.control_flow) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
	} else {
		flat.assign(m_program.srt_reads.size(), 0u);
		active.assign(trace.active.begin(), trace.active.end());
	}
	for (const auto [node, offset]: trace.flat) {
		flat[offset] = static_cast<uint32_t>(m_fast->state[node].value);
	}
}

// SettleReachability over the plan's compiled conditions: assume every block reachable, walk,
// and repeat with the blocks found until they stop shrinking, retrying only unknown values.
void SrtWalker::SettleFastReachability(FastSrtPlan& plan) {
	auto& visited   = m_scratch.visited_blocks;
	auto& reachable = m_scratch.reachable_blocks;
	auto& pending   = m_scratch.pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) m_clean_evaluator->m_reachable_blocks = &visited;
	const bool evaluate = m_runtime.read_specialization_memory != nullptr ||
	                      !m_program.capture_specialization_reads;
	auto* state = m_fast->state.data();
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index]  = 1u;
			const auto& block = m_program.control_flow[index];
			const auto  node  = plan.blocks[index].condition;
			if (!block.condition.IsEmpty() && evaluate && DemandFast(plan, node)) {
				pending.push_back(block.successors[state[node].value != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		// Known values stay true as the possible predecessors shrink; unknowns are retried.
		for (size_t index = 0; index < plan.nodes.size(); ++index) {
			if (state[index].stamp == m_fast->generation && state[index].valid == 0u) {
				state[index].stamp = 0u;
			}
		}
	}
}

bool SrtWalker::WalkFastPlan(FastSrtPlan& plan, std::vector<uint32_t>& flat) {
	const auto refresh = [&](uint32_t slot) {
		if (slot >= plan.slot_nodes.size()) return false;
		const auto node   = plan.slot_nodes[slot];
		const auto offset = m_program.srt_reads[slot].flat_offset;
		if (offset >= flat.size() || !DemandFast(plan, node)) return false;
		flat[offset] = static_cast<uint32_t>(m_fast->state[node].value);
		if (m_recording != nullptr) m_recording->flat.emplace_back(node, offset);
		return true;
	};
	auto& active = m_scratch.active_sources;
	if (!plan.control_flow) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
	} else if (plan.ir_reachability) {
		flat.assign(m_program.srt_reads.size(), 0u);
		active.assign(plan.initial_active.begin(), plan.initial_active.end());
		if (plan.fast_reachability) {
			SettleFastReachability(plan);
		} else {
			SettleReachability();
		}
		const auto& visited = m_scratch.visited_blocks;
		for (uint32_t index = 0; index < visited.size(); ++index) {
			if (!visited[index]) continue;
			const auto& block = m_program.control_flow[index];
			for (const auto source: block.sources) active[source] = 1u;
			for (const auto slot: block.srt_reads) {
				if (!refresh(slot)) return false;
			}
		}
	} else {
		// Same block walk as RefreshFlatBufferSlow.
		flat.assign(m_program.srt_reads.size(), 0u);
		active.assign(plan.initial_active.begin(), plan.initial_active.end());
		auto& visited = m_scratch.visited_blocks;
		auto& pending = m_scratch.pending_blocks;
		visited.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (visited.at(index)) continue;
			visited[index] = 1u;
			const auto& block = m_program.control_flow[index];
			for (const auto source: block.sources) active[source] = 1u;
			for (const auto slot: block.srt_reads) {
				if (!refresh(slot)) return false;
			}
			uint32_t condition = 0;
			bool     decided   = false;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr ||
			     !m_program.capture_specialization_reads)) {
				const auto node = plan.blocks[index].condition;
				if (node == UINT32_MAX) {
					auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
					decided         = predicate.Evaluate(block.condition, condition);
				} else {
					decided   = DemandFast(plan, node);
					condition = static_cast<uint32_t>(m_fast->state[node].value);
					if (m_recording != nullptr && m_skip_conditions != 0u) {
						m_skip_conditions--;
					} else if (m_recording != nullptr) {
						m_recording->steps.push_back(
						    {node, 1u, static_cast<uint8_t>(decided ? 1u : 0u),
						     static_cast<uint8_t>(!decided ? 0u : condition != 0u ? 1u : 2u)});
					}
				}
			}
			if (decided) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
	}
	// Live descriptors are evaluated next through the ordinary walker; compute their words now
	// in the same order, stopping where EvaluateDescriptor would.
	for (uint32_t source = 0; source < plan.source_nodes.size(); ++source) {
		if (!active.empty() && (source >= active.size() || active[source] == 0u)) continue;
		for (const auto node: plan.source_nodes[source]) {
			if (!DemandFast(plan, node)) break;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBufferFast(std::vector<uint32_t>& flat, bool& handled) {
	handled = false;
	if (!m_program.fast_srt.checked.load(std::memory_order_acquire)) {
		static std::mutex build_mutex;
		std::scoped_lock  lock(build_mutex);
		if (!m_program.fast_srt.checked.load(std::memory_order_relaxed)) {
			FastSrtCompiler compiler(m_program);
			m_program.fast_srt.plan = compiler.Build();
			m_program.fast_srt.checked.store(true, std::memory_order_release);
		}
	}
	auto* plan = m_program.fast_srt.plan.get();
	if (plan == nullptr) {
		return false;
	}
	if (m_scratch.fast_plan != plan) {
		m_scratch.fast_plan       = plan;
		m_scratch.fast.state      = plan->initial_state;
		m_scratch.fast.generation = 0;
		m_scratch.fast.traces     = {};
	}
	m_fast = &m_scratch.fast;
	// The plan assumes MaterializeResources' fresh walker pair.
	if (!m_active_mask.IsEmpty() || plan->has_clean != (m_clean_evaluator != nullptr) ||
	    (m_clean_evaluator != nullptr && !m_clean_evaluator->m_active_mask.IsEmpty()) ||
	    m_evaluated || (m_clean_evaluator != nullptr && m_clean_evaluator->m_evaluated)) {
		return false;
	}
	handled = true;

	if (++m_fast->generation == FastSrtPlan::AlwaysValid) {
		for (auto& state: m_fast->state) {
			if (state.stamp != FastSrtPlan::AlwaysValid) state.stamp = 0u;
		}
		m_fast->generation = 1;
	}
	m_mapped_begin   = 0;
	m_mapped_end     = 0;
	m_readable_begin = 0;
	m_readable_end   = 0;
	// Each step evaluates at most once: a trace that stops passes its evaluated steps on to
	// another trace that recorded the same ones, or else to the demand walk.
	const bool          strict   = m_runtime.read_specialization_memory != nullptr;
	auto&               traces   = m_fast->traces;
	const FastSrtTrace* prefix   = nullptr; // holds the steps evaluated so far
	size_t              replayed = 0;
	ReplayStop          stop {};
	bool                done = false;
	for (size_t index = 0; index < traces.size() && !done; ++index) {
		const auto& trace = traces[index];
		if (!trace.ready || trace.strict != strict) continue;
		size_t first = 0;
		if (prefix != nullptr) {
			first = replayed + (stop.evaluated ? 1u : 0u);
			if (trace.steps.size() < first ||
			    !std::equal(prefix->steps.begin(), prefix->steps.begin() + replayed,
			                trace.steps.begin()) ||
			    (stop.evaluated &&
			     trace.steps[replayed] != FastSrtTrace::Step {stop.node, 0u, stop.valid, stop.decision})) {
				continue;
			}
		}
		stop = ReplayFastTrace(*plan, trace, first);
		if (stop.steps == trace.steps.size()) {
			FinishFastTrace(*plan, trace, flat);
			std::rotate(traces.begin(), traces.begin() + index, traces.begin() + index + 1);
			done = true;
		} else {
			prefix   = &trace;
			replayed = stop.steps;
		}
	}
	if (!done) {
		// The demand walk continues from the evaluated steps, which it finds memoized, and records
		// the whole order over the least recently used trace.
		std::vector<FastSrtTrace::Step> steps;
		if (prefix != nullptr) {
			steps.assign(prefix->steps.begin(), prefix->steps.begin() + replayed);
			if (stop.evaluated) {
				steps.push_back({stop.node, 0u, stop.valid, stop.decision});
			}
		}
		auto& trace = traces.back();
		trace.ready = false;
		trace.flat.clear();
		trace.steps = std::move(steps);
		m_skip_conditions = static_cast<size_t>(std::ranges::count_if(
		    trace.steps, [](const FastSrtTrace::Step& step) { return step.condition != 0u; }));
		m_recording       = plan->replayable ? &trace : nullptr;
		const bool walked = WalkFastPlan(*plan, flat);
		m_recording       = nullptr;
		m_skip_conditions = 0;
		if (!walked) {
			return false;
		}
		if (plan->replayable) {
			trace.ready  = true;
			trace.strict = strict;
			trace.active.assign(m_scratch.active_sources.begin(), m_scratch.active_sources.end());
			std::rotate(traces.begin(), traces.end() - 1, traces.end());
		}
	}
	if (m_context.values.size() < m_program.evaluation_value_count) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	auto* clean_context = m_clean_evaluator != nullptr ? &m_clean_evaluator->m_context : nullptr;
	if (clean_context != nullptr && clean_context->values.size() < m_program.evaluation_value_count) {
		clean_context->values.resize(m_program.evaluation_value_count);
	}
	// The ordinary walker's descriptors are read straight from the plan (TryFastDescriptor). Seed
	// the clean walker, whose fill and captured-descriptor evaluation follows, with what it shares.
	m_fast_generation = m_fast->generation;
	if (clean_context != nullptr) {
		for (const auto [index, memo]: plan->clean_seeds) {
			const auto& state = m_fast->state[index];
			if (state.stamp != m_fast->generation) continue;
			if (state.valid != 0) {
				clean_context->values[memo] = {state.value, clean_context->generation};
				continue;
			}
			// A failed read is a completed unknown, as in the IR walk: it is not read again.
			const auto kind = plan->nodes[index].kind;
			if (kind == FastSrtPlan::Kind::RawAddress || kind == FastSrtPlan::Kind::RawConstBuffer ||
			    kind == FastSrtPlan::Kind::RawVector) {
				clean_context->values[memo] = {clean_context->unknown_epoch,
				                               clean_context->generation | 2u};
			}
		}
	}
	return true;
}

SrtWalker::FastDescriptor SrtWalker::TryFastDescriptor(uint32_t source, DescriptorValue& result) {
	auto* plan = m_program.fast_srt.plan.get();
	if (plan == nullptr || m_fast == nullptr || m_fast_generation == 0 ||
	    m_fast_generation != m_fast->generation ||
	    source >= plan->source_nodes.size() || source >= m_program.descriptor_sources.size()) {
		return FastDescriptor::Unknown;
	}
	const auto& nodes      = plan->source_nodes[source];
	const auto& descriptor = m_program.descriptor_sources[source];
	if (nodes.size() != descriptor.dword_count || nodes.empty()) return FastDescriptor::Unknown;
	DescriptorValue value {};
	value.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		const auto& state = m_fast->state[nodes[index]];
		if (state.stamp != m_fast->generation && state.stamp != FastSrtPlan::AlwaysValid) {
			return FastDescriptor::Unknown;
		}
		// The ordinary walk would fail on this word too, after reading it again.
		if (state.valid == 0) return FastDescriptor::Failed;
		value.dwords[index] = static_cast<uint32_t>(state.value);
	}
	result = value;
	return FastDescriptor::Ready;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	// The compiled walker copies the tables directly and seeds the evaluation memo with every
	// value it read: it goes first whenever reads need no callback.
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		return RefreshFlatBufferSlow(flat);
	}
	bool       handled   = false;
	const bool refreshed = RefreshFlatBufferFast(flat, handled);
	return handled ? refreshed : RefreshFlatBufferSlow(flat);
}

void SrtWalker::SettleReachability() {
	auto& visited = m_scratch.visited_blocks;
	auto& reachable = m_scratch.reachable_blocks;
	auto& pending = m_scratch.pending_blocks;
	visited.assign(m_program.control_flow.size(), 1u);
	ResetUnknowns();
	m_reachable_blocks = &visited;
	if (m_clean_evaluator != nullptr) {
		m_clean_evaluator->ResetUnknowns();
		m_clean_evaluator->m_reachable_blocks = &visited;
	}
	for (;;) {
		reachable.assign(m_program.control_flow.size(), 0u);
		pending.clear();
		pending.push_back(0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (reachable.at(index)) continue;
			reachable[index] = 1u;
			const auto& block = m_program.control_flow[index];
			uint32_t condition = 0;
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!block.condition.IsEmpty() &&
			    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads) &&
			    predicate.Evaluate(block.condition, condition)) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
		if (reachable == visited) break;
		visited.swap(reachable);
		if (m_program.control_phi_predecessors.empty()) break;
		// Only unknown values need another evaluation: successful Phi facts remain
		// true as their possible predecessors shrink. Keep uniform-read memos stable.
		ResetUnknowns();
		if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
	}
}

bool SrtWalker::RefreshFlatBufferSlow(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	if (m_program.srt_code && m_runtime.read_memory == nullptr) {
		flat.resize(m_program.srt_reads.size());
		m_scratch.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		return m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		    m_context.values.data(), m_context.generation, m_runtime.user_data.size());
	}
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
	};
	auto& active = m_scratch.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	SettleReachability();
	const auto& visited = m_scratch.visited_blocks;
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources) active.at(source) = 0u;
	for (uint32_t index = 0; index < visited.size(); ++index) {
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		for (const auto slot: block.srt_reads)
			if (!refresh(slot)) return false;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
