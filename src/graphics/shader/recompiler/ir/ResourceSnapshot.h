#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;

	bool operator==(const UniformFill&) const = default;
};

// Merges reads given in walk order, mostly consecutive words of the same tables, into runs of
// contiguous or overlapping neighbors, and returns their bounds. Only the union of the reads
// matters to overlap checks. False for an empty or wrapping read.
inline bool MergeReadRuns(const std::vector<std::pair<uint64_t, uint64_t>>& reads,
                          std::vector<std::pair<uint64_t, uint64_t>>& runs, uint64_t& begin,
                          uint64_t& end) {
	runs.clear();
	begin = UINT64_MAX;
	end   = 0;
	for (const auto [address, size]: reads) {
		if (size == 0 || address > UINT64_MAX - size) {
			return false;
		}
		begin = std::min(begin, address);
		end   = std::max(end, address + size);
		if (!runs.empty()) {
			auto& [last_address, last_size] = runs.back();
			if (address >= last_address && address - last_address <= last_size) {
				last_size = std::max(last_size, address - last_address + size);
				continue;
			}
		}
		runs.emplace_back(address, size);
	}
	return true;
}

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	// MergeReadRuns of specialization_reads, when read_runs_valid.
	std::vector<std::pair<uint64_t, uint64_t>> read_runs;
	uint64_t                                    read_begin      = UINT64_MAX;
	uint64_t                                    read_end        = 0;
	bool                                        read_runs_valid = false;
	UniformFill                 uniform_fill;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
