#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// End of the range from vaddr, within its tracking region, that has no GPU-modified page and
	// so reads without faulting; vaddr when its own page is GPU-modified.
	[[nodiscard]] uint64_t GpuCleanEnd(uint64_t vaddr) {
		const auto index      = vaddr / TRACKER_REGION_SIZE;
		const auto region_end = (index + 1) * TRACKER_REGION_SIZE;
		if (!GuestRange {vaddr, 1}.Valid() || region_end > TRACKER_ADDRESS_SIZE) {
			return vaddr;
		}
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr || !manager->MayHaveGpuModified()) {
			return region_end;
		}
		std::scoped_lock lock(manager->lock);
		return manager->GpuCleanEnd(vaddr);
	}
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Advances whenever a range can have become CPU-dirty, so callers can tell whether a full
	// upload pass may find anything new.
	[[nodiscard]] uint64_t CpuDirtyGeneration() const noexcept {
		return m_cpu_dirty_generation.load(std::memory_order_acquire);
	}
	void BumpCpuDirtyGeneration() noexcept {
		m_cpu_dirty_generation.fetch_add(1, std::memory_order_acq_rel);
	}
	// Records a range whose pages can have become CPU-dirty, then advances the generation. Called
	// after the pages change state, so a caller that reads the generation and then takes the
	// ranges sees every page that generation covers.
	void NoteCpuDirty(uint64_t vaddr, uint64_t size) noexcept {
		{
			std::scoped_lock lock(m_noted_lock);
			const auto end = vaddr + size;
			if (!m_noted.empty() && vaddr <= m_noted.back().second &&
			    end >= m_noted.back().first) {
				m_noted.back().first  = std::min(m_noted.back().first, vaddr);
				m_noted.back().second = std::max(m_noted.back().second, end);
			} else if (m_noted.size() < MAX_NOTED_RANGES) {
				m_noted.emplace_back(vaddr, end);
			} else {
				m_noted_overflow = true;
			}
		}
		BumpCpuDirtyGeneration();
	}
	// Moves out the [begin, end) ranges noted since the last call. Returns false when more were
	// noted than could be listed; the caller then has to treat all memory as possibly dirty.
	bool TakeCpuDirtyRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges) noexcept {
		ranges.clear();
		std::scoped_lock lock(m_noted_lock);
		ranges.swap(m_noted);
		return !std::exchange(m_noted_overflow, false);
	}
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
		NoteCpuDirty(vaddr, size);
	}
	// InvalidateRegion for a CPU write fault on vaddr's page. Guest code that writes one page of
	// a buffer usually writes its neighbours next, often from several threads at once, and each
	// fault is a host exception that the OS dispatches under a process-wide lock. Unless the page
	// needs a flush, the clean pages of its aligned block are unprotected with it; any of them
	// the CPU does not write only costs a redundant upload.
	template <typename Flush>
	void InvalidateWriteFault(uint64_t vaddr, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, 1, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeCpuStateForFault(manager->GetCpuAddr() + offset);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
		constexpr uint64_t block_size = TRACKER_FAULT_BLOCK_PAGES * TRACKER_PAGE_SIZE;
		NoteCpuDirty(vaddr & ~(block_size - 1), block_size);
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;
	std::atomic<uint64_t> m_cpu_dirty_generation = 0;
	// Ranges noted since the last TakeCpuDirtyRanges; past MAX_NOTED_RANGES only the overflow
	// flag is kept.
	static constexpr size_t                     MAX_NOTED_RANGES = 4096;
	TrackingSpinLock                            m_noted_lock;
	std::vector<std::pair<uint64_t, uint64_t>> m_noted;
	bool                                        m_noted_overflow = false;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
