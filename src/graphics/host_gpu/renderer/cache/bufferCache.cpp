#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h" // PERF-INSTR
#include "common/perfCounters.h" // PERF-INSTR
#include <cstdio> // PERF-INSTR
#include <map> // PERF-INSTR
#include <unordered_map> // PERF-INSTR

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <windows.h> // PERF-INSTR
#include <chrono> // PERF-INSTR
#include <cstdio> // PERF-INSTR
namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

// Copies large uploads into staging on several cores. Some scenes rewrite tens of megabytes of
// guest memory per frame, and copying that on the GPU thread alone costs it milliseconds.
class BufferCache::CopyWorkers {
public:
	struct Copy {
		uint8_t*       destination;
		const uint8_t* source;
		uint64_t       size;
	};

	CopyWorkers() {
		// The copies are bound by memory bandwidth, which a few cores saturate.
		const auto hardware = std::max(1u, std::thread::hardware_concurrency());
		const auto count    = std::min(hardware / 4u, 3u);
		m_threads.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~CopyWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(CopyWorkers);

	// The sources are CPU-dirty guest pages, which are never read-protected, so no copy faults
	// into a handler that would need the waiting GPU thread.
	void CopyAll(std::span<const Copy> copies) {
		uint64_t total = 0;
		for (const auto& copy: copies) {
			total += copy.size;
		}
		if (m_threads.empty() || total < ParallelBytes) {
			for (const auto& copy: copies) {
				std::memcpy(copy.destination, copy.source, copy.size);
			}
			return;
		}
		auto batch = std::make_shared<Batch>();
		for (const auto& copy: copies) {
			for (uint64_t offset = 0; offset < copy.size; offset += PieceBytes) {
				batch->pieces.push_back({copy.destination + offset, copy.source + offset,
				                         std::min(PieceBytes, copy.size - offset)});
			}
		}
		Dispatch(batch);
	}

	// Copies guest memory like LibKernel::Memory::CopySparseMemory (resident mappings only),
	// large uploads on several cores. False when a range cannot be read.
	[[nodiscard]] bool CopySparse(std::span<const Copy> copies) {
		uint64_t total = 0;
		for (const auto& copy: copies) {
			total += copy.size;
		}
		if (m_threads.empty() || total < ParallelBytes) {
			for (const auto& copy: copies) {
				if (!Libs::LibKernel::Memory::CopySparseMemory(
				        reinterpret_cast<uint64_t>(copy.source), copy.destination, copy.size)) {
					return false;
				}
			}
			return true;
		}
		auto batch  = std::make_shared<Batch>();
		batch->mode = Mode::Sparse;
		for (const auto& copy: copies) {
			for (uint64_t offset = 0; offset < copy.size; offset += PieceBytes) {
				batch->pieces.push_back({copy.destination + offset, copy.source + offset,
				                         std::min(PieceBytes, copy.size - offset)});
			}
		}
		Dispatch(batch);
		return !batch->failed.load(std::memory_order_acquire);
	}

	// Reads guest backing like LibKernel::Memory::TryReadSparseBacking, large ranges on several
	// cores. Backing reads go to the backing store, not the guest pages, so protection does not
	// matter.
	[[nodiscard]] bool ReadSparseBacking(uint64_t vaddr, uint8_t* destination, uint64_t size) {
		if (m_threads.empty() || size < ParallelBytes) {
			return Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, destination, size);
		}
		auto batch  = std::make_shared<Batch>();
		batch->mode = Mode::Backing;
		for (uint64_t offset = 0; offset < size; offset += PieceBytes) {
			batch->pieces.push_back({destination + offset,
			                         reinterpret_cast<const uint8_t*>(vaddr + offset),
			                         std::min(PieceBytes, size - offset)});
		}
		Dispatch(batch);
		return !batch->failed.load(std::memory_order_acquire);
	}

private:
	static constexpr uint64_t ParallelBytes = MiB;
	static constexpr uint64_t PieceBytes    = 256 * 1024;

	// A worker that wakes after its batch finished still holds that batch, never the next one.
	// Plain copies, backing reads (TryReadSparseBacking) or guest copies (CopySparseMemory).
	enum class Mode { Plain, Backing, Sparse };
	struct Batch {
		std::vector<Copy>   pieces;
		Mode                mode = Mode::Plain;
		std::atomic<bool>   failed {false};
		std::atomic<size_t> next {0};
		std::atomic<size_t> done {0};
	};

	// Hands the batch to the workers, helps, and returns once every piece is done.
	void Dispatch(const std::shared_ptr<Batch>& batch) {
		{
			std::lock_guard lock(m_mutex);
			m_batch = batch;
			m_generation++;
		}
		m_wake.notify_all();
		Work(*batch);
		const auto count = batch->pieces.size();
		for (auto done = batch->done.load(std::memory_order_acquire); done != count;
		     done      = batch->done.load(std::memory_order_acquire)) {
			batch->done.wait(done, std::memory_order_acquire);
		}
	}

	static void Work(Batch& batch) {
		const auto count = batch.pieces.size();
		for (auto index = batch.next.fetch_add(1); index < count; index = batch.next.fetch_add(1)) {
			const auto& piece = batch.pieces[index];
			const auto source = reinterpret_cast<uint64_t>(piece.source);
			if (batch.mode == Mode::Plain) {
				std::memcpy(piece.destination, piece.source, piece.size);
			} else if (!(batch.mode == Mode::Backing
			                 ? Libs::LibKernel::Memory::TryReadSparseBacking(source, piece.destination,
			                                                                 piece.size)
			                 : Libs::LibKernel::Memory::CopySparseMemory(source, piece.destination,
			                                                             piece.size))) {
				batch.failed.store(true, std::memory_order_release);
			}
			if (batch.done.fetch_add(1, std::memory_order_acq_rel) + 1 == count) {
				batch.done.notify_all();
			}
		}
	}

	void Run() {
		uint64_t seen = 0;
		for (;;) {
			std::shared_ptr<Batch> batch;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [&] { return m_stop || m_generation != seen; });
				if (m_stop) {
					return;
				}
				seen  = m_generation;
				batch = m_batch;
			}
			Work(*batch);
		}
	}

	std::mutex               m_mutex;
	std::condition_variable  m_wake;
	std::shared_ptr<Batch>   m_batch;
	uint64_t                 m_generation = 0;
	bool                     m_stop       = false;
	std::vector<std::thread> m_threads;
};

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_copy_workers(std::make_unique<CopyWorkers>()), m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::InvalidateWriteFault(uint64_t vaddr) {
	if (!GuestRange {vaddr, 1}.Valid()) {
		EXIT("BufferCache: invalid write-fault address\n");
	}
	m_memory_tracker.InvalidateWriteFault(vaddr, [this, vaddr] { ReadMemory(vaddr, 1, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (static const double after = [] { const char* t = std::getenv("KYTY_DL_LOG"); return t ? std::atof(t) : -1.0; }(); after >= 0.0) { // PERF-INSTR
		static const auto start = std::chrono::steady_clock::now(); // PERF-INSTR
		static std::atomic<int> logged {0}; // PERF-INSTR
		if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= after && logged.fetch_add(1) < 400) { // PERF-INSTR
			void* frames[10] {}; // PERF-INSTR
			const auto count = RtlCaptureStackBackTrace(0, 10, frames, nullptr); // PERF-INSTR
			const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr)); // PERF-INSTR
			const int near16 = GuestGpu::IsGpuThread() ? (m_gpu_modified_ranges.Intersects(vaddr & ~uint64_t {15}, 32) ? 1 : 0) : -1; // PERF-INSTR
			const int page = GuestGpu::IsGpuThread() ? (m_gpu_modified_ranges.Intersects(vaddr & ~uint64_t {TRACKER_PAGE_SIZE - 1}, TRACKER_PAGE_SIZE) ? 1 : 0) : -1; // PERF-INSTR
			std::fprintf(stderr, "DL addr=%llx size=%llx write=%d gpu=%d near=%d page=%d queued=%d stack:", (unsigned long long)vaddr, (unsigned long long)size, is_write ? 1 : 0, GuestGpu::IsGpuThread() ? 1 : 0, near16, page, GuestGpu::IsGpuThread() ? (HasQueuedDownload(vaddr, 1) ? 1 : 0) : -1); // PERF-INSTR
			for (USHORT f = 0; f < count; f++) std::fprintf(stderr, " %llx", (unsigned long long)(reinterpret_cast<uintptr_t>(frames[f]) - base)); // PERF-INSTR
			std::fprintf(stderr, "\n"); // PERF-INSTR
		} // PERF-INSTR
	} // PERF-INSTR
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		Common::Perf::Add(Common::Perf::SyncDownloads); // PERF-INSTR
		Common::Perf::Add(Common::Perf::SyncDownloadBytes, size); // PERF-INSTR
		Common::Perf::ScopedTimer perf_timer(Common::Perf::SyncDownloadNs); // PERF-INSTR
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		NoteCpuReadBack(vaddr, size);
		if (WaitQueuedDownloads(vaddr, size)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		} else {
			auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
			if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

void BufferCache::NoteCpuReadBack(uint64_t vaddr, uint64_t size) {
	// Few pages are read back each frame; keep the most recent ones.
	constexpr size_t MaxReadBackPages = 64;
	const auto       tick             = m_scheduler.CurrentTick();
	for (auto page = vaddr / TRACKER_PAGE_SIZE; page <= (vaddr + size - 1) / TRACKER_PAGE_SIZE;
	     page++) {
		m_read_back_pages[page] = tick;
	}
	while (m_read_back_pages.size() > MaxReadBackPages) {
		m_read_back_pages.erase(std::ranges::min_element(m_read_back_pages, {}, [](const auto& entry) {
			                        return entry.second;
		                        }));
	}
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	if (m_read_back_pages.empty()) {
		return;
	}
	const auto first = vaddr / TRACKER_PAGE_SIZE;
	const auto last  = (vaddr + size - 1) / TRACKER_PAGE_SIZE;
	if (last - first < m_read_back_pages.size()) {
		for (auto page = first; page <= last; page++) {
			if (m_read_back_pages.contains(page)) {
				m_written_read_back_pages.push_back(page);
			}
		}
		return;
	}
	for (const auto& [page, tick]: m_read_back_pages) {
		if (page >= first && page <= last) {
			m_written_read_back_pages.push_back(page);
		}
	}
}

bool BufferCache::HasQueuedDownload(uint64_t vaddr, uint64_t size) const {
	return !m_queued_download_ranges.Empty() && m_queued_download_ranges.Intersects(vaddr, size);
}

bool BufferCache::WaitQueuedDownloads(uint64_t vaddr, uint64_t size) {
	if (m_queued_downloads.empty()) {
		return false;
	}
	const auto first = vaddr / TRACKER_PAGE_SIZE;
	const auto last  = (vaddr + size - 1) / TRACKER_PAGE_SIZE;
	uint64_t   tick  = 0;
	for (auto page = first; page <= last; page++) {
		if (const auto it = m_queued_downloads.find(page); it != m_queued_downloads.end()) {
			// Bytes written after the download was queued still need the full readback.
			if (m_gpu_modified_ranges.Intersects(page * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE)) {
				return false;
			}
			tick = std::max(tick, it->second);
		}
	}
	if (tick == 0) {
		return false;
	}
	for (auto page = first; page <= last; page++) {
		m_queued_downloads.erase(page);
	}
	m_queued_download_ranges.Subtract(first * TRACKER_PAGE_SIZE,
	                                  (last - first + 1) * TRACKER_PAGE_SIZE);
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	return true;
}

void BufferCache::BeforeSubmit() {
	// Downloads queued by earlier submissions: once published, the CPU may read those pages
	// directly, unless the GPU has written them again since.
	for (auto it = m_queued_downloads.begin(); it != m_queued_downloads.end();) {
		const auto [page, tick] = *it;
		if (!m_scheduler.IsFree(tick)) {
			++it;
			continue;
		}
		m_scheduler.WaitPriorityOperations(tick);
		const auto address = page * TRACKER_PAGE_SIZE;
		if (!m_gpu_modified_ranges.Intersects(address, TRACKER_PAGE_SIZE)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(address, TRACKER_PAGE_SIZE);
		}
		m_queued_download_ranges.Subtract(address, TRACKER_PAGE_SIZE);
		it = m_queued_downloads.erase(it);
	}
	if (m_written_read_back_pages.empty()) {
		return;
	}
	std::ranges::sort(m_written_read_back_pages);
	const auto [unique_end, end] = std::ranges::unique(m_written_read_back_pages);
	m_written_read_back_pages.erase(unique_end, end);
	const auto tick = m_scheduler.CurrentTick();
	for (const auto page: m_written_read_back_pages) {
		const auto address = page * TRACKER_PAGE_SIZE;
		// Buffers cover whole caching pages, so one buffer owns the tracker page.
		const auto* owner = m_page_table.Find(address >> CACHING_PAGEBITS);
		if (owner == nullptr || !*owner || IsBufferInvalid(*owner) ||
		    !m_slot_buffers[*owner].IsInBounds(address, TRACKER_PAGE_SIZE)) {
			continue;
		}
		m_gpu_modified_ranges.ForEachInRange(address, TRACKER_PAGE_SIZE, [&](uint64_t begin, uint64_t end) {
			m_queued_download_ranges.Add(begin, end - begin);
		});
		if (DownloadBufferMemory<true>(m_slot_buffers[*owner], address, TRACKER_PAGE_SIZE)) {
			m_queued_downloads[page] = tick;
		}
	}
	m_written_read_back_pages.clear();
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	// The new buffer may cover CPU-dirty pages that an earlier upload pass found no buffer for.
	m_memory_tracker.NoteCpuDirty(overlap.begin, overlap.end - overlap.begin);
	return id;
}

void BufferCache::DrainCpuDirtyRanges() {
	// Ranges are noted before the generation advances, so reading it first misses none.
	const auto generation = m_memory_tracker.CpuDirtyGeneration();
	if (generation == m_drained_generation) {
		return;
	}
	m_drained_generation = generation;
	if (!m_memory_tracker.TakeCpuDirtyRanges(m_noted_ranges)) {
		m_noted_ranges.clear();
		for (const auto& [address, id]: m_buffers) {
			const auto& buffer = m_slot_buffers[id];
			m_noted_ranges.emplace_back(buffer.CpuAddress(), buffer.CpuAddress() + buffer.Size());
		}
	}
	for (const auto [begin, end]: m_noted_ranges) {
		m_maybe_cpu_dirty.Add(begin, end - begin);
	}
	if (m_sweeping) {
		m_swept_notes.insert(m_swept_notes.end(), m_noted_ranges.begin(), m_noted_ranges.end());
	}
}

void BufferCache::SynchronizeDirtyBuffers(const RangeSet& within) {
	DrainCpuDirtyRanges();
	std::vector<std::pair<uint64_t, uint64_t>> swept;
	m_maybe_cpu_dirty.ForEach([&](uint64_t begin, uint64_t end) { swept.emplace_back(begin, end); });
	m_swept_notes.clear();
	m_sweeping = true;
	for (const auto [begin, end]: swept) {
		within.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t last) {
			SynchronizeBuffersInRange(start, last - start);
		});
	}
	m_sweeping = false;
	// The rest of the swept ranges holds no buffer or is unmapped; a buffer created there notes its
	// range again. Ranges noted during the sweep stay.
	for (const auto [begin, end]: swept) {
		m_maybe_cpu_dirty.Subtract(begin, end - begin);
	}
	for (const auto [begin, end]: m_swept_notes) {
		m_maybe_cpu_dirty.Add(begin, end - begin);
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	DrainCpuDirtyRanges();
	static const bool exp_nofast = std::getenv("KYTY_EXP_NOFAST") != nullptr; // PERF-INSTR
	if (!exp_nofast && !is_written && size != 0 && !m_maybe_cpu_dirty.Intersects(vaddr, size)) {
		return is_texel_buffer && SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (size != 0) {
		m_maybe_cpu_dirty.Subtract(vaddr, size);
	}
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		GpuProfNote(GpuProfUpload); // PERF-INSTR
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		std::vector<CopyWorkers::Copy> host_copies;
		host_copies.reserve(copies.size());
		for (auto& copy: copies) {
			host_copies.push_back({mapped + copy.srcOffset,
			                       reinterpret_cast<const uint8_t*>(buffer.CpuAddress() + copy.dstOffset),
			                       copy.size});
			copy.srcOffset += base_offset;
		}
		if (!m_copy_workers->CopySparse(host_copies)) {
			EXIT("BufferCache: failed to read sparse guest buffer memory\n");
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	DrainCpuDirtyRanges();
	if (!is_written && size <= CACHING_PAGESIZE && m_maybe_cpu_dirty.Intersects(vaddr, size) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		NoteGpuWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!m_copy_workers->ReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// A queued download has not necessarily published its bytes yet.
	return m_gpu_modified_ranges.Intersects(vaddr, size) || HasQueuedDownload(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
