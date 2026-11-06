// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
/// EDEN_NX2_FENCE_LIGHT: keeps the blocks of released fences (object and shared count, one
/// allocation) for the next ones instead of returning them to the heap. Fences are made on the
/// GPU thread and released wherever their last reference goes (the fencing thread's ring).
/// The free list is never destroyed: a fence may outlive static destruction at exit.
template <typename T>
class FencePoolAllocator {
public:
    using value_type = T;

    FencePoolAllocator() noexcept = default;
    template <typename U>
    FencePoolAllocator(const FencePoolAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        if (count == 1) {
            FreeList& list = List();
            std::scoped_lock lock{list.mutex};
            if (!list.blocks.empty()) {
                void* const block = list.blocks.back();
                list.blocks.pop_back();
                return static_cast<T*>(block);
            }
        }
        return std::allocator<T>{}.allocate(count);
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        if (count == 1) {
            FreeList& list = List();
            std::scoped_lock lock{list.mutex};
            if (list.blocks.size() < MaxBlocks) {
                list.blocks.push_back(pointer);
                return;
            }
        }
        std::allocator<T>{}.deallocate(pointer, count);
    }

    template <typename U>
    bool operator==(const FencePoolAllocator<U>&) const noexcept {
        return true;
    }

private:
    static constexpr std::size_t MaxBlocks = 4096;

    struct FreeList {
        std::mutex mutex;
        std::vector<void*> blocks;
    };

    static FreeList& List() {
        static FreeList* const list = [] {
            auto* const created = new FreeList;
            created->blocks.reserve(MaxBlocks);
            return created;
        }();
        return *list;
    }
};
} // Anonymous namespace

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue() {
    if (is_stubbed) {
        return;
    }
    // Get the current tick so we can wait for it
    wait_tick = scheduler.CurrentTick();
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

void InnerFence::Queue(std::chrono::steady_clock::time_point now) {
    if (is_stubbed) {
        return;
    }
    wait_tick = scheduler.CurrentTick();
    if (!scheduler.DeferFlush(now)) {
        scheduler.Flush();
    }
}

bool InnerFence::IsSignaled() const {
    if (is_stubbed) {
        return true;
    }
    return scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    if (is_stubbed) {
        return;
    }
    // A deferred fence flush (Scheduler::DeferFlush) may not be submitted yet. Only the
    // split's back may submit; other threads (the fencing thread) ask it to and wait. If the back
    // has stopped, nothing else uses the scheduler and Wait below submits it.
    if (!scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread() && scheduler.WaitSubmitted(wait_tick)) {
        scheduler.GetMasterSemaphore().Wait(wait_tick);
        return;
    }
    scheduler.Wait(wait_tick);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, const Device& device_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    if (FenceLight()) {
        // A stubbed fence holds no state (no tick, never queued nor waited): one object serves
        // them all. The others reuse the memory of released ones.
        if (is_stubbed) {
            if (!light_stub) {
                light_stub = std::make_shared<InnerFence>(scheduler, true);
            }
            light_stubs_shared.store(light_stubs_shared.load(std::memory_order_relaxed) + 1,
                                     std::memory_order_relaxed);
            return light_stub;
        }
        light_fences_pooled.store(light_fences_pooled.load(std::memory_order_relaxed) + 1,
                                  std::memory_order_relaxed);
        return std::allocate_shared<InnerFence>(FencePoolAllocator<InnerFence>{}, scheduler, false);
    }
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

void FenceManager::QueueFenceAt(Fence& fence, std::chrono::steady_clock::time_point now) {
    fence->Queue(now);
}

bool FenceManager::InFlushWindowAt(std::chrono::steady_clock::time_point now) const {
    return scheduler.InFlushWindow(now);
}

void FenceManager::SubmitDeferredWork() {
    // Only the split's back defers (Scheduler::DeferFlush) and submits.
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.FlushDeferred();
    }
}

std::pair<u64, u64> FenceManager::TakeLightFenceObjects() {
    return {light_stubs_shared.exchange(0, std::memory_order_relaxed),
            light_fences_pooled.exchange(0, std::memory_order_relaxed)};
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

void FenceManager::SubmitDeferredWork() {
    scheduler.FlushDeferred();
}

} // namespace Vulkan
