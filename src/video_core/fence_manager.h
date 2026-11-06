// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <queue>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/gpu.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon {

/// EDEN_NX2_FENCE_LIGHT=1 (default off): lighter fences (FenceManager::SignalFenceImpl).
inline bool FenceLightEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_LIGHT");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}

/// Guest command list submissions (GPU::PushGPUEntries), counted only with EDEN_NX2_FENCE_LIGHT:
/// the first fence the GPU thread signals after one gathers the guest's CPU writes at once.
inline std::atomic<u64> fence_light_submissions{0};

class FenceBase {
public:
    explicit FenceBase(bool is_stubbed_) : is_stubbed{is_stubbed_} {}

    bool IsStubbed() const {
        return is_stubbed;
    }

protected:
    bool is_stubbed;
};

template <typename Traits>
class FenceManager {
    using TFence = typename Traits::FenceType;
    using TTextureCache = typename Traits::TextureCacheType;
    using TBufferCache = typename Traits::BufferCacheType;
    using TQueryCache = typename Traits::QueryCacheType;
    static constexpr bool can_async_check = Traits::HAS_ASYNC_CHECK;

public:
    /// Notify the fence manager about a new frame
    void TickFrame() {
        std::unique_lock lock(ring_guard);
        delayed_destruction_ring.Tick();
    }

    // Unlike other fences, this one doesn't
    void SignalOrdering() {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.AccumulateFlushes();
    }

    void SignalReference() {
        std::function<void()> do_nothing([] {});
        SignalFence(std::move(do_nothing));
    }

    void SyncOperation(std::function<void()>&& func) {
        // Not only the GPU thread queues these (query cache work runs elsewhere too), while
        // SignalFence moves the list into pending_operations: unguarded, the list was corrupted
        // and the fencing thread crashed walking it. Its own lock, held for no callbacks.
        std::scoped_lock lock{uncommitted_guard};
        uncommitted_operations.emplace_back(std::move(func));
    }

    void SignalFence(std::function<void()>&& func) {
        SignalFenceImpl(std::move(func), false);
    }

    /// EDEN_NX2_DOWNLOAD_COALESCE=1 (default off): at low GPU accuracy, a fence made on the split's
    /// back while its submission is deferred anyway (Scheduler::DeferFlush, ~500 us windows)
    /// leaves the buffer downloads uncommitted; the next fence past the window commits them all
    /// at once, before that submission. Syncpoint fences, forced waits (WaitPendingFences) and
    /// the back's idle hook (CommitCoalescedDownloads) always commit.
    static bool CoalesceDownloads() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_DOWNLOAD_COALESCE");
            return value != nullptr && value[0] != '0';
        }();
        return enabled;
    }

    /// Commits the buffer downloads fences left for later (CoalesceDownloads) with a fence of
    /// their own. The split's back calls it before it submits deferred work and sleeps.
    void CommitCoalescedDownloads() {
        if (coalesced_downloads) {
            SignalFenceImpl([] {}, true);
        }
    }

    /// EDEN_NX2_FENCE_LIGHT=1 (default off). Switch 2 titles signal a fence per compute dispatch
    /// (3,000-11,000 a second); each one also gathered the guest's CPU writes (one lock per CPU
    /// core), woke the fencing thread (a system call when it slept) and allocated its fence.
    /// With the switch:
    /// - the gather runs at most once per EDEN_NX2_FENCE_LIGHT_US (default 500 us, the deferred
    ///   submission window), and always at a fence that carries downloads, a syncpoint or forced
    ///   wait (commit_downloads), in accurate fence modes, and at the first fence after the
    ///   guest submitted a command list (fence_light_submissions);
    /// - the fencing thread is notified only while it waits on the condition variable;
    /// - one clock read serves the flush window test, the fence's submission and the throttle;
    /// - stubbed fences share one object and the others come from a free list (backend).
    /// Invariant: the gather applies guest CPU writes into GPU-written buffer pages (the only
    /// writes deferred to it, GPU::OnCPUWrite); every other CPU write is applied when made. A
    /// command recorded after the first fence that follows a guest submission sees every write
    /// made before that submission, as before (the guest orders its writes before the list that
    /// reads them). Fences that commit downloads gather as before, and also before the commit
    /// when a gather was skipped since the last one (LightGatherBeforeDownloads). Not kept: a
    /// write ordered only by a semaphore the GPU acquires (no submission) waits for the first
    /// fence past the interval instead of the first fence after the acquire.
    static bool FenceLight() {
        return FenceLightEnabled();
    }

private:
    void SignalFenceImpl(std::function<void()>&& func, bool commit_downloads) {
        Nx2Profile::Scope profile_scope{Nx2Profile::Zone::Fence};
        // The accuracy can change while the game runs (settings dialog, Settings::UpdateGPUAccuracy
        // at each command list end): read once, every decision of this fence uses one value.
        const bool level_high = Settings::IsGPULevelHigh();
        const bool delay_fence = Settings::IsGPUFenceBehaviorDefault() ? level_high : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        const bool light = FenceLight();
        const auto now = light ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool defer_downloads = CoalesceDownloads() && !commit_downloads && !delay_fence &&
                                     !level_high &&
                                     (light ? InFlushWindowAt(now) : InFlushWindow());
        if (light && light_gather_pending) {
            LightGatherBeforeDownloads(now);
        }
        auto [should_flush, committed_flushes] = CommitCacheFlushes(defer_downloads);
        if (!delay_fence && committed_flushes == 0 && SkipEmptyFence()) {
            Nx2Profile::Scope skipped_scope{Nx2Profile::Zone::FenceSkipped};
            func();
            if (!light || LightGatherDue(now, commit_downloads)) {
                rasterizer.InvalidateGPUCache();
            }
            return;
        }
        skipped_fences = 0;
        should_flush |= query_cache.HasUncommittedFlushes();
        query_cache.CommitAsyncFlushes();
        TFence new_fence = CreateFence(!should_flush);
        if constexpr (can_async_check) {
            guard.lock();
        }
        {
            std::scoped_lock lock{uncommitted_guard};
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(std::move(uncommitted_operations));
            uncommitted_operations.clear(); // moved-from: make it empty explicitly
        }
        pending_flushes.push_back(committed_flushes);
        if (light) {
            QueueFenceAt(new_fence, now);
        } else {
            QueueFence(new_fence);
        }
        if (!delay_fence) {
            func();
        }
        fences.push(std::move(new_fence));
        if (should_flush) {
            rasterizer.FlushCommands();
        }
        if constexpr (can_async_check) {
            // Read under the guard: the fencing thread sets it under the guard before it sleeps,
            // and finds this fence if it takes the guard first.
            const bool wake = !light || fence_thread_waiting;
            guard.unlock();
            if (wake) {
                cv.notify_all();
            }
            if (light) {
                LightCount(wake ? light_stats.notifies : light_stats.notifies_skipped);
            }
        }
        if (delay_fence) {
            // Its callback runs on the fencing thread once the fence's work is done, and the
            // guest or WaitPendingFences may block on that callback: a submission left for
            // later (Scheduler::DeferFlush) would wait for the back to go idle, which it then
            // never does (GPU accuracy switched to High while the game ran, 24 s and more).
            SubmitDeferredWork();
        }
        if (!light || LightGatherDue(now, commit_downloads || delay_fence || committed_flushes != 0)) {
            rasterizer.InvalidateGPUCache();
        }
    }

    static void LightCount(std::atomic<u64>& counter) {
        counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    /// EDEN_NX2_FENCE_LIGHT: whether this fence gathers the guest's CPU writes (FenceLight).
    bool LightGatherDue(std::chrono::steady_clock::time_point now, bool force) {
        static const std::chrono::microseconds interval{[] {
            const char* value = std::getenv("EDEN_NX2_FENCE_LIGHT_US");
            return value == nullptr ? s64{500} : static_cast<s64>(std::atoll(value));
        }()};
        const u64 submissions = fence_light_submissions.load(std::memory_order_acquire);
        const bool submitted = submissions != light_seen_submissions;
        const bool due = force || submitted || now - light_last_gather >= interval;
        if (due) {
            light_seen_submissions = submissions;
            light_last_gather = now;
            light_gather_pending = false;
            LightCount(submitted && !force ? light_stats.gathers_submission : light_stats.gathers);
        } else {
            light_gather_pending = true;
            LightCount(light_stats.gathers_skipped);
        }
        if (now - light_last_log >= std::chrono::seconds{10}) {
            LogLightStats(now);
        }
        return due;
    }

    /// FenceLight, with a gather skipped since the last one: a download committed now is copied
    /// over guest memory later, and would overwrite a CPU write the skipped gather left pending
    /// (the cache would not know the CPU wrote there). Gather first when a cache has downloads
    /// to commit, as the per-fence gather of the previous fence did.
    void LightGatherBeforeDownloads(std::chrono::steady_clock::time_point now) {
        bool downloads = false;
        {
            std::scoped_lock lock{texture_cache.mutex};
            downloads = texture_cache.HasUncommittedFlushes() || texture_cache.HasAsyncFlushWork();
        }
        if (!downloads) {
            std::scoped_lock lock{buffer_cache.mutex};
            downloads = buffer_cache.HasUncommittedFlushes();
        }
        if (!downloads) {
            return;
        }
        rasterizer.InvalidateGPUCache();
        light_seen_submissions = fence_light_submissions.load(std::memory_order_acquire);
        light_last_gather = now;
        light_gather_pending = false;
        LightCount(light_stats.gathers_download);
    }

    void LogLightStats(std::chrono::steady_clock::time_point now) {
        if (light_last_log == std::chrono::steady_clock::time_point{}) {
            light_last_log = now;
            return;
        }
        const double seconds = std::chrono::duration<double>(now - light_last_log).count();
        light_last_log = now;
        const auto take = [seconds](std::atomic<u64>& counter) {
            return static_cast<u64>(
                static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) / seconds);
        };
        const u64 gathers = take(light_stats.gathers);
        const u64 gathers_submission = take(light_stats.gathers_submission);
        const u64 gathers_skipped = take(light_stats.gathers_skipped);
        const u64 gathers_download = take(light_stats.gathers_download);
        const u64 notifies = take(light_stats.notifies);
        const u64 notifies_skipped = take(light_stats.notifies_skipped);
        const auto [stubs_shared, fences_pooled] = TakeLightFenceObjects();
        LOG_INFO(HW_GPU,
                 "EDEN_NX2_FENCE_LIGHT per second: gathers {} (+{} after a guest submission, "
                 "+{} before downloads), gathers skipped {}, notifies {}, notifies skipped {}, "
                 "stub fences shared {}, fences from the free list {}",
                 gathers, gathers_submission, gathers_download, gathers_skipped, notifies,
                 notifies_skipped,
                 static_cast<u64>(static_cast<double>(stubs_shared) / seconds),
                 static_cast<u64>(static_cast<double>(fences_pooled) / seconds));
    }

public:
    void SignalSyncPoint(u32 value) {
        // The id comes from a command list (the 3D engine's sync method, the puller's fence
        // action). Past the syncpoint table, the increment would walk memory after it as an
        // action list; command words read wrong carry any id (Splatoon Raiders on Linux).
        if (value >= Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) {
            LOG_ERROR(HW_GPU, "Signal of syncpoint {} past the table, ignored", value);
            return;
        }
        syncpoint_manager.IncrementGuest(value);
        std::function<void()> func([this, value] { syncpoint_manager.IncrementHost(value); });
        // The guest CPU waits on syncpoints: their fences carry every pending download.
        SignalFenceImpl(std::move(func), true);
    }

    void WaitPendingFences([[maybe_unused]] bool force) {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<true>();
        } else {
            if (!force) {
                return;
            }
            std::mutex wait_mutex;
            std::condition_variable wait_cv;
            std::atomic<bool> wait_finished{};
            std::function<void()> func([&] {
                std::scoped_lock lk(wait_mutex);
                wait_finished.store(true, std::memory_order_relaxed);
                wait_cv.notify_all();
            });
            SignalFence(std::move(func));
            // The fence's commands can be held back by a deferred flush that only this thread
            // submits. The fencing thread waits for that submission before releasing the fence,
            // so waiting here without submitting deadlocks (Ocarina of Time entering gameplay).
            SubmitDeferredWork();
            std::unique_lock lk(wait_mutex);
            wait_cv.wait(
                lk, [&wait_finished] { return wait_finished.load(std::memory_order_relaxed); });
        }
    }

protected:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                          TTextureCache& texture_cache_, TBufferCache& buffer_cache_,
                          TQueryCache& query_cache_)
        : rasterizer{rasterizer_}, gpu{gpu_}, syncpoint_manager{gpu.Host1x().GetSyncpointManager()},
          texture_cache{texture_cache_}, buffer_cache{buffer_cache_}, query_cache{query_cache_} {
        if constexpr (can_async_check) {
            fence_thread =
                std::jthread([this](std::stop_token token) { ReleaseThreadFunc(token); });
        }
    }

    virtual ~FenceManager() {
        if constexpr (can_async_check) {
            fence_thread.request_stop();
            cv.notify_all();
            fence_thread.join();
        }
    }

    /// Creates a Fence Interface, does not create a backend fence if 'is_stubbed' is
    /// true
    virtual TFence CreateFence(bool is_stubbed) = 0;
    /// Queues a fence into the backend if the fence isn't stubbed.
    virtual void QueueFence(TFence& fence) = 0;
    /// Notifies that the backend fence has been signaled/reached in host GPU.
    virtual bool IsFenceSignaled(TFence& fence) const = 0;
    /// Waits until a fence has been signalled by the host GPU.
    virtual void WaitFence(TFence& fence) = 0;
    /// Submits work whose submission the backend deferred (called before blocking on a fence).
    virtual void SubmitDeferredWork() {}

    VideoCore::RasterizerInterface& rasterizer;
    Tegra::GPU& gpu;
    Tegra::Host1x::SyncpointManager& syncpoint_manager;
    TTextureCache& texture_cache;
    TBufferCache& buffer_cache;
    TQueryCache& query_cache;

private:
    template <bool force_wait>
    void TryReleasePendingFences() {
        while (!fences.empty()) {
            TFence& current_fence = fences.front();
            if (ShouldWait() && !IsFenceSignaled(current_fence)) {
                if constexpr (force_wait) {
                    WaitFence(current_fence);
                } else {
                    return;
                }
            }
            PopAsyncFlushes(pending_flushes.front());
            pending_flushes.pop_front();
            auto operations = std::move(pending_operations.front());
            pending_operations.pop_front();
            for (auto& operation : operations) {
                operation();
            }
            {
                std::unique_lock lock(ring_guard);
                delayed_destruction_ring.Push(std::move(current_fence));
            }
            fences.pop();
        }
    }

    void ReleaseThreadFunc(std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPUFencingThread");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        Common::SetCurrentThreadToPerformanceCores();

        // Every fence the host GPU has already passed is released together: Switch 2 titles
        // signal thousands a second and the host GPU completes them in bursts, so each cache's
        // lock is taken once per burst instead of once per fence (the GPU thread, which holds
        // them for nearly every command, spun, slept and woke this thread for each one).
        // EDEN_NX2_FENCE_BATCH=0: one fence at a time (for comparisons).
        static const size_t MaxBatch = [] {
            const char* value = std::getenv("EDEN_NX2_FENCE_BATCH");
            return value != nullptr && value[0] == '0' ? size_t{1} : size_t{64};
        }();
        struct Released {
            TFence fence;
            std::deque<std::function<void()>> operations;
            u8 flushes;
        };
        std::vector<Released> batch;
        batch.reserve(MaxBatch);
        const auto take_front = [&] {
            batch.push_back({std::move(fences.front()), std::move(pending_operations.front()),
                             pending_flushes.front()});
            fences.pop();
            pending_operations.pop_front();
            pending_flushes.pop_front();
        };
        while (!stop_token.stop_requested()) {
            {
                std::unique_lock lock(guard);
                // cv.wait with that predicate, telling SignalFenceImpl when it sleeps (FenceLight
                // notifies only then).
                while (!stop_token.stop_requested() && fences.empty()) {
                    fence_thread_waiting = true;
                    cv.wait(lock);
                    fence_thread_waiting = false;
                }
                if (stop_token.stop_requested()) [[unlikely]] {
                    return;
                }
                take_front();
            }
            if (!batch.front().fence->IsStubbed()) {
                WaitFence(batch.front().fence);
            }
            {
                std::unique_lock lock(guard);
                while (batch.size() < MaxBatch && !fences.empty() &&
                       (fences.front()->IsStubbed() || IsFenceSignaled(fences.front()))) {
                    take_front();
                }
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
            size_t texture_pops = 0;
            size_t buffer_pops = 0;
            for (const Released& released : batch) {
                texture_pops += (released.flushes & TextureFlushes) != 0 ? 1 : 0;
                buffer_pops += (released.flushes & BufferFlushes) != 0 ? 1 : 0;
            }
            if (texture_pops != 0) {
                std::scoped_lock lock{texture_cache.mutex};
                for (size_t i = 0; i < texture_pops; ++i) {
                    texture_cache.PopAsyncFlushes();
                }
            }
            if (buffer_pops != 0) {
                std::scoped_lock lock{buffer_cache.mutex};
                for (size_t i = 0; i < buffer_pops; ++i) {
                    buffer_cache.PopAsyncFlushes();
                }
            }
            // Queries and operations keep their per-fence order.
            for (Released& released : batch) {
                query_cache.PopAsyncFlushes();
                for (auto& operation : released.operations) {
                    operation();
                }
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
            {
                std::unique_lock lock(ring_guard);
                for (Released& released : batch) {
                    delayed_destruction_ring.Push(std::move(released.fence));
                }
            }
            batch.clear();
        }
    }

    bool ShouldWait() const {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        return texture_cache.ShouldWaitAsyncFlushes() || buffer_cache.ShouldWaitAsyncFlushes() ||
               query_cache.ShouldWaitAsyncFlushes();
    }

    static constexpr u8 TextureFlushes = 1;
    static constexpr u8 BufferFlushes = 2;

    /// committed: what CommitAsyncFlushes returned for this fence.
    void PopAsyncFlushes(u8 committed) {
        if (committed & TextureFlushes) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.PopAsyncFlushes();
        }
        if (committed & BufferFlushes) {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_cache.PopAsyncFlushes();
        }
        query_cache.PopAsyncFlushes();
    }

    struct CacheFlushes {
        bool should_flush;  ///< Downloads to wait for: the fence must reach the host GPU
        u8 committed;       ///< TextureFlushes, BufferFlushes: the caches that committed downloads
    };

    /// Commits the texture and buffer caches' downloads for a new fence (the query cache's are
    /// committed once a fence is made). Switch 2 titles signal a fence per compute dispatch
    /// (thousands a second), mostly with nothing or only buffer data to download: each cache's
    /// lock is taken once, alone, and the fencing thread later takes only the lock of the cache
    /// with work, instead of holding both while it copies (the GPU thread's draws and uploads
    /// waited for it).
    CacheFlushes CommitCacheFlushes(bool defer_buffer_downloads) {
        CacheFlushes flushes{};
        {
            std::scoped_lock lock{texture_cache.mutex};
            flushes.should_flush = texture_cache.HasUncommittedFlushes();
            if (texture_cache.HasAsyncFlushWork()) {
                texture_cache.CommitAsyncFlushes();
                flushes.committed |= TextureFlushes;
            }
        }
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.HasUncommittedFlushes()) {
                if (defer_buffer_downloads) {
                    coalesced_downloads = true;
                } else {
                    buffer_cache.CommitAsyncFlushes();
                    flushes.should_flush = true;
                    flushes.committed |= BufferFlushes;
                }
            }
            if (!defer_buffer_downloads) {
                coalesced_downloads = false;
            }
        }
        return flushes;
    }

    /// Whether a fence with no downloads would do nothing, so need not be made: with fast GPU
    /// fences its callback runs at once anyway, and most of the per-dispatch fences of Switch 2
    /// titles have no queries or queued operations either. Making, queueing and retiring them
    /// cost the GPU thread ~5% and woke the fencing thread for each. One in 64 is still made, so
    /// the operations fences carry along (query slot cleanup) keep running.
    bool SkipEmptyFence() {
        if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (skipped_fences >= 64 || query_cache.HasPendingFenceWork()) {
                return false;
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                if (!uncommitted_operations.empty()) {
                    return false;
                }
            }
            ++skipped_fences;
            return true;
        } else {
            return false;
        }
    }

    std::queue<TFence> fences;
    std::deque<std::function<void()>> uncommitted_operations;
    std::deque<std::deque<std::function<void()>>> pending_operations;
    std::deque<u8> pending_flushes; ///< Per fence: the caches that committed downloads
    u32 skipped_fences{}; ///< Empty fences skipped since the last one made (GPU thread only)
    bool coalesced_downloads{}; ///< Buffer downloads left uncommitted (CoalesceDownloads)
    bool fence_thread_waiting{}; ///< The fencing thread sleeps on cv (guarded by guard)

    // FenceLight state (GPU thread only) and counters (logged every 10 s).
    std::chrono::steady_clock::time_point light_last_gather{};
    std::chrono::steady_clock::time_point light_last_log{};
    u64 light_seen_submissions{};
    bool light_gather_pending{}; ///< A gather was skipped since the last one
    struct LightStats {
        std::atomic<u64> gathers{};
        std::atomic<u64> gathers_submission{};
        std::atomic<u64> gathers_skipped{};
        std::atomic<u64> gathers_download{};
        std::atomic<u64> notifies{};
        std::atomic<u64> notifies_skipped{};
    } light_stats;

    std::mutex guard;
    std::mutex ring_guard;
    std::mutex uncommitted_guard; ///< Guards uncommitted_operations
    std::condition_variable cv;

    std::jthread fence_thread;

    DelayedDestructionRing<TFence, 8> delayed_destruction_ring;
};

} // namespace VideoCommon
