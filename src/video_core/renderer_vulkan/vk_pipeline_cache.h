// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct ComputePipelineCacheEntry {
    std::unique_ptr<ComputePipeline> pipeline;
    std::string failure;

    [[nodiscard]] ComputePipeline* Get(bool native, GPUVAddr address,
                                       const ComputePipelineCacheKey& key) const;
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    void TickFrame();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

private:
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    /// EDEN_NX2_PIPELINE_CLEAN: CurrentGraphicsPipeline without FixedPipelineState::Refresh nor
    /// the key comparison while nothing Refresh reads changed since the last lookup.
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineClean();
    /// The registers Refresh reads with these features, as (first word, words) ranges.
    void BuildCleanWatch();
    [[nodiscard]] bool CleanInputsUnchanged() const;
    void TakeCleanSnapshot();

    struct CleanLookup {
        struct Range {
            u32 first;
            u32 count;
        };
        bool built{};
        std::vector<Range> ranges;     ///< Read by every Refresh
        std::vector<Range> xfb_ranges; ///< Read while transform feedback is enabled
        std::vector<u32> snapshot;     ///< ranges' words at the last Refresh
        std::vector<u32> xfb_snapshot;
        std::array<u64, 6> hashes{};   ///< graphics_key.unique_hashes at the last Refresh
        const Tegra::Engines::Maxwell3D* maxwell3d{};
        Maxwell::PrimitiveTopology topology{};
        Tegra::Engines::Maxwell3D::EngineHint engine_state{};
        /// current_pipeline when its key was graphics_key after the last Refresh, else null.
        GraphicsPipeline* pipeline{};
        u64 hits{};
        u64 misses{};
        std::chrono::steady_clock::time_point last_log{};
    } clean;

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    /// The slow path's end: graphics_cache[graphics_key]'s pipeline becomes the current one.
    [[nodiscard]] GraphicsPipeline* MakeCurrent(GraphicsPipeline* pipeline);

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// EDEN_NX2_PIPELINE_ASYNC: what a build off the GPU thread takes in place of maxwell3d.
    struct AsyncBuildContext;

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools = nullptr,
        std::unique_ptr<EmissionBudget::Reservation> reservation = nullptr,
        const AsyncBuildContext* async = nullptr);

    /// Stores a pipeline built at run time in the disk cache (unless it is there already).
    void StoreRuntimePipeline(const GraphicsPipelineCacheKey& key, const GraphicsPipeline& pipeline,
                              GraphicsEnvironments& environments);

    // EDEN_NX2_PIPELINE_ASYNC: graphics pipelines translated and built on the workers.
    struct AsyncGraphicsJob;
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineAsync();
    [[nodiscard]] bool CanDiscardAsyncDraw() const;
    /// Queues graphics_key's build; false when it has to be built on the GPU thread.
    [[nodiscard]] bool QueueAsyncGraphicsPipeline();
    void BuildAsyncGraphicsPipeline(const std::shared_ptr<AsyncGraphicsJob>& job,
                                    bool on_gpu_thread);
    /// Runs the job on the GPU thread if no worker took it yet, else waits for the worker.
    void FinishAsyncGraphicsPipeline(const std::shared_ptr<AsyncGraphicsJob>& job);
    /// Moves the finished builds into graphics_cache (GPU thread).
    void DrainAsyncPipelines();
    void ReportAsyncPipelines(std::chrono::steady_clock::time_point now);

    ComputePipelineCacheEntry CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                     const ShaderInfo* shader);

    ComputePipelineCacheEntry CreateComputePipeline(ShaderPools& pools,
                                                     const ComputePipelineCacheKey& key,
                                                     Shader::Environment& env,
                                                     PipelineStatistics* statistics,
                                                     bool build_in_parallel);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};
    const size_t num_pipeline_workers;
    EmissionBudget emission_budget;

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};

    /// Destroys graphics pipelines unused for a while, once the GPU is done with them.
    void EvictUnusedPipelines();
    u64 frame_count{};
    std::deque<std::pair<u64, std::unique_ptr<GraphicsPipeline>>> evicted_pipelines; ///< (tick, pipeline)
    std::unordered_set<u64> used_and_evicted; ///< Key hashes of pipelines used this session, evicted since

    ::Common::unordered_map<ComputePipelineCacheKey, ComputePipelineCacheEntry> compute_cache;
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    std::mutex spare_pools_mutex;
    std::vector<std::shared_ptr<ShaderPools>> spare_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    /// Key hashes of the graphics pipelines in the disk cache. Only those a recent session used
    /// (used_pipelines) are built at boot; the rest are built when first needed, and not stored
    /// again. Each built pipeline holds driver memory (title 1: 6.5k cached, a quarter used).
    std::unordered_set<u64> disk_pipelines;
    std::unordered_map<u64, u8> used_pipelines; ///< Key hash -> sessions since it was last used
    std::unordered_set<const GraphicsPipeline*> used_this_session;
    std::filesystem::path used_pipelines_filename;
    std::chrono::steady_clock::time_point used_pipelines_saved{};
    void SaveUsedPipelines();

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};

    // EDEN_NX2_PIPELINE_ASYNC state, declared before workers so a job still running while the
    // cache is destroyed (workers joins it) never sees it destroyed.
    u32 async_pipelines{}; ///< 0 off, 1 on, 2 on with native draws skipped while pending too
    /// Builds queued or running (GPU thread only); their graphics_cache entries are null.
    ::Common::unordered_map<GraphicsPipelineCacheKey, std::shared_ptr<AsyncGraphicsJob>>
        async_pending;
    /// Key hashes whose build on a worker failed: built on the GPU thread at their next draw.
    std::unordered_set<u64> async_sync_keys;
    std::mutex async_mutex;
    std::condition_variable async_idle;
    std::vector<std::shared_ptr<AsyncGraphicsJob>> async_completed; ///< Under async_mutex
    std::atomic<size_t> async_completed_count{};
    size_t async_running{}; ///< Jobs running on workers, under async_mutex
    bool async_stopping{};  ///< Under async_mutex: the cache is being destroyed
    struct AsyncStats {
        u32 queued{};
        u32 inline_built{};
        u32 waited{};
        u32 sync_built{};
        u32 fallback{};
        u64 skipped{};
        u64 snapshot_bytes{};
        std::atomic<u32> worker_built{};
        std::chrono::steady_clock::duration gpu_time{};
        std::chrono::steady_clock::time_point window{};
    } async_stats;

    Common::ThreadWorker workers;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;
};

} // namespace Vulkan
