// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// EDEN_NX2_PARALLEL_RECORD=<threads> (2-8): the Vulkan worker records render passes into
/// secondary command buffers on several threads. Recorded commands still run in order on the
/// worker, but through a dispatch table that logs their vkCmd* calls. Inside a render pass calls
/// are only logged; when the pass ends its draws are split into contiguous ranges, each range is
/// recorded into a secondary command buffer that first replays the last call that set each piece
/// of state, and the primary executes the secondaries in order. A call that can't move into a
/// secondary (queries, barriers, transform feedback, timestamps, ...) replays the pass inline.
/// See docs/nx2-parallel-recording.md.
class ParallelRecord {
public:
    /// Hands out secondary command buffers. Thread index i is only used by one thread at a time.
    class SecondarySource {
    public:
        virtual ~SecondarySource() = default;
        virtual VkCommandBuffer Acquire(u32 thread) = 0;
    };

    enum class Fallback : u8 {
        Small,
        Query,
        Barrier,
        TransformFeedback,
        ConditionalRendering,
        Timestamp,
        DebugLabel,
        Event,
        OutsideCommand,
        UnloggedState,
        RenderPassInfo,
        Count,
    };

    enum class Kind : u8 { State, Draw, Action };

    /// State slots a call sets: viewport 3, vertex buffers 0-3, ...
    struct SlotRange {
        u16 first;
        u16 count;
    };

    using ReplayFn = void (*)(const vk::DeviceDispatch&, VkCommandBuffer, const u8*);

    /// 0 when EDEN_NX2_PARALLEL_RECORD is unset or below 2.
    static u32 ThreadsFromEnv();

    /// Draws a pass needs before it is split, and draws each range needs.
    static constexpr u32 DefaultMinPassDraws = 96;
    static constexpr u32 DefaultMinRangeDraws = 32;

    ParallelRecord(const vk::DeviceDispatch& dld, u32 threads, SecondarySource& source,
                   u32 min_pass_draws = DefaultMinPassDraws,
                   u32 min_range_draws = DefaultMinRangeDraws);
    ~ParallelRecord();

    ParallelRecord(const ParallelRecord&) = delete;
    ParallelRecord& operator=(const ParallelRecord&) = delete;

    /// The command buffer recorded commands receive on this thread in place of `primary`.
    vk::CommandBuffer Wrap(VkCommandBuffer primary);

    /// A new primary command buffer begins: nothing is bound in it.
    void BeginPrimary();

    // The rest is called by the dispatch hooks on the thread that called Wrap.

    /// Logs a call: `write` fills `size` payload bytes that `replay` reads back.
    template <typename Write>
    void Log(ReplayFn replay, Kind kind, SlotRange slots, size_t size, Write&& write) {
        const u32 index = static_cast<u32>(ops.size());
        const u32 offset = static_cast<u32>(bytes.size());
        bytes.resize(offset + ((size + 15) & ~size_t{15}));
        write(bytes.data() + offset);
        ops.push_back(Op{replay, offset, slots, kind});
        for (u32 slot = slots.first; slot < u32{slots.first} + slots.count; ++slot) {
            last_writer[slot] = index;
        }
        pass_draws += kind == Kind::Draw ? 1 : 0;
    }
    /// A call that can't move into a secondary: an open pass replays inline.
    void Serial(Fallback reason);
    /// A state call that can't be logged: the rest of this primary records inline.
    void Unlogged(Fallback reason);
    void NoteInlineDraw() {
        ++stats.inline_draws;
    }
    /// A query or conditional rendering began (+1) or ended (-1). Secondaries don't inherit
    /// either, so passes inside one record inline.
    void NoteScope(Fallback kind, s32 delta) {
        (kind == Fallback::Query ? open_queries : open_conditionals) += delta;
    }
    /// True while calls are only logged, not forwarded to the primary.
    [[nodiscard]] bool Capturing() const noexcept {
        return mode == Mode::Capturing;
    }
    /// True when the pass is captured: the call must not reach the primary.
    bool BeginRenderPass(const VkRenderPassBeginInfo& info, VkSubpassContents contents);
    /// True when the end was recorded here.
    bool EndRenderPass();

    [[nodiscard]] const vk::DeviceDispatch& Real() const noexcept {
        return real;
    }

private:
    enum class Mode : u8 { Passthrough, Capturing, SerialPass };

    static constexpr u32 None = ~0U;

    struct Op {
        ReplayFn replay;
        u32 payload;
        SlotRange slots;
        Kind kind;
    };

    struct Job {
        u32 begin{};
        u32 end{};
        std::vector<u32> prefix;
        VkCommandBuffer result{};
        u64 record_ns{};
        bool failed{};
    };

    struct Helper {
        std::thread thread;
        std::atomic<u32> generation{};
    };

    void Prefix(const std::vector<u32>& writers, bool graphics_only, std::vector<u32>& out) const;
    void Replay(VkCommandBuffer cmdbuf, const std::vector<u32>& prefix, u32 begin, u32 end) const;
    void RecordJob(u32 thread, Job& job);
    void FallBack(Fallback reason);
    void FinishParallel();
    void MaybeCompact();
    void HelperLoop(u32 index);
    void Report();

    const vk::DeviceDispatch& real;
    vk::DeviceDispatch capture;
    SecondarySource& source;
    const u32 threads;
    const u32 min_pass_draws;
    const u32 min_range_draws;

    VkCommandBuffer primary{};
    Mode mode = Mode::Passthrough;
    bool state_unknown = false;
    s32 open_queries = 0;
    s32 open_conditionals = 0;

    std::vector<Op> ops;
    std::vector<u8> bytes;
    std::vector<u32> last_writer;
    std::vector<u32> pass_writers;
    std::vector<u32> scratch;

    VkRenderPassBeginInfo pass_info{};
    std::array<VkClearValue, 16> pass_clear_values{};
    u32 pass_first_op = 0;
    u32 pass_draws = 0;

    VkCommandBufferInheritanceInfo inheritance{};
    std::vector<Job> jobs;
    std::vector<std::unique_ptr<Helper>> helpers;
    std::atomic<u32> pending{};
    std::atomic<bool> stopping{};
    std::vector<VkCommandBuffer> secondaries;

    struct Stats {
        u64 passes{};
        u64 parallel_passes{};
        u64 parallel_draws{};
        u64 inline_draws{};
        u64 secondaries{};
        u64 split_ns{};
        u64 wait_ns{};
        u64 failures{};
        std::array<u64, static_cast<size_t>(Fallback::Count)> fallbacks{};
        std::vector<u64> record_ns;
        u64 window_start{};
    } stats;
};

} // namespace Vulkan
