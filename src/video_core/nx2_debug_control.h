// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

/// Requests from the debug server (EDEN_NX2_DEBUG_SERVER, docs/nx2/debug-mcp.md) to the thread
/// that owns the rasterizer. The server posts them from its own thread; RasterizerVulkan takes
/// them at the end of each presented frame, so every cache they touch is used from its owner.
/// Without the server nothing is posted and the per-frame cost is one relaxed atomic load.
namespace VideoCore::Nx2Debug {

/// EXPERIMENT (not for nx2): EDEN_NX2_DMA_WAIT_WRITERS, read at startup and settable live by the
/// debug server's `switch` method, so one session can A/B it.
[[nodiscard]] inline int EnvSwitch(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? 0 : value[0] - '0';
}
inline std::atomic<int> g_dma_wait_writers{EnvSwitch("EDEN_NX2_DMA_WAIT_WRITERS")};
/// EXPERIMENT (not for nx2): EDEN_NX2_SEM_FAST=1, semaphore acquires without draining the back
/// when the value is already met and no queued write overlaps it (puller.cpp).
inline std::atomic<int> g_sem_fast{EnvSwitch("EDEN_NX2_SEM_FAST")};
/// EXPERIMENT (not for nx2): EDEN_NX2_LIST_FAST=1, command list entries skip the wait for queued
/// compute and copies while no page was newly GPU-written for 2 s; =2 also checks each skipped
/// entry on the back and logs misses (dma_pusher.cpp).
inline std::atomic<int> g_list_fast{EnvSwitch("EDEN_NX2_LIST_FAST")};
/// EXPERIMENT (not for nx2): EDEN_NX2_LOOP_SKIP=1, the split's front does not send the instance
/// loop writes the back would ignore (draw.end in an instance loop, unchanged vertex and index
/// buffer firsts and counts) (maxwell_3d.cpp).
inline std::atomic<int> g_loop_skip{EnvSwitch("EDEN_NX2_LOOP_SKIP")};

/// The live switch called `name`, or null.
[[nodiscard]] inline std::atomic<int>* FindSwitch(std::string_view name) {
    if (name == "EDEN_NX2_DMA_WAIT_WRITERS") {
        return &g_dma_wait_writers;
    }
    if (name == "EDEN_NX2_SEM_FAST") {
        return &g_sem_fast;
    }
    if (name == "EDEN_NX2_LIST_FAST") {
        return &g_list_fast;
    }
    if (name == "EDEN_NX2_LOOP_SKIP") {
        return &g_loop_skip;
    }
    return nullptr;
}

struct CaptureRequest {
    u64 id;
    std::filesystem::path directory;
    bool sampled;
    /// The uniform and storage buffers of every draw and dispatch, and dispatch group counts.
    bool buffers;
};

struct CaptureResult {
    std::string error;
};

struct GpuReadRequest {
    u64 id;
    u64 gpu_address;
    u64 size;
};

struct GpuReadResult {
    std::vector<u8> bytes;
    /// Tegra::MemoryManager::GetID of the address space that maps the whole range.
    u64 address_space;
    std::string error;
};

struct RejectChange {
    u64 generation;
    /// Whether a hash left the list, so programs rejected before must be lowered again.
    bool unrejected;
};

/// Opens a debug server session (clears requests and rejections) or ends it (wakes waiters).
void BeginSession();
void EndSession();

/// Rasterizer thread: whether a capture or a reject change is waiting.
[[nodiscard]] bool HasRequests() noexcept;

/// Server thread: queues a frame capture; the rasterizer takes it at the next frame end.
void RequestCapture(CaptureRequest request);
/// Server thread: withdraws a capture that was not taken yet (it timed out).
void CancelCapture(u64 id);
/// Server thread: waits for the capture's manifest to be complete. Empty on timeout or when the
/// session ends.
[[nodiscard]] std::optional<CaptureResult> WaitCapture(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<CaptureRequest> TakeCapture();
/// Frame dump writer thread, once the manifest summary is written (or the capture failed).
void FinishCapture(u64 id, CaptureResult result);

/// Server thread: queues a read of GPU virtual memory; the rasterizer takes it at the next frame
/// end, where the GPU's own reads of its address spaces run.
void RequestGpuRead(GpuReadRequest request);
/// Server thread: withdraws a read that was not taken yet (it timed out).
void CancelGpuRead(u64 id);
/// Server thread: empty on timeout or when the session ends.
[[nodiscard]] std::optional<GpuReadResult> WaitGpuRead(u64 id, std::chrono::milliseconds timeout);
/// Rasterizer thread.
[[nodiscard]] std::optional<GpuReadRequest> TakeGpuRead();
/// Rasterizer thread.
void FinishGpuRead(u64 id, GpuReadResult result);

/// Server thread: adds or removes native program keys or shader hashes (as EDEN_NX2_REJECT_HASH
/// takes them). Returns the generation to wait for.
u64 SetRejected(std::span<const u64> hashes, bool rejected);
[[nodiscard]] std::vector<u64> RejectedHashes();
/// Lowering: whether the debug server rejects this program. Cheap while the list is empty.
[[nodiscard]] bool IsRejected(u64 hash);
[[nodiscard]] bool AnyRejected() noexcept;
/// Rasterizer thread: the pending reject change, once.
[[nodiscard]] std::optional<RejectChange> TakeRejectChange();
/// Rasterizer thread: the cached programs of this generation were dropped.
void NoteRejectApplied(u64 generation);
/// Server thread: waits until the rasterizer applied the generation.
[[nodiscard]] bool WaitRejectApplied(u64 generation, std::chrono::milliseconds timeout);

} // namespace VideoCore::Nx2Debug
