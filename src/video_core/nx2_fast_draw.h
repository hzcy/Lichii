// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#include "common/common_types.h"

/// EDEN_NX2_FAST_DRAW=1: a draw that follows a draw with the same bound state reuses that draw's
/// resolved bindings, recording only the draw (and the index buffer for indexed draws). Anything
/// that could change what the resolve produces between two draws bumps a serial: a 3D register
/// that changes (other than draw parameters), any 3D method with side effects (constant buffer
/// data and binds, uploads, reports, clears, barriers, ...), any other engine, deferred calls on
/// the split's back (except the ones marked GpuSplit::FastDrawNeutral), GPU::TickWork work,
/// memory manager invalidations and fence releases. The rasterizer reuses a draw only while the
/// serial, the command buffer, the render pass and the bound pipeline are the ones the previous
/// draw left.
///
/// EDEN_NX2_FAST_DRAW=2 also takes the delta path: index buffer registers, vertex stream
/// addresses, sizes and limits, constant buffer binds and inline constant buffer data are light
/// changes collected in g_delta instead of breaks, and
/// so is the pre-draw invalidation of that data unless it reaches an image, a shader or a bound
/// vertex buffer (those break). A draw after only light changes keeps the pipeline, image views,
/// samplers and render targets, and resolves the constant buffers and the index buffer again,
/// unless a changed constant buffer holds the pipeline's texture handles.
/// EDEN_NX2_FAST_DRAW_VERIFY=1 resolves those draws in full and counts where the views, samplers
/// or pipeline differ from what the delta path would have kept.
namespace VideoCommon::Nx2FastDraw {

enum class Reason : u8 {
    Register,
    Method,
    ConstantData,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    DeltaOverflow,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",      "method",    "constant buffer data", "inline data",
    "other engine",  "deferred call", "tick work",        "invalidation",
    "fence release", "indirect reset", "frame",           "texture header data",
    "delta overflow",
};

inline int ModeFromEnvironment() {
    const char* const value = std::getenv("EDEN_NX2_FAST_DRAW");
    return value == nullptr ? 0 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
}

inline bool g_enabled = ModeFromEnvironment() != 0;
inline bool g_delta_enabled = ModeFromEnvironment() == 2;
inline const bool g_verify = [] {
    const char* const value = std::getenv("EDEN_NX2_FAST_DRAW_VERIFY");
    return value != nullptr && value[0] == '1';
}();

/// EDEN_NX2_BUFFER_SKIP=1: the buffer cache keeps vertex buffer lookups while their registers and
/// the page table stay the same, skips the upload scan of vertex buffers no CPU write reached, and
/// touches each buffer's LRU entry once a frame.
inline const bool g_buffer_skip = [] {
    const char* const value = std::getenv("EDEN_NX2_BUFFER_SKIP");
    return value != nullptr && value[0] == '1';
}();

/// EDEN_NX2_INLINE_INVALIDATE=1: writes the caches must hear about (inline constant buffer data,
/// inline-to-memory) reach the texture, query and shader caches only on pages those held something
/// on; the buffer cache hears about all of them.
inline const bool g_inline_invalidate = [] {
    const char* const value = std::getenv("EDEN_NX2_INLINE_INVALIDATE");
    return value != nullptr && value[0] == '1';
}();

/// EDEN_NX2_DYNAMIC_SKIP=1: a draw with no dynamic state flag set skips the dynamic state updates,
/// and the vertex input update uses the draw's pipeline instead of looking it up again.
inline const bool g_dynamic_skip = [] {
    const char* const value = std::getenv("EDEN_NX2_DYNAMIC_SKIP");
    return value != nullptr && value[0] == '1';
}();

inline std::atomic<u64> g_serial{1};
/// The last break's reason and, for Register and Method, its 3D method (diagnostics).
inline std::atomic<u32> g_last_break{};

inline void Break(Reason reason, u32 method = 0) {
    if (g_enabled) {
        g_serial.fetch_add(1, std::memory_order_relaxed);
        g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    }
}

inline u64 Serial() {
    return g_serial.load(std::memory_order_relaxed);
}

/// Light changes since the last draw (EDEN_NX2_FAST_DRAW=2). Written and read on the thread that
/// runs the 3D engine and the rasterizer (the split's back).
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    std::array<GPUVAddr, MaxWrites> write_begin{};
    std::array<GPUVAddr, MaxWrites> write_end{};
    u32 num_writes{};
    bool index_buffer{};
    bool vertex_buffers{};
    bool invalidated{};

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && !index_buffer && !vertex_buffers && !invalidated &&
               (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }
};

inline Delta g_delta;

inline void NoteIndexBuffer(u32 method) {
    if (g_delta_enabled) {
        g_delta.index_buffer = true;
    } else {
        Break(Reason::Register, method);
    }
}

/// A vertex stream's address, size or limit (not its stride or enable).
inline void NoteVertexBuffer(u32 method) {
    if (g_delta_enabled) {
        g_delta.vertex_buffers = true;
    } else {
        Break(Reason::Register, method);
    }
}

inline void NoteBind(size_t stage, u32 slot) {
    if (g_delta_enabled) {
        g_delta.binds[stage] |= 1U << slot;
    } else {
        Break(Reason::Method);
    }
}

/// Inline constant buffer data written to [begin, end).
inline void NoteConstantData(GPUVAddr begin, GPUVAddr end) {
    if (!g_delta_enabled) {
        Break(Reason::ConstantData);
        return;
    }
    Delta& delta = g_delta;
    if (delta.num_writes != 0 && delta.write_end[delta.num_writes - 1] == begin) {
        delta.write_end[delta.num_writes - 1] = end;
        return;
    }
    if (delta.num_writes == Delta::MaxWrites) {
        Break(Reason::DeltaOverflow);
        return;
    }
    delta.write_begin[delta.num_writes] = begin;
    delta.write_end[delta.num_writes] = end;
    ++delta.num_writes;
}

/// The pre-draw invalidation of cached writes. Its consumers break for what the delta path
/// doesn't resolve again (images, shaders, bound vertex buffers).
inline void NoteInvalidation() {
    if (g_delta_enabled) {
        g_delta.invalidated = true;
    } else {
        Break(Reason::Invalidation);
    }
}

} // namespace VideoCommon::Nx2FastDraw
