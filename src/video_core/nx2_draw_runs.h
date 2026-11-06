// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_DRAW_RUNS=1 (measurement, off by default): sizes the options for cheaper draws on the
// split's back thread. For one frame every 10 s it logs the runs of consecutive draws whose bound
// state is the same apart from vertex, index and instance offsets (no other 3D state written with
// a new value, same topology, no clear, compute, copy or barrier between them), what changed
// between the draws of a run (constant buffer or inline upload data, constant buffer ranges), what
// ended the runs, and how many 3D register writes with the value the register held still reached
// the back's dirty state. Back thread only (the GPU thread when the split is off).
namespace VideoCommon::Nx2DrawRuns {

inline const bool g_enabled = std::getenv("EDEN_NX2_DRAW_RUNS") != nullptr;

enum class Break : u32 { State, Topology, ConstBufferRange, Clear, Compute, Copy, Barrier, Count };
inline constexpr std::array<const char*, static_cast<size_t>(Break::Count)> break_names{
    "state", "topology", "cb range", "clear", "compute", "copy/blit", "barrier"};
inline constexpr std::array<const char*, 9> bucket_names{"1",     "2",     "3-4",
                                                         "5-8",   "9-16",  "17-32",
                                                         "33-64", "65-128", "129+"};
inline constexpr size_t MethodSlots = 0x1000;

struct Frame {
    std::array<u64, 9> runs{};
    std::array<u64, 9> run_draws{};
    std::array<u64, static_cast<size_t>(Break::Count)> breaks{};
    u64 draws{};
    u64 indirect{};
    u64 identical{};       ///< Run continuations with nothing written but offsets
    u64 data_only{};       ///< Run continuations after constant buffer or inline uploads only
    u64 data_only_bytes{}; ///< Uploaded before those
    u64 data_bytes{};      ///< All constant buffer and inline upload bytes
    u64 same_rebinds{};    ///< Constant buffer binds of the range already bound
    u64 writes{};
    u64 changed{};
    u64 same_dirty{};   ///< Same-value writes processed (dirty flags set)
    u64 same_skipped{}; ///< Same-value writes skipped on the back
    std::array<u32, MethodSlots> same_dirty_by_method{};
};

struct State {
    Frame frame;
    u64 run_length{};
    u64 pending_bytes{};
    u32 topology{~0U};
    Break reason{Break::State};
    bool pending_break{true};
    u64 frames{};
    std::chrono::steady_clock::time_point next_report{};
};

inline State& Get() {
    static State state;
    return state;
}

/// Something that the next draw can't share with the previous one happened.
inline void OnBreak(Break reason) {
    State& state = Get();
    if (!state.pending_break) {
        state.pending_break = true;
        state.reason = reason;
    }
}

/// A 3D register write on the back. `changed`: the register held another value; `processed`:
/// it reached ProcessDirtyRegisters or ProcessMethodCall.
inline void OnWrite(u32 method, bool changed, bool processed) {
    Frame& frame = Get().frame;
    ++frame.writes;
    if (changed) {
        ++frame.changed;
    } else if (processed) {
        ++frame.same_dirty;
        ++frame.same_dirty_by_method[(std::min)(method, u32{MethodSlots - 1})];
    } else {
        ++frame.same_skipped;
    }
}

/// Constant buffer or inline upload data.
inline void OnData(u64 bytes) {
    State& state = Get();
    state.pending_bytes += bytes;
    state.frame.data_bytes += bytes;
}

/// A constant buffer bind: `same` when it binds the range the slot already had.
inline void OnConstBufferBind(bool same) {
    if (same) {
        ++Get().frame.same_rebinds;
    } else {
        OnBreak(Break::ConstBufferRange);
    }
}

inline size_t Bucket(u64 length) {
    size_t bucket = 0;
    for (u64 rest = length - 1; rest != 0 && bucket < 8; rest >>= 1) {
        ++bucket;
    }
    return bucket;
}

inline void EndRun(State& state) {
    if (state.run_length == 0) {
        return;
    }
    const size_t bucket = Bucket(state.run_length);
    ++state.frame.runs[bucket];
    state.frame.run_draws[bucket] += state.run_length;
    ++state.frame.breaks[static_cast<size_t>(state.reason)];
    state.run_length = 0;
}

/// A draw the back executes.
inline void OnDraw(u32 topology, bool indirect) {
    State& state = Get();
    Frame& frame = state.frame;
    ++frame.draws;
    frame.indirect += indirect ? 1 : 0;
    if (topology != state.topology) {
        OnBreak(Break::Topology);
        state.topology = topology;
    }
    if (state.pending_break) {
        EndRun(state);
        state.run_length = 1;
    } else {
        ++state.run_length;
        if (state.pending_bytes != 0) {
            ++frame.data_only;
            frame.data_only_bytes += state.pending_bytes;
        } else {
            ++frame.identical;
        }
    }
    state.pending_break = false;
    state.pending_bytes = 0;
}

/// The frame ended: logs it if 10 s passed since the last report.
inline void OnFrame() {
    State& state = Get();
    Frame& frame = state.frame;
    if (state.run_length != 0) {
        const size_t bucket = Bucket(state.run_length);
        ++frame.runs[bucket];
        frame.run_draws[bucket] += state.run_length;
        state.run_length = 0;
    }
    state.pending_break = true;
    state.pending_bytes = 0;
    const auto now = std::chrono::steady_clock::now();
    if (now < state.next_report || frame.draws == 0) {
        frame = {};
        return;
    }
    state.next_report = now + std::chrono::seconds{10};
    u64 runs{};
    std::string histogram;
    for (size_t i = 0; i < frame.runs.size(); ++i) {
        runs += frame.runs[i];
        histogram += fmt::format(" {}:{}/{}", bucket_names[i], frame.run_draws[i], frame.runs[i]);
    }
    std::string breaks;
    for (size_t i = 0; i < frame.breaks.size(); ++i) {
        breaks += fmt::format(" {} {}", break_names[i], frame.breaks[i]);
    }
    std::array<u32, 6> top{};
    for (u32 method = 0; method < MethodSlots; ++method) {
        if (frame.same_dirty_by_method[method] > frame.same_dirty_by_method[top.back()]) {
            top.back() = method;
            std::ranges::sort(top, [&frame](u32 a, u32 b) {
                return frame.same_dirty_by_method[a] > frame.same_dirty_by_method[b];
            });
        }
    }
    std::string top_methods;
    for (const u32 method : top) {
        if (frame.same_dirty_by_method[method] != 0) {
            top_methods += fmt::format(" {:#x}x{}", method, frame.same_dirty_by_method[method]);
        }
    }
    LOG_INFO(HW_GPU,
             "Draw runs (one frame): draws {} (indirect {}), runs {} (mean {:.1f}) | run length "
             "draws/runs:{} | continuations: identical {}, constant buffer or inline data only {} "
             "({} bytes); all upload bytes {}; same-range constant buffer binds {} | runs ended by:{} "
             "| back 3D register writes {}: changed {}, same value dirtied {} (top{}), same value "
             "skipped {}",
             frame.draws, frame.indirect, runs,
             runs != 0 ? static_cast<double>(frame.draws) / static_cast<double>(runs) : 0.0,
             histogram, frame.identical, frame.data_only, frame.data_only_bytes, frame.data_bytes,
             frame.same_rebinds, breaks, frame.writes, frame.changed, frame.same_dirty,
             top_methods, frame.same_skipped);
    state.frame = {};
}

} // namespace VideoCommon::Nx2DrawRuns
