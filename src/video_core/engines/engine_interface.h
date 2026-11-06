// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bitset>
#include <limits>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Tegra::Engines {

enum class EngineTypes : u32 {
    Nv01Timer,
    KeplerCompute,
    Maxwell3D,
    Fermi2D,
    MaxwellDMA,
    KeplerMemory,
};

class EngineInterface {
public:
    virtual ~EngineInterface() = default;

    /// Write the value to the register identified by method.
    virtual void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) = 0;

    /// Write multiple values to the register identified by method.
    virtual void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) = 0;

    void ConsumeSink(Core::System& system) {
        if (!method_sink.empty()) {
            ConsumeSinkImpl(system);
        }
    }

    /// GpuSplit::PairWrite blocks (EDEN_NX2_RING_PAIRS): writes that only store a register's
    /// value, word i = method (bits 0-15) and value (bits 32-63), applied in order after the
    /// method sink. Engines the split sends pairs to apply them as their sink would.
    virtual void ApplyPairs(Core::System& system, const u64* pairs, u32 count) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, static_cast<u32>(pairs[i] & 0xffff),
                       static_cast<u32>(pairs[i] >> 32), true);
        }
    }

    /// In a method_sink entry: a write the GPU thread split's front found only stores the
    /// register's value although execution_mask has the register (GpuSplit::SinkMethod).
    /// Methods are 16-bit.
    static constexpr u32 SplitPlainTag = 1U << 16;

    std::bitset<(std::numeric_limits<u16>::max)()> execution_mask{};
    /// A Maxwell3D whose CallMethod takes its fast path (Nx2Perf): GpuSplit's back calls it
    /// without the virtual call.
    bool fast_maxwell3d{};
    std::vector<std::pair<u32, u32>> method_sink{};
    GPUVAddr current_dma_segment;
    bool current_dirty{};
protected:
    virtual void ConsumeSinkImpl(Core::System& system) {
        for (auto [method, value] : method_sink) {
            CallMethod(system, method, value, true);
        }
        method_sink.clear();
    }
};

} // namespace Tegra::Engines
