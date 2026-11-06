// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include "common/common_types.h"
#include "core/hle/result.h"

namespace Core {
class System;
}
namespace Kernel {
class KProcess;
class KReadableEvent;
}
namespace AudioCore::Sink {
class Sink;
}

namespace Service::Audio {

struct SysmixOutputCounters {
    u64 ticks{};
    u64 copied_frames{};
    u64 accepted_frames{};
    u64 retried_frames{};
    u64 padded_frames{};
    u64 nonzero_frames{};
    u64 admitted_blocks{};
    u64 backpressure{};
    u64 faults{};
};

class SysmixOutput {
public:
    SysmixOutput(Core::System& system, Kernel::KProcess& process, AudioCore::Sink::Sink& sink,
                 u32 id, u64 buffer, u64 size, u32 creation_flags);
    ~SysmixOutput();
    static Result ValidateRegion(Kernel::KProcess& process, u64 buffer, u64 size,
                                 u32 creation_flags);
    static bool IsFormatSupported(u32 sample_type, u32 channels, u32 rate);
    Result Open(u32 sample_type, u32 channels, u32 rate, u32 frames, u32 prefill, u32 flags);
    void Close();
    Result SetEngaged(bool engaged);
    Result SetDataNotification(u32 mode, s32 threshold);
    Result SetGain(f32 gain, s64 ramp_ns);
    f32 GetGain() const;
    u32 GetState() const;
    void Pump();
    Kernel::KReadableEvent* GetStateEvent();
    Kernel::KReadableEvent* GetDataEvent();
    SysmixOutputCounters GetCounters() const;
    bool IsOwnedBy(const Kernel::KProcess* process) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Service::Audio
