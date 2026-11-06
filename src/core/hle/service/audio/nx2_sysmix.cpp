// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/audio/nx2_sysmix.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/memory.h"

namespace Service::Audio {
namespace {
constexpr u64 ReadOffset = 0x44;
constexpr u64 WriteOffset = 0x1004;
constexpr u64 PayloadOffset = 0x1040;
constexpr u32 BlockFrames = 240;

u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}
}

struct SysmixOutput::Impl {
    Impl(Core::System& system_, Kernel::KProcess& process_, AudioCore::Sink::Sink& sink_,
         u32 id_, u64 buffer_, u64 size_, u32 creation_flags_)
        : system{system_}, process{system.Kernel(), &process_}, sink{sink_},
          context{system, "sysmix"}, id{id_}, buffer{buffer_}, size{size_},
          creation_flags{creation_flags_} {
        state_event = context.CreateEvent("sysmix state");
        data_event = context.CreateEvent("sysmix data");
    }

    ~Impl() {
        Close();
        context.CloseEvent(data_event);
        context.CloseEvent(state_event);
    }

    void Close() {
        std::scoped_lock lock{mutex};
        open = false;
        engaged = false;
        data_mode = 0;
        if (host) {
            sink.CloseStream(host);
            host = nullptr;
        }
        if (state_event) {
            state_event->Signal(system.Kernel());
        }
    }

    Result Fault(Result result = ResultInvalidAddressInfo) {
        faulted = true;
        engaged = false;
        ++counters.faults;
        if (host) {
            host->MarkFaulted();
        }
        state_event->Signal(system.Kernel());
        LOG_ERROR(Service_Audio, "sysmix stream {} fault {:#x}: copied={} accepted={} blocks={}", id,
                  result.raw, counters.copied_frames, counters.accepted_frames,
                  counters.admitted_blocks);
        return result;
    }

    bool ReadOccupancy(u32& read, u32& write, u32& filled) {
        auto& memory = process->GetMemory();
        if (!memory.Read32Acquire(buffer + ReadOffset, read) || read != consumer ||
            !memory.Read32Acquire(buffer + WriteOffset, write) || read >= slots || write >= slots) {
            return false;
        }
        filled = static_cast<u32>((u64{write} + slots - read) % slots);
        return true;
    }

    void Notify(u32 filled) {
        if (data_mode == 2 && slots - 1 - filled >= static_cast<u32>(threshold)) {
            data_event->Signal(system.Kernel());
        }
    }

    // emu_us is the guest clock; the wall clock is the timestamp every log line carries.
    // Comparing their deltas over consecutive windows shows whether emulated time runs at the
    // host's rate (48000 frames of output per second of wall time).
    void LogStatus(u32 filled) {
        const u64 emu_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                system.CoreTiming().GetGlobalTimeNs())
                .count());
        LOG_INFO(Service_Audio,
                 "sysmix stream {} output: copied={} accepted={} nonzero={} blocks={} full={} "
                 "queued={} retried={} padded={} filled={} emu_us={}",
                 id, counters.copied_frames, counters.accepted_frames, counters.nonzero_frames,
                 counters.admitted_blocks, counters.backpressure,
                 host ? host->GetQueueSize() : 0, counters.retried_frames, counters.padded_frames,
                 filled, emu_us);
    }

    Core::System& system;
    Kernel::KScopedAutoObject<Kernel::KProcess> process;
    AudioCore::Sink::Sink& sink;
    KernelHelpers::ServiceContext context;
    Kernel::KEvent* state_event{};
    Kernel::KEvent* data_event{};
    AudioCore::Sink::SinkStream* host{};
    mutable std::mutex mutex;
    u32 id;
    u64 buffer;
    u64 size;
    u32 creation_flags;
    AudioCore::SampleFormat format{};
    u32 frame_bytes{};
    u32 slots{};
    u32 consumer{};
    u32 data_mode{};
    s32 threshold{};
    f32 gain{1.0f};
    bool open{};
    bool engaged{};
    bool faulted{};
    std::array<u8, BlockFrames * 2 * sizeof(f32)> input{};
    std::array<s16, BlockFrames * 2> output{};
    SysmixOutputCounters counters{};
};

SysmixOutput::SysmixOutput(Core::System& system, Kernel::KProcess& process,
                           AudioCore::Sink::Sink& sink, u32 id, u64 buffer, u64 size,
                           u32 creation_flags)
    : impl{std::make_unique<Impl>(system, process, sink, id, buffer, size, creation_flags)} {}

SysmixOutput::~SysmixOutput() = default;

Result SysmixOutput::ValidateRegion(Kernel::KProcess& process, u64 buffer, u64 size,
                                    u32 creation_flags) {
    R_UNLESS(creation_flags == 0x24, ResultNotSupported);
    R_UNLESS(buffer != 0 && (buffer & 0xfff) == 0 && (size & 0xfff) == 0 && size >= 0x2000 &&
                 size <= std::numeric_limits<u64>::max() - buffer &&
                 process.GetMemory().IsValidVirtualAddressRange(buffer, size),
             ResultInvalidAddressInfo);
    R_SUCCEED();
}

bool SysmixOutput::IsFormatSupported(u32 sample_type, u32 channels, u32 rate) {
    return (sample_type == static_cast<u32>(AudioCore::SampleFormat::PcmInt16) ||
            sample_type == static_cast<u32>(AudioCore::SampleFormat::PcmFloat)) &&
           channels == 2 && rate == 48000;
}

Result SysmixOutput::Open(u32 sample_type, u32 channels, u32 rate, u32 frames, u32 prefill,
                          u32 flags) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(!p.open, ResultOperationFailed);
    R_UNLESS(IsFormatSupported(sample_type, channels, rate) && prefill == 0 && flags == 0,
             ResultNotSupported);
    R_UNLESS(frames > 0 && frames < static_cast<u32>(std::numeric_limits<s32>::max()),
             ResultInvalidArgument);
    R_TRY(ValidateRegion(*p.process, p.buffer, p.size, p.creation_flags));
    const u64 slots = u64{frames} + 1;
    const auto format = static_cast<AudioCore::SampleFormat>(sample_type);
    const u32 frame_bytes = channels *
        (format == AudioCore::SampleFormat::PcmInt16 ? sizeof(s16) : sizeof(f32));
    const u64 minimum = AlignUp(0x1000 + AlignUp(0x40 + std::min<u64>(16, slots) * 8, 64) +
                                   AlignUp(slots * frame_bytes, 64), 0x1000);
    R_UNLESS(p.size >= minimum && PayloadOffset + slots * frame_bytes <= p.size,
             ResultInsufficientBuffer);
    R_UNLESS(p.state_event && p.data_event, ResultOutOfSessions);
    u32 read{}, write{};
    auto& memory = p.process->GetMemory();
    R_UNLESS(memory.Read32Acquire(p.buffer + ReadOffset, read) &&
                 memory.Read32Acquire(p.buffer + WriteOffset, write) && read < slots && write < slots,
             ResultInvalidAddressInfo);
    p.host = p.sink.AcquireSinkStream(p.system, 2, "sysmix-" + std::to_string(p.id),
                                     AudioCore::Sink::StreamType::Sysmix);
    if (!p.host || !p.host->IsReady()) {
        if (p.host) {
            p.sink.CloseStream(p.host);
            p.host = nullptr;
        }
        R_RETURN(ResultOperationFailed);
    }
    p.slots = static_cast<u32>(slots);
    p.format = format;
    p.frame_bytes = frame_bytes;
    p.consumer = read;
    p.faulted = false;
    p.open = true;
    p.engaged = false;
    p.gain = 1.0f;
    p.counters = {};
    p.state_event->Signal(p.system.Kernel());
    p.Notify(static_cast<u32>((write + slots - read) % slots));
    LOG_INFO(Service_Audio,
             "sysmix stream {} host ready: sample type {} stereo 48000 slots={} payload={:#x}",
             p.id, sample_type, p.slots, PayloadOffset);
    R_SUCCEED();
}

void SysmixOutput::Close() {
    impl->Close();
}

Result SysmixOutput::SetEngaged(bool engaged) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(p.open && !p.faulted && p.host && p.host->IsReady(), ResultOperationFailed);
    if (p.engaged == engaged) {
        R_SUCCEED();
    }
    if (engaged) {
        p.host->Start();
        R_UNLESS(p.host->IsReady() && !p.host->IsPaused(), ResultOperationFailed);
    } else {
        p.host->Stop();
        R_UNLESS(p.host->IsReady(), p.Fault(ResultOperationFailed));
    }
    p.engaged = engaged;
    p.state_event->Signal(p.system.Kernel());
    R_SUCCEED();
}

Result SysmixOutput::SetDataNotification(u32 mode, s32 threshold) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(mode == 0 || mode == 2, ResultNotSupported);
    R_UNLESS(p.open && !p.faulted && threshold >= 0 && static_cast<u32>(threshold) < p.slots,
             ResultInvalidArgument);
    u32 read{}, write{}, filled{};
    R_UNLESS(p.ReadOccupancy(read, write, filled), p.Fault());
    p.data_mode = mode;
    p.threshold = threshold;
    p.Notify(filled);
    R_SUCCEED();
}

Result SysmixOutput::SetGain(f32 gain, s64 ramp_ns) {
    R_UNLESS(ramp_ns == 0, ResultNotSupported);
    R_UNLESS(std::isfinite(gain) && gain >= 0.0f && gain <= 128.0f, ResultInvalidArgument);
    std::scoped_lock lock{impl->mutex};
    impl->gain = gain;
    R_SUCCEED();
}

f32 SysmixOutput::GetGain() const {
    std::scoped_lock lock{impl->mutex};
    return impl->gain;
}

u32 SysmixOutput::GetState() const {
    std::scoped_lock lock{impl->mutex};
    return impl->open && !impl->faulted ? (impl->engaged ? 2 : 1) : 0;
}

void SysmixOutput::Pump() {
    auto& p = *impl;
    std::unique_lock lock{p.mutex, std::try_to_lock};
    if (!lock || !p.open || p.faulted || p.system.IsShuttingDown()) {
        return;
    }
    ++p.counters.ticks;
    u32 read{}, write{}, filled{};
    if (!p.ReadOccupancy(read, write, filled)) {
        p.Fault();
        return;
    }
    if (!p.engaged) {
        p.Notify(filled);
        return;
    }
    if (!p.host->IsReady()) {
        p.Fault(ResultOperationFailed);
        return;
    }
    const u32 taken = std::min(filled, BlockFrames);
    const u32 first = std::min(taken, p.slots - read);
    const u32 second = taken - first;
    auto& memory = p.process->GetMemory();
    if ((first && !memory.ReadBlock(p.buffer + PayloadOffset + u64{read} * p.frame_bytes,
                                    p.input.data(), first * p.frame_bytes)) ||
        (second && !memory.ReadBlock(p.buffer + PayloadOffset,
                                     p.input.data() + first * p.frame_bytes,
                                     second * p.frame_bytes))) {
        p.Fault();
        return;
    }
    p.counters.copied_frames += taken;
    p.output.fill(0);
    u32 nonzero{};
    for (u32 frame = 0; frame < taken; ++frame) {
        bool audible{};
        for (u32 channel = 0; channel < 2; ++channel) {
            const u32 index = frame * 2 + channel;
            f32 value;
            if (p.format == AudioCore::SampleFormat::PcmInt16) {
                s16 sample;
                std::memcpy(&sample, p.input.data() + index * sizeof(sample), sizeof(sample));
                value = static_cast<f32>(sample) / 32768.0f;
            } else {
                std::memcpy(&value, p.input.data() + index * sizeof(value), sizeof(value));
            }
            if (std::isfinite(value)) {
                const auto sample = static_cast<s32>(std::clamp(value * p.gain, -1.0f, 1.0f) * 32768.0f);
                p.output[index] = static_cast<s16>(std::clamp(sample, -32768, 32767));
                audible |= p.output[index] != 0;
            }
        }
        nonzero += audible;
    }
    AudioCore::Sink::SinkBuffer block{BlockFrames, 0, p.counters.admitted_blocks, false};
    if (!p.host->TryAppendOutputBlock(block, p.output)) {
        if (!p.host->IsReady()) {
            p.Fault(ResultOperationFailed);
            return;
        }
        ++p.counters.backpressure;
        p.counters.retried_frames += taken;
        if (p.counters.backpressure <= 3) {
            LOG_INFO(Service_Audio, "sysmix stream {} host full: accepted={} blocks={}", p.id,
                     p.counters.accepted_frames, p.counters.admitted_blocks);
        }
        if (p.counters.ticks % 400 == 0) {
            p.LogStatus(filled);
        }
        return;
    }
    ++p.counters.admitted_blocks;
    if (taken < BlockFrames) {
        p.counters.padded_frames += BlockFrames - taken;
    }
    const u32 next = (read + taken) % p.slots;
    u32 checked{};
    if (!memory.Read32Acquire(p.buffer + ReadOffset, checked) || checked != read ||
        !memory.WriteExclusive32(p.buffer + ReadOffset, next, read) ||
        !memory.Read32Acquire(p.buffer + ReadOffset, checked) || checked != next) {
        p.Fault();
        return;
    }
    p.consumer = next;
    p.counters.accepted_frames += taken;
    p.counters.nonzero_frames += nonzero;
    if (!p.ReadOccupancy(read, write, filled)) {
        p.Fault();
        return;
    }
    p.Notify(filled);
    if (p.counters.admitted_blocks <= 3 || p.counters.ticks % 400 == 0 ||
        (nonzero && p.counters.nonzero_frames == nonzero)) {
        p.LogStatus(filled);
    }
}

Kernel::KReadableEvent* SysmixOutput::GetStateEvent() {
    return impl->state_event ? &impl->state_event->GetReadableEvent() : nullptr;
}

Kernel::KReadableEvent* SysmixOutput::GetDataEvent() {
    return impl->data_event ? &impl->data_event->GetReadableEvent() : nullptr;
}

SysmixOutputCounters SysmixOutput::GetCounters() const {
    std::scoped_lock lock{impl->mutex};
    return impl->counters;
}

bool SysmixOutput::IsOwnedBy(const Kernel::KProcess* process) const {
    return impl->process.GetPointerUnsafe() == process;
}

} // namespace Service::Audio
