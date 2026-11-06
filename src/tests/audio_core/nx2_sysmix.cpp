// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include "core/core_timing.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/audio/nx2_audio_device_manager.h"
#include "core/hle/service/audio/nx2_sysmix.h"
#include "tests/audio_core/sysmix_test_context.h"

using Service::Audio::SysmixOutput;
using SysmixTests::Context;

namespace {
void Cursor(Context& context, u32 read, u32 write) {
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x44, &read, sizeof(read)));
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1004, &write, sizeof(write)));
}
u32 ReadCursor(Context& context) {
    u32 read{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    return read;
}
void Sample(Context& context, u32 slot, f32 left, f32 right) {
    const std::array<f32, 2> frame{left, right};
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040 + u64{slot} * 8,
                                        frame.data(), sizeof(frame)));
}
}

TEST_CASE("sysmix validates the exact F32 ring allocation and rejects unsupported formats", "[sysmix]") {
    Context context;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(3, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 1, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 44100, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 1, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 1).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 0, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 0xffffffff, 0, 0).IsFailure());
    REQUIRE(context.sink.last == nullptr);
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(context.process->GetReferenceCount() == 2);
    REQUIRE(SysmixOutput::ValidateRegion(*context.process, Context::Base + 1,
                                         Context::Size, 0x24).IsFailure());
    REQUIRE(SysmixOutput::ValidateRegion(*context.process, Context::Base,
                                         Context::Size, 4).IsFailure());
    SysmixOutput short_stream{context.system, *context.process, context.sink, 2, Context::Base,
                              0x5000, 0x24};
    REQUIRE(short_stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
}

TEST_CASE("sysmix copies the physical extra slot and wrap then publishes consumer", "[sysmix]") {
    Context context;
    Cursor(context, 2048, 1);
    Sample(context, 2048, 0.5f, -0.5f);
    Sample(context, 0, 1.0f, -1.0f);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 2);
    REQUIRE(stream.GetCounters().nonzero_frames == 2);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 16384);
    REQUIRE(output[1] == -16384);
    REQUIRE(output[2] == 32767);
    REQUIRE(output[3] == -32768);
    REQUIRE(std::ranges::all_of(std::span{output}.subspan(4), [](s16 value) { return value == 0; }));
    REQUIRE(context.backing[0x5048] == 0);
    REQUIRE(context.backing[0x5fff] == 0);
}

TEST_CASE("sysmix plays the Sports Resort Q15 movie ring including wrap and gain", "[sysmix]") {
    Context context;
    Cursor(context, 960, 1);
    const std::array<s16, 2> last{16384, -16384};
    const std::array<s16, 2> first{32767, -32768};
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040 + 960 * 4,
                                        last.data(), sizeof(last)));
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040, first.data(), sizeof(first)));
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        0x3000, 0x24};
    REQUIRE(stream.Open(2, 2, 48000, 960, 0, 0).IsSuccess());
    REQUIRE(stream.SetGain(0.5f, 0).IsSuccess());
    REQUIRE(stream.SetDataNotification(2, 480).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 2);
    REQUIRE(stream.GetCounters().nonzero_frames == 2);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 8192);
    REQUIRE(output[1] == -8192);
    REQUIRE(output[2] == 16383);
    REQUIRE(output[3] == -16384);
    REQUIRE(std::ranges::all_of(std::span{output}.subspan(4), [](s16 value) { return value == 0; }));
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
}

TEST_CASE("sysmix advertises only implemented stereo PCM formats", "[sysmix]") {
    REQUIRE(SysmixOutput::IsFormatSupported(2, 2, 48000));
    REQUIRE(SysmixOutput::IsFormatSupported(5, 2, 48000));
    for (const u32 format : {0U, 1U, 3U, 4U, 6U, 258U, 261U}) {
        REQUIRE_FALSE(SysmixOutput::IsFormatSupported(format, 2, 48000));
    }
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 1, 48000));
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 6, 48000));
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 2, 44100));
}

TEST_CASE("sysmix mode2 event uses actual free frames even before engagement", "[sysmix]") {
    Context context;
    Cursor(context, 0, 600);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetDataNotification(1, 1536).IsFailure());
    REQUIRE(stream.SetDataNotification(2, -1).IsFailure());
    REQUIRE(stream.SetDataNotification(2, 2049).IsFailure());
    REQUIRE(stream.SetDataNotification(2, 1536).IsSuccess());
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    stream.Pump();
    REQUIRE(ReadCursor(context) == 0);
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 240);
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetEngaged(false).IsSuccess());
    REQUIRE(stream.GetDataEvent()->Clear(context.system.Kernel()).IsSuccess());
    stream.Pump();
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetDataNotification(0, 0).IsSuccess());
    REQUIRE(stream.GetDataEvent()->Clear(context.system.Kernel()).IsSuccess());
    stream.Pump();
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
}

TEST_CASE("sysmix backpressure leaves guest ownership unchanged and resumes in order", "[sysmix]") {
    Context context;
    Cursor(context, 0, 2048);
    for (u32 i = 0; i < 2048; ++i) {
        Sample(context, i, 0.25f, -0.25f);
    }
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    for (u32 i = 0; i < 3; ++i) {
        stream.Pump();
    }
    REQUIRE(ReadCursor(context) == 720);
    stream.Pump();
    REQUIRE(ReadCursor(context) == 720);
    REQUIRE(stream.GetCounters().accepted_frames == 720);
    REQUIRE(stream.GetCounters().backpressure == 1);
    REQUIRE(stream.GetCounters().copied_frames == 960);
    REQUIRE(stream.GetCounters().retried_frames == 240);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    stream.Pump();
    REQUIRE(ReadCursor(context) == 960);
    REQUIRE(context.sink.last->GetQueueSize() == 3);
}

TEST_CASE("sysmix pads a short guest block with silence for the host", "[sysmix]") {
    Context context;
    Cursor(context, 0, 100);
    for (u32 i = 0; i < 100; ++i) {
        Sample(context, i, 0.25f, -0.25f);
    }
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 100);
    REQUIRE(stream.GetCounters().accepted_frames == 100);
    REQUIRE(stream.GetCounters().padded_frames == 140);
    REQUIRE(stream.GetCounters().admitted_blocks == 1);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    for (u32 i = 0; i < 200; ++i) {
        REQUIRE(output[i] == (i % 2 == 0 ? 8192 : -8192));
    }
    REQUIRE(std::ranges::all_of(std::span{output}.subspan(200),
                                [](s16 value) { return value == 0; }));
}

TEST_CASE("sysmix detects unmapped cursor publication even though exclusive store reports success", "[sysmix]") {
    Context context;
    Cursor(context, 0, 240);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    context.sink.last->on_admit = [&] { context.Unmap(Context::Base); };
    stream.Pump();
    REQUIRE(context.Memory().WriteExclusive32(Context::Base + 0x44, 240, 0));
    u32 read{};
    REQUIRE_FALSE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    REQUIRE(stream.GetCounters().accepted_frames == 0);
    REQUIRE(stream.GetCounters().faults == 1);
    REQUIRE(stream.GetCounters().admitted_blocks == 1);
    REQUIRE(stream.GetState() == 0);
    std::array<s16, 480> output{};
    output.fill(99);
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(std::ranges::all_of(output, [](s16 value) { return value == 0; }));
    stream.Pump();
    REQUIRE(stream.GetCounters().faults == 1);
}

TEST_CASE("sysmix rejects unmapped payload and tampered cursors without admission", "[sysmix]") {
    Context context;
    Cursor(context, 2048, 1);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    SECTION("missing payload page") {
        context.Unmap(Context::Base + 0x5000);
    }
    SECTION("producer outside physical modulus") {
        Cursor(context, 2048, 2049);
    }
    SECTION("consumer changed by guest") {
        Cursor(context, 0, 1);
    }
    stream.Pump();
    REQUIRE(stream.GetCounters().faults == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 0);
    REQUIRE(context.sink.last->GetQueueSize() == 0);
}

TEST_CASE("sysmix close releases host stream and allows reopening without stale samples", "[sysmix]") {
    Context context;
    Cursor(context, 0, 1);
    Sample(context, 0, 0.5f, 0.5f);
    {
        SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                            Context::Size, 0x24};
        REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
        REQUIRE(stream.SetEngaged(true).IsSuccess());
        stream.Pump();
        REQUIRE(stream.SetEngaged(false).IsSuccess());
        REQUIRE(context.sink.last->GetQueueSize() == 1);
        stream.Close();
        REQUIRE(context.sink.last == nullptr);
        const auto ticks = stream.GetCounters().ticks;
        stream.Pump();
        REQUIRE(stream.GetCounters().ticks == ticks);
        Cursor(context, 0, 0);
        REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
        REQUIRE(context.sink.last->GetQueueSize() == 0);
    }
    REQUIRE(context.sink.closed == 2);
    REQUIRE(context.process->GetReferenceCount() == 1);
}

TEST_CASE("sysmix requires initialized and successfully started host output", "[sysmix]") {
    Context context;
    context.sink.ready = false;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(context.sink.last == nullptr);
    context.sink.ready = true;
    context.sink.fail_start = true;
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsFailure());
    stream.Pump();
    REQUIRE(stream.GetCounters().admitted_blocks == 0);
}

TEST_CASE("sysmix conversion handles gain clipping and nonfinite values", "[sysmix]") {
    Context context;
    Cursor(context, 0, 2);
    Sample(context, 0, 2.0f, -2.0f);
    Sample(context, 1, std::numeric_limits<f32>::quiet_NaN(),
           std::numeric_limits<f32>::infinity());
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetGain(0.25f, 1).IsFailure());
    REQUIRE(stream.SetGain(0.25f, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 16384);
    REQUIRE(output[1] == -16384);
    REQUIRE(output[2] == 0);
    REQUIRE(output[3] == 0);
}

TEST_CASE("sysmix child lifetime retains registry after user session closes", "[sysmix]") {
    Context context;
    using namespace Service::Audio;
    auto registry = std::make_shared<Nx2SysmixStreams>(context.system);
    std::weak_ptr<Nx2SysmixStreams> weak = registry;
    auto output = std::make_shared<SysmixOutput>(context.system, *context.process, context.sink,
                                                1, Context::Base, Context::Size, 0x24);
    registry->streams.emplace(1, output);
    auto child = std::make_shared<INx2SysmixStreamSession>(context.system, registry, output, 1);
    auto user = std::make_shared<INx2SysmixUserSession>(context.system, registry, &*context.process);
    registry.reset();
    user.reset();
    REQUIRE_FALSE(weak.expired());
    REQUIRE(output->Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(weak.lock()->SetEngaged(1, true, nullptr).IsFailure());
    REQUIRE(weak.lock()->SetEngaged(1, true, &*context.process).IsSuccess());
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().admitted_blocks == 1);
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 10;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().ticks == 2);
    REQUIRE(output->GetCounters().admitted_blocks == 2);
    child.reset();
    REQUIRE(weak.expired());
    REQUIRE(output->GetState() == 0);
    REQUIRE(context.sink.last == nullptr);
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().ticks == 2);
}

TEST_CASE("sysmix does not report successful disengage after backend stop failure", "[sysmix]") {
    Context context;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    context.sink.last->fail_stop = true;
    REQUIRE(stream.SetEngaged(false).IsFailure());
    REQUIRE(stream.GetState() == 0);
    REQUIRE(stream.GetCounters().faults == 1);
    stream.Pump();
    REQUIRE(stream.GetCounters().admitted_blocks == 0);
}
