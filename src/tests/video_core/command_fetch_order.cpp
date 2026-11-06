// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/command_fetch_order.h"
#include "common/settings.h"
#include "core/core.h"
#include "tests/core/system_test_context.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"

TEST_CASE("Command snapshots require fresh non-resumed CPU-only entries",
          "[video_core][command_fetch_order]") {
    for (unsigned bits = 0; bits < 16; ++bits) {
        const bool resumed = (bits & 1) != 0;
        const bool snapshot = (bits & 2) != 0;
        const bool written = (bits & 4) != 0;
        const bool dirty = (bits & 8) != 0;
        REQUIRE(VideoCommon::CanUseSubmittedCommands(resumed, snapshot, written, dirty) ==
                (bits == 2));
    }
}

TEST_CASE("Command fetch waits for a real queued writer after the last host sync",
          "[video_core][command_fetch_order][gpu_split]") {
    ScopedSystemTestContext context;
    Core::System system;
    auto& split = VideoCommon::GpuSplit::Get();
    const auto previous_backend = Settings::values.renderer_backend.GetValue();
    const auto previous_dma = Settings::values.dma_accuracy.GetValue();
    const auto previous_accuracy = Settings::values.current_gpu_accuracy;
    const bool previous_native = Tegra::g_nx2_ampere.exchange(true);
    struct Restore {
        VideoCommon::GpuSplit& split;
        Settings::RendererBackend backend;
        Settings::DmaAccuracy dma;
        Settings::GpuAccuracy accuracy;
        bool native;
        ~Restore() {
            split.Stop();
            Tegra::Nx2ClearGpuWrittenPages();
            Tegra::g_nx2_ampere = native;
            Settings::values.renderer_backend.SetValue(backend);
            Settings::values.dma_accuracy.SetValue(dma);
            Settings::values.current_gpu_accuracy = accuracy;
        }
    } restore{split, previous_backend, previous_dma, previous_accuracy, previous_native};
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    Settings::values.dma_accuracy.SetValue(Settings::DmaAccuracy::Unsafe);
    Settings::values.current_gpu_accuracy = Settings::GpuAccuracy::Low;
    split.MaybeStart(system);
    if (!split.Active()) {
        SKIP("GPU split is disabled by EDEN_NX2_SPLIT; use 1 or 2 to exercise queued writers");
    }

    constexpr u64 address = 0xa000;
    Tegra::Nx2ClearGpuWrittenPages();
    constexpr u32 snapshot = 0;
    u32 host = snapshot;
    split.NoteSync();
    split.Defer([&host] {
        host = 0x12345678;
        Tegra::Nx2MarkGpuWritten(address, sizeof(host));
    });
    split.NoteWriter();
    // One entry is below the ring's publish threshold: only the read-side wait publishes it.
    // The old min(writer, sync, write) target was zero and returned without running the writer.
    REQUIRE_FALSE(Tegra::Nx2IsGpuWritten(address));
    split.WaitForWriters();
    const bool written = Tegra::Nx2IsGpuWritten(address);
    CHECK(written);
    const bool use_snapshot =
        VideoCommon::CanUseSubmittedCommands(false, true, written, false);
    CHECK_FALSE(use_snapshot);
    REQUIRE((use_snapshot ? snapshot : host) == 0x12345678);
}
