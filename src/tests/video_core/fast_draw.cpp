// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_clean_pages.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace FastDraw = VideoCommon::Nx2FastDraw;
constexpr u32 ViewportScaleX = MAXWELL3D_REG_INDEX(viewport_transform);

class NoDma final : public Tegra::Engines::AccelerateDMAInterface {
public:
    bool BufferCopy(GPUVAddr, GPUVAddr, u64) override { return true; }
    bool BufferClear(GPUVAddr, u64, u32) override { return true; }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy&, const Tegra::DMA::ImageOperand&,
                       const Tegra::DMA::BufferOperand&) override { return true; }
    bool BufferToImage(const Tegra::DMA::ImageCopy&, const Tegra::DMA::BufferOperand&,
                       const Tegra::DMA::ImageOperand&) override { return true; }
};

/// Records the fast draw serial each draw sees.
class SerialRasterizer final : public VideoCore::RasterizerInterface {
public:
    std::vector<u64> draws;

    void Draw(bool, u32) override { draws.push_back(FastDraw::Serial()); }
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&& func) override { func(); }
    void SyncOperation(std::function<void()>&& func) override { func(); }
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override { return false; }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(PAddr, u64) override {}
    bool OnCPUWrite(PAddr, u64) override { return false; }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override { return dma; }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    NoDma dma;
};

struct FastDrawFixture {
    FastDrawFixture() : memory{system, device_memory, 32, 1ULL << 31}, maxwell3d{memory} {
        FastDraw::g_enabled = true;
        maxwell3d.BindRasterizer(&rasterizer);
    }
    ~FastDrawFixture() {
        FastDraw::g_enabled = previous;
    }

    u32 Reg(u32 method) const {
        return maxwell3d.regs.reg_array[method];
    }

    void Write(u32 method, u32 value) {
        maxwell3d.CallMethod(system, method, value, true);
    }

    void DrawArrays(u32 first, u32 count) {
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.first), first);
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.count), count);
        Write(MAXWELL3D_REG_INDEX(draw.begin), 0x4); // first instance, triangles
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    /// Whether the last two draws saw the same serial (the second could reuse the first).
    bool Reusable() const {
        const auto& draws = rasterizer.draws;
        REQUIRE(draws.size() >= 2);
        return draws[draws.size() - 1] == draws[draws.size() - 2];
    }

    const bool previous = FastDraw::g_enabled;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory;
    SerialRasterizer rasterizer;
    Maxwell3D maxwell3d;
};

} // namespace

TEST_CASE("Fast draw keeps the serial across draws that change only draw parameters",
          "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.DrawArrays(3, 6);
    REQUIRE(f.rasterizer.draws.size() == 2);
    CHECK(f.Reusable());

    f.Write(ViewportScaleX, f.Reg(ViewportScaleX));
    f.DrawArrays(9, 3);
    CHECK(f.Reusable());

    f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), 0x40);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), 0x1000);
    f.DrawArrays(12, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw breaks reuse on anything that changes what a draw binds", "[fast_draw]") {
    FastDrawFixture f;
    const auto breaks = [&](auto&& change) {
        f.DrawArrays(0, 3);
        change();
        f.DrawArrays(0, 3);
        return !f.Reusable();
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.Write(ViewportScaleX,
                f.Reg(ViewportScaleX) + 1);
    }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 0x12345678); }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config), 0x11); }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(clear_surface), 0x3c); }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(tiled_cache_barrier), 0); }));
    CHECK(breaks([&] { FastDraw::Break(FastDraw::Reason::Fence); }));
}

TEST_CASE("Fast draw serial never moves while disabled", "[fast_draw]") {
    FastDrawFixture f;
    FastDraw::g_enabled = false;
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(blend.enable[0]), f.Reg(MAXWELL3D_REG_INDEX(blend.enable[0])) ^ 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

namespace {

struct DeltaFixture : FastDrawFixture {
    DeltaFixture() {
        FastDraw::g_delta_enabled = true;
        FastDraw::g_delta = {};
    }
    ~DeltaFixture() {
        FastDraw::g_delta_enabled = previous_delta;
        FastDraw::g_delta = {};
    }

    void SetConstantBuffer(GPUVAddr address, u32 size, u32 offset) {
        Write(MAXWELL3D_REG_INDEX(const_buffer.size), size);
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_high), static_cast<u32>(address >> 32));
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), static_cast<u32>(address));
        Write(MAXWELL3D_REG_INDEX(const_buffer.offset), offset);
    }

    const bool previous_delta = FastDraw::g_delta_enabled;
};

} // namespace

TEST_CASE("Fast draw delta mode records index buffer, bind and constant data changes",
          "[fast_draw]") {
    DeltaFixture f;
    f.DrawArrays(0, 3);
    const u64 serial = FastDraw::Serial();
    FastDraw::g_delta = {};

    f.Write(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low),
            f.Reg(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low)) + 0x100);
    CHECK(FastDraw::g_delta.index_buffer);

    f.SetConstantBuffer(0x10000, 0x100, 0);
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31); // slot 3, valid
    CHECK(FastDraw::g_delta.binds[4] == 1U << 3);
    FastDraw::g_delta.binds[4] = 0;
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31);
    CHECK(FastDraw::g_delta.binds[4] == 0);

    f.SetConstantBuffer(0x20000, 0x100, 0x10);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1, 2);
    REQUIRE(FastDraw::g_delta.num_writes == 1);
    CHECK(FastDraw::g_delta.write_begin[0] == 0x20010);
    CHECK(FastDraw::g_delta.write_end[0] == 0x20018);

    CHECK(FastDraw::Serial() == serial);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
    CHECK_FALSE(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw delta mode still breaks on state, texture headers and overflow",
          "[fast_draw]") {
    DeltaFixture f;
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), 0x40000);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_low), 0x50000);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.limit), 7);
    const auto breaks = [&](auto&& change) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        change();
        return FastDraw::Serial() != serial;
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0xe0); // the last texture header
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x50000, 0x1000, 0);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK_FALSE(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0x100); // just past the headers
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x60000, 0x1000, 0);
        for (u32 i = 0; i <= FastDraw::Delta::MaxWrites; ++i) {
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), i * 0x20);
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), i);
        }
    }));
}

TEST_CASE("Fast draw mode 1 breaks on what delta mode records", "[fast_draw]") {
    FastDrawFixture f;
    REQUIRE_FALSE(FastDraw::g_delta_enabled);
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low),
            f.Reg(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low)) + 0x100);
    f.DrawArrays(0, 3);
    CHECK_FALSE(f.Reusable());
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config), 0x11);
    f.DrawArrays(0, 3);
    CHECK_FALSE(f.Reusable());
    CHECK(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw keeps base vertex and instance as draw parameters", "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(global_base_vertex_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_vertex_index)) + 64);
    f.Write(MAXWELL3D_REG_INDEX(global_base_instance_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_instance_index)) + 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw delta mode records vertex stream addresses and sizes", "[fast_draw]") {
    DeltaFixture f;
    const auto stream = [](u32 index, u32 word) {
        return static_cast<u32>(MAXWELL3D_REG_INDEX(vertex_streams) + index * 4 + word);
    };
    const auto light = [&](u32 method) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.Write(method, f.Reg(method) + 0x40);
        return FastDraw::Serial() == serial && FastDraw::g_delta.vertex_buffers;
    };
    CHECK(light(stream(0, 1)));
    CHECK(light(stream(3, 2)));
    CHECK(light(0x600 / 4 + 1));          // SET_VERTEX_STREAM_SIZE_B(0)
    CHECK(light(0x600 / 4 + 31 * 2));     // SET_VERTEX_STREAM_SIZE_A(31)
    CHECK(light(MAXWELL3D_REG_INDEX(vertex_stream_limits) + 5 * 2 + 1));
    CHECK_FALSE(light(stream(0, 0)));     // stride and enable
    CHECK_FALSE(light(stream(0, 3)));     // frequency
    CHECK_FALSE(light(0x600 / 4 + 64));   // past the sizes
}

TEST_CASE("Clean pages remember only pages a cache held nothing on", "[fast_draw]") {
    VideoCommon::Nx2CleanPages pages;
    constexpr DAddr base = 0x10'0000;
    constexpr size_t Page = 0x1000;
    u64 serial = 7;
    std::vector<DAddr> touched{base + Page};
    const auto clean_serial = [&](DAddr addr, size_t size) -> u64 {
        for (const DAddr held : touched) {
            if (held < addr + size && addr < held + 4) {
                return 0;
            }
        }
        return serial;
    };
    CHECK_FALSE(pages.Contains(base, 0x40, serial));
    pages.Fill(base, Page * 2, clean_serial);
    CHECK(pages.Contains(base + 0x100, 0x40, serial));
    // The second page holds something: a range reaching into it is still invalidated.
    CHECK_FALSE(pages.Contains(base + Page - 0x20, 0x40, serial));
    CHECK_FALSE(pages.Contains(base + Page, 0x40, serial));
    // The cache starts holding something new: nothing remembered counts.
    ++serial;
    CHECK_FALSE(pages.Contains(base + 0x100, 0x40, serial));
    // Long ranges aren't remembered.
    touched.clear();
    pages.Fill(base, Page * 17, clean_serial);
    CHECK_FALSE(pages.Contains(base, 0x40, serial));
    pages.Fill(base, Page * 16, clean_serial);
    CHECK(pages.Contains(base, Page * 16, serial));
    // A page that maps to the same entry replaces it.
    pages.Fill(base + Page * VideoCommon::Nx2CleanPages::Size, 0x40, clean_serial);
    CHECK_FALSE(pages.Contains(base, 0x40, serial));
}
