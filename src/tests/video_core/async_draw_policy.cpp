// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/async_draw_policy.h"

TEST_CASE("Native pending draws are never discarded", "[video_core][async_draw]") {
    constexpr std::array<u32, 6> counts{0, 1, 6, 7, 4096, std::numeric_limits<u32>::max()};
    for (const bool depth : {false, true}) {
        for (const u32 indices : counts) {
            for (const u32 vertices : counts) {
                REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, depth, indices, vertices));
            }
        }
    }
}

TEST_CASE("Native pending draws follow the async heuristic when opted in",
          "[video_core][async_draw]") {
    REQUIRE(Vulkan::CanDiscardPendingDraw(true, true, 0, 0, true));
    REQUIRE(Vulkan::CanDiscardPendingDraw(true, false, 7, 7, true));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 6, 100, true));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(true, false, 0, 0, true));
    // The opt-in never changes non-native draws.
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, true, 0, 0, true) ==
            Vulkan::CanDiscardPendingDraw(false, true, 0, 0));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 0, 0, true));
}

TEST_CASE("Legacy pending draws retain the async heuristic", "[video_core][async_draw]") {
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, true, 0, 0));
    REQUIRE(Vulkan::CanDiscardPendingDraw(false, false, 7, 7));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 6, 100));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 100, 6));
    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(false, false, 0, 0));
}

TEST_CASE("Deferred shader emission never makes a pending draw discardable",
          "[video_core][async_draw]") {
    constexpr std::array<u32, 6> counts{0, 1, 6, 7, 4096, std::numeric_limits<u32>::max()};
    for (const bool native : {false, true}) {
        if (!Vulkan::DefersShaderEmission(native, 0, 1)) {
            continue;
        }
        for (const bool depth : {false, true}) {
            for (const u32 indices : counts) {
                for (const u32 vertices : counts) {
                    REQUIRE_FALSE(Vulkan::CanDiscardPendingDraw(native, depth, indices, vertices));
                }
            }
        }
    }
    REQUIRE(Vulkan::DefersShaderEmission(true, 0, 1));
    REQUIRE_FALSE(Vulkan::DefersShaderEmission(false, 0, 1));
}

TEST_CASE("Deferred emission falls back when its worker budget is full",
          "[video_core][async_draw]") {
    for (const size_t capacity : {1U, 2U, 23U}) {
        for (size_t pending = 0; pending < capacity; ++pending) {
            REQUIRE(Vulkan::DefersShaderEmission(true, pending, capacity));
        }
        REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, capacity, capacity));
        REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, capacity + 1, capacity));
    }
    REQUIRE_FALSE(Vulkan::DefersShaderEmission(true, 0, 0));
}
