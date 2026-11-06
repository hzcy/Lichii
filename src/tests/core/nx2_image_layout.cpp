// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_image_layout.h"

namespace {

constexpr std::array<u32, 16> ColorPreference{
    0, 2, (2160U - 1) << 16 | (3840U - 1), 0, 7, 240, 1, 1,
};

}

TEST_CASE("Native color image requirements accept page-aligned storage", "[nx2_image_layout]") {
    const auto layout = Service::Nx2CalculateNativeImageRequirements(ColorPreference, 1);
    REQUIRE(layout.width == 3840);
    REQUIRE(layout.height == 2160);
    REQUIRE(layout.pitch == 0x3C00);
    REQUIRE(layout.block_height_log2 == 7);
    REQUIRE(layout.size == 0x1FE0000);
    REQUIRE(layout.alignment == 0x1000);
    REQUIRE((0x1000 & (layout.alignment - 1)) == 0);
    REQUIRE((0x800 & (layout.alignment - 1)) != 0);
    REQUIRE(0x1000 + layout.size <= 0x3000000);
}

TEST_CASE("Native RGB10A2 storage preserves four-byte geometry", "[nx2_image_layout]") {
    auto preference = ColorPreference;
    preference[6] = 4;
    const auto layout = Service::Nx2CalculateImageLayout(preference.data());
    REQUIRE(layout.width == 3840);
    REQUIRE(layout.height == 2160);
    REQUIRE(layout.pitch == 0x3C00);
    REQUIRE(layout.size == 0x1FE0000);
    REQUIRE(layout.alignment == 0x2000);
}

TEST_CASE("Native FP16 storage uses eight-byte geometry", "[nx2_image_layout]") {
    auto preference = ColorPreference;
    preference[6] = 5;
    preference[5] = 480;
    const auto layout = Service::Nx2CalculateImageLayout(preference.data());
    REQUIRE(layout.width == 3840);
    REQUIRE(layout.height == 2160);
    REQUIRE(layout.pitch == 0x7800);
    REQUIRE(layout.block_height_log2 == 7);
    REQUIRE(layout.size == 0x3FC0000);
    REQUIRE(layout.alignment == 0x2000);
}

TEST_CASE("Native FP16 rejects a pitch sized for four-byte pixels", "[nx2_image_layout]") {
    auto preference = ColorPreference;
    preference[6] = 5;
    preference[4] = 3;
    const auto layout = Service::Nx2CalculateImageLayout(preference.data());
    REQUIRE(layout.pitch == 0x7800);
    REQUIRE(layout.block_height_log2 == 7);
    REQUIRE(layout.size == 0x3FC0000);
}

TEST_CASE("Native image base alignment does not shrink tile geometry", "[nx2_image_layout]") {
    auto preference = ColorPreference;
    preference[4] = 8;
    auto layout = Service::Nx2CalculateNativeImageRequirements(preference, 1);
    REQUIRE(layout.block_height_log2 == 8);
    REQUIRE(layout.size == 0x21C0000);
    REQUIRE(layout.alignment == 0x1000);

    preference = ColorPreference;
    preference[2] = (1080U - 1) << 16 | (1920U - 1);
    preference[5] = 120;
    layout = Service::Nx2CalculateNativeImageRequirements(preference, 1);
    REQUIRE(layout.size == 0x870000);
    REQUIRE(layout.alignment == 0x1000);

    const auto generic = Service::Nx2CalculateImageLayout(ColorPreference.data());
    REQUIRE(generic.size == 0x1FE0000);
    REQUIRE(generic.alignment == 0x2000);
}

TEST_CASE("Unrecognized native image requirements preserve existing layout", "[nx2_image_layout]") {
    for (u32 field = 0; field < ColorPreference.size(); ++field) {
        if (field == 2 || field == 4 || field == 5) {
            continue;
        }
        auto preference = ColorPreference;
        ++preference[field];
        const auto generic = Service::Nx2CalculateImageLayout(preference.data());
        const auto native = Service::Nx2CalculateNativeImageRequirements(preference, 1);
        CAPTURE(field);
        REQUIRE(native.size == generic.size);
        REQUIRE(native.alignment == generic.alignment);
    }
    for (const u32 count : {0U, 2U, 3U}) {
        const auto layout = Service::Nx2CalculateNativeImageRequirements(ColorPreference, count);
        REQUIRE(layout.alignment == 0x2000);
    }
    for (const u32 exponent : {0U, 2U, 9U}) {
        auto preference = ColorPreference;
        preference[4] = exponent;
        const auto generic = Service::Nx2CalculateImageLayout(preference.data());
        const auto native = Service::Nx2CalculateNativeImageRequirements(preference, 1);
        REQUIRE(native.size == generic.size);
        REQUIRE(native.alignment == generic.alignment);
    }
    for (const u32 pitch : {0U, 239U, 4097U}) {
        auto preference = ColorPreference;
        preference[5] = pitch;
        const auto generic = Service::Nx2CalculateImageLayout(preference.data());
        const auto native = Service::Nx2CalculateNativeImageRequirements(preference, 1);
        REQUIRE(native.size == generic.size);
        REQUIRE(native.alignment == generic.alignment);
    }
}
