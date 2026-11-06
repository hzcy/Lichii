// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <span>

#include "common/alignment.h"
#include "common/common_types.h"

namespace Service {

// Pixel format IDs in InterModuleUserImagePreference word 6.
enum class Nx2NativeImageFormat : u32 {
    Rgba8 = 1,
    Rgb10A2 = 4,
    Rgba16Float = 5,
};

struct Nx2ImageLayout {
    u32 width;
    u32 height;
    u64 pitch;
    u32 block_height_log2;
    u64 size;
    u64 alignment;
};

inline Nx2ImageLayout Nx2CalculateImageLayout(const u32* preference) {
    Nx2ImageLayout layout{};
    layout.width = (preference[2] & 0xFFFF) + 1;
    layout.height = (preference[2] >> 16) + 1;
    const u64 pitch_gobs = preference[5];
    const u32 block_height_log2 = preference[4];
    const u32 bytes_per_pixel =
        static_cast<Nx2NativeImageFormat>(preference[6]) == Nx2NativeImageFormat::Rgba16Float ? 8 : 4;
    if (block_height_log2 <= 8 && pitch_gobs >= (u64{layout.width} * bytes_per_pixel + 63) / 64 &&
        pitch_gobs <= 4096) {
        layout.pitch = pitch_gobs * 64;
        layout.block_height_log2 = block_height_log2;
    } else {
        layout.pitch = Common::AlignUp(u64{layout.width} * bytes_per_pixel, 64);
        layout.block_height_log2 = 7;
    }
    const u64 block_rows = u64{1} << layout.block_height_log2;
    layout.size = layout.pitch * Common::AlignUp(u64{layout.height}, block_rows);
    layout.alignment = std::max<u64>(64 * block_rows, 0x1000);
    return layout;
}

inline Nx2ImageLayout Nx2CalculateNativeImageRequirements(std::span<const u32, 16> preference,
                                                         u32 count) {
    auto layout = Nx2CalculateImageLayout(preference.data());
    if (count != 1 || preference[0] != 0 || preference[1] != 2 || preference[3] != 0 ||
        preference[4] < 3 || preference[4] > 8 || preference[6] != 1 || preference[7] != 1 ||
        layout.pitch != u64{preference[5]} * 64 ||
        std::any_of(preference.begin() + 8, preference.end(), [](u32 value) { return value != 0; })) {
        return layout;
    }
    layout.alignment = 0x1000;
    return layout;
}

}
