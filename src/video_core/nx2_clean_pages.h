// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"

namespace VideoCommon {

/// EDEN_NX2_INLINE_INVALIDATE: 4 KiB pages a cache held nothing on, with the cache's page serial
/// at the time. A cache changes its serial whenever it starts holding something new.
struct Nx2CleanPages {
    static constexpr size_t Size = 512;
    static constexpr u64 PageBits = 12;
    /// Ranges over more pages aren't remembered (asking costs more than it saves).
    static constexpr u64 MaxPages = 16;

    struct Entry {
        u64 page = ~u64{0};
        u64 serial{};
    };
    std::array<Entry, Size> entries{};

    [[nodiscard]] bool Contains(DAddr addr, size_t size, u64 serial) const {
        for (u64 page = addr >> PageBits; page <= (addr + size - 1) >> PageBits; ++page) {
            const Entry& entry = entries[page % Size];
            if (entry.page != page || entry.serial != serial) {
                return false;
            }
        }
        return true;
    }

    /// clean_serial(addr, size) returns the cache's serial if it holds nothing there, else 0.
    template <typename CleanSerial>
    void Fill(DAddr addr, size_t size, CleanSerial&& clean_serial) {
        const u64 first = addr >> PageBits;
        const u64 last = (addr + size - 1) >> PageBits;
        if (last - first >= MaxPages) {
            return;
        }
        for (u64 page = first; page <= last; ++page) {
            if (const u64 serial = clean_serial(page << PageBits, size_t{1} << PageBits);
                serial != 0) {
                entries[page % Size] = {page, serial};
            }
        }
    }
};

} // namespace VideoCommon
