// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace VideoCommon {

constexpr bool CanUseSubmittedCommands(bool resumed, bool has_snapshot, bool gpu_written,
                                       bool safe_read_dirty) {
    return !resumed && has_snapshot && !gpu_written && !safe_read_dirty;
}

} // namespace VideoCommon
