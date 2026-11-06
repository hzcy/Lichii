// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "video_core/gpu_split.h"

namespace VideoCommon {

/// What the split's back thread is blocked in, or null. Scheduler::WaitSubmitted names it when
/// it has waited 2 s for the back to submit (the back serves that request between entries, so a
/// back blocked in one entry never does).
inline std::atomic<const char*> back_wait_site{nullptr};

/// Marks a blocking wait on the back thread for the watchdog; nothing on other threads.
class BackWaitScope {
public:
    explicit BackWaitScope(const char* site) : active{GpuSplit::OnBackThread()} {
        if (active) {
            previous = back_wait_site.exchange(site, std::memory_order_relaxed);
        }
    }
    ~BackWaitScope() {
        if (active) {
            back_wait_site.store(previous, std::memory_order_relaxed);
        }
    }
    BackWaitScope(const BackWaitScope&) = delete;
    BackWaitScope& operator=(const BackWaitScope&) = delete;

private:
    bool active;
    const char* previous{};
};

} // namespace VideoCommon
