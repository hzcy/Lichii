// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

/// EDEN_NX2_SAMPLE_PROFILE=<hz> (Windows): a background thread suspends each sampled thread
/// <hz> times a second, unwinds its stack (up to 32 frames) and appends the return addresses to
/// nx2_samples_<pid>.bin in the log directory; tools/nx2/sample_report.py symbolizes and folds
/// them. Sampled threads are picked by name when they set it (EDEN_NX2_SAMPLE_THREADS, a comma
/// list where a trailing * matches a prefix; default GPU,GPUBack,VulkanWorker; adding CPUCore_*
/// samples the guest cores). Sampling starts after EDEN_NX2_SAMPLE_DELAY seconds (default 0)
/// and stops after EDEN_NX2_SAMPLE_SECONDS (default 20). It turns EDEN_NX2_GPU_PROFILE's zone
/// timers off so they don't show up in the samples.
namespace Common::Nx2SampleProfile {

[[nodiscard]] bool Requested();

/// From SetCurrentThreadName: samples the calling thread if its name is selected, and starts
/// the sampler with the first one.
void OnThreadNamed(const char* name);

} // namespace Common::Nx2SampleProfile
