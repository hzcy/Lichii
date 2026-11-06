// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include <SDL3/SDL.h>

#include "common/input.h"

namespace InputCommon {

/// True when the build includes RenHDrum support (ENABLE_RENHDRUM).
bool IsRenHDrumBuilt();

/// True when renhdrum.dll (librenhdrum.so / .dylib) was found and loaded. Loads it on first call.
bool IsRenHDrumLoaded();

/**
 * Vibration for one SDL controller through RenHDrum (https://github.com/trzz1214/RenHDrum):
 * Switch Pro Controller / Joy-Con get HD Rumble packets, a wired DualSense gets audio haptics on
 * its voice coils. Settings come from Settings::values and are re-read on every vibration.
 */
class RenHDrumPad {
public:
    /// nullptr when RenHDrum is not loaded or the controller is not one it drives.
    static std::shared_ptr<RenHDrumPad> Create(SDL_Joystick* joystick, SDL_Gamepad* gamepad);

    ~RenHDrumPad();

    RenHDrumPad(const RenHDrumPad&) = delete;
    RenHDrumPad& operator=(const RenHDrumPad&) = delete;

    /// Releases the engine and the audio stream. Call before the SDL joystick is closed.
    void Shutdown();

    /// True while RenHDrum plays this controller's vibration.
    bool IsActive() const;

    /// HLE thread. True when RenHDrum took the vibration; false means use the standard rumble.
    bool Vibrate(std::int64_t now_us, bool is_right, const Common::Input::VibrationStatus& status);

    /// Input thread, about once per millisecond: due HD Rumble packets and DualSense audio retries.
    void Tick(std::int64_t now_us);

private:
    struct Impl;
    explicit RenHDrumPad(std::unique_ptr<Impl> impl_);

    std::unique_ptr<Impl> impl;
};

} // namespace InputCommon
