// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QDialog>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QListWidget;
class QSlider;
class QSpinBox;
class QStackedWidget;

/// Settings for RenHDrum: HD Rumble on Switch Pro Controller / Joy-Con and the DualSense voice
/// coil haptics. The SDL driver reads them on every vibration, so connected controllers pick up
/// a change as soon as the dialog is accepted.
class ConfigureHdRumble : public QDialog {
    Q_OBJECT

public:
    explicit ConfigureHdRumble(QWidget* parent);
    ~ConfigureHdRumble() override;

    void ApplyConfiguration();

private:
    QWidget* CreateSwitchPage();
    QWidget* CreateDualSensePage();
    void LoadConfiguration();
    void RestoreDefaults();
    void UpdateEnabledState();

    QCheckBox* hd_enabled{};
    QSlider* gain_slider{};
    QSpinBox* gain{};
    QSlider* boost_slider{};
    QSpinBox* boost{};
    QListWidget* page_list{};
    QStackedWidget* pages{};

    QGroupBox* wired_group{};
    QSpinBox* usb_interval{};
    QCheckBox* envelope_5ms{};
    QCheckBox* frequency_in_packet{};
    QCheckBox* crisp_notes{};

    QGroupBox* bt_group{};
    QComboBox* bt_level{};
    QCheckBox* bt_after_input{};
    QCheckBox* bt_fast{};
    QCheckBox* bt_crisp_notes{};

    QCheckBox* ds_enabled{};
    QSlider* ds_volume_slider{};
    QSpinBox* ds_volume{};
    QCheckBox* ds_crisp_notes{};
};
