// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>

#include "common/settings.h"
#include "input_common/drivers/renhdrum_pad.h"
#include "yuzu/configuration/configure_hd_rumble.h"

namespace {
// Room for "1000%" plus the arrows, whatever the font and scaling
void FitPercentSpinBox(QSpinBox* box) {
    box->setMinimumWidth(box->fontMetrics().horizontalAdvance(QStringLiteral("10000%")) + 40);
}

QLabel* MakeNote(const QString& text) {
    auto* label = new QLabel(text);
    label->setWordWrap(true);
    label->setStyleSheet(QStringLiteral("color: palette(mid);"));
    return label;
}
} // namespace

ConfigureHdRumble::ConfigureHdRumble(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("HD Rumble"));
    setMinimumSize(640, 460);

    auto* root = new QVBoxLayout(this);

    // Header: master switch
    auto* header = new QHBoxLayout;
    auto* title = new QLabel(tr("<b>HD Rumble</b>"));
    hd_enabled = new QCheckBox(tr("Enabled"));
    hd_enabled->setToolTip(tr("Off: the controllers use the emulator's standard rumble."));
    header->addWidget(title);
    header->addStretch();
    header->addWidget(hd_enabled);
    root->addLayout(header);
    if (!InputCommon::IsRenHDrumLoaded()) {
        auto* missing = new QLabel(
            tr("renhdrum.dll was not found next to the emulator, so the controllers use the "
               "standard rumble. Get it from <a href=\"https://github.com/trzz1214/RenHDrum/"
               "releases\">the RenHDrum releases</a> and restart the emulator."));
        missing->setWordWrap(true);
        missing->setOpenExternalLinks(true);
        root->addWidget(missing);
    }

    // Intensity: all controllers. Saved per game when the game has its own value.
    auto* gain_row = new QHBoxLayout;
    gain_slider = new QSlider(Qt::Horizontal);
    gain_slider->setRange(25, 1000);
    gain = new QSpinBox;
    gain->setRange(25, 1000);
    gain->setSingleStep(25);
    gain->setSuffix(QStringLiteral("%"));
    FitPercentSpinBox(gain);
    connect(gain_slider, &QSlider::valueChanged, gain, &QSpinBox::setValue);
    connect(gain, qOverload<int>(&QSpinBox::valueChanged), gain_slider, &QSlider::setValue);
    gain_row->addWidget(new QLabel(tr("Intensity")));
    gain_row->addWidget(gain_slider, 1);
    gain_row->addWidget(gain);
    root->addLayout(gain_row);

    auto* boost_row = new QHBoxLayout;
    boost_slider = new QSlider(Qt::Horizontal);
    boost_slider->setRange(0, 100);
    boost = new QSpinBox;
    boost->setRange(0, 100);
    boost->setSingleStep(5);
    boost->setSuffix(QStringLiteral("%"));
    FitPercentSpinBox(boost);
    boost->setToolTip(tr("Lifts weak vibrations more than strong ones. 0% keeps the game's "
                         "curve."));
    connect(boost_slider, &QSlider::valueChanged, boost, &QSpinBox::setValue);
    connect(boost, qOverload<int>(&QSpinBox::valueChanged), boost_slider, &QSlider::setValue);
    boost_row->addWidget(new QLabel(tr("Weak rumble boost")));
    boost_row->addWidget(boost_slider, 1);
    boost_row->addWidget(boost);
    root->addLayout(boost_row);
    const bool per_game = !Settings::values.hd_rumble_gain.UsingGlobal();
    root->addWidget(MakeNote(
        per_game ? tr("Intensity: this game has its own value, and changes here apply to it.")
                 : tr("Intensity multiplies the game's rumble. Some Switch 2 games (Ocarina of "
                      "Time) send very weak values that the Joy-Con 2 still plays strongly: "
                      "try 400-600% for them.")));

    // Body: controller list on the left, its settings on the right
    auto* body = new QHBoxLayout;
    page_list = new QListWidget;
    page_list->addItem(tr("Switch Controller"));
    page_list->addItem(tr("DualSense"));
    page_list->setFixedWidth(170);
    pages = new QStackedWidget;
    pages->addWidget(CreateSwitchPage());
    pages->addWidget(CreateDualSensePage());
    connect(page_list, &QListWidget::currentRowChanged, pages, &QStackedWidget::setCurrentIndex);
    page_list->setCurrentRow(0);
    body->addWidget(page_list);
    body->addWidget(pages, 1);
    root->addLayout(body, 1);

    root->addWidget(MakeNote(
        tr("Changes reach connected controllers on their next vibration. To feel them now, open "
           "the Vibration settings and hold a button on the controller.")));

    auto* footer = new QHBoxLayout;
    auto* credit = new QLabel(tr("Powered by RenHDrum"));
    credit->setStyleSheet(QStringLiteral("color: palette(mid);"));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel |
                                         QDialogButtonBox::RestoreDefaults);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked, this,
            &ConfigureHdRumble::RestoreDefaults);
    footer->addWidget(credit);
    footer->addStretch();
    footer->addWidget(buttons);
    root->addLayout(footer);

    connect(hd_enabled, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);
    connect(ds_enabled, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);
    connect(envelope_5ms, &QCheckBox::toggled, this, &ConfigureHdRumble::UpdateEnabledState);

    LoadConfiguration();
}

ConfigureHdRumble::~ConfigureHdRumble() = default;

QWidget* ConfigureHdRumble::CreateSwitchPage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(MakeNote(tr("Pro Controller and Joy-Con: the game's HD Rumble sent with "
                                  "full detail instead of a plain motor rumble.")));

    wired_group = new QGroupBox(tr("Wired (USB)"));
    auto* wired = new QFormLayout(wired_group);
    usb_interval = new QSpinBox;
    usb_interval->setRange(5, 100);
    usb_interval->setSuffix(tr(" ms"));
    usb_interval->setToolTip(tr("Time between rumble packets. 15 ms is the console's rate."));
    envelope_5ms = new QCheckBox(tr("5 ms envelope (three samples per packet)"));
    frequency_in_packet = new QCheckBox(tr("Frequency changes inside a packet"));
    crisp_notes = new QCheckBox(tr("Crisp notes (stronger attack, softer sustain)"));
    wired->addRow(tr("Packet interval"), usb_interval);
    wired->addRow(envelope_5ms);
    wired->addRow(frequency_in_packet);
    wired->addRow(crisp_notes);
    layout->addWidget(wired_group);

    bt_group = new QGroupBox(tr("Bluetooth"));
    auto* bt = new QFormLayout(bt_group);
    bt_level = new QComboBox;
    bt_level->addItem(tr("1 - Stable (60 ms)"), 1);
    bt_level->addItem(tr("2 - Balanced (45 ms)"), 2);
    bt_level->addItem(tr("3 - Quality (35 ms)"), 3);
    bt_level->setToolTip(tr("Lower levels send less often, which keeps input lag down on a busy "
                            "Bluetooth link."));
    bt_after_input = new QCheckBox(tr("Send right after an input report"));
    bt_fast = new QCheckBox(tr("5 ms envelope while the link is idle"));
    bt_crisp_notes = new QCheckBox(tr("Crisp notes"));
    bt->addRow(tr("Level"), bt_level);
    bt->addRow(bt_after_input);
    bt->addRow(bt_fast);
    bt->addRow(bt_crisp_notes);
    layout->addWidget(bt_group);

    layout->addStretch();
    return page;
}

QWidget* ConfigureHdRumble::CreateDualSensePage() {
    auto* page = new QWidget;
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(MakeNote(tr("Rumble through the haptic voice coils of the PS5 controller "
                                  "(DualSense and DualSense Edge), as 48,000 samples per second "
                                  "of audio.")));

    auto* group = new QGroupBox(tr("Wired (USB)"));
    auto* form = new QFormLayout(group);
    ds_enabled = new QCheckBox(tr("Use the haptic voice coils"));
    ds_enabled->setToolTip(tr("Off: the DualSense uses its standard rumble."));
    ds_volume_slider = new QSlider(Qt::Horizontal);
    ds_volume_slider->setRange(0, 100);
    ds_volume = new QSpinBox;
    ds_volume->setRange(0, 100);
    ds_volume->setSuffix(QStringLiteral("%"));
    FitPercentSpinBox(ds_volume);
    connect(ds_volume_slider, &QSlider::valueChanged, ds_volume, &QSpinBox::setValue);
    connect(ds_volume, qOverload<int>(&QSpinBox::valueChanged), ds_volume_slider,
            &QSlider::setValue);
    auto* volume_row = new QHBoxLayout;
    volume_row->addWidget(ds_volume_slider, 1);
    volume_row->addWidget(ds_volume);
    ds_crisp_notes = new QCheckBox(tr("Crisp notes"));
    form->addRow(ds_enabled);
    form->addRow(tr("Volume"), volume_row);
    form->addRow(ds_crisp_notes);
    layout->addWidget(group);

    layout->addWidget(MakeNote(tr("On Bluetooth the DualSense uses its standard rumble.")));
    layout->addStretch();
    return page;
}

void ConfigureHdRumble::LoadConfiguration() {
    const auto& s = Settings::values;
    hd_enabled->setChecked(s.hd_rumble_enabled.GetValue());
    gain->setValue(std::clamp(s.hd_rumble_gain.GetValue(), 25, 1000));
    boost->setValue(std::clamp(s.hd_rumble_low_boost.GetValue(), 0, 100));
    usb_interval->setValue(std::clamp(s.hd_rumble_usb_interval_ms.GetValue(), 5, 100));
    envelope_5ms->setChecked(s.hd_rumble_envelope_5ms.GetValue());
    frequency_in_packet->setChecked(s.hd_rumble_frequency_in_packet.GetValue());
    crisp_notes->setChecked(s.hd_rumble_crisp_notes.GetValue());
    bt_level->setCurrentIndex(std::clamp(s.hd_rumble_bluetooth_level.GetValue(), 1, 3) - 1);
    bt_after_input->setChecked(s.hd_rumble_bt_after_input.GetValue());
    bt_fast->setChecked(s.hd_rumble_bt_fast.GetValue());
    bt_crisp_notes->setChecked(s.hd_rumble_bt_crisp_notes.GetValue());
    ds_enabled->setChecked(s.dualsense_haptics_enabled.GetValue());
    ds_volume->setValue(std::clamp(s.dualsense_haptics_volume.GetValue(), 0, 100));
    ds_crisp_notes->setChecked(s.dualsense_crisp_notes.GetValue());
    UpdateEnabledState();
}

void ConfigureHdRumble::RestoreDefaults() {
    hd_enabled->setChecked(true);
    gain->setValue(100);
    boost->setValue(0);
    usb_interval->setValue(15);
    envelope_5ms->setChecked(true);
    frequency_in_packet->setChecked(true);
    crisp_notes->setChecked(true);
    bt_level->setCurrentIndex(2);
    bt_after_input->setChecked(true);
    bt_fast->setChecked(true);
    bt_crisp_notes->setChecked(true);
    ds_enabled->setChecked(true);
    ds_volume->setValue(100);
    ds_crisp_notes->setChecked(true);
    UpdateEnabledState();
}

void ConfigureHdRumble::UpdateEnabledState() {
    const bool on = hd_enabled->isChecked();
    pages->setEnabled(on);
    page_list->setEnabled(on);
    // The extra wired cards only run on top of the 5 ms envelope
    frequency_in_packet->setEnabled(envelope_5ms->isChecked());
    crisp_notes->setEnabled(envelope_5ms->isChecked());
    ds_volume_slider->setEnabled(ds_enabled->isChecked());
    ds_volume->setEnabled(ds_enabled->isChecked());
    ds_crisp_notes->setEnabled(ds_enabled->isChecked());
}

void ConfigureHdRumble::ApplyConfiguration() {
    auto& s = Settings::values;
    const bool on = hd_enabled->isChecked();
    s.hd_rumble_enabled.SetValue(on);
    s.hd_rumble_gain.SetValue(gain->value());
    s.hd_rumble_low_boost.SetValue(boost->value());
    s.hd_rumble_usb_interval_ms.SetValue(usb_interval->value());
    s.hd_rumble_envelope_5ms.SetValue(envelope_5ms->isChecked());
    s.hd_rumble_frequency_in_packet.SetValue(frequency_in_packet->isChecked());
    s.hd_rumble_crisp_notes.SetValue(crisp_notes->isChecked());
    s.hd_rumble_bluetooth_level.SetValue(bt_level->currentData().toInt());
    s.hd_rumble_bt_after_input.SetValue(bt_after_input->isChecked());
    s.hd_rumble_bt_fast.SetValue(bt_fast->isChecked());
    s.hd_rumble_bt_crisp_notes.SetValue(bt_crisp_notes->isChecked());
    // The master switch also turns the DualSense haptics off (the SDL driver checks both)
    s.dualsense_haptics_enabled.SetValue(ds_enabled->isChecked());
    s.dualsense_haptics_volume.SetValue(ds_volume->value());
    s.dualsense_crisp_notes.SetValue(ds_crisp_notes->isChecked());
}
