// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <vector>
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader::Ampere {
struct RegisterPressure {
    Instruction instruction;
    std::vector<u64> prefix, suffix;
    bool valid = true;
};
inline u64 NativeRegisterCopy(u32 destination_bank, u32 destination, u32 source_bank, u32 source) {
    return 0x5c98000000070000ULL | (u64{10} << 44) | destination | (u64{destination_bank} << 8) |
           (u64{source_bank} << 10) | (u64{source} << 20);
}
// Preserve high guest registers in SSA while borrowing aligned low quartets
// for one native instruction. The expanded instruction owns its temporary bank.
inline RegisterPressure PrepareRegisterPressure(const Instruction& input, u32 reserved) {
    RegisterPressure result{input};
    if (reserved < 4 || reserved > 246 || (reserved & 3)) {
        result.valid = false;
        return result;
    }
    struct Operand {
        u32 bit, count;
    };
    std::vector<Operand> operands;
    const auto operand = [&](u32 bit, u32 count = 1) { operands.push_back({bit, count}); };
    const u32 opcode = input.Bits(0, 12);
    switch (opcode) {
    case 0xa02:
    case 0xc3e:
    case 0x802:
    case 0xb08:
    case 0xd08:
    case 0x908:
    case 0x919:
        operand(16);
        break;
    case 0xa0c:
    case 0x804:
    case 0x80b:
    case 0x80c:
        operand(24);
        break;
    case 0x20c:
    case 0x20b:
        operand(24);
        operand(32);
        break;
    case 0x807:
    case 0x808:
    case 0x923:
    case 0x820:
    case 0xa20:
    case 0x421:
        operand(16);
        operand(24);
        break;
    case 0x207:
    case 0x208:
    case 0x220:
    case 0x221:
    case 0x209:
        operand(16);
        operand(24);
        operand(32);
        break;
    case 0x805:
        operand(16, 2);
        break;
    case 0xc17:
    case 0x817:
        operand(16);
        operand(24);
        break;
    case 0xa24:
    case 0xa23:
    case 0x623:
    case 0x423:
        operand(16);
        operand(24);
        operand(64);
        break;
    case 0x621:
        operand(16);
        operand(24);
        break;
    case 0x321:
        operand(16, input.Bits(74, 2) + 1);
        operand(24);
        operand(32);
        break;
    case 0x322:
        operand(24);
        operand(32, input.Bits(74, 2) + 1);
        break;
    case 0x927:
        operand(24);
        operand(32);
        break;
    case 0x9a8: // ATOMG: result in Rd, data operand at bit 32.
        operand(16);
        operand(32);
        break;
    case 0xb6c:
    case 0xf6c: // TXD: results in Rd/Rd+1 and the second pair, coordinates in Ra, gradients in Rb..Rb+3.
        operand(16, 2);
        operand(24, 2);
        operand(32, 4);
        operand(64, 2);
        break;
    case 0x224:
    case 0x223:
        operand(16);
        operand(24);
        operand(32);
        operand(64);
        break;
    case 0x81a:
        operand(16);
        operand(24);
        break;
    case 0x308:
    case 0x202:
    case 0x243:
    case 0x245:
    case 0x305:
    case 0x306:
    case 0x307:
        operand(16);
        operand(32);
        break;
    case 0xb05:
    case 0xd05:
        operand(16);
        break;
    case 0x59f:
        operand(24, 2);
        operand(32);
        break;
    case 0x98e:
        operand(24, input.Bits(72, 1) ? 2 : 1);
        operand(32, input.Bits(73, 3) == 2 || input.Bits(73, 3) == 5 ? 2 : 1);
        break;
    case 0x433:
    case 0x633:
        operand(16);
        operand(24);
        break;
    case 0x233:
        operand(16);
        operand(24);
        operand(32);
        break;
    case 0x810:
    case 0xa10:
    case 0xc10:
    case 0x812:
    case 0x824:
    case 0x823:
    case 0x831:
    case 0x424:
    case 0x624:
        operand(16);
        operand(24);
        operand(64);
        break;
    case 0x230:
    case 0x231:
        operand(16);
        operand(24);
        operand(32);
        operand(64);
        break;
    case 0x419:
    case 0x819:
        operand(16);
        operand(24);
        operand(64);
        break;
    case 0x210:
    case 0x211:
    case 0x212:
        operand(16);
        operand(24);
        operand(32);
        operand(64);
        break;
    case 0xa11:
        operand(16);
        operand(24);
        operand(64);
        break;
    case 0x225:
    case 0x227:
    case 0x625:
    case 0x825:
        operand(16, opcode == 0x227 ? 1 : 2);
        operand(24);
        if (opcode == 0x225 || opcode == 0x227) {
            operand(32);
            operand(64, 2);
        }
        if (opcode == 0x625)
            operand(64);
        if (opcode == 0x825)
            operand(64, 2);
        break;
    case 0x83b:
        operand(16, 1U << input.Bits(72, 2));
        operand(24);
        break;
    case 0x23c:
        operand(16, 2);
        operand(24, input.Bits(75, 1) ? 4 : 2);
        operand(32, input.Bits(75, 1) ? 2 : 1);
        operand(64, 2);
        break;
    case 0x388:
    case 0x984:
    case 0x988: {
        const u32 width = input.Bits(73, 3), count = width == 6 ? 4 : width == 5 ? 2 : 1;
        operand(opcode == 0x984 ? 16 : 32, count);
        operand(24);
        break;
    }
    case 0x387:
    case 0x983:
    case 0x987: {
        const u32 width = input.Bits(73, 3), count = width == 6 ? 4 : width == 5 ? 2 : 1;
        operand(opcode == 0x983 ? 16 : 32, count);
        operand(24);
        break;
    }
    case 0x381:
    case 0x386:
    case 0x981:
    case 0x986: {
        const u32 width = input.Bits(73, 3), count = width == 6 ? 4 : width == 5 ? 2 : 1;
        operand(opcode == 0x386 || opcode == 0x986 ? 32 : 16, count);
        operand(24, opcode == 0x381 || opcode == 0x386 || input.Bits(90, 1) ? 2 : 1);
        break;
    }
    default:
        result.valid = false;
        return result;
    }
    std::array<bool, 256> used{};
    std::array<bool, 64> groups{};
    for (const auto [bit, count] : operands) {
        const u32 reg = input.Bits(bit, 8);
        if (reg == 255)
            continue;
        if (reg + count > 255 ||
            (reg < reserved && reg + count > reserved) ||
            (reg >= reserved && reg / 4 != (reg + count - 1) / 4)) {
            result.valid = false;
            return result;
        }
        for (u32 component = 0; component < count; ++component) {
            used[reg + component] = true;
            if (reg + component >= reserved)
                groups[(reg + component) / 4] = true;
        }
    }
    std::array<u32, 64> mapped{};
    for (u32 group = 0; group < groups.size(); ++group) {
        if (!groups[group])
            continue;
        u32 borrow = reserved;
        while (borrow >= 4) {
            borrow -= 4;
            if (!used[borrow] && !used[borrow + 1] && !used[borrow + 2] && !used[borrow + 3])
                break;
        }
        if (used[borrow] || used[borrow + 1] || used[borrow + 2] || used[borrow + 3]) {
            result.valid = false;
            return result;
        }
        mapped[group] = borrow;
        for (u32 component = 0; component < 4; ++component) {
            const u32 low = borrow + component, high = group * 4 + component;
            used[low] = true;
            result.prefix.push_back(NativeRegisterCopy(2, low, 0, low));
            result.prefix.push_back(NativeRegisterCopy(0, low, 1, high));
            result.suffix.push_back(NativeRegisterCopy(1, high, 0, low));
            result.suffix.push_back(NativeRegisterCopy(0, low, 2, low));
        }
    }
    for (const auto [bit, count] : operands) {
        const u32 reg = input.Bits(bit, 8);
        if (reg == 255 || !groups[reg / 4])
            continue;
        const u32 replacement = mapped[reg / 4] + reg % 4;
        const u32 word = bit / 32, shift = bit % 32;
        result.instruction.words[word] =
            (result.instruction.words[word] & ~(255U << shift)) | (replacement << shift);
    }
    return result;
}
} // namespace Shader::Ampere
