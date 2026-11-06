#pragma GCC diagnostic ignored "-Wshadow"
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.
#pragma once

#include <cstdlib>
#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>
#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/geometry_exit.h"
#include "shader_recompiler/frontend/ampere/register_pressure.h"

namespace Shader::Ampere {

enum class TessellationStage {
    None,
    Control,
    Evaluation,
};

struct LoweringOptions {
    bool geometry{};
    bool tensor{};
    /// Reject matrix multiplies instead of emulating them (KeplerCompute::EmulatesNativeTensorPrograms).
    bool reject_matrix{};
    u64 native_code_address{};
    std::span<const u32> verified_traps{};
    TessellationStage tessellation_stage = TessellationStage::None;
};

struct LoweredProgram {
    std::vector<u64> instructions;
    std::vector<u32> source_instruction_offsets;
    u32 source_size{};
    u32 required_texture_buffer = ~0U;
    // UR28 and up seen without a mirror: lowering again with this many registers below R216
    // reserved for them (even, so pairs stay adjacent) maps them.
    u32 high_uniform_registers{};
    u32 error_offset{};
    std::string_view error;

    explicit operator bool() const { return error.empty(); }
};

// Lower only operations whose semantics are representable by the existing
// Maxwell frontend. These are new instructions, not reinterpretations of the
// native bytes. Scheduling/control words are added by the environment adapter.
inline LoweredProgram LowerToMaxwellWith(std::span<const Instruction> code, u32 high_uniforms,
                                        LoweringOptions options = {}) {
    const bool geometry_stage = options.geometry, compact_matrix = options.tensor;
    const auto verified_traps = options.verified_traps;
    const auto tessellation_stage = options.tessellation_stage;
    const u64 native_code_address = options.native_code_address;
    LoweredProgram result;
    // UR28 and up (when used) live just below R216: UR28 at R(216 - high_uniforms).
    const u32 high_uniform_base = 216 - high_uniforms;
    struct Relocation { u32 output_index; s64 target; bool call = false; u32 output_target = ~0U; };
    std::vector<Relocation> relocations;
    std::vector<u32> barrier_tokens;
    std::vector<u32> call_targets;
    std::vector<u32> call_links;
    bool main_exited = false;
    bool computed_branch = false; // a jump table: the program ends at its self-branch, not an EXIT
    bool uses_uniform_bank = false;
    const bool uses_async_copy = std::any_of(code.begin(), code.end(), [](const auto& instruction) {
        return instruction.Bits(0, 12) == 0xfae;
    });
    for (const auto& instruction : code) {
        const u32 op = instruction.Bits(0, 12);
        uses_uniform_bank |= op == 0xf66 || op == 0xf6c;
        uses_uniform_bank |= op == 0xc3e || op == 0x499 || op == 0x2bf || op == 0x4a4;
        uses_uniform_bank |= op == 0x987 ||
            ((op == 0x387 || op == 0x983) && instruction.Bits(91, 1));
        uses_uniform_bank |= op == 0x3c4 || op == 0x2bd || op == 0x896 || op == 0xf60 || op == 0xc11 || op == 0xd08 || op == 0xc0c || op == 0x986 || op == 0x9a8 || op == 0xc12 || op == 0xc17 || op == 0xe24 || op == 0xc24 || op == 0xd00 || op == 0xd09 || op == 0x9c3 || op == 0xc45 || op == 0x2a4 || op == 0x891 || op == 0x890 || op == 0x290 || op == 0x28c || op == 0x88c || op == 0x287 || op == 0x887 || op == 0x958 || op == 0xab9 || op == 0xabb || op == 0x899 || op == 0xc20 || op == 0xe21 || op == 0xe23 || op == 0xc02 ||
                             op == 0x886 || op == 0x3c2 || op == 0xc82 || op == 0x882 || op == 0x292 || op == 0x892 || op == 0x291 || op == 0x981 || op == 0xc13 || op == 0xc10 || op == 0xc23 || op == 0x526 || op == 0x2be || op == 0x8a4 || op == 0xd05 || (op == 0x984 && instruction.Bits(91, 1)) || (op == 0x988 && instruction.Bits(91, 1)) || op == 0x89c || op == 0x883 || op == 0x2a5 || op == 0xd06 || op == 0xf63 || op == 0xf66 || op == 0xf69 || op == 0xc0b || op == 0x8a5 || op == 0x299 || op == 0xc09 || op == 0xc07 || (op == 0x81c && instruction.Bits(67, 1)) || op == 0xc0a ||
                             (op == 0x83b && instruction.Bits(91, 1));
    }
    const auto address_of = [](u32 index) { return 0x50U + index * 8 + ((index + 1) / 3) * 8; };
    const auto fail = [&](u32 index, std::string_view reason) {
        result.instructions.clear();
        result.source_instruction_offsets.clear();
        result.error_offset = index * sizeof(Instruction);
        result.error = reason;
        return result;
    };
    const auto finish = [&](u32 index) {
        for (const auto& relocation : relocations) {
            if (relocation.target > index) return fail(index, "native branch or call leaves captured program");
            const u32 target = relocation.output_target != ~0U ? relocation.output_target :
                result.source_instruction_offsets[static_cast<size_t>(relocation.target)];
            const s64 relative = s64{address_of(target)} - address_of(relocation.output_index) - 8;
            if (relative < -0x800000 || relative > 0x7fffff) return fail(index, "lowered branch distance");
            result.instructions[relocation.output_index] |= (static_cast<u64>(relative) & 0xffffff) << 20;
        }
        result.source_size = (index + 1) * sizeof(Instruction);
        return result;
    };
    if (code.empty() || code.size() > 65536) return fail(0, "invalid native program size");
    std::optional<GeometryExitEnvelope> geometry_exit;
    if (code.front().Bits(0, 12) == 0x34e) {
        if (!geometry_stage) return fail(0, "native program counter load outside geometry stage");
        geometry_exit = AnalyzeGeometryExit(code);
        if (!*geometry_exit)
            return fail(geometry_exit->error_offset / sizeof(Instruction), geometry_exit->error);
    }
    std::vector<bool> control_entries(code.size()), dead_instructions(code.size()), checked_returns(code.size());
    std::vector<u32> bounded_links(code.size(), ~0U), direct_targets(code.size(), ~0U);
    std::vector<std::pair<u32, u32>> control_edges;
    bool bounded_control = !geometry_stage;
    control_entries[0] = true;
    const auto write_count = [](const Instruction& instruction) -> std::optional<u32> {
        switch (instruction.Bits(0, 12)) {
        case 0x202: case 0x802: case 0x803: case 0x424: case 0x724: case 0x824: case 0xe24: case 0x227:
        case 0x810: case 0x819: case 0x845: case 0x906: case 0x820:
        case 0x812: case 0x589:
        case 0x212: case 0x207: case 0x807: case 0x208: case 0x808:
        case 0x220: case 0xa20: case 0x221: case 0x421: case 0x621:
        case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23:
        case 0x243: case 0x245: case 0x306:
        case 0x919: case 0xc12: case 0x300: case 0x210: case 0x308:
            return instruction.Bits(16, 8) == 255 ? 0U : 1U;
        case 0x225: case 0x625: case 0x825:
            return 2;
        case 0x305:
            return instruction.Bits(75, 2) == 3 ? 2U : 1U;
        case 0x83b:
            return instruction.Bits(72, 2) == 3 ? std::nullopt : std::optional<u32>{1U << instruction.Bits(72, 2)};
        case 0x381: case 0x981: case 0x984: case 0xb82: {
            const u32 width = instruction.Bits(73, 3);
            if (width > 6) return std::nullopt;
            return width == 6 ? 4U : width == 5 ? 2U : 1U;
        }
        case 0x20c: case 0x80c: case 0x81c: case 0x387: case 0x388: case 0xb1d: case 0x348: case 0x918:
        case 0x886: case 0x95b:
            return 0;
        default:
            return std::nullopt;
        }
    };
    const auto ordinary_operands = [&](const Instruction& instruction) {
        for (const u32 bit : {16U, 24U, 32U, 64U}) {
            if (bit == 32 && instruction.Bits(9, 3) != 1) continue;
            const u32 reg = instruction.Bits(bit, 8);
            if (reg >= (uses_uniform_bank ? high_uniform_base : 246U) && reg != 255) return false;
        }
        return true;
    };
    for (u32 index = 0; index < code.size(); ++index) {
        const auto& instruction = code[index];
        const u32 opcode = instruction.Bits(0, 12);
        if (opcode == 0x944 || opcode == 0x945 || opcode == 0x947) {
            if (opcode == 0x944 && index == 0) bounded_control = false;
            const u32 mode = instruction.words[2] & ~0x3ffffU;
            if (mode != (opcode == 0x944 ? 0x03c00000U : 0x03800000U) &&
                !(opcode == 0x947 && mode == 0x0b800000)) bounded_control = false;
            const u64 raw = (instruction.words[1] & (opcode == 0x947 ? ~3U : ~0U)) |
                            (u64{instruction.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16 || static_cast<u64>(target / 16) >= code.size()) {
                bounded_control = false;
            } else {
                direct_targets[index] = static_cast<u32>(target / 16);
                control_entries[direct_targets[index]] = true;
                control_edges.emplace_back(index, direct_targets[index]);
            }
            if (opcode == 0x944 && index + 1 < code.size()) {
                control_entries[index + 1] = true;
                control_edges.emplace_back(index, index + 1);
            }
        } else if ((opcode & 0x1ff) >= 0x140 && (opcode & 0x1ff) <= 0x15f &&
                   opcode != 0x941 && opcode != 0x942 && opcode != 0x948 &&
                   opcode != 0x348 && opcode != 0x94d && opcode != 0x950 && opcode != 0x95b) {
            bounded_control = false;
        }
    }
    for (u32 index = 1; bounded_control && index < code.size(); ++index) {
        if (code[index].Bits(0, 12) != 0x944) continue;
        if (code[index].words[0] != 0x7944) { bounded_control = false; break; }
        std::array<bool, 256> overwritten{};
        u32 setup = index;
        for (u32 prior = index; prior-- > 0;) {
            const auto& instruction = code[prior];
            const u32 destination = instruction.Bits(16, 8);
            if ((instruction.words[0] & ~0x00ff0000U) == 0x00007802 &&
                instruction.words[1] == (index + 1) * 16 && instruction.words[2] == 0xf00 &&
                !(instruction.words[3] & 0xc00001ffU) && ordinary_operands(instruction) &&
                destination < 246 && !overwritten[destination]) {
                bounded_links[index] = destination;
                setup = prior;
                break;
            }
            const auto count = write_count(instruction);
            if (!count || !ordinary_operands(instruction) || destination + *count > 255) break;
            for (u32 component = 0; component < *count; ++component)
                overwritten[destination + component] = true;
        }
        if (bounded_links[index] == ~0U) { bounded_control = false; break; }
        for (u32 entry = setup + 1; entry <= index; ++entry)
            if (control_entries[entry]) bounded_control = false;
    }
    for (u32 index = 0; bounded_control && index < code.size(); ++index) {
        if (code[index].Bits(0, 12) != 0x944) continue;
        const u32 link = bounded_links[index], target = direct_targets[index];
        if (target == 0) { bounded_control = false; break; }
        const auto& previous = code[target - 1];
        const u32 previous_opcode = previous.Bits(0, 12);
        if (previous.Bits(12, 4) != 7 || previous.Bits(87, 4) != 7 ||
            (previous_opcode != 0x947 && previous_opcode != 0x94d && previous_opcode != 0x950)) {
            bounded_control = false;
            break;
        }
        for (const auto& [source, destination] : control_edges)
            if (destination == target && code[source].Bits(0, 12) != 0x944) bounded_control = false;
        std::array<bool, 256> aliases{};
        aliases[link] = true;
        bool returned = false;
        for (u32 pc = target; pc < code.size(); ++pc) {
            const auto& instruction = code[pc];
            if (pc != target && control_entries[pc]) break;
            if (instruction.Bits(0, 12) == 0x950) {
                const u64 raw = instruction.words[1] | (u64{instruction.words[2] & 0x3ffff} << 32);
                const s64 relative = static_cast<s64>(raw << 14) >> 14;
                returned = (instruction.words[0] & 0x00ffffffU) == 0x00007950 &&
                           (instruction.words[2] & ~0x3ffffU) == 0x03c00000 &&
                           !(instruction.words[3] & 0xc00001ffU) &&
                           s64{pc} * 16 + 16 + relative == 0 &&
                           aliases[instruction.Bits(24, 8)];
                if (returned) checked_returns[pc] = true;
                break;
            }
            const auto count = write_count(instruction);
            const u32 destination = instruction.Bits(16, 8);
            if (!count || !ordinary_operands(instruction) || destination + *count > 255 ||
                (*count && destination <= link && link < destination + *count)) break;
            bool copy = false;
            if (instruction.Bits(12, 4) == 7) {
                if ((instruction.words[0] & 0xff00ffffU) == 0x00007202U &&
                    instruction.words[1] < 256 && instruction.words[2] == 0xf00)
                    copy = aliases[instruction.words[1]];
                if ((instruction.words[0] & 0xff00ffffU) == 0xff007224U &&
                    instruction.words[1] == 0xff && (instruction.words[2] & ~0xffU) == 0x078e0000)
                    copy = aliases[instruction.Bits(64, 8)];
            }
            for (u32 component = 0; component < *count; ++component)
                aliases[destination + component] = component == 0 && copy;
        }
        if (!returned) bounded_control = false;
    }
    for (u32 index = 0; bounded_control && index < code.size(); ++index)
        if (code[index].Bits(0, 12) == 0x950 && !checked_returns[index]) bounded_control = false;
    for (u32 index = 0; bounded_control && index < code.size(); ++index) {
        const auto& instruction = code[index];
        const u32 target = direct_targets[index];
        if (instruction.Bits(0, 12) != 0x947 || instruction.Bits(12, 4) != 7 ||
            instruction.Bits(32, 2) == 2 ||
            instruction.Bits(16, 8) || instruction.Bits(87, 4) != 7 || target <= index ||
            target == ~0U || (instruction.words[3] & 0xc00001ffU) ||
            (instruction.Bits(91, 1) ? instruction.Bits(24, 8) >= 64 : instruction.Bits(24, 8) != 0))
            continue;
        bool skipped_call = false;
        for (u32 pc = index + 1; pc < target; ++pc)
            skipped_call |= code[pc].Bits(0, 12) == 0x944;
        if (!skipped_call) continue;
        bool entered = false;
        for (const auto& [source, destination] : control_edges)
            entered |= index < destination && destination < target &&
                       !(index < source && source < target);
        if (!entered)
            for (u32 pc = index + 1; pc < target; ++pc) dead_instructions[pc] = true;
    }
    for (u32 index = 0; index < code.size(); ++index) {
        auto insn = code[index];
        result.source_instruction_offsets.push_back(static_cast<u32>(result.instructions.size()));
        if (dead_instructions[index]) continue;
        if (geometry_exit && (index < GeometryExitSetupInstructions ||
            index * sizeof(Instruction) == geometry_exit->handler + sizeof(Instruction)))
            continue;
        u32 opcode = insn.Bits(0, 12);
        if (insn.Bits(12, 4) == 15) continue; // @!PT: never executes (compiler padding)
        std::optional<u32> active_link;
        u32 active_entry = 0;
        if (main_exited && (!geometry_exit || index * sizeof(Instruction) < geometry_exit->handler)) {
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] <= index && call_targets[call] >= active_entry) {
                    active_entry = call_targets[call];
                    active_link = call_links[call];
                }
            }
            // RET, BRA, NOP and ISETP/FSETP (predicate outputs) have no destination GPR; bits 16..23
            // are not a write to R0. IMAD.WIDE writes a register pair.
            const bool no_destination = opcode == 0x950 || opcode == 0x947 || opcode == 0x918 ||
                                        (opcode & 0x1ff) == 0x00c || (opcode & 0x1ff) == 0x00b;
            const u32 written = write_count(insn).value_or(1);
            if (active_link && !no_destination &&
                insn.Bits(16, 8) <= *active_link && *active_link < insn.Bits(16, 8) + written)
                return fail(index, "native callee overwrites link register");
        }
        RegisterPressure pressure{insn};
        bool requires_shadow = false;
        for (u32 bit : {16U, 24U, 32U, 64U}) {
            if (opcode == 0x81c) break;
            // Bits 32-39 name a register in register forms and in these opcodes: TEX/TLD, the
            // stores, and the data operand of ISBEWR (0x927), RED (0x98e) and SURED (0x59f), all
            // three listed with operand 32 in RegisterPressure.
            if (bit == 32 && insn.Bits(9, 3) != 1 && opcode != 0xb60 && opcode != 0xb66 && opcode != 0x986 && opcode != 0x987 && opcode != 0xf66 && opcode != 0x988 && opcode != 0x927 && opcode != 0x98e && opcode != 0x59f) continue;
            const u32 reg = insn.Bits(bit, 8);
            if (reg >= (uses_uniform_bank ? high_uniform_base : 246U) && reg < 255)
                requires_shadow = true;
        }
        if (requires_shadow) {
            if (!uses_uniform_bank || (high_uniform_base & 3))
                return fail(index, "native program uses lowering scratch registers");
            pressure = PrepareRegisterPressure(insn, high_uniform_base);
            if (!pressure.valid) return fail(index, "native register shadowing mode");
            insn = pressure.instruction;
            result.instructions.insert(result.instructions.end(), pressure.prefix.begin(), pressure.prefix.end());
        }
        const size_t instruction_start = result.instructions.size();
        const size_t relocation_start = relocations.size();
        // UR0-UR21 live in R216-R237 and UR22-UR27 in R240-R245 (R239 holds the uniform
        // predicates), so even/odd pairs stay adjacent; UR28 and up below R216. Sites that
        // still check reg < 16 rely on UR0-UR15 being R216-R231.
        const auto uniform_register = [&](u32 reg) -> u32 {
            if (reg < 22) return 216 + reg;
            if (reg < 28) return 218 + reg;
            if (reg == 63) return 255;
            if (reg < 63 && reg - 28 < high_uniforms) return high_uniform_base + (reg - 28);
            if (reg < 63) result.high_uniform_registers = (std::max)(result.high_uniform_registers, (reg - 26) & ~1U);
            return ~0U;
        };
        bool rewritten_uniform = false;
        bool uniform_guard = false;
        u32 uniform_output = 7; // ULOP3's UPn output: computed in P(6-n), then merged into R239
        if (opcode == 0x2a4 || opcode == 0x4a4 || opcode == 0x8a4 || opcode == 0x891 || opcode == 0x890 || opcode == 0x290 ||
            opcode == 0x292 || opcode == 0x892 || opcode == 0x291 || opcode == 0x896 || opcode == 0x8a5) {
            if (insn.Bits(12, 4) != 7 && opcode != 0x292 && opcode != 0x892) {
                const bool guarded_add = (opcode == 0x290 || opcode == 0x890) &&
                    !insn.Bits(74, 1) && insn.Bits(81, 6) == 0x3f &&
                    insn.Bits(77, 4) == 15 && insn.Bits(87, 4) == 15;
                const bool guarded_lea = (opcode == 0x291 || opcode == 0x891) &&
                    !insn.Bits(74, 1) && !insn.Bits(80, 1) && insn.Bits(81, 3) == 7;
                const bool guarded_multiply = opcode == 0x2a4 &&
                    (insn.words[2] & ~0x080002ffU) == 0x078e0000U;
                if (!guarded_add && !guarded_lea && !guarded_multiply)
                    return fail(index, "native uniform arithmetic predication");
            }
            uniform_guard = insn.Bits(12, 4) != 7;
            rewritten_uniform = true;
            const u32 dst = uniform_register(insn.Bits(16, 8));
            const u32 a = uniform_register(insn.Bits(24, 8));
            const u32 c = uniform_register(insn.Bits(64, 8));
            if (dst == ~0U || a == ~0U || c == ~0U) return fail(index, "native uniform arithmetic register");
            insn.words[0] = (insn.words[0] & 0xffff) | (dst << 16) | (a << 24);
            insn.words[2] = (insn.words[2] & ~0x080000ffU) | c;
            if (opcode == 0x2a4 || opcode == 0x290 || opcode == 0x292 || opcode == 0x291) {
                const u32 b = uniform_register(insn.Bits(32, 8));
                if (b == ~0U) return fail(index, "native uniform arithmetic second register");
                insn.words[1] = (insn.words[1] & ~0xffU) | b;
            }
            if ((opcode == 0x292 || opcode == 0x892) && insn.Bits(80, 4) != 0xe) {
                if (insn.Bits(80, 1)) return fail(index, "native uniform logic predicate output");
                uniform_output = insn.Bits(81, 3);
                insn.words[2] = (insn.words[2] & ~(7U << 17)) | ((6 - uniform_output) << 17);
            }
            opcode = opcode == 0x2a4 ? 0x224 : opcode == 0x4a4 ? 0x424 : opcode == 0x8a4 ? 0x824 : opcode == 0x891 ? 0x811 : opcode == 0x890 ? 0x810
                   : opcode == 0x292 ? 0x212 : opcode == 0x892 ? 0x812 : opcode == 0x291 ? 0x211
                   : opcode == 0x896 ? 0x816 : opcode == 0x8a5 ? 0x825 : 0x210;
        }
        // 0x899 (USHF) lowers to one SHR/SHL, so a uniform guard is swapped in like these.
        const bool uniform_predication = opcode == 0xabb || opcode == 0xab9 || opcode == 0xc82 || opcode == 0x882 ||
                                         opcode == 0x899 || opcode == 0x299 || opcode == 0x499;
        const u32 native_predicate = insn.Bits(12, 4);
        const bool guarded_uniform = (uniform_predication || uniform_guard) && native_predicate != 7;
        // A uniform guard UPn lives in R239 bit 6-n and is swapped into P(6-n) for the op.
        const u32 guard_slot = 6 - (native_predicate & 7);
        u64 predicate = u64{guarded_uniform ? ((native_predicate & 8) | guard_slot) : native_predicate} << 16;
        const u64 dest = insn.Bits(16, 8);
        const auto base = [&](u64 opcode_bits) { return opcode_bits | predicate | dest; };
        // NVN2 fullscreen vertex shaders read bank 24, which the game never binds (the console reads
        // zeros from an unbound bank); Maxwell has 18, so read it from bank 17, also never bound.
        const auto bank = [&]() -> u32 {
            const u32 value = insn.Bits(54, 5);
            return value == 24 ? 17U : value;
        };
        const auto cbuf = [&]() -> u64 {
            return (u64{insn.Bits(38, 16) / 4} << 20) | (u64{bank()} << 34);
        };
        const auto valid_cbuf = [&] {
            return !insn.Bits(91, 1) && bank() < 18 &&
                   (insn.Bits(38, 16) % 4) == 0;
        };
        std::string_view operand_error;
        const auto emit = [&](u64 word) { result.instructions.push_back(word); };
        // Bound Maxwell texture instructions share one descriptor constant buffer per program.
        const auto bind_texture_buffer = [&](u32 binding) {
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return false;
            result.required_texture_buffer = binding;
            return true;
        };
        // P(guard slot) while a uniform guard is swapped in; P6 for lowerings that use it as
        // scratch (R2UR, MATCH, HFMA2.RELU), which save and restore it with the defaults below.
        const u32 guard_mask = guard_slot < 7 ? 1U << guard_slot : 0U;
        const auto save_predicate = [&](u32 mask = 1U << 6) {
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
        };
        const auto restore_predicate = [&](u32 mask = 1U << 6) {
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
        };
        if (guarded_uniform) {
            save_predicate(guard_mask);
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{guard_mask} << 20));
        }
        const u32 output_mask = uniform_output == 7 ? 0U : 1U << (6 - uniform_output);
        if (output_mask) {
            // Keep the guest P(6-n) in R254 and start from the current UPn (a false guard keeps it).
            emit(0x38e8000000070000ULL | 254 | (u64{255} << 8) | (u64{output_mask} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{output_mask} << 20));
        }
        // Pd = LUT(a, b, c) over predicate operands (index | negate << 3): each input becomes
        // 0 or ~0 (SEL), and LOP3 sets Pd when the result is nonzero.
        const auto predicate_lut = [&](u32 lut, u32 a, u32 b, u32 c, u32 target,
                                       bool uniform_c = false, u32 second_target = 7, u32 second_lut = 0) {
            const u32 inputs[3]{a, b, c};
            for (u32 i = 0; i < 3; ++i) {
                if (i == 2 && uniform_c) {
                    // UPn from R239 bit 6-n as 0 or ~0 (UPT is ~0); a negation flips the table.
                    if ((c & 7) == 7) {
                        emit(0x0100000000000000ULL | predicate | 252 | (u64{15} << 12) | (u64{~0U} << 20));
                    } else {
                        emit(0x3800000000000000ULL | predicate | 252 | (u64{239} << 8) |
                             (u64{(6 - (c & 7)) | (1U << 8)} << 20));
                        emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) |
                             (u64{252} << 20) | (u64{1} << 48));
                    }
                    if (c & 8) {
                        lut = ((lut & 0x55) << 1) | ((lut & 0xaa) >> 1);
                        second_lut = ((second_lut & 0x55) << 1) | ((second_lut & 0xaa) >> 1);
                    }
                    continue;
                }
                emit(0x38a0000000000000ULL | predicate | (250 + i) | (u64{255} << 8) |
                     (u64{0x7ffff} << 20) | (u64{1} << 56) | (u64{inputs[i] ^ 8U} << 39));
            }
            if (second_target != 7) {
                const u32 first_mask = target == 7 ? 0U : 1U << target;
                const u32 second_mask = 1U << second_target;
                for (u32 i = 0; i < 2; ++i) {
                    const u32 reg = 248 + i;
                    emit(0x5be0000000000000ULL | predicate | reg | (u64{250} << 8) |
                         (u64{251} << 20) | (u64{i == 0 ? lut : second_lut} << 28) |
                         (u64{252} << 39) | (u64{7} << 48));
                    emit(0x0400000000000000ULL | predicate | reg | (u64{reg} << 8) |
                         (u64{i == 0 ? first_mask : second_mask} << 20));
                }
                emit(0x5c40000000000000ULL | predicate | 248 | (u64{248} << 8) |
                     (u64{249} << 20) | (u64{1} << 41) | (u64{7} << 48));
                emit(0x38f0000000000000ULL | predicate | (u64{248} << 8) |
                     (u64{first_mask | second_mask} << 20));
                return;
            }
            emit(0x5be0000000000000ULL | predicate | 255 | (u64{250} << 8) | (u64{251} << 20) |
                 (u64{lut} << 28) | (u64{3} << 36) | (u64{252} << 39) | (u64{target} << 48));
        };
        const auto mov_imm = [&](u32 reg, u32 value) {
            emit(0x0100000000000000ULL | predicate | reg | (u64{15} << 12) | (u64{value} << 20));
        };
        const auto materialize = [&](u32 kind, u32 reg, u32 scratch, bool absolute, bool negate) -> u32 {
            if (kind == 3) {
                const u32 mapped = uniform_register(reg);
                if (mapped == ~0U) { operand_error = "native uniform source register"; return 255; }
                reg = mapped;
            } else if (kind == 1) {
                mov_imm(scratch, insn.words[1]);
                reg = scratch;
            } else if (kind == 2) {
                if (!valid_cbuf()) { operand_error = "native constant buffer address"; return 255; }
                emit(0x4c98000000000000ULL | predicate | scratch | (u64{15} << 39) | cbuf());
                reg = scratch;
            }
            if (absolute || negate) {
                if (reg != scratch) emit(0x5c98000000000000ULL | predicate | scratch | (u64{reg} << 20) | (u64{15} << 39));
                if (absolute) emit(0x0400000000000000ULL | predicate | scratch | (u64{scratch} << 8) | (u64{0x7fffffff} << 20));
                if (negate) emit(0x0400000000000000ULL | predicate | scratch | (u64{scratch} << 8) | (u64{0x80000000} << 20) | (u64{2} << 53));
                reg = scratch;
            }
            return reg;
        };
        const auto operand = [&](u32 which, bool floating = true) -> u32 {
            const u32 form = insn.Bits(9, 3);
            u32 reg{}, kind{}, abs{}, neg{};
            if (which == 0) {
                reg = insn.Bits(24, 8); abs = insn.Bits(73, 1); neg = insn.Bits(72, 1);
            } else {
                if (form != 1 && form != 2 && form != 3 && form != 4 && form != 5 && form != 6 && form != 7) {
                    operand_error = "native ALU source form"; return 255;
                }
                const bool wide = which == 1 ? (form == 4 || form == 5 || form == 6) : (form == 2 || form == 3 || form == 7);
                if (wide) {
                    kind = form == 6 || form == 7 ? 3 : (form == 4 || form == 2) ? 1 : 2;
                    if (kind == 3) reg = insn.Bits(32, 8);
                    if (kind == 2 || kind == 3) { abs = insn.Bits(62, 1); neg = insn.Bits(63, 1); }
                } else {
                    const bool upper = which == 1 ? (form == 2 || form == 3 || form == 7) : true;
                    reg = insn.Bits(upper ? 64 : 32, 8);
                    abs = insn.Bits(upper ? 74 : 62, 1); neg = insn.Bits(upper ? 75 : 63, 1);
                }
            }
            return materialize(kind, reg, 250 + which, floating && abs, floating && neg);
        };
        struct HalfSource { u32 reg, swizzle; bool absolute, negate; };
        const auto half_source = [&](u32 which) -> HalfSource {
            if (which == 0) return {insn.Bits(24, 8), insn.Bits(74, 2),
                                    insn.Bits(73, 1) != 0, insn.Bits(72, 1) != 0};
            const u32 form = insn.Bits(9, 3);
            const bool wide = which == 1 ? (form == 4 || form == 5) : (form == 2 || form == 3);
            if (wide) {
                const bool immediate = form == 2 || form == 4;
                const u32 reg = materialize(immediate ? 1 : 2, 255, 250 + which, false, false);
                return {reg, immediate ? 0U : insn.Bits(60, 2),
                        !immediate && insn.Bits(62, 1), !immediate && insn.Bits(63, 1)};
            }
            const bool upper = which == 2 || form == 2 || form == 3;
            return {insn.Bits(upper ? 64 : 32, 8), insn.Bits(upper ? 81 : 60, 2),
                    insn.Bits(upper ? 83 : 62, 1) != 0, insn.Bits(upper ? 84 : 63, 1) != 0};
        };
        const auto extract_half = [&](HalfSource source, u32 lane, u32 target) {
            const bool high = source.swizzle == 3 || (source.swizzle == 0 && lane == 1);
            emit(0x5ca8000000000000ULL | predicate | target | (u64{2} << 8) |
                 (u64{source.swizzle == 1 ? 2U : 1U} << 10) | (u64{source.reg} << 20) |
                 (u64{high} << 41) | (u64{source.negate} << 45) | (u64{source.absolute} << 49));
        };
        const auto xmad = [&](u32 target, u32 a, u32 b, u32 c, bool a_high,
                              bool b_high, u32 mode, bool immediate) {
            emit((immediate ? 0x3600000000000000ULL : 0x5b00000000000000ULL) |
                 predicate | target | (u64{a} << 8) |
                 (u64{immediate ? ((b >> (b_high ? 16 : 0)) & 0xffff) : b} << 20) |
                 (u64{!immediate && b_high} << 35) | (u64{c} << 39) |
                 (u64{mode} << 50) | (u64{a_high} << 53));
        };
        const auto multiply_low = [&](u32 a, u32 b, u32 c, bool immediate) {
            xmad(250, a, b, c, false, false, 0, immediate);
            xmad(251, a, b, 255, true, false, 0, immediate);
            xmad(251, a, b, 251, false, true, 0, immediate);
            emit(base(0x5c18000000000000ULL) | (u64{251} << 8) |
                (u64{250} << 20) | (u64{16} << 39));
        };
        const auto right_shift_u64_low = [&](u32 target, u32 low, u32 high, u32 shift) {
            mov_imm(250, 32);
            emit(0x5c10000000000000ULL | predicate | 251 | (u64{250} << 8) |
                 (u64{shift} << 20) | (u64{1} << 48));
            emit(0x5c28000000000000ULL | predicate | 250 | (u64{low} << 8) | (u64{shift} << 20));
            emit(0x5c48000000000000ULL | predicate | 251 | (u64{high} << 8) | (u64{251} << 20));
            emit(0x5c40000000000000ULL | predicate | 250 | (u64{250} << 8) |
                 (u64{251} << 20) | (u64{1} << 41) | (u64{7} << 48));
            emit(0x1c00000000000000ULL | predicate | 251 | (u64{shift} << 8) |
                 (u64{0xffffffe0U} << 20));
            emit(0x5c28000000000000ULL | predicate | 251 | (u64{high} << 8) | (u64{251} << 20));
            emit(0x5c40000000000000ULL | predicate | target | (u64{250} << 8) |
                 (u64{251} << 20) | (u64{1} << 41) | (u64{7} << 48));
        };
        switch (opcode) {
        case 0x304: { // F2F.F16.F32, scalar round-to-nearest conversion.
            if (insn.words[2] != 0x00200800 || insn.Bits(24, 8) ||
                (insn.words[1] & ~0xc00000ffU))
                return fail(index, "native scalar float conversion mode");
            emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{2} << 10) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(62, 1)} << 49) |
                 (u64{insn.Bits(63, 1)} << 45));
            break;
        }
        case 0xc3e: {
            const u32 source = uniform_register(insn.Bits(32, 8));
            if (insn.Bits(24, 8) != 255 || (insn.words[1] & ~0xffU) ||
                insn.words[2] != 0x080000ff || (insn.words[3] & 0xc00001ffU) || source == ~0U)
                return fail(index, "native uniform float packing mode");
            emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{2} << 10) |
                 (u64{source} << 20));
            emit(0x5800000000000000ULL | predicate | 250 | (u64{source} << 8) |
                 (u64{source} << 20) | (u64{7} << 39) | (u64{8} << 48));
            mov_imm(251, 0x7e00);
            emit(0x5c40000000000000ULL | predicate | 250 | (u64{250} << 8) |
                 (u64{251} << 20) | (u64{7} << 48));
            emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{250} << 20) |
                 (u64{1} << 41) | (u64{7} << 48));
            break;
        }
        case 0x23e: case 0xa3e: { // F2FP.PACK_AB / MERGE_C, round-to-nearest
            if (insn.Bits(62, 2) || insn.Bits(73, 5) || insn.Bits(79, 2) ||
                insn.Bits(81, 10) || insn.Bits(91, 1)) {
                return fail(index, "native float packing modifiers");
            }
            const bool merge = insn.Bits(78, 1);
            if (insn.Bits(72, 1)) return fail(index, "native float packing selector");
            const u32 a = merge ? operand(2, false) : operand(0, false);
            const u32 b = operand(1, false);
            // Canonical F2F writes the converted half in its low 16 bits.
            if (merge) {
                emit(0x5c98000000000000ULL | predicate | 250 | (u64{a} << 20) | (u64{15} << 39));
            } else {
                emit(0x5ca8000000000000ULL | predicate | 250 | (u64{1} << 8) |
                     (u64{2} << 10) | (u64{a} << 20));
            }
            emit(0x5ca8000000000000ULL | predicate | 251 | (u64{1} << 8) |
                 (u64{2} << 10) | (u64{b} << 20));
            emit(base(0x5c18000000000000ULL) | (u64{250} << 8) | (u64{251} << 20) | (u64{16} << 39));
            break;
        }
        case 0x232: case 0x832: case 0xa32: { // HMUL2
            if (insn.Bits(78, 2) || insn.Bits(81, 10) || insn.Bits(91, 1))
                return fail(index, "native half multiply mode");
            const auto a = half_source(0), b = half_source(1);
            const u32 precision = insn.Bits(76, 1) ? 2 : insn.Bits(80, 1);
            emit(base(0x5d08000000000000ULL) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{a.negate != b.negate} << 31) |
                 (u64{insn.Bits(77, 1)} << 32) | (u64{precision} << 39) |
                 (u64{a.absolute} << 44) | (u64{a.swizzle} << 47));
            break;
        }
        case 0x230: case 0x430: case 0x630: { // HADD2
            if (insn.Bits(76, 1) || insn.Bits(79, 1) || insn.Bits(81, 10) || insn.Bits(91, 1))
                return fail(index, "native half add mode");
            const auto a = half_source(0);
            const auto b = half_source(opcode == 0x230 ? 1 : 2);
            if (insn.Bits(78, 1)) {
                if (a.reg != 255 || !a.negate || a.absolute || a.swizzle != 0 ||
                    b.swizzle < 2 || insn.Bits(77, 1) || insn.Bits(80, 1))
                    return fail(index, "native float-output half add mode");
                extract_half(b, 0, static_cast<u32>(dest));
            } else {
                emit(base(0x5d10000000000000ULL) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                     (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{b.negate} << 31) |
                     (u64{insn.Bits(77, 1)} << 32) | (u64{insn.Bits(80, 1)} << 39) |
                     (u64{a.negate} << 43) | (u64{a.absolute} << 44) | (u64{a.swizzle} << 47));
            }
            break;
        }
        case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31: { // HFMA2
            const bool relu = insn.Bits(79, 1) != 0;
            // With .RELU, bits 87-90 are a predicate operand: the clamp applies only where it holds.
            if (insn.Bits(78, 1) || insn.Bits(85, 1) || insn.Bits(91, 1) ||
                (relu ? insn.Bits(77, 1) != 0 : insn.Bits(87, 4) != 0))
                return fail(index, "native half fused multiply-add mode");
            if (relu && native_predicate != 7) {
                // Guard the whole expansion before its temporary P6 updates.
                relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                emit(0xe24000000000000fULL | (u64{native_predicate ^ 8U} << 16));
                predicate = u64{7} << 16;
            }
            auto a = half_source(0);
            auto b = half_source(1);
            const auto c = half_source(2);
            if (insn.Bits(86, 1)) {
                if (b.swizzle != 0 || b.absolute || b.reg >= 250)
                    return fail(index, "native half mixed-sign source mode");

                // H0_NH1 swaps the FP16 halves and negates the new low half.
                emit(0x36f8000000000000ULL | predicate | 251 | (u64{b.reg} << 8) |
                     (u64{16} << 20) | (u64{b.reg} << 39));
                emit(0x0400000000000000ULL | predicate | 251 | (u64{251} << 8) |
                     (u64{0x8000} << 20) | (u64{2} << 53));
                b.reg = 251;
            }
            if (b.absolute || c.absolute || (a.absolute && a.reg >= 248))
                return fail(index, "native half fused multiply-add absolute source");
            if (a.absolute) {
                emit(0x0400000000000000ULL | predicate | 250 | (u64{a.reg} << 8) |
                     (u64{a.swizzle == 1 ? 0x7fffffffU : 0x7fff7fffU} << 20));
                a.reg = 250;
                a.absolute = false;
            }
            const u32 precision = insn.Bits(76, 1) ? 2 : insn.Bits(80, 1);
            emit(0x5d00000000000000ULL | predicate | (relu ? 254U : dest) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{c.negate} << 30) | (u64{a.negate != b.negate} << 31) |
                 (u64{insn.Bits(77, 1)} << 32) | (u64{c.swizzle} << 35) |
                 (u64{precision} << 37) | (u64{c.reg} << 39) | (u64{a.swizzle} << 47));
            if (relu) {
                // Ampere's trailing predicate enables ReLU; it is not a predicate output.
                // Keep the ordinary FMA result when it is false, including negative zero.
                const u32 relu_predicate = insn.Bits(87, 4);
                if (relu_predicate != 7) {
                    emit(base(0x5c98000000000000ULL) | (u64{254} << 20) | (u64{15} << 39));
                    relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                    emit(0xe24000000000000fULL | (u64{relu_predicate ^ 8U} << 16));
                }
                // Clamp the rounded binary16 lanes, preserving positive infinity
                // and canonicalizing either sign of NaN without saturating to one.
                save_predicate();
                for (u32 lane = 0; lane < 2; ++lane) {
                    emit(0x3828000000070000ULL | 251 | (u64{254} << 8) | (u64{lane * 16} << 20));
                    emit(0x0400000000070000ULL | 251 | (u64{251} << 8) | (u64{0xffff} << 20));
                    emit(0x0400000000070000ULL | 252 | (u64{251} << 8) | (u64{0x7fff} << 20));
                    mov_imm(250, 0x7c00);
                    emit(0x5b60000000070000ULL | 7 | (u64{6} << 3) | (u64{250} << 8) |
                         (u64{252} << 20) | (u64{7} << 39) | (u64{1} << 49));
                    emit(0x3828000000070000ULL | 252 | (u64{251} << 8) | (u64{15} << 20));
                    mov_imm(250, 1);
                    emit(0x5c10000000070000ULL | 252 | (u64{252} << 8) | (u64{250} << 20) | (u64{1} << 48));
                    emit(0x5c40000000070000ULL | 251 | (u64{251} << 8) | (u64{252} << 20) | (u64{7} << 48));
                    mov_imm(250, 0x7fff);
                    emit(0x5ca0000000070000ULL | 251 | (u64{250} << 8) | (u64{251} << 20) | (u64{6} << 39));
                    if (lane == 0) {
                        emit(base(0x5c98000000000000ULL) | (u64{251} << 20) | (u64{15} << 39));
                    } else {
                        emit(0x3848000000070000ULL | 251 | (u64{251} << 8) | (u64{16} << 20));
                        emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{251} << 20) |
                             (u64{1} << 41) | (u64{7} << 48));
                    }
                }
                restore_predicate();
            }
            break;
        }
        case 0x233: case 0x433: case 0x633: { // HSET2, packed half comparisons and predicate combination.
            const u32 combine = insn.Bits(69, 2);
            if (insn.Bits(64, 5) || combine > 2 || insn.Bits(81, 6) || insn.Bits(91, 1) ||
                (opcode == 0x233 && (insn.words[1] & 0x0fffff00U)))
                return fail(index, "native packed half comparison mode");
            const auto a = half_source(0), b = half_source(opcode == 0x233 ? 1 : 2);
            emit(base(0x5d18000000000000ULL) | (u64{a.reg} << 8) | (u64{b.reg} << 20) |
                 (u64{b.swizzle} << 28) | (u64{b.absolute} << 30) | (u64{b.negate} << 31) |
                 (u64{insn.Bits(76, 4)} << 35) | (u64{insn.Bits(87, 4)} << 39) |
                 (u64{a.negate} << 43) | (u64{a.absolute} << 44) | (u64{combine} << 45) |
                 (u64{a.swizzle} << 47) | (u64{insn.Bits(71, 1)} << 49) | (u64{insn.Bits(80, 1)} << 50));
            break;
        }
        case 0x240: case 0x840: { // HMNMX2, exact half promotion for min/max
            if (insn.Bits(76, 11) || insn.Bits(91, 1))
                return fail(index, "native half min/max mode");
            const auto a = half_source(0);
            const auto b = opcode == 0x840 ? HalfSource{255, 0, false, false} : half_source(1);
            if (a.swizzle == 1 || b.swizzle == 1 || (b.reg >= 250 && b.reg != 255))
                return fail(index, "native half min/max source mode");
            for (u32 lane = 0; lane < 2; ++lane) {
                extract_half(a, lane, 250);
                if (opcode == 0x840) {
                    const u32 half = (insn.words[1] >> (lane * 16)) & 0xffff;
                    mov_imm(251, half);
                    extract_half(HalfSource{251, 2, false, false}, 0, 251);
                } else {
                    extract_half(b, lane, 251);
                }
                const u32 target = lane == 0 ? 252 : 250;
                emit(0x5c60000000000000ULL | predicate | target | (u64{250} << 8) | (u64{251} << 20) |
                     (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(0x5ca8000000000000ULL | predicate | target | (u64{1} << 8) |
                     (u64{2} << 10) | (u64{target} << 20));
            }
            emit(base(0x5c18000000000000ULL) | (u64{250} << 8) | (u64{252} << 20) | (u64{16} << 39));
            break;
        }
        case 0xc24: case 0xe24: { // IMAD low word with a uniform multiplier or addend.
            // 0xe24 keeps the uniform addend in bits 32-37; bit 63 negates it ("RZ, RZ, -UR4").
            const bool negate_c = opcode == 0xe24 && insn.Bits(63, 1) != 0;
            if ((insn.words[2] & ~0x080002ffU) != 0x078e0000 ||
                (insn.words[1] & ~(0xffU | (negate_c ? 0x80000000U : 0U))))
                return fail(index, "native uniform-addend multiply mode");
            const u32 a = operand(0, false), b = operand(1, false);
            u32 c = operand(2, false);
            if (negate_c) {
                emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) | (u64{c} << 20) |
                     (u64{1} << 48));
                c = 252;
            }
            multiply_low(a, b, c, false);
            break;
        }
        case 0x224: { // IMAD register / zero-product move
            if (insn.Bits(24, 8) == 255 && insn.words[1] == 255 && insn.Bits(75, 1) &&
                (insn.words[2] & ~0xaffU) == 0x078e0000) {
                emit(base(0x5c10000000000000ULL) | (u64{255} << 8) |
                    (u64{insn.Bits(64, 8)} << 20) | (u64{1} << 48));
                break;
            }
            if (insn.Bits(24, 8) == 255 && insn.words[1] == 255 &&
                (insn.words[2] & ~0xffU) == 0x078e0000) {
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{insn.Bits(64, 8)} << 20));
                break;
            }
            if (insn.Bits(24, 8) == 255 && insn.words[1] == 255 && (insn.words[2] & ~0x07800affU) == 0x000e0400 &&
                (!rewritten_uniform || insn.Bits(87, 3) == 7)) {
                // IMAD.X RZ, RZ, c, Pn: the high word of a 64-bit add (bit 75 inverts c).
                u32 c = insn.Bits(64, 8);
                mov_imm(250, 1);
                emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                     (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                if (insn.Bits(75, 1)) {
                    emit(0x0400000000000000ULL | predicate | 251 | (u64{c} << 8) | (u64{0xffffffffU} << 20) | (u64{2} << 53));
                    c = 251;
                }
                emit(base(0x5c10000000000000ULL) | (u64{c} << 8) | (u64{250} << 20));
                break;
            }
            if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || (insn.words[1] & ~0xffU))
                return fail(index, "native integer register multiply mode");
            multiply_low(insn.Bits(24, 8), insn.Bits(32, 8), insn.Bits(64, 8), false);
            break;
        }
        case 0x2a5: { // UIMAD.WIDE.U32 UR(d):UR(d+1) = URa * URb + URc:UR(c+1)
            const u32 a = uniform_register(insn.Bits(24, 8)), b = uniform_register(insn.Bits(32, 8));
            const u32 c = insn.Bits(64, 8);
            const u32 c_low = uniform_register(c), c_high = c == 63 ? 255U : uniform_register(c + 1);
            const u32 d_low = uniform_register(insn.Bits(16, 8)), d_high = uniform_register(insn.Bits(16, 8) + 1);
            if (native_predicate != 7 || (insn.words[1] & ~0xffU) || (insn.words[2] & ~0xffU) != 0x0f8e0000 ||
                (insn.Bits(16, 8) & 1) || (c != 63 && (c & 1)) || a == ~0U || b == ~0U || c_low == ~0U ||
                c_high == ~0U || d_low == ~0U || d_high == ~0U || d_low == 255)
                return fail(index, "native uniform wide multiply mode");
            // High word in R253 (base-2^16 long multiplication, as IMAD.HI), low word in R250.
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 250, true, false, 2, false);
            xmad(250, a, b, 251, false, true, 1, false);
            xmad(252, a, b, 251, true, true, 2, false);
            emit(0x3828000000070000ULL | 250 | (u64{250} << 8) | (u64{16} << 20));
            emit(0x5c10000000070000ULL | 253 | (u64{250} << 8) | (u64{252} << 20));
            xmad(250, a, b, 255, false, false, 0, false);
            xmad(251, a, b, 255, true, false, 0, false);
            xmad(251, a, b, 251, false, true, 0, false);
            emit(0x5c18000000070000ULL | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            emit(0x5c10000000070000ULL | 250 | (u64{250} << 8) | (u64{c_low} << 20) | (u64{1} << 47));
            emit(0x5c10000000070000ULL | d_high | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 43));
            emit(0x5c10000000070000ULL | d_low | (u64{250} << 8) | (u64{255} << 20));
            break;
        }
        case 0x226: { // IDP.2A.LO.U16.U8: a.lo16 * b.byte0 + a.hi16 * b.byte1 + c
            if ((insn.words[2] & ~0xffU) != 0x1000 || (insn.words[1] & ~0xffU))
                return fail(index, "native packed dot-product mode");
            const u32 a = insn.Bits(24, 8), b = insn.Bits(32, 8), c = insn.Bits(64, 8);
            for (u32 component = 0; component < 2; ++component) {
                emit(0x3828000000000000ULL | predicate | (250 + component) | (u64{b} << 8) | (u64{component * 8} << 20));
                emit(0x0400000000000000ULL | predicate | (250 + component) | (u64{250 + component} << 8) | (u64{0xff} << 20));
            }
            xmad(252, a, 250, c, false, false, 0, false);
            xmad(static_cast<u32>(dest), a, 251, 252, true, false, 0, false);
            break;
        }
        case 0x225: case 0x227: case 0x625: case 0x825: case 0x827: case 0xa27: {
            // IMAD.HI Rd / IMAD.WIDE Rd:Rd+1 = a * b + c:c+1 (.U32 or signed, bit 73), with an optional
            // carry-out (bits 81-83). b is a register, an immediate (0x8xx) or a cbuf (0xa27); 0x625 takes
            // b from bits 64-71 and adds a cbuf pair.
            const bool wide = opcode == 0x225 || opcode == 0x625 || opcode == 0x825;
            const bool immediate = opcode == 0x825 || opcode == 0x827, signed_product = insn.Bits(73, 1);
            const u32 c = opcode == 0x625 ? 255U : insn.Bits(64, 8), carry_output = insn.Bits(81, 3);
            if ((insn.words[2] & ~0x000e02ffU) != 0x07800000 ||
                ((opcode == 0x225 || opcode == 0x227) && (insn.words[1] & ~0xffU)) ||
                ((opcode == 0xa27 || opcode == 0x625) && (!valid_cbuf() || insn.Bits(62, 2))) ||
                (opcode == 0x625 && insn.Bits(38, 16) > 0xfff8) || (c != 255 && (c & 1)) ||
                (wide && (dest == 255 || (dest & 1))) || (signed_product && carry_output != 7))
                return fail(index, "native high multiply mode");
            const u32 a = insn.Bits(24, 8);
            const u32 b = immediate ? insn.words[1] : opcode == 0xa27 ? 254U : insn.Bits(opcode == 0x625 ? 64 : 32, 8);
            if (opcode == 0xa27) emit(0x4c98000000000000ULL | predicate | 254 | (u64{15} << 39) | cbuf());
            if (signed_product) {
                // The signed high word is the unsigned one minus (a < 0 ? b : 0) + (b < 0 ? a : 0),
                // which R253 holds (R252 is free until the multiplication).
                const u32 b_register = immediate ? 254U : b;
                if (immediate) mov_imm(254, b);
                emit(0x3828000000000000ULL | predicate | 253 | (u64{a} << 8) | (u64{31} << 20) | (u64{1} << 48));
                emit(0x5c40000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{b_register} << 20) | (u64{7} << 48));
                emit(0x3828000000000000ULL | predicate | 252 | (u64{b_register} << 8) | (u64{31} << 20) | (u64{1} << 48));
                emit(0x5c40000000000000ULL | predicate | 252 | (u64{252} << 8) | (u64{a} << 20) | (u64{7} << 48));
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{252} << 20));
            }
            // Base-2^16 long multiplication. Each intermediate sum fits U32.
            xmad(250, a, b, 255, false, false, 0, immediate);
            xmad(251, a, b, 250, true, false, 2, immediate);
            xmad(250, a, b, 251, false, true, 1, immediate);
            xmad(252, a, b, 251, true, true, 2, immediate);
            emit(0x3828000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{16} << 20));
            if (signed_product)
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{253} << 20) | (u64{1} << 48));
            if (!wide && c == 255 && carry_output == 7) {
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{252} << 20));
                break;
            }
            // High word in R253 and low word in R250 (as UIMAD.WIDE.U32), then c:c+1 is added.
            emit(0x5c10000000000000ULL | predicate | 253 | (u64{250} << 8) | (u64{252} << 20));
            xmad(250, a, b, 255, false, false, 0, immediate);
            xmad(251, a, b, 255, true, false, 0, immediate);
            xmad(251, a, b, 251, false, true, 0, immediate);
            emit(0x5c18000000000000ULL | predicate | 250 | (u64{251} << 8) | (u64{250} << 20) | (u64{16} << 39));
            u32 c_low = c, c_high = c == 255 ? 255U : c + 1;
            if (opcode == 0x625) {
                emit(0x4c98000000000000ULL | predicate | 252 | (u64{15} << 39) | cbuf());
                emit(0x4c98000000000000ULL | predicate | 254 | (u64{15} << 39) | (cbuf() + (u64{1} << 20)));
                c_low = 252;
                c_high = 254;
            }
            const u32 high = wide ? static_cast<u32>(dest) + 1 : static_cast<u32>(dest);
            emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{c_low} << 20) | (u64{1} << 47));
            if (carry_output == 7) {
                emit(0x5c10000000000000ULL | predicate | high | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 43));
                if (wide) emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{255} << 20));
                break;
            }
            // Maxwell IADD has no .X with .CC: R251 takes the low carry, R254 counts the high word's.
            emit(0x5c10000000000000ULL | predicate | 251 | (u64{255} << 8) | (u64{255} << 20) | (u64{1} << 43));
            emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{c_high} << 20) | (u64{1} << 47));
            emit(0x5c10000000000000ULL | predicate | 254 | (u64{255} << 8) | (u64{255} << 20) | (u64{1} << 43));
            emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{251} << 20) | (u64{1} << 47));
            emit(0x5c10000000000000ULL | predicate | 254 | (u64{254} << 8) | (u64{255} << 20) | (u64{1} << 43));
            emit(0x5c98000000000000ULL | predicate | high | (u64{253} << 20) | (u64{15} << 39));
            if (wide) emit(base(0x5c98000000000000ULL) | (u64{250} << 20) | (u64{15} << 39));
            emit(0x5b60000000000000ULL | predicate | 7 | (u64{carry_output} << 3) | (u64{254} << 8) |
                 (u64{255} << 20) | (u64{7} << 39) | (u64{5} << 49));
            break;
        }
        case 0x95c:
            if (insn.words[0] != 0x795c || insn.words[1] != 4 || insn.words[2] != 0x00300000 ||
                std::find(verified_traps.begin(), verified_traps.end(), index * sizeof(Instruction)) == verified_traps.end())
                return fail(index, "native runtime breakpoint/trap");
            emit(0x50b0000000000f00ULL | predicate);
            break;
        case 0xa24: { // IMAD, constant-buffer multiplier
            if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || !valid_cbuf() || insn.Bits(62, 2))
                return fail(index, "native integer constant-buffer multiply mode");
            emit(0x4c98000000000000ULL | predicate | 252 | (u64{15} << 39) | cbuf());
            multiply_low(insn.Bits(24, 8), 252, insn.Bits(64, 8), false);
            break;
        }
        case 0x424: case 0x624: { // IMAD.MOV.U32: zero product plus immediate/cbuf
            if (opcode == 0x624 && insn.Bits(24, 8) == 255 &&
                (insn.words[2] & ~0x07800000U) == 0x000e06ffU) {
                const u32 source = operand(2, false);
                mov_imm(250, 1);
                emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                    (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                emit(base(0x5c10000000000000ULL) | (u64{source} << 8) | (u64{250} << 20));
                break;
            }
            if (insn.Bits(24, 8) != 255 || insn.Bits(64, 8) != 255 || insn.words[2] != 0x078e00ff) {
                if ((insn.words[2] & ~0x2ffU) != 0x078e0000 || (opcode == 0x624 && insn.Bits(62, 2)))
                    return fail(index, "native multiply wide addend mode");
                const u32 a = operand(0, false), b = operand(1, false), c = operand(2, false);
                multiply_low(a, b, c, false);
                break;
            }
            if (insn.Bits(24, 8) != 255 || insn.Bits(64, 8) != 255 ||
                insn.words[2] != 0x078e00ff)
                return fail(index, "native integer move mode");
            if (opcode == 0x624) {
                if (!valid_cbuf() || insn.Bits(62, 2))
                    return fail(index, "native integer move constant buffer");
                emit(base(0x4c98000000000000ULL) | (u64{15} << 39) | cbuf());
            } else {
                emit(base(0x0100000000000000ULL) | (u64{15} << 12) |
                     (u64{insn.words[1]} << 20));
            }
            break;
        }
        case 0x824: { // Captured low-word IMAD forms; bit 75 negates the addend.
            const bool negate_c = insn.Bits(75, 1) != 0;
            if (insn.words[1] == 1 && (insn.words[2] & ~(0x07800800U | 0xffU)) == 0x000e0600U) {
                // IMAD.X Rd, Ra, 1, Rc, Pn: Ra + Rc plus the carry predicate in bits 87-90 (~Rc with bit 75).
                u32 c = insn.Bits(64, 8);
                mov_imm(252, 1);
                emit(0x5ca0000000000000ULL | predicate | 252 | (u64{252} << 8) | (u64{255} << 20) |
                     (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                if (negate_c) {
                    emit(0x0400000000000000ULL | predicate | 251 | (u64{c} << 8) | (u64{0xffffffffU} << 20) | (u64{2} << 53));
                    c = 251;
                }
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{c} << 20));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{252} << 20));
                break;
            }
            if (insn.words[1] == 1 && (insn.words[2] & 0xfffff700U) == 0x078e0200) {
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(64, 8)} << 20) | (u64{negate_c} << 48));
                break;
            }
            const u32 multiplier = insn.words[1];
            if (insn.Bits(64, 8) == 255 && !negate_c && multiplier > 1 && !(multiplier & (multiplier - 1)) &&
                (insn.words[2] & ~0x2ffU) == 0x078e0000) {
                // IMAD.SHL by a power of two: one shift (also the form Eden's jump-table tracking
                // expects between the clamp and the table load).
                u32 shift = 0;
                while ((1U << shift) != multiplier) ++shift;
                emit(base(0x3848000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{shift} << 20));
                break;
            }
            if (insn.words[1] != 48 || insn.words[2] != 0x078e02ff) {
                if ((insn.words[2] & ~0xaffU) != 0x078e0000)
                    return fail(index, "native integer multiply mode");
                u32 c = insn.Bits(64, 8);
                if (negate_c && c != 255) {
                    emit(0x5c10000000000000ULL | predicate | 252 | (u64{255} << 8) | (u64{c} << 20) |
                         (u64{1} << 48));
                    c = 252;
                }
                multiply_low(insn.Bits(24, 8), insn.words[1], c, true);
                break;
            }
            const u64 source = insn.Bits(24, 8);
            emit(0x5c18000000000000ULL | predicate | 250 | (source << 8) | (source << 20) | (u64{1} << 39));
            emit(base(0x3848000000000000ULL) | (u64{250} << 8) | (u64{4} << 20));
            break;
        }
        case 0x806: { // VOTE: register ballot, ordinary predicate input and output.
            if (insn.words[1] || insn.Bits(72, 2) > 2 || (insn.words[2] & ~0x078e0300U))
                return fail(index, "native vote mode");
            emit(base(0x50d8000000000000ULL) | (u64{insn.Bits(87, 4)} << 39) |
                 (u64{insn.Bits(81, 3)} << 45) | (u64{insn.Bits(72, 2)} << 48));
            break;
        }
        case 0x886: { // VOTEU ballot and independent uniform predicate output.
            const u32 output = insn.Bits(81, 3), target = insn.Bits(16, 8);
            const u32 mapped_target = uniform_register(target);
            if (mapped_target == ~0U || insn.Bits(24, 8) || insn.words[1] ||
                (insn.words[2] & ~0x078e0300U) || insn.Bits(72, 2) > 2 || native_predicate != 7)
                return fail(index, "native uniform vote mode");
            const u32 temporary = output == 7 ? 7 : 6 - output;
            const u32 mask = 1U << temporary;
            if (output != 7) emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
            emit(0x50d8000000070000ULL | mapped_target |
                (u64{insn.Bits(87, 4)} << 39) | (u64{temporary} << 45) | (u64{insn.Bits(72, 2)} << 48));
            if (output != 7) {
                emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
            }
            break;
        }
        case 0x3c2: { // R2UR: broadcast the first lane satisfying the execution predicate.
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || target == 255 || insn.words[1] || insn.words[2] != 0x000e0000)
                return fail(index, "native register-to-uniform mode");
            save_predicate();
            emit(0x50d8000000070000ULL | 250 | (u64{native_predicate} << 39) |
                 (u64{6} << 45) | (u64{1} << 48));
            emit(0x5c10000000070000ULL | 251 | (u64{250} << 8) | (u64{255} << 20) | (u64{1} << 49));
            emit(0x5c40000000070000ULL | 250 | (u64{250} << 8) | (u64{251} << 20) | (u64{7} << 48));
            emit(0x5c30000000070000ULL | 250 | (u64{250} << 20));
            emit(0xef10000000070000ULL | 251 | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{250} << 20) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            if (native_predicate == 7) {
                // Every lane executing it is in the ballot, so no lane-less fallback can be taken.
                // Leaving the old value selectable would make it a source of the result for the
                // texture and global memory tracking (a handle traced to the uniform's last load).
                emit(0x5c98000000070000ULL | target | (u64{15} << 39) | (u64{251} << 20));
            } else {
                emit(0x5ca0000000070000ULL | target | (u64{251} << 8) |
                     (u64{target} << 20) | (u64{6} << 39));
            }
            restore_predicate();
            break;
        }
        case 0x81c: { // PLOP3.LUT; bit 67 makes the third source a uniform predicate.
            const u32 target = insn.Bits(81, 3);
            const u32 second_target = insn.Bits(84, 3);
            const u32 lut = insn.Bits(64, 3) | (insn.Bits(72, 5) << 3);
            if (second_target != 7) {
                if (target == second_target || insn.Bits(24, 8) || insn.words[1] ||
                    insn.Bits(91, 5) || (insn.words[3] & 0xc00001ffU))
                    return fail(index, "native dual predicate LUT mode");
                predicate_lut(lut, insn.Bits(87, 4), insn.Bits(77, 4), insn.Bits(68, 4),
                              target, insn.Bits(67, 1) != 0, second_target, insn.Bits(16, 8));
                break;
            }
            if (target == 7 || insn.Bits(84, 3) != 7 ||
                insn.Bits(16, 8) || insn.words[1])
                return fail(index, "native predicate LUT mode");
            if (!insn.Bits(67, 1) && (lut == 0xfe || lut == 0xa8)) {
                emit(0x5090000000000000ULL | predicate | (u64{target} << 3) | 7 |
                     (u64{insn.Bits(87, 4)} << 12) | (u64{1} << 24) |
                     (u64{insn.Bits(77, 4)} << 29) | (u64{insn.Bits(68, 4)} << 39) |
                     (u64{lut == 0xfe} << 45));
                break;
            }
            if (!insn.Bits(67, 1) && (lut == 0x80 ||
                (lut == 8 && insn.Bits(68, 4) == 7 && insn.Bits(77, 4) == 7))) {
                emit(0x5090000000000000ULL | predicate | (u64{target} << 3) | 7 |
                     (u64{insn.Bits(87, 4) ^ (lut == 8 ? 8U : 0U)} << 12) | (u64{insn.Bits(77, 4)} << 29) |
                     (u64{insn.Bits(68, 4)} << 39));
                break;
            }
            predicate_lut(lut, insn.Bits(87, 4), insn.Bits(77, 4), insn.Bits(68, 4), target,
                          insn.Bits(67, 1) != 0);
            break;
        }
        case 0x89c: { // UPLOP3.LUT: load the uniform predicates into P0-P6, combine, merge back.
            const u32 target = insn.Bits(81, 3);
            const u32 lut = insn.Bits(64, 3) | (insn.Bits(72, 5) << 3);
            if (insn.Bits(84, 3) != 7 || insn.Bits(67, 1) || insn.Bits(16, 8) || insn.words[1])
                return fail(index, "native uniform predicate LUT mode");
            if (target == 7) break;
            const auto slot = [](u32 operand) { return (operand & 7) == 7 ? operand : (operand & 8) | (6 - (operand & 7)); };
            predicate = u64{slot(native_predicate)} << 16; // a UPn guard is loaded as well
            emit(0x38e8000000070000ULL | 254 | (u64{255} << 8) | (u64{0x7f} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{0x7f} << 20));
            predicate_lut(lut, slot(insn.Bits(87, 4)), slot(insn.Bits(77, 4)), slot(insn.Bits(68, 4)), 6 - target);
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{1U << (6 - target)} << 20));
            emit(0x38f0000000070000ULL | (u64{254} << 8) | (u64{0x7f} << 20));
            break;
        }
        case 0x883: { // UP2UR UR, UPR, URZ, mask: bit k of the result is UPk (R239 bit 6-k).
            const u32 target = uniform_register(insn.Bits(16, 8)), mask = insn.words[1];
            if (target == ~0U || insn.Bits(24, 8) != 63 || native_predicate != 7 ||
                insn.words[2] != 0x08000000 || mask > 0x7f)
                return fail(index, "native uniform predicate move mode");
            if (target == 255) break;
            mov_imm(target, 0);
            for (u32 k = 0; k < 7; ++k) {
                if (!((mask >> k) & 1)) continue;
                emit(0x3800000000070000ULL | 250 | (u64{239} << 8) | (u64{(6 - k) | (1U << 8)} << 20));
                emit(0x36f0000000070000ULL | target | (u64{250} << 8) | (u64{k | (1U << 8)} << 20) |
                     (u64{target} << 39));
            }
            break;
        }
        case 0x882: { // UMOV immediate, into any uniform register (Hyrule Warriors: @!UP0 UMOV UR59, 0x20).
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || insn.words[2]) return fail(index, "native uniform immediate move mode");
            if (target != 255) mov_imm(target, insn.words[1]);
            break;
        }
        case 0xc82: { // UMOV UR, UR (or URZ), including uniform execution predicates.
            // Any uniform destination (Hyrule Warriors: @UP0 UMOV UR59, UR58).
            const u32 source = insn.words[1] < 64 ? uniform_register(insn.words[1]) : ~0U;
            const u32 target = uniform_register(static_cast<u32>(dest));
            if (target == ~0U || source == ~0U || insn.words[2] != 0x08000000)
                return fail(index, "native uniform move mode");
            if (target != 255)
                emit(0x5c98000000000000ULL | predicate | target | (u64{source} << 20) | (u64{15} << 39));
            break;
        }
        case 0xab9: case 0xabb: { // ULDC, direct or indexed.
            const u32 uniform_dest = insn.Bits(16, 8), index_reg = insn.Bits(24, 8);
            const u32 width = insn.Bits(73, 3);
            const bool subword_load = width == 0 || width == 2;
            const u32 offset = insn.Bits(38, 16);
            const u32 alignment = width == 0 ? 1U : width == 2 ? 2U : 4U;
            const u32 target = uniform_register(uniform_dest);
            const u32 index_target = opcode == 0xabb ? uniform_register(index_reg) : 255U;
            if (target == ~0U || target == 255 || (!subword_load && width != 4 && width != 5) ||
                (opcode == 0xab9 && subword_load && offset >= 0x8000) ||
                (width == 5 && ((uniform_dest & 1) || uniform_register(uniform_dest + 1) != target + 1)) ||
                // An index register can align the offset; the host reads the aligned word either way.
                bank() >= 18 || (opcode == 0xab9 && offset % alignment) || insn.Bits(62, 2) ||
                (insn.words[2] & ~0xe00U) != (opcode == 0xab9 ? 0U : 0x08000000U) ||
                index_target == ~0U)
                return fail(index, "native uniform constant load mode");
            if (opcode == 0xab9 && width == 4) {
                emit(0x4c98000000000000ULL | predicate | target | (u64{15} << 39) | cbuf());
            } else {
                emit(0xef90000000000000ULL | predicate | target |
                     (u64{index_target} << 8) | (u64{offset} << 20) |
                     (u64{bank()} << 36) | (u64{width} << 48));
            }
            break;
        }
        case 0x899: { // USHF.L.U32; accept a zero word outside the selected half.
            const u32 dst = insn.Bits(16, 8), low = insn.Bits(24, 8), high = insn.Bits(64, 8);
            const u32 dst_reg = uniform_register(dst), low_reg = uniform_register(low);
            const u32 high_reg = uniform_register(high);
            const bool dst_ok = dst != 63 && dst_reg != ~0U;
            const bool low_ok = low != 63 && low_reg != ~0U, high_ok = high != 63 && high_reg != ~0U;
            if (dst_ok && low_reg != ~0U && high_reg != ~0U &&
                insn.words[2] == (0x08001200U | high)) {
                mov_imm(252, insn.words[1]);
                right_shift_u64_low(dst_reg, low_reg, high_reg, 252);
                break;
            }
            // A UPn guard (Hyrule Warriors: @UP1 USHF.R.U32.HI) is swapped into the predicate above.
            if (insn.words[2] == (0x08011400U | high) && low == 63 && high_ok &&
                dst_ok && insn.words[1] <= 32) {
                emit(0x3828000000000000ULL | predicate | dst_reg |
                     (u64{high_reg} << 8) | (u64{insn.words[1]} << 20) | (u64{1} << 48));
                break;
            }
            // Logical right shifts of one register (bit 76 right, bit 80 high word, bits 73-74
            // U32): USHF.R.U32 URd, URa, n, URZ and USHF.R.U32.HI URd, URZ, n, URb.
            const bool right_low = insn.words[2] == 0x0800163f && low_ok;
            const bool right_high = insn.words[2] == (0x08011600U | high) && low == 63 && high_ok;
            if ((right_low || right_high) && dst_ok && insn.words[1] <= 32) {
                emit(0x3828000000000000ULL | predicate | dst_reg |
                     (u64{right_high ? high_reg : low_reg} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            const bool high_result = insn.words[2] == (0x08010600U | high) && low == 63 && high_ok;
            const bool low_result = insn.words[2] == 0x0800063f && low_ok;
            if (!dst_ok || insn.words[1] > 32 || (!high_result && !low_result))
                return fail(index, "native uniform shift mode");
            emit(0x3848000000000000ULL | predicate | dst_reg |
                 (u64{high_result ? high_reg : low_reg} << 8) | (u64{insn.words[1]} << 20));
            break;
        }
        case 0xb82: { // LDC indexed byte offset; bits 78-79 the mode as Maxwell's (2 = IS)
            // .IS: Ra + offset addresses all constant buffers as one space, the buffer in its high
            // half (Kirby Air Riders' vertex shaders), which Eden's Maxwell LDC implements.
            const u32 width = insn.Bits(73, 3), mode = insn.Bits(78, 2);
            if (width > 5 || (mode != 0 && mode != 2) ||
                (insn.words[2] & ~0xce00U) || (insn.words[1] & 0xf800003fU) ||
                bank() >= 18 ||
                (width == 5 && ((dest & 1) || dest >= 254)))
                return fail(index, "native indexed constant load mode");
            emit(base(0xef90000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(38, 16)} << 20) | (u64{bank()} << 36) | (u64{mode} << 44) | (u64{width} << 48));
            break;
        }
        case 0xf89: { // SHFL with immediate lane and segmentation/clamp mask, any mode.
            const u32 lane = insn.Bits(53, 5), mask = insn.Bits(40, 13), mode = insn.Bits(58, 2);
            if (insn.words[2] != 0x000e0000 || insn.Bits(32, 8) || insn.Bits(60, 4))
                return fail(index, "native immediate shuffle mode");
            emit(base(0xef10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{lane} << 20) | (u64{1} << 28) | (u64{1} << 29) |
                 (u64{mode} << 30) | (u64{mask} << 34) | (u64{7} << 48));
            break;
        }
        case 0x822: { // FSWZADD: per lane 0 = a+b, 1 = a-b, 2 = b-a, 3 = b (nvdisasm PP/PN/NP/ZP).
            const u32 swizzle = insn.words[1];
            if (swizzle > 0xff || insn.words[2] > 255)
                return fail(index, "native swizzled add mode");
            // Maxwell's lane codes 1 and 2 are the other way round (1 = b-a, 2 = a-b).
            const u32 maxwell = ((swizzle & 0x55) << 1) | ((swizzle & 0xaa) >> 1);
            emit(base(0x50f8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(64, 8)} << 20) | (u64{maxwell} << 28));
            break;
        }
        case 0xf6c: {
            const u32 handle = uniform_register(insn.Bits(40, 6));
            const u32 coordinates = insn.Bits(24, 8), derivatives = insn.Bits(32, 8);
            const u32 dimension = insn.Bits(61, 3), second_dest = insn.Bits(64, 8);
            const u32 mask = insn.Bits(72, 4), channels = static_cast<u32>(std::popcount(mask));
            const u32 register_limit = high_uniform_base;
            if ((dimension != 1 && dimension != 5) || insn.Bits(46, 15) ||
                (insn.words[2] & ~0x04000fffU) != 0x081e0000U || mask == 0 ||
                (insn.words[3] & 0xc00001ffU) || handle == ~0U || handle == 255 ||
                coordinates + (dimension == 5 ? 3U : 2U) > register_limit ||
                derivatives + 4 > register_limit ||
                dest + (std::min)(channels, 2U) > register_limit || (channels > 1 && (dest & 1)) ||
                (channels > 2 ? second_dest != 255 &&
                    (second_dest + channels - 2 > register_limit || (channels == 4 && (second_dest & 1)) ||
                     (second_dest < dest + 2 && dest < second_dest + channels - 2))
                              : second_dest != 255))
                return fail(index, "native uniform gradient texture mode");
            emit(0x5c98000000000000ULL | predicate | 248 | (u64{15} << 39) | (u64{handle} << 20));
            for (u32 i = 0; i < (dimension == 5 ? 3U : 2U); ++i)
                emit(0x5c98000000000000ULL | predicate | (249 + i) | (u64{15} << 39) |
                     (u64{coordinates + i} << 20));
            emit(0xde40000000000000ULL | predicate | 248 | (u64{248} << 8) |
                 (u64{derivatives} << 20) | (u64{dimension == 5 ? 3U : 2U} << 28) |
                 (u64{mask} << 31) | (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
            for (u32 i = 0; i < channels; ++i) {
                if (i >= 2 && second_dest == 255) break;
                emit(0x5c98000000000000ULL | predicate | (i < 2 ? dest + i : second_dest + i - 2) |
                     (u64{15} << 39) | (u64{248 + i} << 20));
            }
            break;
        }
        case 0xb6c: { // TXD 2D or ARRAY_2D, any channels; the two destination pairs anywhere.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 coordinates = insn.Bits(24, 8), derivatives = insn.Bits(32, 8);
            // Ampere dimensions: 1 = 2D, 5 = ARRAY_2D; Maxwell TXD's are 2 and 3. The coordinates
            // (x, y[, layer]) and the four gradients sit in the same registers for both.
            const u32 dimension = insn.Bits(61, 3);
            const u32 high = insn.Bits(64, 8); // destination of the third and fourth channels
            // Bit 90 is forwarded below (Maxwell's NODEP); Splatoon Raiders sets it. Bits 72-75 are
            // the channel mask: the first two enabled channels go to Rd, Rd+1, the others to the
            // second destination (RZ drops them; a third channel alone may be in an odd register).
            const u32 mask = insn.Bits(72, 4), channels = static_cast<u32>(std::popcount(mask));
            if ((insn.words[2] & ~(0xfffU | (1U << 26))) != 0x001e0000 || mask == 0 || insn.Bits(59, 2) ||
                (dimension != 1 && dimension != 5) || binding >= 18 || offset > 0x1fff ||
                (channels > 1 && (dest & 1)) || dest >= 248 || (channels <= 2 && high != 255) ||
                (high != 255 && ((channels > 3 && (high & 1)) || high >= 248)) ||
                coordinates >= 248 || derivatives >= 247)
                return fail(index, "native gradient texture mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native texture descriptor buffer conflict");
            result.required_texture_buffer = binding;
            // Maxwell writes the channels to consecutive registers: when the second destination
            // doesn't follow Rd (Splatoon Raiders: R12/R18, R18/R44), write them to scratch and move
            // each into place.
            const bool consecutive = channels <= 2 || high == dest + 2;
            const u64 target = consecutive ? dest : 248;
            emit(0xde00000000000000ULL | predicate | target | (u64{coordinates} << 8) |
                 (u64{derivatives} << 20) | (u64{dimension == 5 ? 3U : 2U} << 28) | (u64{mask} << 31) |
                 (u64{offset} << 36) | (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
            if (!consecutive) {
                for (u32 i = 0; i < channels; ++i) {
                    if (i >= 2 && high == 255) break;
                    const u32 to = i < 2 ? static_cast<u32>(dest) + i : high + (i - 2);
                    emit(0x5c98000000000000ULL | predicate | to | (u64{15} << 39) | (u64{248 + i} << 20));
                }
            }
            break;
        }
        case 0x989: { // SHFL, immediate lane and register clamp/segmentation mask, any mode.
            if ((insn.words[1] & 0xf01fffffU) || (insn.words[2] & ~0x000e00ffU))
                return fail(index, "native register-mask shuffle mode");
            emit(base(0xef10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(53, 5)} << 20) | (u64{1} << 28) | (u64{insn.Bits(58, 2)} << 30) |
                 (u64{insn.Bits(64, 8)} << 39) | (u64{insn.Bits(81, 3)} << 48));
            break;
        }
        case 0x589: { // SHFL, register lane and immediate clamp/segmentation mask, any mode.
            const u32 lane = insn.Bits(32, 8);
            if ((insn.words[2] & ~0x000e0000U) || insn.Bits(53, 5) || insn.Bits(60, 4) ||
                (insn.words[3] & 0xc00001ffU) ||
                (lane >= (uses_uniform_bank ? high_uniform_base : 246U) && lane != 255))
                return fail(index, "native shuffle mode");
            emit(base(0xef10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{lane} << 20) |
                 (u64{1} << 29) | (u64{insn.Bits(58, 2)} << 30) |
                 (u64{insn.Bits(40, 13)} << 34) | (u64{insn.Bits(81, 3)} << 48));
            break;
        }
        case 0x98e: { // RED.E.{ADD..XOR}.STRONG.GPU U32 (op in bits 87-89), register address.
            const u32 type = insn.Bits(73, 3);
            const u32 operation = insn.Bits(87, 3);
            const bool uniform_address = insn.Bits(24, 8) == 255;
            const u32 variable = 0x03800e00U | (uniform_address ? 0x3fU : 0U);
            const u32 mode = uniform_address ? 0x0c10e100U : 0x0010e100U;
            const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
            if ((insn.words[2] & ~variable) != mode ||
                (type != 0 && type != 1 && type != 3) || (type == 3 && operation != 0))
                return fail(index, "native global reduction mode");
            const u32 address = uniform_address ? uniform_register(insn.Bits(64, 6))
                                                : insn.Bits(24, 8);
            if (address == ~0U || (address & 1) || offset < -0x80000 || offset > 0x7ffff)
                return fail(index, "native global reduction address");
            emit(0xebf8000000000000ULL | predicate | insn.Bits(32, 8) | (u64{address} << 8) |
                 (u64{type} << 20) | (u64{operation} << 23) |
                 (u64{static_cast<u32>(offset) & 0xfffff} << 28) | (u64{1} << 48));
            break;
        }
        case 0x2bd: { // UFLO.U32, the highest set-bit index (.SH, bit 74: its shift amount).
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 source = uniform_register(insn.Bits(32, 8));
            if (dest == 63 || target == ~0U || source == ~0U || (insn.words[1] & ~0xffU) ||
                (insn.words[2] & ~0x400U) != 0x080e0000 || native_predicate != 7)
                return fail(index, "native uniform leading bit mode");
            emit(0x5c30000000000000ULL | predicate | target | (u64{source} << 20) |
                 (u64{insn.Bits(74, 1)} << 41));
            break;
        }
        case 0x2be: { // UBREV, as a bit-reversing zero shift like BREV.
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 source = uniform_register(insn.Bits(32, 8));
            if (dest == 63 || target == ~0U || source == ~0U || (insn.words[1] & ~0xffU) ||
                insn.words[2] != 0x08000000 || native_predicate != 7)
                return fail(index, "native uniform bit reverse mode");
            emit(0x3828000000000000ULL | predicate | target | (u64{source} << 8) |
                 (u64{1} << 39) | (u64{1} << 40));
            break;
        }
        case 0x2bf: { // UPOPC, as POPC of the uniform's register.
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 source = uniform_register(insn.Bits(32, 8));
            if (dest == 63 || target == ~0U || source == ~0U || (insn.words[1] & ~0x800000ffU) ||
                insn.words[2] != 0x08000000 || native_predicate != 7)
                return fail(index, "native uniform population count mode");
            emit(0x5c08000000000000ULL | predicate | target | (u64{source} << 20) |
                 (u64{insn.Bits(63, 1)} << 40));
            break;
        }
        case 0x3a1: { // MATCH.ALL (.U32, or .U64 on bit 73) with only the all-equal predicate retained.
            const u32 source = insn.Bits(24, 8), target = insn.Bits(81, 3);
            const u32 words = insn.Bits(73, 1) ? 2 : 1;
            const u32 scratch = target == 6 ? 5U : 6U; // the comparison, saved around the match
            if (dest != 255 || source >= 254 || (words == 2 && (source & 1)) || target == 7 ||
                native_predicate != 7 || insn.words[1] || (insn.words[2] & ~0xe0200U) != 0)
                return fail(index, "native warp match mode");
            save_predicate(1U << scratch);
            // Compare every lane with the lowest active one, then require all to agree.
            emit(0x50d8000000070000ULL | 250 | (u64{7} << 39) | (u64{7} << 45) | (u64{1} << 48));
            emit(0x5c10000000070000ULL | 251 | (u64{250} << 8) | (u64{255} << 20) | (u64{1} << 49));
            emit(0x5c40000000070000ULL | 250 | (u64{250} << 8) | (u64{251} << 20) | (u64{7} << 48));
            emit(0x5c30000000070000ULL | 250 | (u64{250} << 20));
            for (u32 word = 0; word < words; ++word)
                emit(0xef10000000070000ULL | (251 + word) | (u64{source + word} << 8) |
                    (u64{250} << 20) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x5b60000000070000ULL | 7 | (u64{scratch} << 3) | (u64{source} << 8) |
                (u64{251} << 20) | (u64{7} << 39) | (u64{2} << 49));
            if (words == 2)
                emit(0x5b60000000070000ULL | 7 | (u64{scratch} << 3) | (u64{source + 1} << 8) |
                    (u64{252} << 20) | (u64{scratch} << 39) | (u64{2} << 49));
            emit(0x50d8000000070000ULL | 255 | (u64{scratch} << 39) | (u64{target} << 45));
            restore_predicate(1U << scratch);
            break;
        }
        case 0x80d: { // CSMTEST: a test of shading-mode state Eden does not model, taken as false.
            // Kirby Air Riders' vertex-side shaders run "CSMTEST.A.FCSM_TR P0" then "@P0 EXIT":
            // never leave early. Outputs of PT discard the result.
            const u32 first = insn.Bits(81, 3), second = insn.Bits(84, 3);
            if (insn.Bits(87, 4) != 7 || insn.Bits(16, 16))
                return fail(index, "native CSM test mode");
            for (const u32 target : {first, second})
                if (target != 7)
                    emit(0x5090000000000000ULL | predicate | (u64{target} << 3) | 7 | (u64{7} << 12) |
                         (u64{1} << 15) | (u64{7} << 29) | (u64{7} << 39));
            break;
        }
        case 0x3c4: { // REDUX.SUM (modulo 2^32), .MIN/.MAX (unsigned), .OR or .XOR over the actual active lanes.
            const u32 source = insn.Bits(24, 8), target = uniform_register(static_cast<u32>(dest));
            const u32 mode = insn.words[2] & ~0x200U;
            const bool maximum = mode == 0x14000, minimum = mode == 0x10000;
            const bool signed_order = insn.Bits(73, 1);
            // OR and XOR: inactive lanes contribute zero, as for the sum. Maxwell LOP: 1 = OR, 2 = XOR.
            const u32 logical = insn.words[2] == 0x4000 ? 1U : insn.words[2] == 0x8000 ? 2U : 0U;
            if (target == ~0U || target == 255 || insn.words[1] ||
                (insn.words[2] != 0xc000 && !maximum && !minimum && !logical) ||
                (signed_order && !maximum && !minimum) || native_predicate != 7)
                return fail(index, "native warp sum mode");
            emit(0x50d8000000070000ULL | 250 | (u64{7} << 39) | (u64{7} << 45) | (u64{1} << 48));
            mov_imm(target, minimum ? ~0U : 0U);
            for (u32 lane = 0; lane < 32; ++lane) {
                emit(0xef10000000070000ULL | 251 | (u64{source} << 8) | (u64{lane} << 20) |
                    (u64{1} << 28) | (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                if (signed_order)
                    emit(0x0400000000070000ULL | 251 | (u64{251} << 8) |
                         (u64{0x80000000U} << 20) | (u64{2} << 53));
                emit(0x3828000000070000ULL | 252 | (u64{250} << 8) | (u64{lane} << 20));
                emit(0x0400000000070000ULL | 252 | (u64{252} << 8) | (u64{1} << 20));
                emit(0x5c10000000070000ULL | 252 | (u64{255} << 8) | (u64{252} << 20) | (u64{1} << 48));
                // Inactive lanes: 0, or ~0 for the minimum (OR with the inverted lane mask).
                emit(0x5c40000000070000ULL | 251 | (u64{251} << 8) | (u64{252} << 20) | (u64{7} << 48) |
                     (u64{minimum} << 40) | (u64{minimum} << 41));
                if (minimum) {
                    emit(0x5c20000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20) | (u64{7} << 39));
                } else if (maximum) {
                    emit(0x5c20000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20) | (u64{15} << 39));
                } else if (logical) {
                    emit(0x5c40000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20) |
                        (u64{logical} << 41) | (u64{7} << 48));
                } else {
                    emit(0x5c10000000070000ULL | target | (u64{target} << 8) | (u64{251} << 20));
                }
            }
            if (signed_order)
                emit(0x0400000000070000ULL | target | (u64{target} << 8) |
                     (u64{0x80000000U} << 20) | (u64{2} << 53));
            break;
        }
        case 0xf8c: { // ATOMS.POPC.INC.32 with no returned value: one increment per active lane.
            const u32 stride = insn.Bits(78, 2), offset = insn.Bits(40, 24);
            if (dest != 255 || (insn.words[1] & 0xff) || (offset & 3) ||
                (insn.words[2] & ~0xc000U) != 0x0d00003f)
                return fail(index, "native shared increment mode");
            u32 address = insn.Bits(24, 8);
            if (stride && address != 255) {
                emit(0x3848000000000000ULL | predicate | 251 | (u64{address} << 8) |
                    (u64{stride + 1} << 20));
                address = 251;
            }
            mov_imm(250, 1);
            emit(0xec00000000000000ULL | predicate | 255 | (u64{address} << 8) | (u64{250} << 20) |
                (u64{offset >> 2} << 30));
            break;
        }
        case 0x38c: { // ATOMS (ADD/MIN/MAX/INC/DEC/AND/OR/XOR/EXCH, U32 or S32), aligned byte offset.
            // The operation (bits 87-90) is numbered as in Maxwell ATOMS (52-55); bit 73 is S32.
            const u32 offset = insn.Bits(40, 24), operation = insn.Bits(87, 4);
            if ((insn.words[2] & ~0x07800200U) || operation > 8 || (offset & 3))
                return fail(index, "native shared atomic mode");
            emit(0xec00000000000000ULL | predicate | dest |
                (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(32, 8)} << 20) |
                (u64{insn.Bits(73, 1)} << 28) | (u64{offset >> 2} << 30) | (u64{operation} << 52));
            break;
        }
        case 0x3a8: { // ATOMG.E.STRONG.GPU U32 at a 64-bit register address plus an offset.
            const u32 address = insn.Bits(24, 8), operation = insn.Bits(87, 4);
            const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
            if ((insn.words[2] & ~0x07800000U) != 0x001ee100 || operation > 8 || (address & 1) || address >= 246 ||
                offset < -0x80000 || offset > 0x7ffff)
                return fail(index, "native global atomic register mode");
            emit(0xed00000000000000ULL | predicate | dest | (u64{address} << 8) |
                (u64{insn.Bits(32, 8)} << 20) | (u64{static_cast<u32>(offset) & 0xfffff} << 28) |
                (u64{1} << 48) | (u64{operation} << 52));
            break;
        }
        case 0x9a8: { // ATOMG.E.ADD.STRONG.GPU U32 through a uniform pointer plus an offset.
            const u32 address = uniform_register(insn.Bits(64, 6));
            const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
            if (insn.Bits(24, 8) != 255 || offset < -0x80000 || offset > 0x7ffff ||
                (insn.words[2] & ~0x3fU) != 0x081ee140 || address == ~0U || (address & 1))
                return fail(index, "native global atomic mode");
            emit(0xed00000000000000ULL | predicate | dest | (u64{address} << 8) |
                (u64{insn.Bits(32, 8)} << 20) | (u64{static_cast<u32>(offset) & 0xfffff} << 28) |
                (u64{1} << 48));
            break;
        }
        case 0x998:
        case 0x99c: {
            const bool store = opcode == 0x99c;
            const u32 data = insn.Bits(store ? 32 : 16, 8);
            const u32 coordinate = insn.Bits(24, 8), handle = insn.Bits(64, 8);
            const u32 channels = insn.Bits(72, 4), dimension = insn.Bits(61, 3);
            const u32 count = static_cast<u32>(std::popcount(channels));
            const u32 reserved = uses_uniform_bank ? high_uniform_base : 246U;
            if ((insn.words[1] & ~(store ? 0xe00000ffU : 0xe0000000U)) ||
                (insn.words[2] & ~0xfffU) != (store ? 0x00100000U : 0x001e0000U) ||
                (insn.words[3] & 0xc00001ffU) || dimension < 3 || dimension > 5 ||
                channels == 0 || handle >= reserved || data + count > reserved ||
                coordinate + (dimension >= 4 ? 3U : 2U) > reserved ||
                (store && (dest != 0 || (channels != 1 && channels != 3 && channels != 15))) ||
                (!store && (data % (count == 3 ? 4U : count))))
                return fail(index, "native bindless formatted surface mode");
            emit((store ? 0xeb20000000000000ULL : 0xeb00000000000000ULL) | predicate |
                data | (u64{coordinate} << 8) | (u64{channels} << 20) |
                (u64{dimension} << 33) | (u64{handle} << 39));
            break;
        }
        case 0xb97: { // SULD.P.1D/buffer/2D/2D_ARRAY/3D.IGN, direct descriptor, channel mask at 72-75.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 coordinate = insn.Bits(24, 8);
            const u32 channels = insn.Bits(72, 4);
            const u32 dimension = insn.Bits(61, 3); // Maxwell's dimension codes.
            const u32 reserved = uses_uniform_bank ? high_uniform_base : 246U;
            const u32 count = static_cast<u32>(std::popcount(channels));
            if ((insn.words[2] & ~0xf00U) != 0x001e0000 || channels == 0 || dimension == 2 || dimension > 5 ||
                binding >= 18 || offset > 0x1fff || insn.Bits(59, 2) || dest == 255 ||
                coordinate > (dimension >= 4 ? 252U : dimension == 3 ? 253U : 254U) ||
                (dimension < 2 && (coordinate >= reserved || dest + count > reserved ||
                                   dest % (count == 3 ? 4U : count))))
                return fail(index, "native formatted surface load mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native surface descriptor buffer conflict");
            result.required_texture_buffer = binding;
            emit(base(0xeb00000000000000ULL) | (u64{coordinate} << 8) |
                (u64{channels} << 20) | (u64{dimension} << 33) | (u64{offset} << 36) | (u64{1} << 51));
            break;
        }
        case 0xb9d: {
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 data = insn.Bits(32, 8), coordinate = insn.Bits(24, 8);
            const u32 reserved = uses_uniform_bank ? high_uniform_base : 246U;
            if (insn.words[2] != 0x0010a900 || insn.Bits(61, 3) != 3 ||
                insn.Bits(16, 8) != 0 || binding >= 18 || offset > 0x1fff ||
                insn.Bits(59, 2) || data >= reserved || coordinate + 1 >= reserved)
                return fail(index, "native byte-addressed surface store mode");
            if (!bind_texture_buffer(binding))
                return fail(index, "native byte-addressed surface descriptor buffer conflict");
            emit(0xeb38000000000000ULL | predicate | data | (u64{coordinate} << 8) |
                (u64{4} << 20) | (u64{1} << 23) | (u64{3} << 33) | (u64{offset} << 36));
            break;
        }
        case 0xb9b: { // SUST.P.1D/buffer/2D/3D.IGN, direct descriptor.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 data = insn.Bits(32, 8), coordinate = insn.Bits(24, 8);
            const u32 dimension = insn.Bits(61, 3);
            const u32 mask = insn.Bits(72, 4);
            const u32 mode = insn.words[2] & ~0xf00U;
            const u32 reserved = uses_uniform_bank ? high_uniform_base : 246U;
            const bool strong = mode == 0x0010a000; // Strong stores are fenced on both sides.
            if (!mask || (mode != 0x00100000 && !strong) || dimension == 2 || dimension > 5 ||
                binding >= 18 || offset > 0x1fff || insn.Bits(59, 2) || data > 252 ||
                coordinate > (dimension >= 4 ? 252U : dimension == 3 ? 253U : 254U) ||
                (dimension < 2 && (coordinate >= reserved ||
                                   data + static_cast<u32>(std::popcount(mask)) > reserved ||
                                   (mask != 1 && mask != 3 && mask != 15))))
                return fail(index, "native formatted surface store mode");
            // A second descriptor buffer: SUST.B with the handle, loaded from that buffer, in R250.
            const bool bindless = !bind_texture_buffer(binding);
            if (bindless) {
                emit(0x4c98000000070000ULL | 250 | (u64{15} << 39) | (u64{offset} << 20) | (u64{binding} << 34));
            }
            if (strong) emit(0xef98000000000000ULL | predicate | (u64{1} << 8)); // MEMBAR.GL
            emit(0xeb20000000000000ULL | predicate | data | (u64{coordinate} << 8) |
                (u64{mask} << 20) | (u64{dimension} << 33) |
                (bindless ? u64{250} << 39 : (u64{offset} << 36) | (u64{1} << 51)));
            if (strong) emit(0xef98000000000000ULL | predicate | (u64{1} << 8));
            break;
        }
        case 0x593: {
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 data = insn.Bits(32, 8), coordinate = insn.Bits(24, 8);
            const u32 dimension = insn.Bits(61, 3), operation = insn.Bits(87, 4);
            const u32 coordinates = dimension >= 4 ? 3U : dimension == 3 ? 2U : 1U;
            const u32 reserved = uses_uniform_bank ? high_uniform_base : 246U;
            if ((insn.words[2] & ~0x07800000U) != 0x001ee000 || operation > 8 ||
                dimension == 2 || dimension > 5 || binding >= 18 || offset > 0x1fff ||
                insn.Bits(59, 2) || coordinate + coordinates > reserved ||
                (data != 255 && data >= reserved) || (dest != 255 && dest >= reserved) ||
                (insn.words[3] & 0xc00001ffU))
                return fail(index, "native formatted surface atomic mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native surface descriptor buffer conflict");
            result.required_texture_buffer = binding;
            emit(base(0xea00000000000000ULL) | (u64{coordinate} << 8) | (u64{data} << 20) |
                 (u64{operation} << 29) | (u64{dimension} << 33) | (u64{offset} << 36));
            break;
        }
        case 0x59f: { // SURED.D.STRONG.GPU.IGN U32, direct descriptor, buffer or 2D.
            const u32 binding = insn.Bits(54, 5), offset = insn.Bits(40, 14);
            const u32 data = insn.Bits(32, 8), coordinate = insn.Bits(24, 8);
            const u32 dimension = insn.Bits(61, 3);
            if ((insn.words[2] & ~0x03800000U) != 0x0010e000 || (dimension != 1 && dimension != 3) || binding >= 18 ||
                offset > 0x1fff || insn.Bits(59, 2) || coordinate > 253)
                return fail(index, "native surface reduction mode");
            if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                return fail(index, "native surface descriptor buffer conflict");
            result.required_texture_buffer = binding;
            // Maxwell SURED: operand 0-7, coordinates 8-15, op 24, U32 zero at 20,
            // type 33 (1 = buffer, 3 = 2D), bound offset 36, bound 51.
            emit(0xeb40000000000000ULL | predicate | data | (u64{coordinate} << 8) |
                (u64{insn.Bits(87, 3)} << 24) | (u64{dimension} << 33) | (u64{offset} << 36) | (u64{1} << 51));
            break;
        }
        case 0x301: // BREV register.
            if (insn.words[2] || insn.words[1] > 255) return fail(index, "native bit reverse mode");
            emit(base(0x3828000000000000ULL) | (u64{insn.words[1]} << 8) |
                (u64{1} << 39) | (u64{1} << 40));
            break;
        case 0x219: case 0xa19: case 0xc19: { // U32 register (or cbuf, 0xa19, or uniform, 0xc19) shifts with a zero word outside the selected result.
            const u32 low = insn.Bits(24, 8), high = insn.Bits(64, 8);
            const bool constant = opcode == 0xa19;
            // A constant shift amount is loaded into R251 once the form is known.
            const u32 shift = opcode == 0xc19 ? uniform_register(insn.Bits(32, 8)) : constant ? 251U : insn.Bits(32, 8);
            const bool shift_operand = constant ? valid_cbuf() && !insn.Bits(62, 2) : insn.words[1] <= 255;
            const auto load_shift = [&] { if (constant) materialize(2, 0, 251, false, false); };
            const bool right = insn.Bits(76, 1) != 0, high_result = insn.Bits(80, 1) != 0;
            if (opcode == 0x219 && shift_operand &&
                (insn.words[2] & ~0xffU) == 0x00001200U) {
                right_shift_u64_low(static_cast<u32>(dest), low, high, shift);
                break;
            }
            if (opcode == 0x219 && low == 255 && shift_operand &&
                (insn.words[2] & ~0x8ffU) == 0x00011400U &&
                !(insn.words[3] & 0xc00001ffU)) {
                const bool wrap = insn.Bits(75, 1) != 0;
                if (!wrap)
                    emit(0x3820000000000000ULL | predicate | 250 | (u64{shift} << 8) |
                         (u64{31} << 20) | (u64{7} << 39));
                emit(base(0x5c28000000000000ULL) | (u64{high} << 8) |
                     (u64{wrap ? shift : 250U} << 20) | (u64{1} << 39) | (u64{1} << 48));
                break;
            }
            // SHF.R.U32 of the low word with a zero high word: a plain logical right shift.
            if (right && !high_result && high == 255 && insn.Bits(73, 2) == 3 && shift != ~0U &&
                shift_operand &&
                (insn.words[2] & ~(0x000118ffU | (opcode == 0xc19 ? 0x08000000U : 0U))) == 0x00000600U) {
                load_shift();
                emit(base(0x5c28000000000000ULL) | (u64{low} << 8) | (u64{shift} << 20));
                break;
            }
            if (!shift_operand || insn.Bits(73, 2) != 3 || shift == ~0U ||
                (insn.words[2] & ~(0x000118ffU | (opcode == 0xc19 ? 0x08000000U : 0U))) != 0x00000600U ||
                (right ? (!high_result || low != 255) : (high_result ? low != 255 : high != 255)))
                return fail(index, "native register shift mode");
            load_shift();
            emit(base(right ? 0x5cf8000000000000ULL : 0x5bf8000000000000ULL) |
                (u64{right ? high : 255U} << 8) | (u64{shift} << 20) |
                (u64{right ? 255U : (high_result ? high : low)} << 39) | (u64{insn.Bits(75, 1)} << 50));
            break;
        }
        case 0x300: case 0xd00: { // FLO, including a uniform source.
            if ((insn.words[2] & ~(0x600U | (opcode == 0xd00 ? 0x08000000U : 0U))) != 0x000e0000U)
                return fail(index, "native leading bit mode");
            const u32 source = operand(1, false);
            emit(base(0x5c30000000000000ULL) | (u64{source} << 20) |
                (u64{insn.Bits(63, 1)} << 40) | (u64{insn.Bits(74, 1)} << 41) |
                (u64{insn.Bits(73, 1)} << 48));
            break;
        }
        case 0x309: case 0xd09: { // POPC, including a uniform source.
            if (insn.words[2] != (opcode == 0xd09 ? 0x08000000U : 0U))
                return fail(index, "native population count mode");
            const u32 source = operand(1, false);
            emit(base(0x5c08000000000000ULL) | (u64{source} << 20) | (u64{insn.Bits(63, 1)} << 40));
            break;
        }
        case 0x918:
            if (insn.words[1] || insn.words[2]) return fail(index, "native NOP mode");
            emit(0x50b0000000000f00ULL | predicate);
            break;
        case 0x992: { // MEMBAR.SC: preserve workgroup/device visibility.
            const u32 scope = insn.Bits(76, 3);
            if (insn.words[1] || insn.words[2] != (scope << 12) ||
                (scope != 0 && scope != 2 && scope != 3 && scope != 5))
                return fail(index, "native memory barrier scope");
            const u32 legacy_scope = scope == 0 ? 0 : scope == 2 ? 1 : scope == 3 ? 2 : 3;
            emit(0xef98000000000000ULL | predicate | (u64{legacy_scope} << 8));
            break;
        }
        case 0x9ab: // ERRBAR: wait for prior memory operations on the host backend.
            if (insn.words[1] || insn.words[2]) return fail(index, "native error barrier mode");
            emit(0xef98000000000000ULL | predicate | (u64{1} << 8));
            break;
        case 0xb1d: // BAR.SYNC 0 (bit 80: .DEFER_BLOCKING), full workgroup.
            if (insn.words[1] || (insn.words[2] & ~0x10000U) || native_predicate != 7)
                return fail(index, "native workgroup barrier mode");
            emit(0xf0a8000000000000ULL | predicate | (u64{1} << 43) |
                (u64{1} << 44) | (u64{1} << 39) | (u64{7} << 39));
            break;
        case 0xfae: { // LDGSTS.E.BYPASS.128 [shared], [global.64], enable
            // Complete copies synchronously. Disabled sources zero-fill without touching global memory.
            const u32 shared = static_cast<u32>(dest), global = insn.Bits(24, 8);
            const u32 limit = uses_uniform_bank ? high_uniform_base : 246U;
            const bool promoted_copy = (insn.words[2] & ~0x0780003fU) == 0x08101d40U;
            const u32 descriptor = insn.Bits(64, 6);
            const bool descriptor_copy = (insn.words[2] & ~0x0380007fU) == 0x0c101c00U &&
                                         descriptor < 63 && !(descriptor & 1);
            if ((insn.words[1] & 0x0000f000U) ||
                (!promoted_copy && !descriptor_copy) ||
                shared >= limit || (global & 1) || global + 1 >= limit)
                return fail(index, "native global-to-shared copy mode");
            const s32 global_offset = static_cast<s32>(insn.Bits(32, 12) << 20) >> 20;
            const s32 shared_offset = static_cast<s32>(insn.Bits(48, 16) << 16) >> 12;
            for (u32 component = 0; component < 4; ++component) mov_imm(248 + component, 0);
            const u32 enable = insn.Bits(87, 3) |
                               (insn.Bits(promoted_copy ? 90 : 70, 1) << 3);
            const u32 skip = static_cast<u32>(result.instructions.size());
            emit(0);
            emit(0xeed0000000000000ULL | predicate | 248 | (u64{global} << 8) |
                 (u64{static_cast<u32>(global_offset) & 0xffffffU} << 20) |
                 (u64{1} << 45) | (u64{6} << 48));
            const u32 target = static_cast<u32>(result.instructions.size());
            const u32 distance = address_of(target) - address_of(skip) - 8;
            result.instructions[skip] = 0xe24000000000000fULL |
                                        (u64{enable ^ 8U} << 16) | (u64{distance} << 20);
            emit(0xef58000000000000ULL | predicate | 248 | (u64{shared} << 8) |
                 (u64{static_cast<u32>(shared_offset) & 0xffffffU} << 20) | (u64{6} << 48));
            break;
        }
        case 0x388: case 0x984: case 0x988: { // Shared memory load/store with scaled register offsets.
            const u32 width = insn.Bits(73, 3), stride = insn.Bits(78, 2);
            // LDS [Ra.Xn + URb + offset]: bit 91 adds a uniform register (bits 32-39). STS with one
            // (0x988, Hyrule Warriors: STS [UR9+0x130], R32) has it in bits 64-71, the data at 32.
            const bool uniform_base = (opcode == 0x984 || opcode == 0x988) && insn.Bits(91, 1);
            const u32 base_register = uniform_base ? uniform_register(insn.Bits(opcode == 0x988 ? 64 : 32, 8)) : 255U;
            const u32 ignored = uniform_base ? (opcode == 0x988 ? 0x080000ffU : 0x08000000U) : 0U;
            if (width > 6 || (insn.words[2] & ~ignored) != ((width << 9) | (stride << 14)) ||
                (opcode == 0x984 && !uniform_base && insn.Bits(32, 8)) || (opcode == 0x988 && !uniform_base) ||
                base_register == ~0U)
                return fail(index, "native shared memory addressing mode");
            u32 address = insn.Bits(24, 8);
            if (stride && address != 255) {
                emit(0x3848000000000000ULL | predicate | 250 |
                    (u64{address} << 8) | (u64{stride + 1} << 20));
                address = 250;
            }
            if (base_register != 255) {
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{address} << 8) | (u64{base_register} << 20));
                address = 250;
            }
            const bool store = opcode != 0x984;
            const u32 data = store ? insn.Bits(32, 8) : static_cast<u32>(dest);
            emit((store ? 0xef58000000000000ULL : 0xef48000000000000ULL) |
                predicate | data | (u64{address} << 8) | (u64{insn.Bits(40, 24)} << 20) |
                (u64{width} << 48));
            break;
        }
        case 0x23a: { // MOVM.16.MT88: transpose an 8x8 16-bit matrix held one row pair per lane.
            // Lane t holds row t/4, columns 2(t%4)..+1; the result is the transposed matrix in the
            // same layout, so each half comes from lane 8(t%4) + t/8 (+4) at half (t/4)%2.
            const u32 source = insn.Bits(24, 8);
            if (insn.words[1] || insn.words[2] || source >= 246 || dest >= 246)
                return fail(index, "native matrix move mode");
            emit(0xf0c8000000000000ULL | predicate | 246);                                        // laneid
            emit(0x0400000000000000ULL | predicate | 249 | (u64{246} << 8) | (u64{3} << 20));
            emit(0x3848000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{3} << 20));     // 8(t%4)
            emit(0x3828000000000000ULL | predicate | 247 | (u64{246} << 8) | (u64{3} << 20));     // t/8
            emit(0x5c10000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{247} << 20));
            emit(0xef10000000000000ULL | predicate | 247 | (u64{source} << 8) | (u64{249} << 20) |
                 (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x1c00000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{4} << 20));
            emit(0xef10000000000000ULL | predicate | 248 | (u64{source} << 8) | (u64{249} << 20) |
                 (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
            emit(0x3828000000000000ULL | predicate | 249 | (u64{246} << 8) | (u64{2} << 20));
            emit(0x0400000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{1} << 20));
            emit(0x3848000000000000ULL | predicate | 249 | (u64{249} << 8) | (u64{4} << 20));     // 0 or 16
            emit(0x5c28000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{249} << 20));
            emit(0x0400000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{0xffff} << 20));
            emit(0x5c28000000000000ULL | predicate | 248 | (u64{248} << 8) | (u64{249} << 20));
            emit(0x5c18000000000000ULL | predicate | dest | (u64{248} << 8) | (u64{247} << 20) | (u64{16} << 39));
            break;
        }
        case 0x23c: { // HMMA.1688/.16816.F16, software warp expansion (no tensor extension required).
            if (options.reject_matrix) return fail(index, "native matrix multiply (not emulated on this host)");
            const u32 a = insn.Bits(24, 8), b = insn.Bits(32, 8), c = insn.Bits(64, 8);
            const u32 mode = insn.words[2] & ~0xffU;
            const u32 depth = mode == 0 ? 8U : 16U;
            const u32 a_count = depth / 4, b_count = depth / 8;
            const u32 limit = uses_uniform_bank ? high_uniform_base : 246U;
            if (native_predicate != 7 || (mode != 0 && mode != 0x800) ||
                (insn.words[1] & ~0xffU) || (insn.words[3] & 0xc00001ffU) ||
                dest % 2 || dest + 2 > limit || a % a_count || a + a_count > limit ||
                b % b_count || b + b_count > limit || (c != 255 && (c % 2 || c + 2 > limit)))
                return fail(index, "native half matrix multiply mode");
            if (compact_matrix) {
                // A software collective, not an NV cooperative-matrix operation. The backend
                // shares a bounded loop helper instead of duplicating hundreds of instructions
                // at every HMMA site. Bit 48 tags software; bit 49 selects K=8 rather than K=16.
                emit(0x5d30000000000000ULL | predicate | dest | (u64{a} << 8) |
                     (u64{b} << 20) | (u64{c} << 39) | (u64{1} << 48) |
                     (u64{depth == 8} << 49));
                break;
            }
            // Four accumulators and five temporaries, all in R246..R254. In particular, R240..245
            // can hold live GPRs or emulated uniform registers and must not be borrowed.
            const auto shuffle = [&](u32 target, u32 source, u32 lane) {
                emit(0xef10000000000000ULL | predicate | target | (u64{source} << 8) |
                     (u64{lane} << 20) | (u64{1} << 29) | (u64{0x1f} << 34) | (u64{7} << 48));
            };
            const auto round_half = [&](u32 reg) {
                emit(0x5ca8000000000000ULL | predicate | reg | (u64{1} << 8) |
                     (u64{2} << 10) | (u64{reg} << 20));
            };
            for (u32 row = 0; row < 2; ++row)
                for (u32 col = 0; col < 2; ++col)
                    extract_half({c == 255 ? 255 : c + row, 0, false, false}, col, 246 + row * 2 + col);
            for (u32 pair = 0; pair < depth / 2; ++pair) {
                emit(0xf0c8000000000000ULL | predicate | 250); // LaneId
                emit(0x0400000000000000ULL | predicate | 251 | (u64{250} << 8) | (u64{3} << 20));
                emit(0x3848000000000000ULL | predicate | 251 | (u64{251} << 8) | (u64{3} << 20));
                emit(0x0400000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{0x1c} << 20));
                mov_imm(254, pair & 3);
                for (u32 reg : {250U, 251U})
                    emit(0x5c10000000000000ULL | predicate | reg | (u64{reg} << 8) | (u64{254} << 20));
                for (u32 row = 0; row < 2; ++row) {
                    for (u32 half = 0; half < 2; ++half) {
                        shuffle(252, a + (pair >= 4 ? 2 : 0) + row, 250);
                        extract_half({252, 0, false, false}, half, 252);
                        for (u32 col = 0; col < 2; ++col) {
                            mov_imm(254, col * 4);
                            emit(0x5c10000000000000ULL | predicate | 254 | (u64{251} << 8) | (u64{254} << 20));
                            shuffle(253, b + pair / 4, 254);
                            extract_half({253, 0, false, false}, half, 253);
                            const u32 acc = 246 + row * 2 + col;
                            emit(0x5980000000000000ULL | predicate | acc | (u64{252} << 8) |
                                 (u64{253} << 20) | (u64{acc} << 39));
                        }
                    }
                }
                if ((pair & 3) == 3) {
                    for (u32 reg = 246; reg < 250; ++reg) {
                        round_half(reg);
                        if (pair + 1 < depth / 2) extract_half({reg, 2, false, false}, 0, reg);
                    }
                }
            }
            for (u32 row = 0; row < 2; ++row)
                emit(0x5c18000000000000ULL | predicate | (dest + row) |
                     (u64{247 + row * 2} << 8) | (u64{246 + row * 2} << 20) | (u64{16} << 39));
            break;
        }
        case 0x83b: { // LDSM.16.M88.{1,2,4} (ldmatrix): 8x8 16-bit matrices from shared memory.
            // Lanes 8m..8m+7 give the row addresses of matrix m; lane t receives the 32 bits at
            // row t/4, byte (t%4)*4 of each matrix in Rd+m. Emulated with SHFL.IDX + LDS.32.
            const u32 count = 1U << insn.Bits(72, 2);
            const bool transpose = insn.Bits(78, 1) != 0;
            const bool uniform_offset = insn.Bits(91, 1);
            if (native_predicate != 7 || insn.Bits(72, 2) == 3 ||
                (insn.words[2] & ~0x08004300U) || (!uniform_offset && insn.Bits(32, 8)) ||
                dest % count || dest + count > (uses_uniform_bank ? high_uniform_base : 246U))
                return fail(index, "native matrix load mode");
            u32 address = insn.Bits(24, 8);
            const u64 offset = insn.Bits(40, 24);
            if (uniform_offset) {
                const u32 uniform = uniform_register(insn.Bits(32, 8));
                if (uniform == ~0U) return fail(index, "native matrix load uniform offset");
                emit(0x5c10000000000000ULL | predicate | 254 | (u64{address} << 8) | (u64{uniform} << 20));
                address = 254;
            } else if (address != 255 && address >= dest && address < dest + count) {
                emit(0x5c10000000000000ULL | predicate | 254 | (u64{address} << 8) | (u64{255} << 20));
                address = 254;
            }
            if (transpose) {
                // .MT88: lane t receives stored rows 2(t%4) and 2(t%4)+1 at column t/4 (16 bits each).
                emit(0xf0c8000000000000ULL | predicate | 246);                                    // laneid
                emit(0x0400000000000000ULL | predicate | 247 | (u64{246} << 8) | (u64{3} << 20));
                emit(0x3848000000000000ULL | predicate | 247 | (u64{247} << 8) | (u64{1} << 20)); // row 2(t%4)
                emit(0x3828000000000000ULL | predicate | 248 | (u64{246} << 8) | (u64{2} << 20));
                emit(0x3848000000000000ULL | predicate | 248 | (u64{248} << 8) | (u64{1} << 20)); // byte 2(t/4)
                for (u32 matrix = 0; matrix < count; ++matrix) {
                    for (u32 half = 0; half < 2; ++half) {
                        emit(0x1c00000000000000ULL | predicate | 249 | (u64{247} << 8) |
                             (u64{matrix * 8 + half} << 20));
                        emit(0xef10000000000000ULL | predicate | 250 | (u64{address} << 8) | (u64{249} << 20) |
                             (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                        emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{248} << 20));
                        emit(0xef48000000000000ULL | predicate | (251 + half) | (u64{250} << 8) |
                             (offset << 20) | (u64{2} << 48));                                  // LDS.U16
                    }
                    emit(0x5c18000000000000ULL | predicate | (dest + matrix) | (u64{252} << 8) |
                         (u64{251} << 20) | (u64{16} << 39));
                }
                break;
            }
            emit(0xf0c8000000000000ULL | predicate | 250);                                    // S2R laneid
            emit(0x3828000000000000ULL | predicate | 251 | (u64{250} << 8) | (u64{2} << 20)); // row
            emit(0x0400000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{3} << 20));
            emit(0x3848000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{2} << 20)); // column bytes
            for (u32 matrix = 0; matrix < count; ++matrix) {
                u32 lane = 251;
                if (matrix) {
                    emit(0x1c00000000000000ULL | predicate | 252 | (u64{251} << 8) | (u64{matrix * 8} << 20));
                    lane = 252;
                }
                emit(0xef10000000000000ULL | predicate | 253 | (u64{address} << 8) | (u64{lane} << 20) |
                     (u64{1} << 29) | (u64{31} << 34) | (u64{7} << 48));
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{250} << 20));
                emit(0xef48000000000000ULL | predicate | (dest + matrix) | (u64{253} << 8) |
                     (offset << 20) | (u64{4} << 48));
            }
            break;
        }
        case 0x919: case 0x9c3: { // S2R / S2UR: special registers the Maxwell S2R handles.
            // Same numbering as Maxwell: lane id (0), invocation id/info, Y direction, thread kill
            // (17-19, 29), W scale factors (30-31, Kirby Air Riders), thread and workgroup ids,
            // lane masks (56-60).
            const u32 special = insn.Bits(72, 8);
            if (special == 132 && !insn.words[1] && insn.words[2] == (special << 8)) {
                // SR_VARIABLE_RATE: without variable-rate shading every fragment is 1x1, its width
                // and height in bytes 1 and 2 (Nintendo Switch Sports Resort multiplies them).
                const u32 target = opcode == 0x9c3 ? uniform_register(static_cast<u32>(dest)) : static_cast<u32>(dest);
                if (target == ~0U) return fail(index, "native uniform special destination");
                mov_imm(target, 0x00010100);
                break;
            }
            if (insn.words[1] || insn.words[2] != (special << 8) ||
                (special != 0 && (special < 17 || special > 19) && (special < 29 || special > 35) &&
                 special != 37 && special != 38 && special != 39 &&
                 special != 56 && special != 57 && special != 58 && special != 59 && special != 60))
                return fail(index, "native special register");
            const u32 target = opcode == 0x9c3 ? uniform_register(static_cast<u32>(dest)) : static_cast<u32>(dest);
            if (target == ~0U) return fail(index, "native uniform special destination");
            emit(0xf0c8000000000000ULL | predicate | target | (u64{special} << 20));
            break;
        }
        case 0x28c: case 0x88c: {
            const u32 output = insn.Bits(81, 3);
            const u32 a = uniform_register(insn.Bits(24, 8)), b = opcode == 0x88c ? 252U : uniform_register(insn.Bits(32, 8));
            const u32 combine = insn.Bits(87, 3), combine_op = insn.Bits(74, 2);
            if (output > 6 || insn.Bits(84, 3) != 7 || insn.Bits(72, 1) || combine_op == 3 ||
                a == ~0U || b == ~0U)
                return fail(index, "native uniform comparison mode");
            const u32 temporary = 6 - output, combine_slot = combine == 7 ? 7U : 6 - combine;
            const u32 combine_mask = combine == 7 ? 0U : 1U << combine_slot;
            const bool guarded = native_predicate != 7;
            const u32 temporary_mask = 1U << temporary;
            const u32 load_mask = combine_mask | (guarded ? temporary_mask | 1U << guard_slot : 0U);
            const u32 mask = temporary_mask | load_mask;
            if (opcode == 0x88c)
                emit(0x0100000000000000ULL | (u64{7} << 16) | 252 | (u64{15} << 12) |
                     (u64{insn.words[1]} << 20));
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
            if (load_mask) emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{load_mask} << 20));
            emit(0x5b60000000000000ULL |
                (u64{guarded ? (native_predicate & 8) | guard_slot : 7U} << 16) | 7 | (u64{temporary} << 3) |
                (u64{a} << 8) | (u64{b} << 20) | (u64{combine_slot} << 39) |
                (u64{insn.Bits(90, 1)} << 42) | (u64{combine_op} << 45) | (u64{insn.Bits(73, 1)} << 48) |
                (u64{insn.Bits(76, 3)} << 49));
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{temporary_mask} << 20));
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
            break;
        }
        case 0x287: case 0x887: { // USEL, UP0/UP1; preserve ordinary predicates.
            const u32 condition = insn.Bits(87, 3);
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 a = uniform_register(insn.Bits(24, 8)), b = opcode == 0x887 ? 252U : uniform_register(insn.Bits(32, 8));
            if (condition > 6 || target == ~0U || a == ~0U || b == ~0U ||
                (insn.words[2] & ~0x07800000U) != 0x08000000U)
                return fail(index, "native uniform select mode");
            // The clamp's UISETP may sit a few instructions back, with only instructions between
            // that write no uniform state and do not branch (Nintendo Switch Sports' jump tables),
            // and no branch seen so far landing between them.
            u32 compare = index;
            for (u32 back = 1; back <= 8 && back <= index; ++back) {
                const u32 earlier = code[index - back].Bits(0, 12), datapath = earlier & 0x1ff;
                if (earlier == 0x88c) {
                    compare = index - back;
                    break;
                }
                if ((datapath >= 0x080 && datapath <= 0x0bf) || (datapath >= 0x1c2 && datapath <= 0x1c4) ||
                    (earlier >= 0x940 && earlier <= 0x95f))
                    break;
            }
            const bool target_between = std::any_of(relocations.begin(), relocations.end(),
                [&](const Relocation& relocation) { return relocation.target > compare && relocation.target <= index; });
            if (opcode == 0x887 && native_predicate == 7 && compare < index && !target_between &&
                condition == 0 && insn.words[1] <= 0x7ffff &&
                code[compare].Bits(12, 4) == 7 && code[compare].Bits(24, 8) == insn.Bits(24, 8) &&
                code[compare].words[1] == insn.words[1] && code[compare].words[2] == 0x0bf01070 &&
                insn.words[2] == 0x08000000) {
                // Preserve the unsigned clamp idiom for indirect-table analysis.
                emit(0x3820000000070000ULL | target | (u64{a} << 8) |
                     (u64{insn.words[1]} << 20) | (u64{7} << 39));
                break;
            }
            if (opcode == 0x887)
                emit(0x0100000000000000ULL | (u64{7} << 16) | 252 | (u64{15} << 12) |
                     (u64{insn.words[1]} << 20));
            const u32 temporary = 6 - condition;
            const u32 mask = (1U << temporary) | (native_predicate != 7 ? 1U << guard_slot : 0U);
            emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
            emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{mask} << 20));
            emit(0x5ca0000000000000ULL | (u64{(native_predicate & 8) | (native_predicate != 7 ? guard_slot : 7U)} << 16) |
                target | (u64{a} << 8) | (u64{b} << 20) |
                (u64{temporary} << 39) | (u64{insn.Bits(90, 1)} << 42));
            emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
            break;
        }
        case 0x381: case 0x981: { // LDG.E of 8 to 128 bits, register or uniform 64-bit address.
            // Sizes as Maxwell's: U8, S8, U16, S16, 32, 64, 128.
            const u32 size = insn.Bits(73, 3);
            const bool scoped_descriptor = opcode == 0x981 &&
                insn.words[2] == 0x0c1ebd00;
            u32 address = insn.Bits(24, 8);
            if (size > 6 || (opcode == 0x381 && insn.Bits(32, 8)) ||
                (size == 5 && (dest & 1)) || (size == 6 && (dest & 3)))
                return fail(index, "native global load mode");
            if (opcode == 0x381) {
                // Bit 77 (strong/scoped ordering) needs nothing extra from a Maxwell LDG.
                if ((insn.words[2] & ~0x2e00U) != 0x001e8100 || (address & 1) || address >= 246)
                    return fail(index, "native global load mode");
            } else {
                // The uniform base sits where a register source would (bits 32-37).
                const u32 uniform_address =
                    insn.Bits(38, 2) ? ~0U : uniform_register(insn.Bits(32, 6));
                // desc[URn][Ra.64]: the uniform pair is a memory (cache policy) descriptor and
                // Ra the whole address.
                const bool descriptor_address = scoped_descriptor ||
                    (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c1e1100;
                if ((!descriptor_address &&
                     (insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U | 0x2000U)) != 0x081e8100) ||
                    uniform_address == ~0U || (uniform_address & 1))
                    return fail(index, "native global load uniform addressing mode");
                if (descriptor_address) {
                    if ((address & 1) || address >= 246)
                        return fail(index, "native global load register address pair");
                } else if (address == 255) {
                    address = uniform_address;
                } else {
                    if (insn.Bits(90, 1) && ((address & 1) || address >= 246))
                        return fail(index, "native global load register address pair");
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{uniform_address} << 8) |
                         (u64{address} << 20) | (u64{1} << 47));
                    emit(0x5c10000000000000ULL | predicate | 251 | (u64{uniform_address + 1} << 8) |
                         (u64{insn.Bits(90, 1) ? address + 1 : 255} << 20) | (u64{1} << 43));
                    address = 250;
                }
            }
            emit(base(0xeed0000000000000ULL) | (u64{address} << 8) |
                 (u64{insn.Bits(40, 24)} << 20) | (u64{1} << 45) |
                 (u64{scoped_descriptor} << 46) | (u64{size} << 48));
            break;
        }
        case 0x386: case 0x986: { // Global store of 32/64/128 bits, 64-bit address.
            u32 data = insn.Bits(32, 8);
            const u32 size = insn.Bits(73, 3);
            if (size > 6 ||
                (data != 255 && ((size == 5 && (data & 1)) || (size == 6 && (data & 3)) ||
                                 data + (size < 4 ? 1U : 1U << (size - 4)) > 246)))
                return fail(index, "native global store size");
            if (data == 255 && size > 4) {
                // Zeros wider than one register: store from zeroed scratch registers (aligned to the
                // access: R248-R251 for 128 bits, R250-R251 for 64).
                data = size == 6 ? 248 : 250;
                for (u32 i = 0; i < (1U << (size - 4)); ++i) mov_imm(data + i, 0);
            }
            u32 address = insn.Bits(24, 8);
            bool extended = insn.Bits(72, 1) != 0;
            if (opcode == 0x986) {
                const u32 uniform_address = uniform_register(insn.Bits(64, 6));
                // desc[URn][Ra.64]: the uniform pair is a memory (cache policy) descriptor and
                // Ra the whole address.
                const bool descriptor_address = (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c101100;
                if ((!descriptor_address && (insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U)) != 0x0810a100) ||
                    uniform_address == ~0U || (uniform_address & 1) || !extended)
                    return fail(index, "native global store uniform addressing mode");
                if (descriptor_address) {
                    if ((address & 1) || address >= 246)
                        return fail(index, "native global store register address pair");
                } else if (address == 255) address = uniform_address;
                else {
                    if (insn.Bits(90, 1) && ((address & 1) || address >= 254))
                        return fail(index, "native global store register address pair");
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{uniform_address} << 8) |
                        (u64{address} << 20) | (u64{1} << 47));
                    emit(0x5c10000000000000ULL | predicate | 251 | (u64{uniform_address + 1} << 8) |
                        (u64{insn.Bits(90, 1) ? address + 1 : 255} << 20) | (u64{1} << 43));
                    address = 250;
                }
            } else if ((insn.words[2] & ~0xe00U) != 0x0010a100) {
                return fail(index, "native global store register addressing mode");
            }
            emit(0xeed8000000000000ULL | predicate | data |
                (u64{address} << 8) | (u64{insn.Bits(40, 24)} << 20) |
                (u64{extended} << 45) | (u64{size} << 48));
            break;
        }
        case 0x95b: // Fragment discard; execution predicate is preserved.
            if (insn.words[1] || insn.words[2] != 0x03800000)
                return fail(index, "native fragment discard mode");
            emit(0xe33000000000000fULL | predicate);
            break;
        case 0x311: { // F2I.FTZ.U64.TRUNC, register source, destination pair
            if (insn.words[2] != 0x0021d800 || (insn.words[1] & ~0xffU) || insn.Bits(24, 8) ||
                (dest & 1) || dest >= 246)
                return fail(index, "native unsigned 64-bit float conversion mode");
            emit(base(0x5cb0000000000000ULL) | (u64{3} << 8) | (u64{2} << 10) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{3} << 39) | (u64{1} << 44));
            break;
        }
        case 0x312: { // I2F.F32.(U|S)64, register pair source; sign (74) and rounding (78-79) as Maxwell's
            if ((insn.words[2] & ~0xc400U) != 0x00301000 || (insn.words[1] & ~0xffU) || insn.Bits(24, 8) ||
                (insn.Bits(32, 8) & 1) || insn.Bits(32, 8) >= 254)
                return fail(index, "native 64-bit integer-to-float mode");
            emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{3} << 10) | (u64{insn.Bits(74, 1)} << 13) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(78, 2)} << 39));
            break;
        }

        case 0x243:
            // F2IP.U8.F32 Rd, Ra, Rb, RZ: Rb's byte in bits 0-7 and Ra's in bits 8-15. The default
            // rounding (bits 78-79 clear) packs unorm values already scaled by 255.
            if ((insn.words[1] & ~0xffU) ||
                (insn.words[2] != 0x0000c4ffU && insn.words[2] != 0x000004ffU))
                return fail(index, "native packed float-to-byte mode");
            if (insn.Bits(24, 8) != 255 || insn.words[2] == 0x000004ffU) {
                const auto to_byte = [&](u32 source, u32 target) {
                    emit(0x5800000000000000ULL | predicate | target | (u64{source} << 8) |
                         (u64{source} << 20) | (u64{7} << 39) | (u64{7} << 48));
                    emit(0x5c40000000000000ULL | predicate | target | (u64{source} << 8) |
                         (u64{target} << 20) | (u64{7} << 48));
                    mov_imm(252, 0);
                    emit(0x5c60000000000000ULL | predicate | target | (u64{target} << 8) |
                         (u64{252} << 20) | (u64{15} << 39));
                    mov_imm(252, 0x437f0000U);
                    emit(0x5c60000000000000ULL | predicate | target | (u64{target} << 8) |
                         (u64{252} << 20) | (u64{7} << 39));
                    emit(0x5cb0000000000000ULL | predicate | target | (u64{2} << 8) | (u64{2} << 10) |
                         (u64{target} << 20) | (u64{insn.Bits(78, 2)} << 39));
                };
                to_byte(insn.Bits(32, 8), 250);
                if (insn.Bits(24, 8) == 255) {
                    emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                    break;
                }
                to_byte(insn.Bits(24, 8), 251);
                emit(base(0x5c18000000000000ULL) | (u64{251} << 8) | (u64{250} << 20) | (u64{8} << 39));
                break;
            }
            insn.words[0] &= 0x00ffffffU;
            insn.words[2] = 0x0021e000U;
            opcode = 0x305;
            [[fallthrough]];
        case 0x305: case 0xb05: case 0xd05: { // F2I NTZ, register/constant-buffer/uniform source
            const u32 mode = insn.words[2] & ~(opcode == 0xd05 ? 0x08000000U : 0U);
            const bool signed_half = opcode == 0x305 && (mode == 0x00103100 || mode == 0x00107100);
            const bool half = signed_half || (opcode == 0x305 && (mode & ~0x100U) == 0x0010f000);
            const bool small = (mode & ~0xc000U) == 0x00212800;
            const bool short_trunc = opcode == 0x305 && (mode & ~0x100U) == 0x0021e800;
            const bool absolute = opcode == 0x305 && !half && insn.Bits(62, 1);
            const bool negate = opcode == 0x305 && !half && insn.Bits(63, 1);
            const bool byte = (mode & ~0x10100U) == 0x0020e000;
            const bool word = (mode & ~0x1c100U) == 0x00203000;
            const bool extended_word =
                word && (insn.Bits(78, 2) == 2 || (insn.Bits(78, 2) == 3 && !insn.Bits(80, 1)));
            if ((!half && !small && !short_trunc && !byte && !word) ||
                (signed_half && (insn.Bits(24, 8) || (insn.words[3] & 0xc00001ffU))) ||
                ((byte || small || short_trunc || extended_word) &&
                 (insn.Bits(24, 8) || (insn.words[3] & 0xc00001ffU) ||
                  (opcode == 0xb05 &&
                   (insn.Bits(63, 1) || (!small && insn.Bits(62, 1)) ||
                    (small && insn.Bits(59, 3)))) ||
                  insn.Bits(91, 1) != static_cast<u32>(opcode == 0xd05))) ||
                (opcode != 0xb05 && (insn.words[1] & ~(half ? 0x100000ffU :
                    0xffU | (absolute ? 0x40000000U : 0U) | (negate ? 0x80000000U : 0U)))) ||
                (opcode == 0xd05 && uniform_register(insn.Bits(32, 8)) == ~0U)) {
                return fail(index, "native float-to-integer conversion mode");
            }
            const bool is_signed = insn.Bits(72, 1);
            u32 source = opcode == 0xb05 ? materialize(2, 0, 252, small && insn.Bits(62, 1), false)
                       : opcode == 0xd05 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            if (half) {
                extract_half({source, insn.Bits(60, 1) ? 3U : 2U, false, false}, 0, 252);
                source = 252;
            }
            if (absolute || negate) source = materialize(0, source, 249, absolute, negate);
            if (signed_half) {
                mov_imm(253, 0x47800000);
                emit(0x5800000000000000ULL | predicate | 250 | (u64{252} << 8) |
                     (u64{253} << 20) | (u64{7} << 39) | (u64{1} << 48) | (u64{1} << 54));
                emit(0x5800000000000000ULL | predicate | 251 | (u64{252} << 8) |
                     (u64{253} << 20) | (u64{7} << 39) | (u64{4} << 48));
                mov_imm(253, 0xc7800000);
                emit(0x5800000000000000ULL | predicate | 253 | (u64{252} << 8) |
                     (u64{253} << 20) | (u64{7} << 39) | (u64{1} << 48));
                emit(0x5c40000000000000ULL | predicate | 252 | (u64{252} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                emit(base(0x5cb0000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) |
                     (u64{1} << 12) | (u64{252} << 20) | (u64{insn.Bits(78, 2)} << 39));
                mov_imm(250, 0x7fffffff);
                emit(0x5c40000000000000ULL | predicate | 251 | (u64{251} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                mov_imm(250, 0x80000000);
                emit(0x5c40000000000000ULL | predicate | 253 | (u64{253} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{251} << 20) |
                     (u64{1} << 41) | (u64{7} << 48));
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{253} << 20) |
                     (u64{1} << 41) | (u64{7} << 48));
                break;
            }
            if (byte) {
                emit(0x5800000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{source} << 20) | (u64{7} << 39) | (u64{7} << 48));
                emit(0x5c40000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                mov_imm(252, is_signed ? 0xc3000000U : 0U);
                emit(0x5c60000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{15} << 39));
                mov_imm(252, is_signed ? 0x42fe0000U : 0x437f0000U);
                emit(0x5c60000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39));
                source = 250;
            } else if (!is_signed) {
                // Clear NaNs without consuming a guest predicate. Clamp the
                // conversion below 2^32, retaining an overflow mask for saturation.
                emit(0x5800000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{source} << 20) | (u64{7} << 39) | (u64{7} << 48));
                emit(0x5c40000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{250} << 20) | (u64{7} << 48));
                if (!small && !short_trunc) {
                mov_imm(252, 0x4f800000);
                emit(0x5800000000000000ULL | predicate | 251 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39) | (u64{6} << 48));
                mov_imm(252, 0x4f7fffff);
                emit(0x5c60000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{252} << 20) | (u64{7} << 39));
                }
                source = 250;
            }
            emit(base(0x5cb0000000000000ULL) | (u64{small || short_trunc ? 1U : 2U} << 8) | (u64{2} << 10) |
                 (u64{is_signed} << 12) | (u64{source} << 20) |
                 (u64{insn.Bits(78, 2)} << 39) | (u64{insn.Bits(80, 1)} << 44));
            if (!is_signed && !small && !short_trunc && !byte) {
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{251} << 20) |
                     (u64{1} << 41) | (u64{7} << 48));
            }
            break;
        }
        case 0xb6f: case 0x370: { // TXQ texture dimensions (0x370: TXQ.B).
            // TXQ.B takes the handle in Ra and the level in Ra+1, as Maxwell's. Up to two outputs
            // go to Rd, Rd+1 directly; more, or a second destination (bits 64-71) for the third and
            // fourth, come out packed in scratch and are copied as for TEX.
            const bool bindless = opcode == 0x370;
            const u32 binding = insn.Bits(54, 5); // descriptor buffer, as for TLD
            const u32 mask = insn.Bits(72, 4), components = static_cast<u32>(std::popcount(mask));
            const u32 second_dest = insn.Bits(64, 8);
            const bool direct = second_dest == 255;
            if (insn.Bits(59, 5) != (bindless ? 1U : 0U) || mask == 0 || (direct && components > 2) ||
                insn.Bits(76, 14) || insn.Bits(91, 1) || (!bindless && binding >= 18) ||
                (bindless && (insn.Bits(32, 27) || insn.Bits(24, 8) >= 245)) ||
                (!direct && (second_dest >= 245 || dest >= 245)) ||
                insn.Bits(40, 14) > 0x1fff || (direct && components == 2 && (dest & 1)) || dest >= 254)
                return fail(index, "native texture dimension query mode");
            if (!bindless) {
                if (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding)
                    return fail(index, "native texture descriptor buffer conflict");
                result.required_texture_buffer = binding;
            }
            emit((bindless ? 0xdf50000000000000ULL : 0xdf48000000000000ULL) | predicate | (direct ? dest : 250) |
                 (u64{insn.Bits(24, 8)} << 8) | (u64{1} << 22) | (u64{mask} << 31) |
                 (u64{bindless ? 0U : insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49));
            for (u32 i = 0; !direct && i < components; ++i) {
                const u32 target = i < 2 ? (dest == 255 ? 255U : static_cast<u32>(dest) + i) : second_dest + (i - 2);
                if (target != 255)
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb66: { // TLD.SCR, direct texel fetch at level zero (.LZ) or a given level (.LL)
            u32 mask = insn.Bits(72, 4);
            const bool level = insn.Bits(87, 3) == 3; // .LL: the level follows the coordinates
            // Bit 76: a texel offset after the coordinates and level. 2D only, as Maxwell TLDS
            // 2D.LZ.AOFFI (x, y in Ra, Ra+1, offsets in Rb) or 2D.LL.AOFFI (level, offsets in Rb, Rb+1).
            const bool offsets = insn.Bits(76, 1) != 0;
            // Bit 79 (.F16): half2 results, (c0, c1) in the first register and (c2, c3) in the
            // second, as Maxwell TLDS.F16 (bit 59 clear) stores them.
            const bool half = insn.Bits(79, 1) != 0;
            const u32 dimension = insn.Bits(61, 3);
            const bool array = dimension == 5;
            const bool volume = dimension == 2;
            u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            u32 second_dest = insn.Bits(64, 8);
            // TLDS has no single-register G+B or R+B swizzle (Hyrule Warriors fetches G+B): take
            // alpha too, as the three-channel form, with it going to scratch R246.
            if (second_dest == 255 && (mask == 5 || mask == 6)) {
                mask |= 8;
                second_dest = 246;
            }
            const bool split = second_dest != 255;
            constexpr std::array<u32, 8> rg_masks{1, 2, 4, 8, 3, 9, 10, 12};
            constexpr std::array<u32, 5> rgba_masks{7, 11, 13, 14, 15};
            u32 swizzle = ~0U;
            if (split) {
                for (u32 i = 0; i < rgba_masks.size(); ++i) if (rgba_masks[i] == mask) swizzle = i;
            } else {
                for (u32 i = 0; i < rg_masks.size(); ++i) if (rg_masks[i] == mask) swizzle = i;
            }
            if (!insn.Bits(60, 1) || (dimension != 0 && dimension != 1 && !array && !volume) ||
                swizzle == ~0U || (insn.Bits(76, 5) & ~0x9U) || (half && array && level) ||
                (offsets && (dimension != 1 || (source_a & 1) || source_a >= 254 || source_b == 255 ||
                             (level && ((source_b & 1) || source_b >= 254)))) ||
                insn.Bits(81, 3) != 7 || (insn.Bits(87, 3) != 1 && !level) || (level && volume) ||
                (level && dimension == 1 && ((source_a & 1) || source_a >= 254)) ||
                (level && array && ((source_b & 1) || source_b >= 254)) ||
                insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) >= 18 ||
                dest == 255 || (!half && (mask & (mask - 1)) && ((dest & 1) || dest >= 254)) ||
                (!half && split && mask == 15 && ((second_dest & 1) || second_dest >= 254)) ||
                (array && ((source_a & 1) || source_a >= 249)) ||
                (volume && ((source_a & 1) || source_a >= 254))) {
                return fail(index, "native texel fetch mode or descriptor binding");
            }
            const u32 binding = insn.Bits(54, 5);
            if (!bind_texture_buffer(binding)) {
                if (level || offsets || half)
                    return fail(index, "native texture descriptor buffer conflict");
                // A second descriptor buffer: TLD.B (LZ) with the coordinates in R250.. (array
                // index or Z after Ra) and the handle, loaded from that buffer, in R253.
                const auto copy = [&](u32 target, u32 source) {
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
                };
                copy(250, source_a);
                if (array || volume) copy(251, source_a + 1);
                if (dimension != 0) copy(array || volume ? 252 : 251, source_b);
                emit(0x4c98000000070000ULL | 253 | (u64{15} << 39) | (u64{insn.Bits(40, 14)} << 20) |
                     (u64{binding} << 34));
                emit(0xdd00000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{253} << 20) |
                     (u64{array ? 3U : volume ? 4U : dimension == 0 ? 0U : 2U} << 28) | (u64{mask} << 31) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
                const u32 components = static_cast<u32>(std::popcount(mask));
                for (u32 i = 0; i < components; ++i)
                    copy(i < 2 || !split ? static_cast<u32>(dest) + i : second_dest + (i - 2), 250 + i);
                break;
            }
            if (array && level) {
                // TLDS has no array form with a level: TLD takes (layer, x, y) from Ra and the level
                // from Rb (layer, x in Ra, Ra+1; y, level in Rb, Rb+1 here). Results come out packed,
                // then go where the Ampere destinations (Rd, Rd+1, then the second pair) take them.
                for (u32 i = 0; i < 2; ++i) {
                    emit(0x5c98000000000000ULL | predicate | (250 + i) | (u64{15} << 39) | (u64{source_a + i} << 20));
                    emit(0x5c98000000000000ULL | predicate | (252 + i) | (u64{15} << 39) | (u64{source_b + i} << 20));
                }
                emit(0xdc00000000000000ULL | predicate | 246 | (u64{250} << 8) | (u64{253} << 20) |
                     (u64{3} << 28) | (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51) | (u64{1} << 55));
                const u32 components = static_cast<u32>(std::popcount(mask));
                for (u32 i = 0; i < components; ++i) {
                    const u32 target = !split || i < 2 ? static_cast<u32>(dest) + i : second_dest + (i - 2);
                    if (target != 255 && (split || i < 2))
                        emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{246 + i} << 20));
                }
                break;
            }
            if (array) {
                emit(0x5c98000000000000ULL | predicate | 250 | (u64{15} << 39) | (u64{source_a + 1} << 20));
                emit(0x5c98000000000000ULL | predicate | 251 | (u64{15} << 39) | (u64{source_b} << 20));
                source_b = 250;
            }
            emit((0xda00000000000000ULL & ~(u64{half} << 59)) | predicate | dest | (u64{source_a} << 8) |
                 (u64{source_b} << 20) | (u64{second_dest} << 28) |
                 (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                 (u64{swizzle} << 50) |
                 (u64{offsets ? (level ? 12U : 4U)
                      : array ? 8U : volume ? 7U : dimension == 0 ? (level ? 1U : 0U) : (level ? 5U : 2U)} << 53));
            break;
        }
        case 0x367: { // TLD.B at level zero (.LZ), the handle in a register.
            // The values are the coordinates, then the handle, split over Ra and Rb as for TEX.B
            // (ARRAY_2D: layer, x | y, handle); Maxwell TLD.B takes the handle first in Rb.
            constexpr std::array<u32, 8> coordinate_count{1, 2, 3, 0, 0, 3, 0, 0};
            constexpr std::array<u32, 8> maxwell_type{0, 2, 4, 0, 0, 3, 0, 0};
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 dimension = insn.Bits(61, 3), coordinates = coordinate_count[dimension];
            const u32 count = coordinates + 1, in_a = (count + 1) / 2;
            if (coordinates == 0 || !insn.Bits(59, 1) || !insn.Bits(60, 1) || insn.Bits(40, 19) ||
                insn.Bits(76, 5) || insn.Bits(81, 3) != 7 || insn.Bits(87, 3) != 1 || insn.Bits(91, 1) ||
                mask == 0 || source_a == 255 || source_a + in_a > 246 || source_b == 255 ||
                source_b + (count - in_a) > 246)
                return fail(index, "native bindless texel fetch mode");
            const auto value = [&](u32 i) { return i < in_a ? source_a + i : source_b + (i - in_a); };
            const auto copy = [&](u32 target, u32 source) {
                emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
            };
            for (u32 i = 0; i < coordinates; ++i) copy(250 + i, value(i));
            copy(246, value(coordinates));
            emit(0xdd00000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{246} << 20) |
                 (u64{maxwell_type[dimension]} << 28) | (u64{mask} << 31) | (u64{insn.Bits(90, 1)} << 49) |
                 (u64{7} << 51));
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = second_dest != 255;
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255) copy(target, 250 + i);
            }
            break;
        }
        case 0xf66: { // TLD.SCR.LZ with the handle in a uniform register (bit 91).
            // Maxwell TLD.B with the coordinates in R250.. (as 0xb66's second-buffer form: x | y,
            // layer, x | y or x, y | z over Ra and Rb) and the handle in Rb.
            const u32 handle = uniform_register(insn.Bits(40, 6));
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 dimension = insn.Bits(61, 3);
            const bool array = dimension == 5, volume = dimension == 2;
            const bool split = second_dest != 255;
            const u32 components = static_cast<u32>(std::popcount(mask));
            if (handle == ~0U || handle >= 246 || insn.Bits(46, 13) || !insn.Bits(91, 1) ||
                !insn.Bits(60, 1) || insn.Bits(59, 1) || insn.Bits(76, 5) || insn.Bits(81, 3) != 7 ||
                insn.Bits(87, 3) != 1 || (dimension != 0 && dimension != 1 && !array && !volume) ||
                mask == 0 || ((array || volume) && source_a + 1 >= high_uniform_base) ||
                (dest != 255 && dest + (split ? (std::min)(components, 2U) : components) > high_uniform_base) ||
                (split && components > 2 && second_dest + components - 2 > high_uniform_base))
                return fail(index, "native bindless texel fetch mode");
            const auto copy = [&](u32 target, u32 source) {
                emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
            };
            copy(250, source_a);
            if (array || volume) copy(251, source_a + 1);
            if (dimension != 0) copy(array || volume ? 252 : 251, source_b);
            copy(246, handle);
            emit(0xdd00000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{246} << 20) |
                 (u64{array ? 3U : volume ? 4U : dimension == 0 ? 0U : 2U} << 28) | (u64{mask} << 31) |
                 (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
            for (u32 i = 0; i < components; ++i) {
                const u32 target = !split || i < 2 ? (dest == 255 ? 255U : static_cast<u32>(dest) + i)
                                                   : second_dest + (i - 2);
                if (target != 255) copy(target, 250 + i);
            }
            break;
        }
        case 0x91a: { // DEPBAR.LE SBn, count, {list}: scoreboard wait, represented by IR data dependencies.
            // Any scoreboard (bits 44-46), count (38-43) and wait list (32-37) is the same no-op here.
            // DEPBAR.LE SB0, n (scoreboard 0x8000) waits on LDGSTS copy groups. The copies below
            // complete synchronously, so only make their shared memory stores visible.
            const u32 scoreboard = insn.words[1] & ~0xfffU;
            if (insn.words[0] != 0x791a || (insn.words[1] & ~0x7fffU) != 0x8000 || insn.words[2]) {
                return fail(index, "native dependency barrier mode");
            }
            if (uses_async_copy && scoreboard == 0x8000) emit(0xef98000000000000ULL | predicate); // MEMBAR.CTA
            break;
        }
        case 0x9af: { // LDGDEPBAR: commit a copy group (the copies are already complete).
            if (insn.words[0] != 0x79af || insn.words[1] || insn.words[2])
                return fail(index, "native copy-group commit mode");
            emit(0xef98000000000000ULL | predicate);
            break;
        }
        case 0xb69: case 0xf69: case 0x36a: { // TMML (LOD query), 1D/2D/3D/CUBE. Results are 8.8 fixed point, as on Maxwell.
            // 0xf69: the handle in a uniform register (bit 91); 0x36a, TMML.B: the handle in Rb and
            // every coordinate in Ra. Both become Maxwell TMML.B with the handle in Rb.
            const bool bindless = opcode != 0xb69;
            const u32 handle = opcode == 0xf69 ? uniform_register(insn.Bits(40, 6))
                             : opcode == 0x36a ? insn.Bits(32, 8) : 255;
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 binding = insn.Bits(54, 5), dimension = insn.Bits(61, 3);
            // Coordinates: all from Ra when Rb is RZ, else split as for TEX (2D: Ra, Rb).
            constexpr std::array<u32, 4> coordinate_count{1, 2, 3, 3};
            const u32 coordinates = dimension < 4 ? coordinate_count[dimension] : 0;
            const u32 in_a = source_b == 255 || opcode == 0x36a ? coordinates : (coordinates + 1) / 2;
            if (coordinates == 0 || insn.Bits(76, 4) || insn.Bits(80, 10) || mask == 0 ||
                insn.Bits(59, 1) != (opcode == 0x36a ? 1U : 0U) || source_a + in_a > 246 ||
                (in_a < coordinates && source_b + (coordinates - in_a) > 246) ||
                (opcode == 0xf69 && (handle >= 246 || insn.Bits(46, 13) || !insn.Bits(91, 1))) ||
                (opcode == 0x36a && (handle >= 246 || insn.Bits(40, 19) || insn.Bits(91, 1))) ||
                (!bindless && (insn.Bits(40, 14) > 0x1fff || binding >= 18 ||
                               (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding))))
                return fail(index, "native texture LOD query mode");
            if (!bindless) result.required_texture_buffer = binding;
            for (u32 i = 0; i < coordinates; ++i)
                emit(0x5c98000000000000ULL | predicate | (250 + i) | (u64{15} << 39) |
                     (u64{i < in_a ? source_a + i : source_b + (i - in_a)} << 20));
            if (bindless) emit(0x5c98000000000000ULL | predicate | 246 | (u64{15} << 39) | (u64{handle} << 20));
            emit((bindless ? 0xdf60000000000000ULL : 0xdf58000000000000ULL) | predicate | 250 | (u64{250} << 8) |
                 (u64{bindless ? 246U : 255U} << 20) | (u64{dimension * 2} << 28) | (u64{mask} << 31) |
                 (u64{bindless ? 0U : insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49));
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = second_dest != 255;
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255)
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb63: case 0xf63: { // TLD4 (gather); component at bits 87-88.
            // 0xf63: the handle in a uniform register (bit 91), Maxwell TLD4.B with the handle in Rb.
            const bool bindless = opcode == 0xf63;
            const u32 handle = bindless ? uniform_register(insn.Bits(40, 6)) : 255;
            const u32 mask = insn.Bits(72, 4), second_dest = insn.Bits(64, 8);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            const u32 binding = insn.Bits(54, 5);
            // Bit 76: a texel offset after the coordinates (x, y in Ra, Ra+1, offsets in Rb, as
            // for TEX); Maxwell TLD4 takes it as AOFFI (bits 54-55) from Rb.
            const bool offsets = insn.Bits(76, 1) != 0;
            // Bits 76-77 = 2: per-tap offsets (PTP), x and y in Ra, Ra+1 and the packed offsets in
            // Rb, Rb+1. The host gather needs them as constants, so they must be proven immediates.
            const bool per_tap = insn.Bits(76, 2) == 2;
            const bool depth = insn.Bits(78, 1) != 0;
            const bool half = insn.Bits(79, 1) != 0;
            const u32 register_limit = uses_uniform_bank ? high_uniform_base : 246U;
            const bool array = insn.Bits(61, 3) == 5;
            if (bindless && array) {
                const u32 count = static_cast<u32>(std::popcount(mask));
                const u32 first_count = second_dest == 255 ? count : std::min(count, 2U);
                const u32 second_count = count - first_count;
                if (!insn.Bits(60, 1) ||
                    (insn.words[2] & ~0x05800fffU) != 0x081e0000U ||
                    source_a + 2 > register_limit || source_b >= register_limit ||
                    (dest != 255 && dest + first_count > register_limit) ||
                    (second_dest != 255 && second_dest + second_count > register_limit) ||
                    (dest != 255 && second_dest != 255 && second_count != 0 &&
                     second_dest < dest + first_count && dest < second_dest + second_count) ||
                    (insn.words[3] & 0xc00001ffU))
                    return fail(index, "native uniform array texture gather mode");
            }
            if ((insn.Bits(61, 3) != 1 && !array) || (insn.Bits(77, 1) && !per_tap) || insn.Bits(81, 3) != 7 ||
                insn.Bits(59, 1) ||
                (depth && (bindless || array || offsets || per_tap || half ||
                           !insn.Bits(60, 1) || insn.Bits(80, 1) || insn.Bits(91, 1) ||
                           (source_a != 255 && source_a + 2 > register_limit) ||
                           (source_b != 255 && source_b >= register_limit))) ||
                (array && !bindless && (!insn.Bits(60, 1) ||
                           (insn.words[2] & ~0x058000ffU) != 0x001e0f00U ||
                           source_a + 2 > register_limit || source_b >= register_limit ||
                           dest + 2 > register_limit || second_dest + 2 > register_limit ||
                           (second_dest < dest + 2 && dest < second_dest + 2) ||
                           (insn.words[3] & 0xc00001ffU))) ||
                (half && (bindless || offsets || !insn.Bits(60, 1) || insn.Bits(80, 1) ||
                          insn.Bits(91, 1) ||
                          (source_b != 255 && source_b >= register_limit) ||
                          mask != 15 || dest >= register_limit || second_dest >= register_limit)) ||
                (offsets && (bindless || source_b == 255 || (source_a != 255 && source_a >= 245))) ||
                (per_tap && (bindless || depth || insn.Bits(80, 1) || insn.Bits(91, 1))) ||
                insn.Bits(89, 1) || mask == 0 ||
                (bindless ? (handle >= 246 || insn.Bits(46, 13) || !insn.Bits(91, 1))
                          : (insn.Bits(40, 14) > 0x1fff || binding >= 18 ||
                             (result.required_texture_buffer != ~0U && result.required_texture_buffer != binding))))
                return fail(index, "native texture gather mode");
            if (per_tap) {
                if (!insn.Bits(60, 1) || mask != 15 || source_a + 1 >= register_limit ||
                    source_b + 1 >= register_limit || (dest != 255 && dest + 1 >= register_limit) ||
                    (second_dest != 255 && second_dest + 1 >= register_limit))
                    return fail(index, "native per-tap gather operands");
                std::array<std::optional<u32>, 2> packed_offsets;
                bool imad_offsets = false;
                bool predicate_clobbered = false;
                if (!bounded_control)
                    return fail(index, "native per-tap gather control-flow proof");
                for (u32 cursor = index; cursor > 0 && (!packed_offsets[0] || !packed_offsets[1]);) {
                    if (control_entries[cursor])
                        return fail(index, "native per-tap gather control-flow proof");
                    const auto& definition = code[--cursor];
                    if (definition.Bits(12, 4) == 15) continue;
                    const u32 op = definition.Bits(0, 12), target = definition.Bits(16, 8);
                    u32 written_predicates = ~0U;
                    switch (op) {
                    case 0x802: case 0x805: case 0x819: case 0x823: case 0x91a:
                    case 0x945: case 0x947:
                        written_predicates = 0;
                        break;
                    case 0x212: case 0x812: case 0x224: case 0x424: case 0x624:
                    case 0x824: case 0xb63:
                        written_predicates = 1U << definition.Bits(81, 3);
                        break;
                    case 0x810: case 0x81c:
                        written_predicates = (1U << definition.Bits(81, 3)) |
                                             (1U << definition.Bits(84, 3));
                        break;
                    }
                    predicate_clobbered |= (written_predicates & (1U << (native_predicate & 7))) != 0;
                    if (op == 0xb63) {
                        const u32 count = static_cast<u32>(std::popcount(definition.Bits(72, 4)));
                        const u32 high = definition.Bits(64, 8);
                        const bool packed = definition.Bits(79, 1);
                        const u32 first_count = packed ? (high == 255 ? (count + 1) / 2 : 1U) : high == 255 ? count : std::min(count, 2U);
                        const u32 second_count = packed ? (count > 2 ? 1U : 0U) : count > 2 ? count - 2 : 0U;
                        for (u32 i = 0; i < 2; ++i) {
                            if (packed_offsets[i]) continue;
                            const u32 reg = source_b + i;
                            if ((target != 255 && reg >= target && reg < target + first_count) ||
                                (high != 255 && reg >= high && reg < high + second_count))
                                return fail(index, "native per-tap gather offset clobber");
                        }
                        continue;
                    }
                    auto count = write_count(definition);
                    switch (op) {
                    case 0x827: case 0x307: case 0x211: case 0x811: case 0xa11: case 0x624: case 0x831:
                    case 0x832: case 0x224:
                        count = 1;
                        break;
                    case 0x805:
                        count = 2;
                        break;
                    case 0x945: case 0x947: case 0x91a:
                        count = 0;
                        break;
                    }
                    if (!count) return fail(index, "native per-tap gather offset proof");
                    for (u32 i = 0; i < 2; ++i) {
                        if (packed_offsets[i] || target == 255 || source_b + i < target || source_b + i >= target + *count)
                            continue;
                        const bool move = op == 0x802 && !definition.Bits(24, 8) && definition.words[2] == 0xf00;
                        const bool imad = op == 0x424 && definition.Bits(24, 8) == 255 && definition.words[2] == 0x078e00ff;
                        const bool same_guard = imad && definition.Bits(12, 4) == native_predicate &&
                                                !predicate_clobbered;
                        if ((!move && !imad) || (definition.Bits(12, 4) != 7 && !same_guard) ||
                            (definition.words[3] & 0xc00001ffU))
                            return fail(index, "native per-tap gather nonconstant offsets");
                        imad_offsets |= imad;
                        packed_offsets[i] = definition.words[1];
                    }
                }
                if (!half && !imad_offsets) {
                    bool terminal = false;
                    for (const auto& instruction : code) {
                        if (instruction.Bits(12, 4) == 15) continue;
                        const u32 op = instruction.Bits(0, 12);
                        if (op == 0x94d && instruction.Bits(12, 4) == 7) {
                            terminal = true;
                            break;
                        }
                        if (op != 0x326 && op != 0x802 && op != 0x307 && op != 0x421 &&
                            op != 0xa20 && op != 0xb63 && op != 0x221 && op != 0x823 &&
                            op != 0x423 && op != 0x223 && op != 0x308 && op != 0x220 && op != 0x820)
                            return fail(index, "native per-tap gather control-flow proof");
                    }
                    if (!terminal) return fail(index, "native per-tap gather terminal proof");
                }
                for (const auto& packed : packed_offsets) {
                    if (!packed) return fail(index, "native per-tap gather missing offsets");
                    // Each byte is a signed 6-bit offset; the gather reads only the low six bits.
                    for (u32 shift = 0; shift < 32; shift += 8) {
                        const u32 value = (*packed >> shift) & 255;
                        if (value > 31 && value < 224)
                            return fail(index, "native per-tap gather offset range");
                    }
                }
                for (u32 i = 0; i < packed_offsets.size(); ++i)
                    emit(0x010000000007f000ULL | (248 + i) | (u64{*packed_offsets[i]} << 20));
            }
            if (!bindless) result.required_texture_buffer = binding;
            // Two coordinates: x in Ra, y in Rb (the coordinate list splits evenly across them);
            // with an offset, x and y in Ra, Ra+1 and the offset in Rb.
            const u32 y = !offsets && !per_tap && !depth ? source_b : source_a == 255 ? 255U : source_a + 1;
            emit(0x5c98000000000000ULL | predicate | 250 | (u64{15} << 39) | (u64{source_a} << 20));
            emit(0x5c98000000000000ULL | predicate | 251 | (u64{15} << 39) |
                 (u64{array ? source_a + 1 : y} << 20));
            if (array)
                emit(0x5c98000000000000ULL | predicate | 252 | (u64{15} << 39) | (u64{source_b} << 20));
            if (offsets || depth) {
                emit(0x5c98000000000000ULL | predicate | 246 | (u64{15} << 39) | (u64{source_b} << 20));
            }
            if (bindless) {
                emit(0x5c98000000000000ULL | predicate | 246 | (u64{15} << 39) | (u64{handle} << 20));
                emit(0xdec0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{246} << 20) |
                     (u64{array ? 3U : 2U} << 28) | (u64{mask} << 31) | (u64{insn.Bits(87, 2)} << 38) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{7} << 51));
            } else {
            emit(0xc800000000000000ULL | predicate | 250 | (u64{250} << 8) |
                 (u64{per_tap ? 248U : offsets || depth ? 246U : 255U} << 20) | (u64{array ? 3U : 2U} << 28) | (u64{mask} << 31) |
                 (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) | (u64{depth} << 50) | (u64{7} << 51) |
                 (u64{per_tap ? 2U : offsets ? 1U : 0U} << 54) | (u64{insn.Bits(87, 2)} << 56));
            }
            if (half) {
                for (u32 i = 0; i < 2; ++i) {
                    emit(0x5ca8000000000000ULL | predicate | 246 | (u64{1} << 8) | (u64{2} << 10) |
                         (u64{250 + i * 2} << 20));
                    emit(0x5ca8000000000000ULL | predicate | 247 | (u64{1} << 8) | (u64{2} << 10) |
                         (u64{251 + i * 2} << 20));
                    emit(0x5c18000000000000ULL | predicate | (i == 0 ? dest : second_dest) |
                         (u64{247} << 8) | (u64{246} << 20) | (u64{16} << 39));
                }
                break;
            }
            const u32 components = static_cast<u32>(std::popcount(mask));
            const bool split = second_dest != 255;
            for (u32 i = 0; i < components; ++i) {
                u32 target = 255;
                if (!split || i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                else target = second_dest + (i - 2);
                if (target != 255)
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{250 + i} << 20));
            }
            break;
        }
        case 0xb60: case 0xf60: case 0x361: { // TEX; 0xf60: handle in a uniform register; 0x361: TEX.B
            // Bindless forms become Maxwell TEX.B, the handle first among the meta registers (the
            // texture pass tracks it back to its constant buffer words). 0x361 lists the handle
            // right after the coordinates, then the LOD and the offsets (see general_tex).
            const bool bindless = opcode != 0xb60;
            const u32 handle = opcode == 0xf60 ? uniform_register(insn.Bits(40, 6)) : 255;
            if (bindless && ((opcode == 0xf60 && (handle == ~0U || handle >= 246 || insn.Bits(46, 13))) ||
                             (opcode == 0x361 && (insn.Bits(40, 19) || insn.Bits(87, 3) > 3))))
                return fail(index, "native bindless texture mode");
            const u32 mask = insn.Bits(72, 4);
            const u32 second_dest = insn.Bits(64, 8);
            constexpr std::array<u32, 8> rg_masks{1, 2, 4, 8, 3, 9, 10, 12};
            constexpr std::array<u32, 5> rgba_masks{7, 11, 13, 14, 15};
            u32 swizzle = ~0U;
            const bool split = second_dest != 255;
            const bool depth_compare = insn.Bits(78, 1);
            const u32 dimension = insn.Bits(61, 3), lod = insn.Bits(87, 3);
            const u32 source_a = insn.Bits(24, 8), source_b = insn.Bits(32, 8);
            if (!bindless && dimension == 5 && depth_compare && lod == 0 && !split && mask == 1 &&
                (result.required_texture_buffer == ~0U ||
                 result.required_texture_buffer == insn.Bits(54, 5))) {
                if (!insn.Bits(60, 1) || insn.Bits(76, 2) || insn.Bits(81, 3) != 7 ||
                    insn.Bits(59, 1) || insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) >= 18 ||
                    (source_a & 1) || (source_b & 1) || source_a >= 248 || source_b >= 248 || dest >= 248)
                    return fail(index, "native array depth comparison sample mode");
                if (!bind_texture_buffer(insn.Bits(54, 5)))
                    return fail(index, "native texture descriptor buffer conflict");
                for (u32 i = 0; i < 3; ++i) {
                    const u32 source = i < 2 ? source_a + i : source_b;
                    emit(0x5c98000000000000ULL | predicate | (250 + i) |
                         (u64{15} << 39) | (u64{source} << 20));
                }
                emit(0xc000000000000000ULL | predicate | dest | (u64{250} << 8) |
                     (u64{source_b + 1} << 20) | (u64{3} << 28) | (u64{1} << 31) |
                     (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                     (u64{1} << 50) | (u64{7} << 51));
                break;
            }
            // General form: gather the native operand list (array index first, coordinates,
            // then LOD/bias and the depth reference) into Maxwell TEX's coordinate and meta
            // registers, sample into scratch and copy the packed components out. Native packs
            // up to four values as halves of Ra and Rb (1+1, 2+1, 2+2), five or more as 4+n.
            const auto general_tex = [&]() -> bool {
                constexpr std::array<u32, 8> coordinate_count{1, 2, 3, 3, 2, 3, 0, 4};
                constexpr std::array<u32, 8> maxwell_type{0, 2, 4, 6, 1, 3, 0, 7};
                const bool offsets = insn.Bits(76, 1) != 0;
                // Bit 59 is .B, set in 0x361's encoding.
                if (dimension == 6 || insn.Bits(77, 1) || insn.Bits(81, 3) != 7 ||
                    insn.Bits(59, 1) != (opcode == 0x361 ? 1U : 0U) ||
                    (!bindless && (insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) >= 18)) ||
                    (lod > 3 && lod != 6 && lod != 7) || (depth_compare && mask != 1) || mask == 0)
                    return false;
                const u32 coordinates = coordinate_count[dimension];
                const bool has_lod = lod == 2 || lod == 3 || lod == 6 || lod == 7;
                // 0x361's handle is one more value after the coordinates: 2D.DC is (x, y | handle,
                // reference), ARRAY_2D is (layer, x | y, handle); with five or more values the
                // coordinates keep Ra (CUBE.LL is (x, y, z | handle, LOD)).
                const u32 handle_slot = opcode == 0x361 ? 1 : 0;
                const u32 count = coordinates + handle_slot + (has_lod ? 1 : 0) + (offsets ? 1 : 0) +
                                  (depth_compare ? 1 : 0);
                // Unpacked 0x361 (bit 60 clear): Ra holds the coordinates, Rb the handle and the rest.
                const u32 in_a = opcode == 0x361 && !insn.Bits(60, 1) ? coordinates
                               : count <= 4 ? (count + 1) / 2 : opcode == 0x361 ? coordinates : 4;
                const auto value = [&](u32 i) -> u32 {
                    if (i < in_a) return source_a == 255 ? 255U : source_a + i;
                    return source_b == 255 ? 255U : source_b + (i - in_a);
                };
                if ((source_a != 255 && source_a + in_a > 246) ||
                    (count > in_a && source_b != 255 && source_b + (count - in_a) > 246))
                    return false;
                const u32 texture_handle = opcode == 0x361 ? value(coordinates) : handle;
                if (bindless && texture_handle >= 246) return false;
                // A second descriptor buffer is read bindless: TEX.B takes the handle, loaded from
                // that buffer, as its first meta register.
                const u32 binding = insn.Bits(54, 5);
                const bool second_buffer = !bindless && !bind_texture_buffer(binding);
                const bool tex_b = bindless || second_buffer;
                const auto copy = [&](u32 target, u32 source) {
                    emit(0x5c98000000000000ULL | predicate | target | (u64{15} << 39) | (u64{source} << 20));
                };
                // Coordinates to R250.., then LOD/bias, offsets and reference to R246...
                for (u32 i = 0; i < coordinates; ++i) copy(250 + i, value(i));
                u32 meta = 246;
                u32 next = coordinates + handle_slot;
                if (bindless) copy(meta++, texture_handle);
                else if (second_buffer)
                    emit(0x4c98000000070000ULL | meta++ | (u64{15} << 39) | (u64{insn.Bits(40, 14)} << 20) |
                         (u64{binding} << 34));
                if (has_lod) copy(meta++, value(next++));
                if (offsets) copy(meta++, value(next++));
                if (depth_compare) copy(meta++, value(next++));
                if (tex_b) {
                    // Maxwell TEX.B: AOFFI at 36 and the LOD mode at 37 instead of 54 and 55.
                    emit(0xde80000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{246} << 20) |
                         (u64{maxwell_type[dimension]} << 28) | (u64{mask} << 31) | (u64{offsets} << 36) |
                         (u64{lod} << 37) | (u64{insn.Bits(90, 1)} << 49) | (u64{depth_compare} << 50) |
                         (u64{7} << 51));
                } else {
                emit(0xc000000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{meta == 246 ? 255U : 246U} << 20) | (u64{maxwell_type[dimension]} << 28) |
                     (u64{mask} << 31) | (u64{insn.Bits(40, 14)} << 36) |
                     (u64{insn.Bits(90, 1)} << 49) | (u64{depth_compare} << 50) | (u64{7} << 51) |
                     (u64{offsets} << 54) | (u64{lod} << 55));
                }
                const u32 components = static_cast<u32>(std::popcount(mask));
                if (insn.Bits(79, 1)) {
                    // FP16 results: (c0, c1) packed into the first destination, (c2, c3) into
                    // the second, the first of each pair in the low half as Maxwell TEXS.F16 packs
                    // (F2F writes the half in the low 16 bits; ISCADD Rd = (Ra << 16) + Rb merges).
                    const auto pack = [&](u32 target, u32 low, u32 high) {
                        if (target == 255) return;
                        emit(0x5ca8000000000000ULL | predicate | 246 | (u64{1} << 8) | (u64{2} << 10) | (u64{low} << 20));
                        emit(0x5ca8000000000000ULL | predicate | 247 | (u64{1} << 8) | (u64{2} << 10) | (u64{high} << 20));
                        emit(0x5c18000000000000ULL | predicate | target | (u64{247} << 8) | (u64{246} << 20) | (u64{16} << 39));
                    };
                    pack(static_cast<u32>(dest), 250, components > 1 ? 251U : 255U);
                    if (components > 2) {
                        pack(split ? second_dest : (dest == 255 ? 255U : static_cast<u32>(dest) + 1),
                             252, components > 3 ? 253U : 255U);
                    }
                    return true;
                }
                for (u32 i = 0; i < components; ++i) {
                    u32 target = 255;
                    if (!split) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                    else if (i < 2) target = dest == 255 ? 255U : static_cast<u32>(dest) + i;
                    else target = second_dest + (i - 2);
                    if (target != 255) copy(target, 250 + i);
                }
                return true;
            };
            if (split) {
                for (u32 i = 0; i < rgba_masks.size(); ++i) if (rgba_masks[i] == mask) swizzle = i;
            } else {
                for (u32 i = 0; i < rg_masks.size(); ++i) if (rg_masks[i] == mask) swizzle = i;
            }
            // SM70 scalar coordinates and split RG/BA destinations match TEXS.
            // Descriptor offsets are words in both encodings, but only native
            // explicitly encodes the constant-buffer binding.
            if (bindless || !insn.Bits(60, 1) || (dimension != 1 && dimension != 5) ||
                swizzle == ~0U || insn.Bits(76, 2) ||
                insn.Bits(81, 3) != 7 || (lod > 1 && (lod != 3 || dimension != 1 || depth_compare)) ||
                (lod == 3 && ((source_a & 1) || source_a >= 249)) ||
                insn.Bits(59, 1) ||
                insn.Bits(40, 14) > 0x1fff || insn.Bits(54, 5) != 2 ||
                dest == 255 || (mask & (mask - 1) && ((dest & 1) || dest >= 254)) ||
                (split && mask == 15 && ((second_dest & 1) || second_dest >= 254)) ||
                (depth_compare && (dimension != 1 || split || mask != 1 || (source_a & 1) || source_a >= 249)) ||
                (dimension == 5 && ((source_a & 1) || source_a >= 249)) || bindless ||
                !bind_texture_buffer(2)) {
                if (!general_tex()) return fail(index, "native texture mode or descriptor binding");
                break;
            }
            emit((0xd800000000000000ULL & ~(u64{insn.Bits(79, 1)} << 59)) | predicate | dest | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(64, 8)} << 28) |
                 (u64{insn.Bits(40, 14)} << 36) | (u64{insn.Bits(90, 1)} << 49) |
                 (u64{swizzle} << 50) | (u64{dimension == 5 ? (lod ? 8U : 7U) : depth_compare ? (lod ? 6U : 4U) : (lod == 3 ? 3U : lod ? 2U : 1U)} << 53));
            break;
        }
        case 0x816: { // Captured PRMT half-word concatenation
            if (insn.words[1] > 0xffff || (insn.words[2] & ~0xffU)) return fail(index, "native permutation mode");
            if (insn.words[1] != 0x5410 && insn.words[1] != 0x5432 && insn.words[1] != 0x7610 && insn.words[1] != 0x7632) {
                // Byte i of the result is byte (s & 7) of c:a, or its sign bit repeated when s & 8.
                const u32 a = insn.Bits(24, 8), c = insn.Bits(64, 8);
                if (a >= 250 && a != 255) return fail(index, "native permutation source");
                mov_imm(250, 0);
                for (u32 i = 0; i < 4; ++i) {
                    const u32 select = (insn.words[1] >> (i * 4)) & 0xf;
                    const u32 source = select & 4 ? c : a, position = (select & 3) * 8;
                    emit(0x3800000000000000ULL | predicate | 251 | (u64{source} << 8) |
                         (u64{select & 8 ? (position + 7) | (1U << 8) : position | (8U << 8)} << 20) |
                         (u64{(select >> 3) & 1} << 48));
                    emit(0x36f0000000000000ULL | predicate | 250 | (u64{251} << 8) |
                         (u64{(i * 8) | (8U << 8)} << 20) | (u64{250} << 39));
                }
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{255} << 20));
                break;
            }
            const u32 a = insn.Bits(24, 8), b = insn.Bits(64, 8);
            if (insn.words[1] != 0x5432 && insn.words[1] != 0x7632) {
                emit(0x0400000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{0xffff} << 20));
            } else {
                emit(0x3828000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{16} << 20));
            }
            if (insn.words[1] == 0x7610 || insn.words[1] == 0x7632) {
                emit(0x0400000000000000ULL | predicate | 251 | (u64{b} << 8) | (u64{0xffff0000} << 20));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{251} << 20));
            } else {
                emit(base(0x5c18000000000000ULL) | (u64{b} << 8) | (u64{250} << 20) | (u64{16} << 39));
            }
            break;
        }
        case 0x307: case 0xb07: { // FRND.F32, round to an integral floating-point value (0xb07: cbuf)
            if (opcode == 0x307 &&
                (insn.words[2] == 0x00100800 || insn.words[2] == 0x0010c800)) {
                if (insn.Bits(24, 8) || (insn.words[1] & ~0x100000ffU) ||
                    (insn.words[3] & 0xc00001ffU))
                    return fail(index, "native half round mode");
                emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{1} << 10) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{8 + insn.Bits(78, 2)} << 39) |
                     (u64{insn.Bits(60, 1)} << 41));
                break;
            }
            if (insn.Bits(75, 2) != 2 || insn.Bits(84, 2) != 2 || insn.Bits(81, 3) ||
                insn.Bits(86, 6)) return fail(index, "native floating round format");
            if (opcode == 0xb07 && !valid_cbuf()) return fail(index, "native constant buffer address");
            emit(base(opcode == 0xb07 ? 0x4ca8000000000000ULL : 0x5ca8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) |
                 (opcode == 0xb07 ? cbuf() : u64{insn.Bits(32, 8)} << 20) | (u64{8 + insn.Bits(78, 2)} << 39) |
                 (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(63, 1)} << 45) |
                 (u64{insn.Bits(62, 1)} << 49));
            break;
        }
        case 0x324: case 0x924: {
            if (!geometry_stage) return fail(index, "native geometry output outside geometry stage");
            const u32 type = insn.Bits(78, 2);
            if (insn.words[2] & ~0xc000U)
                return fail(index, "native geometry output mode");
            if (type == 0 && (opcode != 0x324 || dest != 255 || insn.words[1] != 255))
                return fail(index, "native geometry finalization mode");
            const u64 handle = u64{insn.Bits(24, 8)} << 8;
            const u64 flags = u64{type} << 39;
            if (opcode == 0x324) {
                if (insn.words[1] & ~0xffU)
                    return fail(index, "native geometry output stream");
                emit(base(0xfbe0000000000000ULL) | handle | (u64{insn.words[1]} << 20) | flags);
            } else {
                if (insn.words[1] > 0x7ffff)
                    return fail(index, "native geometry output stream");
                emit(base(0xf6e0000000000000ULL) | handle | (u64{insn.words[1]} << 20) | flags);
            }
            break;
        }
        case 0x321: // ALD
        case 0x322: { // AST
            const bool patch = insn.Bits(76, 1) != 0;
            const bool output = insn.Bits(79, 1) != 0;
            const u32 address = insn.Bits(40, 10);
            const u32 components = insn.Bits(74, 2) + 1;
            const bool tessellation_coordinates =
                tessellation_stage == TessellationStage::Evaluation && opcode == 0x321 &&
                !patch && address >= 0x2f0 && address + components * 4 <= 0x2f8;
            if (insn.Bits(77, 1) || (patch && tessellation_stage == TessellationStage::None) ||
                (output && ((!patch && !tessellation_coordinates) || opcode != 0x321)) ||
                (patch && opcode == 0x321 &&
                 output != (tessellation_stage == TessellationStage::Control)) ||
                (patch && opcode == 0x322 && tessellation_stage != TessellationStage::Control)) {
                return fail(index, "native indexed/patch attribute access");
            }
            if (address % 4 || address + components * 4 > 0x400 || insn.Bits(24, 8) != 255) {
                return fail(index, "native attribute address or indexing");
            }
            // a[0x5c] has no Maxwell meaning (Ampere uses it for a per-primitive hint such as the
            // shading rate). Nothing reads it back; dropping the store keeps full-rate shading.
            if (opcode == 0x322 && address == 0x5c && components == 1) break;
            u32 value_reg = opcode == 0x321 ? static_cast<u32>(dest) : insn.Bits(32, 8);
            if (value_reg == 255) {
                if (opcode == 0x321) break; // load into RZ: nothing to do
                // AST of RZ writes zeros: store them from zeroed scratch registers.
                for (u32 i = 0; i < components; ++i) mov_imm(250 + i, 0);
                value_reg = 250;
            }
            if (value_reg + components > 255) return fail(index, "native attribute register range");
            const u32 vertex = opcode == 0x321 ? insn.Bits(32, 8) : insn.Bits(64, 8);
            const u64 translated = (opcode == 0x321 ? 0xefd8000000000000ULL : 0xeff0000000000000ULL) |
                predicate | value_reg | (u64{255} << 8) | (u64{address} << 20) |
                (u64{patch} << 31) | (u64{output} << 32) |
                (u64{vertex} << 39) | (u64{components - 1} << 47);
            result.instructions.push_back(translated);
            break;
        }
        case 0x387: case 0x983: case 0x987: {
            // The size is bits 73-75 (0-3 = U8, S8, U16, S16, then 4, 5, 6 = 32, 64, 128 bits).
            // A load's bit 85 (.LU) is a cache hint.
            const bool store = opcode != 0x983;
            const u32 size = insn.Bits(73, 3);
            const bool uniform_address = insn.Bits(91, 1) != 0;
            const u32 data = insn.Bits(32, 8), offset = insn.Bits(40, 24);
            u32 address = insn.Bits(24, 8);
            const u32 words = size == 6 ? 4 : size == 5 ? 2 : 1, value = store ? data : static_cast<u32>(dest);
            const u32 bytes = size < 2 ? 1 : size < 4 ? 2 : 4;
            const u32 register_limit = uses_uniform_bank ? high_uniform_base : 246U;
            const u32 value_limit = uses_uniform_bank ? high_uniform_base : 248U;
            const u32 allowed = (store ? 0xe00U : 0x200e00U) |
                (uniform_address ? 0x08000000U | (store ? 0x3fU : 0U) : 0U);
            if ((address != 255 && address >= register_limit) || (store && insn.Bits(16, 8)) ||
                (!store && data && !uniform_address) ||
                (value != 255 && (value + words > value_limit || (value & (words - 1)))) ||
                (!store && value == 255) || (opcode == 0x987 && !uniform_address) ||
                (offset & (bytes - 1)) || offset + bytes * words > 0x800000 || size > 6 ||
                (insn.words[2] & ~allowed) != 0x00100000U)
                return fail(index, "native local scalar spill mode");
            if (uniform_address) {
                const u32 uniform = uniform_register(store ? insn.Bits(64, 6) : data);
                if (uniform == ~0U) return fail(index, "native local spill uniform address");
                if (address == 255) {
                    address = uniform;
                } else if (uniform != 255) {
                    emit(0x5c10000000000000ULL | predicate | 252 | (u64{uniform} << 8) |
                         (u64{address} << 20));
                    address = 252;
                }
            }
            if (!store && words > 1 && address >= value && address < value + words) {
                emit(0x5c98000000000000ULL | predicate | 252 | (u64{15} << 39) | (u64{address} << 20));
                address = 252;
            }
            for (u32 word = 0; word < words; ++word)
                emit((store ? 0xef50000000000000ULL : 0xef40000000000000ULL) | predicate |
                     (value == 255 ? 255U : value + word) | (u64{address} << 8) |
                     (u64{offset + 4 * word} << 20) | (u64{size < 4 ? size : 4U} << 48));
            break;
        }
        case 0x803: { // P2R PR, register merge source, immediate mask, byte select (76-77)
            if (insn.words[1] > 0x7f || (insn.words[2] & ~0x3000U))
                return fail(index, "native predicate-to-register mode");
            emit(base(0x38e8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1]} << 20) | (u64{insn.Bits(76, 2)} << 41));
            break;
        }
        case 0x804: { // R2P PR, register, immediate mask, byte select (76-77)
            if (insn.Bits(16, 8) || insn.words[1] > 0x7f || (insn.words[2] & ~0x3000U))
                return fail(index, "native register-to-predicate mode");
            emit(0x38f0000000000000ULL | predicate | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1]} << 20) | (u64{insn.Bits(76, 2)} << 41));
            break;
        }
        case 0xc02: // MOV uniform register
        case 0x202: // MOV register
        case 0x802: // MOV immediate
        case 0xa02: { // MOV constant buffer
            if (insn.Bits(72, 4) != 15 || (insn.Bits(76, 16) & ~(opcode == 0xc02 ? 0x8000U : 0U))) return fail(index, "native MOV modifiers");
            if (opcode == 0x802) {
                result.instructions.push_back(base(0x0100000000000000ULL) | (u64{15} << 12) |
                    (u64{insn.words[1]} << 20));
            } else if (opcode == 0xc02) {
                // Any uniform register (Hyrule Warriors: MOV R18, UR17; MOV R105, UR42).
                const u32 source = insn.words[1] < 64 ? uniform_register(insn.words[1]) : ~0U;
                if (source == ~0U || insn.words[2] != 0x08000f00) return fail(index, "native uniform move source");
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{source} << 20));
            } else if (opcode == 0x202) {
                if (insn.Bits(40, 24)) return fail(index, "native MOV register modifiers");
                result.instructions.push_back(base(0x5c98000000000000ULL) | (u64{15} << 39) |
                    (u64{insn.Bits(32, 8)} << 20));
            } else {
                if (!valid_cbuf() || insn.Bits(62, 2)) return fail(index, "native MOV constant buffer");
                result.instructions.push_back(base(0x4c98000000000000ULL) | (u64{15} << 39) | cbuf());
            }
            break;
        }
        case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23: { // FFMA
            if (insn.Bits(76, 1) || (insn.Bits(81, 11) & ~(opcode == 0xe23 || opcode == 0xc23 ? 0x400U : 0U))) return fail(index, "native FFMA denormal/source mode");
            const u32 a = operand(0), b = operand(1), c = operand(2);
            emit(base(0x5980000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) | (u64{c} << 39) |
                (u64{insn.Bits(77, 1)} << 50) | (u64{insn.Bits(78, 2)} << 51) | (u64{insn.Bits(80, 1)} << 53));
            break;
        }
        case 0x220: case 0x820: case 0xa20: case 0xc20: { // FMUL
            const u32 scale = insn.Bits(84, 3);
            if (insn.Bits(76, 1) || scale == 0 || insn.Bits(81, 3) ||
                insn.Bits(87, 4) || insn.Bits(91, 1) != static_cast<u32>(opcode == 0xc20) ||
                (scale != 4 && (!insn.Bits(80, 1) || insn.Bits(78, 2))))
                return fail(index, "native FMUL scale/mode");
            const u32 a = operand(0), b = operand(1);
            emit(base(0x5c68000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(78, 2)} << 39) | (u64{std::array<u32, 8>{0, 3, 2, 1, 0, 6, 5, 4}[scale]} << 41) | (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(77, 1)} << 50));
            break;
        }
        case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21: { // FADD
            if (insn.Bits(81, 10) || insn.Bits(91, 1) != static_cast<u32>(opcode == 0xe21))
                return fail(index, "native FADD mode");
            const u32 a = operand(0), b = operand(insn.Bits(9, 3) == 2 || insn.Bits(9, 3) == 3 || insn.Bits(9, 3) == 7 ? 2 : 1);
            emit(base(0x5c58000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(78, 2)} << 39) | (u64{insn.Bits(80, 1)} << 44) | (u64{insn.Bits(77, 1)} << 50));
            break;
        }
        case 0x209: case 0x809: case 0xa09: case 0xc09: { // FMNMX, including a uniform source
            if (insn.Bits(74, 6) || insn.Bits(82, 5) || (opcode != 0xc09 && insn.Bits(91, 1)))
                return fail(index, "native floating min/max mode");
            const u32 a = operand(0), b = operand(1);
            // .NAN (bit 81): an unordered pair gives the canonical NaN. R252 = ~0 when both are numbers.
            const bool propagate_nan = insn.Bits(81, 1);
            if (propagate_nan)
                emit(0x5800000000000000ULL | predicate | 252 | (u64{a} << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{7} << 48));
            emit(base(0x5c60000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42) |
                 (u64{insn.Bits(80, 1)} << 44));
            if (propagate_nan) {
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{252} << 20) | (u64{7} << 48));
                emit(0x0400000000000000ULL | predicate | 252 | (u64{252} << 8) | (u64{0xffffffffU} << 20) | (u64{2} << 53));
                emit(0x0400000000000000ULL | predicate | 252 | (u64{252} << 8) | (u64{0x7fffffffU} << 20));
                emit(base(0x5c40000000000000ULL) | (dest << 8) | (u64{252} << 20) | (u64{1} << 41) | (u64{7} << 48));
            }
            break;
        }
        case 0x308: case 0x908: case 0xb08: case 0xd08: {
            const u32 operation = insn.Bits(74, 6);
            if (opcode == 0x908 && (insn.Bits(24, 8) || (insn.words[2] & ~0xfc00U)))
                return fail(index, "native immediate MUFU mode");
            // Bit 91 is part of the uniform-register form's encoding.
            if (operation > 9 || operation == 6 || operation == 7 ||
                (operation == 9 && insn.Bits(73, 1)) ||
                insn.Bits(80, 12) != (opcode == 0xd08 ? 0x800U : 0U))
                return fail(index, "native MUFU mode");
            // SM70+ MUFU.COS/SIN take the angle in turns (compilers multiply by 1/2pi first);
            // Eden's Maxwell MUFU takes radians (its RRO is a plain move), so scale by 2pi.
            const bool turns = operation == 0 || operation == 1;
            const auto to_radians = [&](u32 source) {
                emit(0x1e00000000000000ULL | predicate | 250 | (u64{source} << 8) |
                     (u64{0x40c90fdbU} << 20));
                return 250U;
            };
            if (insn.Bits(73, 1)) {
                if (opcode != 0x308 || insn.Bits(60, 2) > 1)
                    return fail(index, "native half special-function source");
                emit(0x5ca8000000000000ULL | predicate | 250 | (u64{2} << 8) | (u64{1} << 10) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(60, 1)} << 41) |
                     (u64{insn.Bits(63, 1)} << 45) | (u64{insn.Bits(62, 1)} << 49));
                if (turns) to_radians(250);
                emit(0x5080000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{operation} << 20));
                emit(base(0x5ca8000000000000ULL) | (u64{1} << 8) | (u64{2} << 10) | (u64{250} << 20));
            } else {
                u32 src = opcode == 0xd08 ? uniform_register(insn.Bits(32, 8)) : operand(1);
                if (src == ~0U) return fail(index, "native MUFU uniform source");
                if (turns) src = to_radians(src);
                emit(base(0x5080000000000000ULL) | (u64{src} << 8) | (u64{operation} << 20));
            }
            break;
        }
        case 0x526: { // IPA.PASS / CONSTANT with the attribute indexed by a uniform register
            const u32 index_register = uniform_register(insn.Bits(24, 8));
            if (index_register == ~0U || insn.Bits(32, 8) != 255 ||
                (insn.Bits(76, 4) != 0 && insn.Bits(76, 4) != 4) || insn.Bits(81, 3) != 7)
                return fail(index, "native indexed IPA mode");
            if (index_register == 255) {
                emit(base(0xe000000000000000ULL) | (u64{255} << 8) | (u64{255} << 20) |
                     (u64{insn.Bits(64, 8)} << 30) | (u64{insn.Bits(78, 1)} << 55));
                break;
            }
            // Maxwell IPA.IDX reads the whole byte address from the index register.
            emit(0x1c00000000000000ULL | predicate | 250 | (u64{index_register} << 8) |
                 (u64{insn.Bits(64, 8) * 4} << 20));
            emit(base(0xe000000000000000ULL) | (u64{250} << 8) | (u64{255} << 20) |
                 (u64{insn.Bits(64, 8)} << 30) | (u64{1} << 38) | (u64{insn.Bits(78, 1)} << 55));
            break;
        }
        case 0x326: { // IPA.PASS / CONSTANT, default or centroid sample location
            // Bits 76-77 are the sample mode (title 1 interpolates most varyings at the centroid),
            // bit 78 the constant mode; Maxwell keeps them at bits 52-53 and 55.
            if (insn.Bits(32, 8) != 255 || insn.Bits(76, 2) > 1 || insn.Bits(79, 1) ||
                insn.Bits(81, 3) != 7)
                return fail(index, "native IPA interpolation mode");
            emit(base(0xe000000000000000ULL) | (u64{255} << 8) | (u64{255} << 20) | (u64{insn.Bits(64, 8)} << 30) |
                 (u64{insn.Bits(76, 2)} << 52) | (u64{insn.Bits(78, 1)} << 55));
            break;
        }
        case 0x81a: {
            if ((insn.words[2] & ~0x200U) || insn.words[1] > 32 || dest >= 255)
                return fail(index, "native sign extend mode");
            emit(base(0x3800000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{insn.words[1] << 8} << 20) | (u64{insn.Bits(73, 1)} << 48));
            break;
        }
        case 0x299: { // USHF by a uniform amount, one register shifted (SHL/SHR, bit 75 wraps).
            const u32 dst = uniform_register(insn.Bits(16, 8)), low = uniform_register(insn.Bits(24, 8));
            const u32 shift = uniform_register(insn.Bits(32, 8)), high = uniform_register(insn.Bits(64, 8));
            const bool right = insn.Bits(76, 1) != 0, high_result = insn.Bits(80, 1) != 0;
            if (dst == ~0U || dst == 255 || low == ~0U || shift == ~0U || high == ~0U ||
                insn.Bits(73, 2) != 3 ||
                (insn.words[2] & ~0x000118ffU) != 0x08000600U)
                return fail(index, "native uniform register shift mode");
            u32 source;
            if (high_result && low == 255) source = high;        // (high:0) >> or << s, high word
            else if (!high_result && high == 255) source = low;  // (0:low) >> or << s, low word
            else return fail(index, "native uniform funnel shift");
            emit((right ? 0x5c28000000000000ULL : 0x5c48000000000000ULL) | predicate | dst |
                 (u64{source} << 8) | (u64{shift} << 20) | (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x499: {
            const u32 target = uniform_register(static_cast<u32>(dest));
            const u32 shift = uniform_register(insn.Bits(64, 8));
            if (insn.Bits(24, 8) != 63 || (insn.words[2] & ~0x8ffU) != 0x08010600 ||
                (insn.words[3] & 0xc00001ffU) || target == ~0U || target == 255 || shift == ~0U)
                return fail(index, "native uniform immediate high-word shift mode");
            mov_imm(250, insn.words[1]);
            emit(0x5c48000000000000ULL | predicate | target | (u64{250} << 8) |
                 (u64{shift} << 20) | (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x419: { // SHF.L.U32.HI Rd, RZ, Rb, imm: W uses Rb & 31; otherwise min(Rb, 32).
            if (insn.Bits(24, 8) != 255 || (insn.words[2] & ~0x8ffU) != 0x00010600 ||
                (!insn.Bits(75, 1) && (insn.words[3] & 0xc00001ffU)))
                return fail(index, "native funnel shift immediate mode");
            mov_imm(250, insn.words[1]);
            emit(base(0x5c48000000000000ULL) | (u64{250} << 8) | (u64{insn.Bits(64, 8)} << 20) |
                 (u64{insn.Bits(75, 1)} << 39));
            break;
        }
        case 0x805: { // CS2R SRZ
            if (insn.Bits(72, 8) != 255 || !insn.Bits(80, 1) || dest >= 249) return fail(index, "native special-register read");
            mov_imm(static_cast<u32>(dest), 0);
            mov_imm(static_cast<u32>(dest) + 1, 0);
            break;
        }
        case 0x845: case 0x906: {
            const bool round_up = opcode == 0x906;
            if (insn.Bits(24, 8) || (insn.words[3] & 0xc00001ffU) ||
                (insn.words[2] & ~0xc400U) != 0x00201000U)
                return fail(index, "native immediate integer-to-float conversion mode");
            if (insn.words[2] != (round_up ? 0x00209000U : 0x0020d000U)) {
                mov_imm(250, insn.words[1]);
                emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) |
                     (u64{insn.Bits(74, 1)} << 13) | (u64{250} << 20) |
                     (u64{insn.Bits(78, 2)} << 39));
                break;
            }
            const u32 value = insn.words[1];
            u32 converted = 0;
            if (value) {
                const u32 shift = std::countl_zero(value);
                const u32 normalized = value << shift;
                converted = ((127U + 31U - shift) << 23) | ((normalized >> 8) & 0x7fffffU);
                converted += round_up && (normalized & 255U) != 0;
            }
            mov_imm(static_cast<u32>(dest), converted);
            break;
        }
        case 0x306: case 0xd06: { // I2F.F32.U8, round-to-nearest (0xd06: uniform source)
            const u32 source = opcode == 0xd06 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            const u64 is_signed = insn.Bits(74, 1);
            if (opcode == 0x306 &&
                (insn.words[2] == 0x00200c00 || insn.words[2] == 0x00209400)) {
                const bool half = insn.words[2] == 0x00200c00;
                if (insn.Bits(24, 8) || (insn.words[1] & ~0xffU) || (insn.words[3] & 0xc00001ffU))
                    return fail(index, "native scalar integer-to-float conversion mode");
                emit(base(0x5cb8000000000000ULL) | (u64{half ? 1U : 2U} << 8) | (u64{2} << 10) |
                     (u64{1} << 13) | (u64{source} << 20) | (u64{half ? 0U : 2U} << 39));
                break;
            }
            if (((opcode == 0x306 && (insn.words[2] & ~0x400U) == 0x00101000) ||
                 (opcode == 0xd06 && insn.words[2] == 0x08101000U && !insn.Bits(61, 1) &&
                  !insn.Bits(24, 8) && !(insn.words[3] & 0xc00001ffU))) &&
                !(insn.words[1] & 0xcfffff00U) && source != ~0U) {
                // I2F.U16/S16 Rd, Rs(.H1): signedness and the halfword selector are independent.
                emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{1} << 10) |
                     (u64{insn.Bits(74, 1)} << 13) | (u64{source} << 20) |
                     (u64{insn.Bits(60, 2) * 2} << 41));
                break;
            }
            // I2F.{U,S}8 Rd, Rs.B0-B3: the byte selector (bits 60-61) as Maxwell's (41-42).
            if ((insn.words[2] & ~0x400U) != (opcode == 0xd06 ? 0x08001000U : 0x1000U) ||
                (insn.words[1] & 0xcfffff00U) || source == ~0U)
                return fail(index, "native byte-to-float conversion mode");
            emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (is_signed << 13) | (u64{source} << 20) |
                 (u64{insn.Bits(60, 2)} << 41));
            break;
        }
        case 0x245: case 0xa45: case 0xc45: { // I2FP.F32.U32
            // Bits 78-79: rounding (RN/RM/RP/RZ), as Maxwell I2F's bits 39-40.
            if ((insn.words[2] & ~(0x400U | 0xc000U | (opcode == 0xc45 ? 0x08000000U : 0U))) != 0x00201000) return fail(index, "native packed integer-to-float format");
            const u32 src = operand(1, false);
            emit(base(0x5cb8000000000000ULL) | (u64{2} << 8) | (u64{2} << 10) | (u64{insn.Bits(74, 1)} << 13) | (u64{src} << 20) |
                 (u64{insn.Bits(78, 2)} << 39));
            break;
        }
        case 0x207: case 0x807: case 0xa07: case 0xc07:
        case 0x208: case 0x808: case 0xa08: { // Bitwise SEL / FSEL
            // FSEL applies the floating operand modifiers (negate/absolute) before selecting.
            const bool floating = (opcode & 0x1ff) == 0x008;
            const u32 a = operand(0, floating), b = operand(1, floating);
            emit(base(0x5ca0000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
            break;
        }
        case 0x20a: case 0x80a: case 0xa0a: case 0xc0a: { // FSET.BF, combined with Pc (bits 87-90) by bits 74-75 (0xc0a: uniform b)
            if (insn.Bits(64, 8) || insn.Bits(74, 2) == 3 || insn.Bits(81, 6) || (opcode != 0xc0a && insn.Bits(91, 1)))
                return fail(index, "native floating comparison output mode");
            const u32 a = operand(0), b = operand(1);
            emit(base(0x5800000000000000ULL) | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{insn.Bits(87, 4)} << 39) | (u64{insn.Bits(74, 2)} << 45) |
                 (u64{insn.Bits(76, 4)} << 48) | (u64{1} << 52) | (u64{insn.Bits(80, 1)} << 55));
            break;
        }
        case 0x234: case 0x434: { // HSETP2 with a register (B swizzle 60-61, abs 62, neg 63) or an
                      // immediate half pair: A swizzle 74-75 (neg 72, abs 73),
                      // combine 69-70, H_AND 71, compare 76-79, FTZ 80, P/Q 81-86, input 87-90.
            u64 source = u64{250} << 20;
            if (opcode == 0x234) {
                if ((insn.words[1] & 0x0fffff00U) || insn.Bits(64, 5))
                    return fail(index, "native half comparison mode");
                source = (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(60, 2)} << 28) |
                         (u64{insn.Bits(62, 1)} << 30) | (u64{insn.Bits(63, 1)} << 31);
            } else {
                mov_imm(250, insn.words[1]);
            }
            emit(0x5d20000000000000ULL | predicate | insn.Bits(84, 3) | (u64{insn.Bits(81, 3)} << 3) |
                 (u64{insn.Bits(80, 1)} << 6) | (u64{insn.Bits(24, 8)} << 8) | source |
                 (u64{insn.Bits(76, 4)} << 35) | (u64{insn.Bits(87, 3)} << 39) |
                 (u64{insn.Bits(90, 1)} << 42) | (u64{insn.Bits(72, 1)} << 43) |
                 (u64{insn.Bits(73, 1)} << 44) | (u64{insn.Bits(69, 2)} << 45) |
                 (u64{insn.Bits(74, 2)} << 47) | (u64{insn.Bits(71, 1)} << 49));
            break;
        }
        case 0x20b: case 0x80b: case 0xa0b: case 0xc0b: // FSETP
        case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: { // ISETP
            const bool floating = (opcode & 0x1ff) == 0xb;
            if (!floating && insn.Bits(72, 1)) {
                // ISETP.EX (bit 72): the high-word half of a 64-bit compare. The low words'
                // result comes in as the predicate at bits 68-70 (negated by 71), e.g.
                // ISETP.NE P0 = lo != 0, then ISETP.NE.EX P0 = (hi != 0) OR P0. EQ: high equal
                // AND low; NE: high different OR low; the orderings: high strictly ordered OR
                // (high equal AND low).
                const u32 compare = insn.Bits(76, 3);
                const u32 destination = insn.Bits(81, 3);
                const u32 destination_b = insn.Bits(84, 3);
                const u32 low = insn.Bits(68, 4);
                const u32 accumulate = insn.Bits(87, 4);
                const u32 combine = insn.Bits(74, 2);
                if (compare == 0 || compare == 7 || combine == 3)
                    return fail(index, "native extended integer comparison mode");
                const u32 a = operand(0, false), b = operand(1, false);
                u32 temporary = 0;
                while (temporary == destination || temporary == destination_b ||
                       temporary == (low & 7) || temporary == (accumulate & 7) ||
                       temporary == insn.Bits(12, 3))
                    ++temporary;
                const u32 mask = 1U << temporary;
                emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
                const auto isetp = [&](u32 op, u32 input, u32 combine) {
                    return 0x5b60000000000000ULL | predicate | 7 | (u64{temporary} << 3) |
                           (u64{a} << 8) | (u64{b} << 20) | (u64{input & 7} << 39) |
                           (u64{input >> 3} << 42) | (u64{combine} << 45) |
                           (u64{insn.Bits(73, 1)} << 48) | (u64{op} << 49);
                };
                constexpr u32 And = 0, Or = 1, Lt = 1, Eq = 2, Gt = 4, Ne = 5;
                if (compare == Eq) {
                    emit(isetp(Eq, low, And));
                } else if (compare == Ne) {
                    emit(isetp(Ne, low, Or));
                } else {
                    emit(isetp(Eq, low, And));
                    emit(isetp(compare <= 3 ? Lt : Gt, temporary, Or));
                }
                emit(0x5090000000000000ULL | predicate | destination_b |
                     (u64{destination} << 3) | (u64{temporary} << 12) | (u64{7} << 29) |
                     (u64{accumulate & 7} << 39) | (u64{accumulate >> 3} << 42) |
                     (u64{combine} << 45));
                emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
                break;
            }
            const u32 a = operand(0, floating), b = operand(1, floating);
            const u32 combine = insn.Bits(74, 2);
            if (combine > 2) return fail(index, "native predicate combination");
            u64 out = (floating ? 0x5bb0000000000000ULL : 0x5b60000000000000ULL) | predicate |
                insn.Bits(84, 3) | (u64{insn.Bits(81, 3)} << 3) | (u64{a} << 8) | (u64{b} << 20) |
                (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42) | (u64{combine} << 45);
            if (floating) out |= (u64{insn.Bits(80, 1)} << 47) | (u64{insn.Bits(76, 4)} << 48);
            else out |= (u64{insn.Bits(73, 1)} << 48) | (u64{insn.Bits(76, 3)} << 49);
            emit(out);
            break;
        }
        case 0x212: case 0x812: case 0xa12: case 0xc12: { // LOP3.LUT, any table, optional P = (result != 0)
            const u32 lut = insn.Bits(72, 8);
            const u32 pred_target = insn.Bits(81, 3);
            if (insn.Bits(80, 1) || insn.Bits(87, 4) != 15 ||
                (opcode == 0x212 && insn.Bits(40, 24)) || (opcode == 0xa12 && insn.Bits(62, 2)))
                return fail(index, "native LUT/predicate mode");
            u32 b = 255;
            if (opcode == 0x212) {
                b = insn.Bits(32, 8);
            } else if (opcode == 0x812) {
                mov_imm(251, insn.words[1]);
                b = 251;
            } else if (opcode == 0xc12) {
                b = uniform_register(insn.Bits(32, 8));
                if (b == ~0U) return fail(index, "native uniform logic source");
            } else {
                b = operand(1, false);
            }
            const u32 a = insn.Bits(24, 8), c = insn.Bits(64, 8);
            emit(0x5be0000000000000ULL | predicate | dest | (u64{a} << 8) | (u64{b} << 20) |
                 (u64{lut} << 28) | (u64{pred_target == 7 ? 0U : 3U} << 36) | (u64{c} << 39) |
                 (u64{pred_target} << 48));
            break;
        }
        case 0x213: case 0xa13: case 0xc13: { // IABS: max(x, 0 - x), signed.
            if (insn.words[2] & ~(opcode == 0xc13 ? 0x08000000U : 0U))
                return fail(index, "native integer absolute mode");
            const u32 source = operand(1, false);
            emit(0x5c10000000000000ULL | predicate | 250 | (u64{255} << 8) | (u64{source} << 20) |
                 (u64{1} << 48));
            emit(base(0x5c20000000000000ULL) | (u64{source} << 8) | (u64{250} << 20) |
                 (u64{15} << 39) | (u64{1} << 48));
            break;
        }
        case 0x217: case 0x817: case 0xa17: case 0xc17: { // IMNMX register, immediate, cbuf or uniform operand.
            if ((insn.words[2] & ~(0x04000200U | (opcode == 0xc17 ? 0x08000000U : 0U))) != 0x03800000 ||
                (opcode != 0xa17 && opcode != 0x817 && insn.words[1] > 255U) ||
                (opcode == 0xa17 && (!valid_cbuf() || insn.Bits(62, 2))))
                return fail(index, "native integer min/max mode");
            // An immediate is a signed 20-bit value: 19 bits at 20, the sign in bit 56.
            const bool short_immediate = opcode == 0x817 &&
                                         (insn.words[1] <= 0x7ffffU || insn.words[1] >= 0xfff80000U);
            if (opcode == 0x817 && !short_immediate) mov_imm(251, insn.words[1]);
            const u32 source = opcode == 0xc17   ? uniform_register(insn.words[1])
                               : opcode == 0xa17 ? operand(1, false)
                               : opcode == 0x817 ? (short_immediate ? insn.words[1] & 0x7ffffU : 251U)
                                                 : insn.words[1];
            if (source == ~0U) return fail(index, "native uniform min/max operand");
            emit(base(short_immediate ? 0x3820000000000000ULL : 0x5c20000000000000ULL) |
                 (u64{insn.Bits(24, 8)} << 8) | (u64{source} << 20) |
                 (u64{short_immediate ? insn.words[1] >> 31 : 0U} << 56) |
                 (u64{insn.Bits(87, 4)} << 39) | (u64{insn.Bits(73, 1)} << 48));
            break;
        }
        case 0x819: { // SHF.HI, immediate shift
            const bool is_signed = insn.Bits(73, 2) == 2;
            if (!insn.Bits(80, 1) && insn.Bits(76, 1) && insn.Bits(73, 2) == 1 &&
                !insn.Bits(75, 1) && insn.words[1] <= 32) {
                // SHF.R.U64 funnel: the translator reads bits 20-38 as the immediate, so the U64
                // clamp (bit 38) would turn every shift into 63; shifts up to 32 need no clamp.
                emit(base(0x38f8000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.words[1]} << 20) | (u64{insn.Bits(64, 8)} << 39));
                break;
            }
            if (!insn.Bits(80, 1) && !insn.Bits(76, 1) && insn.Bits(64, 8) == 255 &&
                insn.Bits(73, 2) == 3 && !insn.Bits(75, 1) && insn.words[1] <= 32) {
                emit(base(0x3848000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            if (!insn.Bits(80, 1) && insn.Bits(76, 1) && insn.Bits(64, 8) == 255 &&
                (insn.Bits(73, 2) == 3 || is_signed) && !insn.Bits(75, 1) && insn.words[1] <= 32) {
                // SHF.R low word with a zero high word: a logical right shift.
                emit(base(0x3828000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20));
                break;
            }
            // U64 left shifts through 32 have the same high word as a 32-bit funnel shift.
            const bool u64_left = insn.Bits(73, 2) == 1 && !insn.Bits(76, 1);
            if ((insn.Bits(73, 2) != 3 && !is_signed && !u64_left) || insn.Bits(75, 1) || !insn.Bits(80, 1) || insn.words[1] > 32) return fail(index, "native funnel-shift mode");
            const u32 low = insn.Bits(24, 8), high = insn.Bits(64, 8), shift = insn.words[1];
            if (insn.Bits(76, 1)) {
                if (low != 255) return fail(index, "native nonzero low right-shift word");
                emit(base(0x3828000000000000ULL) | (u64{high} << 8) | (u64{shift} << 20) |
                     (u64{is_signed} << 48));
            } else {
                if (is_signed) return fail(index, "native signed left funnel shift");
                emit(base(0x36f8000000000000ULL) | (u64{low} << 8) | (u64{high} << 39) | (u64{shift} << 20));
            }
            break;
        }
        case 0x211: case 0x811: case 0xa11: case 0xc11: { // LEA (0xa11: cbuf, 0xc11: uniform Rb)
            if (insn.Bits(80, 1)) {
                // LEA.HI Rd, Ra, Rb (or an immediate, 0x811), Rc, s: Rb + high word of (Rc:Ra << s);
                // bit 73 sign-extends Ra (Rc is RZ) and .X (bit 74) adds the carry predicate in bits 87-90.
                const bool extended = insn.Bits(74, 1) != 0;
                const u32 high = insn.Bits(64, 8);
                if (insn.Bits(72, 1) || insn.Bits(81, 3) != 7 ||
                    (high != 255 && insn.Bits(73, 1)) ||
                    (opcode == 0x211 && (insn.Bits(62, 1) || (insn.Bits(63, 1) && extended))) ||
                    (opcode == 0xa11 && insn.Bits(62, 2)) ||
                    (opcode == 0xc11 && (insn.words[1] & ~0x3fU)))
                    return fail(index, "native high effective address mode");
                u32 b = opcode == 0xc11 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
                if (opcode == 0xa11) b = operand(1, false);
                if (b == ~0U) return fail(index, "native uniform LEA register");
                if (opcode == 0x811) {
                    b = insn.words[1] ? 251U : 255U;
                    if (b == 251) mov_imm(251, insn.words[1]);
                }
                const u64 shift = (u64{32 - insn.Bits(75, 5)} << 20) | (u64{insn.Bits(73, 1)} << 48);
                // The high word is a right shift of Ra, or with Rc a left funnel shift (SHF.L).
                const auto high_word = [&](u64 target) {
                    if (high == 255) emit(target | 0x3828000000000000ULL | (u64{insn.Bits(24, 8)} << 8) | shift);
                    else emit(target | 0x36f8000000000000ULL | (u64{insn.Bits(24, 8)} << 8) | (u64{high} << 39) |
                              (u64{insn.Bits(75, 5)} << 20));
                };
                if (b == 255 && !extended) {
                    high_word(base(0));
                } else {
                    high_word(predicate | 250);
                    if (extended) {
                        mov_imm(252, 1);
                        emit(0x5ca0000000000000ULL | predicate | 252 | (u64{252} << 8) | (u64{255} << 20) |
                             (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                        emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{252} << 20));
                    }
                    emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{b} << 20) |
                         (u64{opcode == 0x211 ? insn.Bits(63, 1) : 0U} << 48));
                }
                break;
            }
            // A carry out (bits 81-83) is the low half of a 64-bit address; IADD3.X adds it next.
            const u32 carry_out = insn.Bits(81, 3);
            if (insn.Bits(72, 1) || insn.Bits(74, 1) || insn.Bits(80, 1) ||
                (carry_out != 7 && ((opcode != 0x811 && insn.Bits(63, 1)) ||
                                    (rewritten_uniform && opcode != 0x211))) ||
                ((opcode == 0x211 || opcode == 0xa11) && insn.Bits(62, 1)) || (opcode == 0xc11 && (insn.words[1] & ~0x8000003fU)))
                return fail(index, "native LEA carry/high mode");
            // 0x211 bit 63 negates Rb (LEA Rd, Ra, -Rb, s), as Maxwell ISCADD's bit 48.
            u32 b = opcode == 0xc11 ? uniform_register(insn.Bits(32, 8)) : insn.Bits(32, 8);
            if (opcode == 0xa11) b = operand(1, false);
            if (b == ~0U) return fail(index, "native uniform LEA register");
            if (opcode == 0x811) { mov_imm(251, insn.words[1]); b = 251; }
            if (opcode == 0xa11 && carry_out != 7) {
                // The cbuf is snapshotted in R251. Write the result before the carry predicate,
                // which can also be the instruction's guard.
                emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{carry_out} << 3) |
                     (dest << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
                break;
            }
            if (carry_out != 7) {
                // The sum in R250, carry = sum < Rb (unsigned), as IADD3's carry out; a uniform
                // predicate goes through P(6-n) into R239.
                const u32 target = rewritten_uniform ? 6 - carry_out : carry_out, mask = 1U << target;
                emit(0x5c18000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39));
                if (rewritten_uniform) emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{250} << 8) |
                     (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
                if (rewritten_uniform) {
                    emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                    emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20));
                }
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                break;
            }
            emit(base(0x5c18000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                 (u64{b} << 20) | (u64{insn.Bits(75, 5)} << 39) |
                 (u64{opcode != 0x811 ? insn.Bits(63, 1) : 0U} << 48));
            break;
        }
        case 0x210: case 0x810: case 0xa10: case 0xc10: { // IADD3 with zero third source and no carry outputs
            if (rewritten_uniform && opcode == 0x210 && insn.Bits(74, 1) &&
                insn.Bits(81, 6) == 0x3f) {
                if (guarded_uniform || (insn.words[2] & ~0x0781e0ffU) != 0x007e0400U ||
                    (insn.words[1] & ~0xffU))
                    return fail(index, "native uniform extended add mode");
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(32, 8)} << 20));
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{insn.Bits(64, 8)} << 20));
                u32 mask = 0;
                for (u32 bit : {77U, 87U}) {
                    const u32 source = insn.Bits(bit, 3);
                    if (source != 7) mask |= 1U << (6 - source);
                }
                save_predicate(mask);
                emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{mask} << 20));
                mov_imm(252, 1);
                for (u32 bit : {77U, 87U}) {
                    const u32 source = insn.Bits(bit, 3);
                    emit(0x5ca0000000000000ULL | predicate | 251 | (u64{252} << 8) |
                         (u64{255} << 20) | (u64{source == 7 ? 7U : 6 - source} << 39) |
                         (u64{insn.Bits(bit + 3, 1)} << 42));
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) |
                         (u64{251} << 20));
                }
                restore_predicate(mask);
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                break;
            }
            if (rewritten_uniform && opcode == 0x210 && insn.Bits(81, 6) != 0x3f) {
                const u32 first = insn.Bits(81, 3), second = insn.Bits(84, 3);
                if (guarded_uniform || (insn.words[2] & ~0x007e00ffU) != 0x0781e000U ||
                    (insn.words[1] & ~0xffU) || (first != 7 && first == second))
                    return fail(index, "native uniform add carry mode");
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{1} << 47));
                emit(0x5c10000000000000ULL | predicate | 251 | (u64{255} << 8) |
                     (u64{255} << 20) | (u64{1} << 43));
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{250} << 8) |
                     (u64{insn.Bits(64, 8)} << 20) | (u64{1} << 47));
                emit(0x5c10000000000000ULL | predicate | 251 | (u64{251} << 8) |
                     (u64{255} << 20) | (u64{1} << 43));
                const u32 mask = (first == 7 ? 0U : 1U << (6 - first)) |
                                 (second == 7 ? 0U : 1U << (6 - second));
                save_predicate(mask);
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                for (u32 output = 0; output < 2; ++output) {
                    const u32 target = output ? second : first;
                    if (target == 7) continue;
                    mov_imm(252, output + 1);
                    emit(0x5b60000000000000ULL | predicate | 7 | (u64{6 - target} << 3) |
                         (u64{251} << 8) | (u64{252} << 20) | (u64{7} << 39) | (u64{6} << 49));
                }
                emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                restore_predicate(mask);
                break;
            }
            // 64-bit adds: IADD3 with a carry-out predicate (bits 81-83), then IADD3.X (bit 74) with
            // a carry-in predicate (bits 87-89, negate 90). Uniform forms keep UPn in R239 bit 6-n.
            if ((opcode == 0x210 || opcode == 0x810) && insn.Bits(64, 8) == 255 && !insn.Bits(73, 1) &&
                !insn.Bits(75, 2) && insn.Bits(84, 3) == 7 && insn.Bits(77, 4) == 15 && !insn.Bits(72, 1) &&
                (insn.Bits(81, 3) != 7 || insn.Bits(74, 1)) && (opcode == 0x810 || !insn.Bits(63, 1))) {
                const bool carry_in = insn.Bits(74, 1) != 0;
                const u32 carry_out = insn.Bits(81, 3);
                const u32 a = insn.Bits(24, 8);
                if ((carry_in && carry_out != 7) || (opcode == 0x210 && (insn.words[1] & ~0xffU)))
                    return fail(index, "native integer carry mode");
                if (opcode == 0x810) {
                    emit(0x1c00000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{insn.words[1]} << 20));
                } else {
                    emit(0x5c10000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{insn.Bits(32, 8)} << 20));
                }
                const auto save = [&](u32 mask) { emit(0x38e8000000070000ULL | 253 | (u64{255} << 8) | (u64{mask} << 20)); };
                const auto restore = [&](u32 mask) { emit(0x38f0000000070000ULL | (u64{253} << 8) | (u64{mask} << 20)); };
                if (!carry_in) {
                    // carry = (a + b) < a, unsigned
                    const u32 target = rewritten_uniform ? 6 - carry_out : carry_out;
                    const u32 mask = 1U << target;
                    if (rewritten_uniform) save(mask);
                    emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{250} << 8) |
                         (u64{a} << 20) | (u64{7} << 39) | (u64{1} << 49));
                    if (rewritten_uniform) {
                        emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{mask} << 20));
                        restore(mask);
                    }
                    emit(base(0x5c98000000000000ULL) | (u64{15} << 39) | (u64{250} << 20));
                    break;
                }
                const u32 condition = insn.Bits(87, 3);
                const u32 source = rewritten_uniform && condition != 7 ? 6 - condition : condition;
                const u32 mask = 1U << source;
                mov_imm(252, 1);
                if (rewritten_uniform && condition != 7) {
                    save(mask);
                    emit(0x38f0000000070000ULL | (u64{239} << 8) | (u64{mask} << 20));
                }
                emit(0x5ca0000000000000ULL | predicate | 251 | (u64{252} << 8) | (u64{255} << 20) |
                     (u64{source} << 39) | (u64{insn.Bits(90, 1)} << 42));
                if (rewritten_uniform && condition != 7) restore(mask);
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{251} << 20));
                break;
            }
            // Three sources, negations or two carries: the sum builds up in R253 (.CC) with the carry
            // count in R254 (.X). Pu/Pv (bits 81/84) are set when the count reaches 1/2, and IADD3.X
            // adds both carry-ins (bits 87, 77). A negated source is ~x, plus 1 unless extended.
            const bool extended = insn.Bits(74, 1) != 0;
            const bool negate_a = insn.Bits(72, 1), negate_b = opcode != 0x810 && insn.Bits(63, 1), negate_c = insn.Bits(75, 1);
            if ((extended || insn.Bits(81, 6) != 0x3f) &&
                (insn.Bits(64, 8) != 255 || insn.Bits(84, 3) != 7 || negate_a || negate_b || negate_c ||
                 (extended && insn.Bits(77, 4) != 15))) {
                const u32 first_output = insn.Bits(81, 3), second_output = insn.Bits(84, 3);
                if (rewritten_uniform || guarded_uniform || insn.Bits(73, 1) || insn.Bits(76, 1) ||
                    (first_output != 7 && first_output == second_output) ||
                    (!extended && (insn.Bits(77, 4) != 15 || insn.Bits(87, 4) != 15)) ||
                    (opcode == 0x210 && (insn.words[1] & ~0x800000ffU)) || (opcode == 0xa10 && insn.Bits(62, 1)) ||
                    (opcode == 0xc10 && (insn.words[1] & ~0x8000003fU)))
                    return fail(index, "native three-input carry mode");
                u32 a = insn.Bits(24, 8), b = operand(1, false), c = insn.Bits(64, 8);
                const auto invert = [&](u32& source, u32 target) {
                    emit(0x0400000000000000ULL | predicate | target | (u64{source} << 8) | (u64{0xffffffffU} << 20) | (u64{2} << 53));
                    source = target;
                };
                if (negate_a) invert(a, 250);
                if (negate_b) invert(b, 251);
                if (negate_c) invert(c, 252);
                const auto accumulate = [&](u32 source) {
                    emit(0x5c10000000000000ULL | predicate | 253 | (u64{253} << 8) | (u64{source} << 20) | (u64{1} << 47));
                    emit(0x5c10000000000000ULL | predicate | 254 | (u64{254} << 8) | (u64{255} << 20) | (u64{1} << 43));
                };
                emit(0x5c10000000000000ULL | predicate | 253 | (u64{a} << 8) | (u64{b} << 20) | (u64{1} << 47));
                emit(0x5c10000000000000ULL | predicate | 254 | (u64{255} << 8) | (u64{255} << 20) | (u64{1} << 43));
                accumulate(c);
                if (extended) {
                    for (const u32 carry : {insn.Bits(87, 4), insn.Bits(77, 4)}) {
                        if (carry == 15) continue;
                        mov_imm(250, 1);
                        if (carry != 7)
                            emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                                 (u64{carry & 7} << 39) | (u64{carry >> 3} << 42));
                        accumulate(250);
                    }
                } else if (const u32 ones = u32{negate_a} + u32{negate_b} + u32{negate_c}) {
                    mov_imm(250, ones);
                    accumulate(250);
                }
                if (dest != 255) emit(base(0x5c98000000000000ULL) | (u64{253} << 20) | (u64{15} << 39));
                // An output that is also the guard predicate is written last.
                const bool guard_first = (native_predicate & 7) != 7 && first_output == (native_predicate & 7);
                for (const u32 output : {guard_first ? 1U : 0U, guard_first ? 0U : 1U}) {
                    const u32 target = output ? second_output : first_output;
                    if (target == 7) continue;
                    mov_imm(250, output + 1);
                    emit(0x5b60000000000000ULL | predicate | 7 | (u64{target} << 3) | (u64{254} << 8) |
                         (u64{250} << 20) | (u64{7} << 39) | (u64{6} << 49));
                }
                break;
            }
            if ((opcode == 0xa10 || opcode == 0xc10) && insn.Bits(81, 3) != 7) { // cbuf or uniform b
                if (insn.Bits(64, 8) != 255 || insn.Bits(73, 4) || insn.Bits(84, 7) != 0x7f ||
                    insn.Bits(77, 4) != 15 || insn.Bits(63, 1) || insn.Bits(72, 1) ||
                    (opcode == 0xc10 && (insn.words[1] & ~0x3fU)))
                    return fail(index, "native constant add carry mode");
                const u32 b = operand(1, false);
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{b} << 20));
                emit(0x5b60000000000000ULL | predicate | 7 | (u64{insn.Bits(81, 3)} << 3) |
                    (dest << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{1} << 49));
                break;
            }
            if ((opcode == 0xa10 || opcode == 0xc10) && insn.Bits(74, 1)) { // IADD3.X Ra + cbuf or UR + carry predicate
                if ((insn.words[2] & ~(0x07800000U | (opcode == 0xc10 ? 0x08000000U : 0U))) != 0x007fe4ffU ||
                    (opcode == 0xc10 && (insn.words[1] & ~0x3fU)))
                    return fail(index, "native extended constant add mode");
                const u32 a = insn.Bits(24, 8), b = operand(1, false);
                mov_imm(250, 1);
                emit(0x5ca0000000000000ULL | predicate | 250 | (u64{250} << 8) | (u64{255} << 20) |
                    (u64{insn.Bits(87, 3)} << 39) | (u64{insn.Bits(90, 1)} << 42));
                if (a != 255) emit(0x5c10000000000000ULL | predicate | 250 | (u64{a} << 8) | (u64{250} << 20));
                emit(base(0x5c10000000000000ULL) | (u64{b} << 8) | (u64{250} << 20));
                break;
            }
            // Third source, optionally negated (bit 75; 63 and 72 negate b and a).
            if (insn.Bits(64, 8) != 255 && (insn.Bits(73, 4) & ~4U) == 0 && insn.Bits(77, 7) == 0x7f && insn.Bits(84, 7) == 0x7f) {
                const u32 b = operand(1, false);
                emit(0x5c10000000000000ULL | predicate | 250 | (u64{insn.Bits(24, 8)} << 8) |
                    (u64{b} << 20) | (u64{negate_b} << 48) | (u64{insn.Bits(72, 1)} << 49));
                emit(base(0x5c10000000000000ULL) | (u64{250} << 8) | (u64{insn.Bits(64, 8)} << 20) |
                     (u64{insn.Bits(75, 1)} << 48));
                break;
            }
            if (insn.Bits(64, 8) != 255 || insn.Bits(73, 4) || insn.Bits(77, 7) != 0x7f || insn.Bits(84, 7) != 0x7f) return fail(index, "native integer carry/third source");
            if (opcode == 0xa10 || opcode == 0xc10) {
                if (opcode == 0xc10 && (insn.words[1] & ~0x8000003fU)) return fail(index, "native uniform add operand");
                const u32 b = operand(1, false);
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{b} << 20) |
                    (u64{insn.Bits(63, 1)} << 48) | (u64{insn.Bits(72, 1)} << 49));
            } else if (opcode == 0x210) {
                if (insn.words[1] & ~0x800000ffU) return fail(index, "native integer add register modifiers");
                emit(base(0x5c10000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) |
                     (u64{insn.Bits(32, 8)} << 20) | (u64{insn.Bits(63, 1)} << 48) | (u64{insn.Bits(72, 1)} << 49));
            } else {
                emit(base(0x1c00000000000000ULL) | (u64{insn.Bits(24, 8)} << 8) | (u64{insn.words[1]} << 20) | (u64{insn.Bits(72, 1)} << 56));
            }
            break;
        }
        case 0x949: case 0x958: { // Captured BRX / BRXU with absolute native-PC table values
            if (insn.Bits(12, 4) != 7 || (insn.words[2] & ~0x3ffffU) != (opcode == 0x958 ? 0x0b800000U : 0x03800000U))
                return fail(index, "native computed branch mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            if (s64{index} * 16 + 16 + relative != 0)
                return fail(index, "native computed branch relative base");
            const s64 offset = -s64{address_of(static_cast<u32>(result.instructions.size()))} - 8;
            if (offset < -0x800000) return fail(index, "native computed branch range");
            const u32 source = opcode == 0x958 ? uniform_register(insn.Bits(24, 8)) : insn.Bits(24, 8);
            if (source == ~0U) return fail(index, "native uniform computed branch source");
            emit(0xe25000000000000fULL | predicate | (u64{source} << 8) |
                 ((static_cast<u64>(offset) & 0xffffff) << 20));
            computed_branch = true;
            break;
        }
        case 0x944: { // Verified CALL.REL.NOINC with R55 holding the return PC
            // A predicated NOINC forward transfer can terminate a loop at its
            // reconvergence point without a return-link setup or matching RET.
            if (native_predicate != 7 && insn.words[1] == 16 && insn.words[2] == 0x03c00000 &&
                index + 2 < code.size() && code[index + 1].Bits(0, 12) == 0x947) {
                relocations.push_back({static_cast<u32>(result.instructions.size()), index + 2});
                emit(0xe24000000000000fULL | predicate);
                break;
            }
            const bool adjacent = index != 0 &&
                (code[index - 1].words[0] & ~0x00ff0000U) == 0x00007802 &&
                code[index - 1].Bits(16, 8) < 250 &&
                code[index - 1].words[1] == (index + 1) * 16 &&
                code[index - 1].words[2] == 0xf00;
            if (index == 0 || insn.words[0] != 0x7944 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000 || (!adjacent && !bounded_control))
                return fail(index, "native call linkage mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) return fail(index, "native call destination alignment");
            const u32 link = adjacent ? code[index - 1].Bits(16, 8) : bounded_links[index];
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] == target / 16 && call_links[call] != link)
                    return fail(index, "native inconsistent call linkage");
            }
            call_targets.push_back(static_cast<u32>(target / 16));
            call_links.push_back(link);
            relocations.push_back({static_cast<u32>(result.instructions.size()), target / 16, true});
            emit(0xe26000000007000fULL);
            break;
        }
        case 0x950: { // Verified RET.REL.NODEC epilogue restoring the call link
            const u32 return_reg = insn.Bits(24, 8);
            // The return register is the link itself, or the last write to it in this function
            // is a copy of the link (MOV or IMAD.MOV.U32 Rd, RZ, RZ, Rlink).
            const auto copies_link = [&](const Instruction& prior) {
                if (prior.Bits(16, 8) != return_reg || prior.Bits(12, 4) != 7) return false;
                if ((prior.words[0] & 0xff00ffffU) == 0x00007202U)
                    return prior.words[1] == *active_link && prior.words[2] == 0xf00;
                if ((prior.words[0] & 0xff00ffffU) == 0xff007224U)
                    return prior.words[1] == 0xff && prior.words[2] == (0x078e0000U | *active_link);
                return false;
            };
            bool linked = active_link && return_reg == *active_link;
            for (u32 prior = index; active_link && !linked && prior-- > active_entry;) {
                if (copies_link(code[prior])) linked = true;
                else if (code[prior].Bits(16, 8) == return_reg) break;
            }
            if (!main_exited || !linked ||
                (insn.words[0] & 0x00ffffffU) != 0x00007950 || return_reg >= 249 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000 || !barrier_tokens.empty())
                return fail(index, "native return linkage mode");
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            if (s64{index} * 16 + 16 + relative != 0)
                return fail(index, "native return relative base");
            emit(0xe32000000007000fULL);
            bool complete = !computed_branch && !geometry_exit;
            for (u32 target : call_targets) complete &= target <= index;
            if (complete) return finish(index);
            break;
        }
        case 0x945: case 0x941: case 0x942: case 0x948: // (+ WARPSYNC)
            // BSSY / BSYNC / BREAK only manage warp reconvergence: BSYNC waits for the other
            // threads and falls through, BREAK drops the thread from that wait. Eden runs threads
            // independently, so all three are no-ops (a SYNC would wrongly jump, and a branch
            // out of a lowered SSY region would leave its token on Eden's flow stack).
            break;
        case 0x947: { // BRA
            // The offset's two low bits: .DIV (2) branches when the warp has diverged, .CONV (3)
            // when its active threads are the mask in the uniform register at 24-31 (bit 91;
            // URZ without it). The compiler puts warp-synchronous fast paths behind them (shuffles
            // on a converged warp; the other path calls a WARPSYNC helper). Eden runs the warp's
            // threads converged there: .CONV always branches and .DIV never does.
            const u32 mode = insn.Bits(32, 2);
            if ((mode <= 1 && insn.Bits(91, 1)) ||
                (mode == 1 && (insn.Bits(82, 5) || insn.Bits(92, 13))))
                return fail(index, "native branch convergence mode");
            if (mode == 2) break;
            u64 branch_predicate = predicate;
            if (opcode == 0x947 && insn.Bits(87, 4) != 7) {
                if (insn.Bits(12, 4) != 7) {
                    relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                    emit(0xe24000000000000fULL | (u64{insn.Bits(12, 4) ^ 8U} << 16));
                }
                branch_predicate = u64{insn.Bits(87, 4)} << 16;
            } else if (insn.Bits(87, 4) != 7) {
                return fail(index, "native extra branch condition");
            }
            const u64 raw = (insn.words[1] & ~3U) | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) return fail(index, "native branch destination alignment");
            relocations.push_back({static_cast<u32>(result.instructions.size()), target / 16});
            emit(0xe24000000000000fULL | branch_predicate);
            if (computed_branch && target == s64{index} * 16 && branch_predicate == (u64{7} << 16))
                return finish(index); // the closing self-branch
            break;
        }
        case 0x34e: {
            if (!geometry_stage) return fail(index, "unsupported native opcode");
            if (!native_code_address || insn.Bits(24, 8) || insn.words[1] || insn.words[2] ||
                (dest != 255 && dest + 1 >= (uses_uniform_bank ? high_uniform_base : 246U)) ||
                native_predicate != 7)
                return fail(index, "native program counter load mode");
            if (dest == 255) break;
            const u64 address = native_code_address + u64{index} * sizeof(Instruction);
            mov_imm(static_cast<u32>(dest), static_cast<u32>(address));
            mov_imm(static_cast<u32>(dest) + 1, static_cast<u32>(address >> 32));
            break;
        }
        case 0x923: case 0x927: { // ISBERD / ISBEWR: task and mesh programs' stage buffer.
            if (geometry_stage || tessellation_stage != TessellationStage::None) {
                if (opcode != 0x923 || insn.words[1] || insn.words[2] ||
                    insn.Bits(24, 8) == 255)
                    return fail(index, tessellation_stage != TessellationStage::None
                                           ? "native tessellation stage buffer read mode"
                                           : "native geometry stage buffer read mode");
                emit(base(0x5c98000000000000ULL) | (u64{15} << 39) |
                     (u64{insn.Bits(24, 8)} << 20));
                break;
            }
            // Ampere: Ra 24-31, offset 40-63, ISBERD dest 16-23, ISBEWR data 32-39; 75 = .32 (else
            // U8), 76-77 = 3 .ATTR (0 the map region), 78 = .SKEW, 79 = .O (the output buffer).
            // Kept as an nx2 pseudo instruction (Maxwell 0xefc0, free there) that the frontend
            // reads for the task or mesh stage: data/dest 0-7, Ra 8-15, offset 20-43, .32 44,
            // .ATTR 45, .SKEW 46, .O 47, write 48. Addresses are linear bytes in the region.
            const bool write = opcode == 0x927;
            const u32 region = insn.Bits(76, 2);
            const bool output = insn.Bits(79, 1) != 0;
            if ((insn.words[2] & ~0xf800U) != 0 || (region != 0 && region != 3) || write != output ||
                (!write && region != 3) || (!write && dest == 255))
                return fail(index, "native stage buffer access mode");
            emit(0xefc0000000000000ULL | predicate | (write ? u64{insn.Bits(32, 8)} : dest) |
                 (u64{insn.Bits(24, 8)} << 8) | (u64{insn.Bits(40, 24)} << 20) |
                 (u64{insn.Bits(75, 1)} << 44) | (u64{region == 3} << 45) |
                 (u64{insn.Bits(78, 1)} << 46) | (u64{output} << 47) | (u64{write} << 48));
            break;
        }
        case 0x94d: { // EXIT
            if (geometry_exit && index * sizeof(Instruction) < geometry_exit->handler) {
                relocations.push_back({static_cast<u32>(result.instructions.size()),
                                       static_cast<s64>(geometry_exit->handler / sizeof(Instruction))});
                emit(0xe24000000000000fULL | predicate);
                const bool pending_main_branch = std::any_of(
                    relocations.begin(), relocations.end(), [&](const Relocation& relocation) {
                        return !relocation.call && relocation.target > index &&
                               relocation.target !=
                                   static_cast<s64>(geometry_exit->handler / sizeof(Instruction));
                    });
                if (geometry_exit->has_returns && native_predicate == 7 && !main_exited && !pending_main_branch &&
                    !call_targets.empty()) {
                    for (u32 target : call_targets) {
                        if (target <= index)
                            return fail(index, "native call target overlaps main program");
                    }
                    main_exited = true;
                }
                break;
            }
            if (insn.Bits(84, 2)) {
                return fail(index, "native conditional/special EXIT");
            }
            if (insn.Bits(87, 4) != 7) {
                // @Pg EXIT Pc: exits when both hold; skip over a Pc-predicated EXIT unless Pg.
                if (native_predicate != 7) {
                    relocations.push_back({static_cast<u32>(result.instructions.size()), index + 1});
                    emit(0xe24000000000000fULL | (u64{native_predicate ^ 8U} << 16));
                }
                emit(0xe30000000000000fULL | (u64{insn.Bits(87, 4)} << 16));
                break;
            }
            result.instructions.push_back(0xe30000000000000fULL | predicate);
            if (insn.Bits(12, 4) != 7) break;
            if (!barrier_tokens.empty()) return fail(index, "native unclosed reconvergence scope");
            if (call_targets.empty()) {
                const bool pending_branch = std::any_of(relocations.begin(), relocations.end(),
                    [&](const Relocation& relocation) { return relocation.target > index; });
                if (!pending_branch && !computed_branch) return finish(index);
                break;
            }
            // An EXIT with main-program branches still pending past it (not calls) ends one path
            // of the main program; the functions start after the last of them.
            const bool pending_main_branch = std::any_of(relocations.begin(), relocations.end(),
                [&](const Relocation& relocation) { return !relocation.call && relocation.target > index; });
            if (!main_exited && pending_main_branch) break;
            if (main_exited) {
                // A library function may end the shader instead of returning (Eden inlines
                // calls, so this is an ordinary exit at that point).
                bool complete = std::none_of(relocations.begin(), relocations.end(),
                    [&](const Relocation& relocation) { return relocation.target > index; });
                for (u32 target : call_targets) complete &= target <= index;
                if (complete && !computed_branch) return finish(index);
                break;
            }
            for (u32 target : call_targets) {
                if (target <= index) return fail(index, "native call target overlaps main program");
            }
            // The main program goes on past an EXIT that a branch still jumps over.
            const bool pending_branch = std::any_of(relocations.begin(), relocations.end(),
                [&](const Relocation& relocation) {
                    return relocation.target > index &&
                           std::find(call_targets.begin(), call_targets.end(),
                                     static_cast<u32>(relocation.target)) == call_targets.end();
                });
            if (!pending_branch) main_exited = true;
            break;
        }
        default:
            return fail(index, "unsupported native opcode");
        }
        if (!operand_error.empty()) return fail(index, operand_error);
        if (output_mask) {
            emit(0x38e8000000070000ULL | 239 | (u64{239} << 8) | (u64{output_mask} << 20));
            emit(0x38f0000000070000ULL | (u64{254} << 8) | (u64{output_mask} << 20));
        }
        if (guarded_uniform) restore_predicate(guard_mask);
        if (!pressure.suffix.empty()) {
            // Branches that skip the rest of this instruction land on its copies back.
            for (size_t relocation = relocation_start; relocation < relocations.size(); ++relocation) {
                auto& pending = relocations[relocation];
                if (!pending.call && pending.output_index >= instruction_start && pending.target == index + 1)
                    pending.output_target = static_cast<u32>(result.instructions.size());
            }
            result.instructions.insert(result.instructions.end(), pressure.suffix.begin(), pressure.suffix.end());
        }
    }
    return fail(static_cast<u32>(code.size()), main_exited ? "native program lacks captured library return" : "native program has no unconditional EXIT");
}

inline LoweredProgram LowerToMaxwell(std::span<const Instruction> code,
                                    LoweringOptions options = {}) {
    // Programs using uniform registers above UR27 are lowered again with mirrors for them.
    u32 high_uniforms = 0;
    for (;;) {
        LoweredProgram lowered = LowerToMaxwellWith(code, high_uniforms, options);
        if (lowered || lowered.high_uniform_registers <= high_uniforms) return lowered;
        high_uniforms = lowered.high_uniform_registers;
    }
}

inline LoweredProgram LowerToMaxwellWith(std::span<const Instruction> code, u32 high_uniforms,
                                        bool geometry_stage, bool compact_matrix,
                                        std::span<const u32> verified_traps = {},
                                        TessellationStage tessellation_stage = TessellationStage::None) {
    return LowerToMaxwellWith(code, high_uniforms,
                             {.geometry = geometry_stage, .tensor = compact_matrix,
                              .verified_traps = verified_traps,
                              .tessellation_stage = tessellation_stage});
}

inline LoweredProgram LowerToMaxwell(std::span<const Instruction> code, bool geometry_stage,
                                    bool compact_matrix = false,
                                    std::span<const u32> verified_traps = {},
                                    TessellationStage tessellation_stage = TessellationStage::None) {
    return LowerToMaxwell(code, {.geometry = geometry_stage, .tensor = compact_matrix,
                                .verified_traps = verified_traps,
                                .tessellation_stage = tessellation_stage});
}

} // namespace Shader::Ampere
