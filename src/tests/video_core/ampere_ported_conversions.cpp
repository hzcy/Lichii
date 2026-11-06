// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

TEST_CASE("Ampere byte F2I clamps before converting and preserves predicates",
          "[video_core][ampere]") {
    for (const bool is_signed : {false, true}) {
        for (const u32 ftz : {0U, 0x10000U}) {
            for (const u32 predicate : {1U, 7U, 9U}) {
                const std::array<Instruction, 2> code{{
                    {{0x00150305U | (predicate << 12), 3,
                      0x0020e000U | ftz | (is_signed ? 0x100U : 0U), 0x0001e400}},
                    {{0x0000794d, 0, 0x03800000, 0}},
                }};
                const auto lowered = LowerToMaxwell(code);
                REQUIRE(static_cast<bool>(lowered));
                const u64 guard = u64{predicate} << 16;
                const u32 lower = is_signed ? 0xc3000000U : 0U;
                const u32 upper = is_signed ? 0x42fe0000U : 0x437f0000U;
                const std::array<u64, 7> expected{
                    0x5800000000000000ULL | guard | 250 | (3ULL << 8) | (3ULL << 20) |
                        (7ULL << 39) | (7ULL << 48),
                    0x5c40000000000000ULL | guard | 250 | (3ULL << 8) | (250ULL << 20) |
                        (7ULL << 48),
                    0x0100000000000000ULL | guard | 252 | (15ULL << 12) | (u64{lower} << 20),
                    0x5c60000000000000ULL | guard | 250 | (250ULL << 8) | (252ULL << 20) |
                        (15ULL << 39),
                    0x0100000000000000ULL | guard | 252 | (15ULL << 12) | (u64{upper} << 20),
                    0x5c60000000000000ULL | guard | 250 | (250ULL << 8) | (252ULL << 20) |
                        (7ULL << 39),
                    0x5cb0000000000000ULL | guard | 21 | (2ULL << 8) | (2ULL << 10) |
                        (u64{is_signed} << 12) | (250ULL << 20) | (3ULL << 39) |
                        (u64{ftz != 0} << 44),
                };
                REQUIRE(lowered.instructions.size() == expected.size() + 1);
                REQUIRE(std::equal(expected.begin(), expected.end(), lowered.instructions.begin()));
                REQUIRE(std::bit_cast<float>(lower) == (is_signed ? -128.0f : 0.0f));
                REQUIRE(std::bit_cast<float>(upper) == (is_signed ? 127.0f : 255.0f));
            }
        }
    }
}

TEST_CASE("Ampere byte F2I materializes constant and uniform sources",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x00157b05, 0x00400400, 0x0021e100, 0x0001e400}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    const auto constant = LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(constant));
    REQUIRE(constant.instructions.front() ==
            (0x4c98000000070000ULL | 252 | (15ULL << 39) | (4ULL << 20) | (1ULL << 34)));
    for (const u32 source : {0U, 21U, 22U, 27U, 63U}) {
        auto uniform = code;
        uniform[0].words = {0x00157d05, source, 0x0821e100, 0x0001e400};
        const auto lowered = LowerToMaxwell(uniform);
        REQUIRE(static_cast<bool>(lowered));
        const u32 mapped = source < 22 ? source + 216 : source < 28 ? source + 218 : 255;
        REQUIRE(((lowered.instructions.front() >> 8) & 255) == mapped);
        REQUIRE(((lowered.instructions.front() >> 20) & 255) == mapped);
    }
}

TEST_CASE("Ampere byte F2I rejects other widths, rounding, modifiers and register aliases",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x00157305, 3, 0x0021e100, 0x0001e400}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    for (const u32 mode : {0x0021c100U, 0x0021a100U, 0x00216100U, 0x0021e300U,
                           0x0021e900U, 0x0061e100U, 0x0821e100U}) {
        auto invalid = code;
        invalid[0].words[2] = mode;
        const auto lowered = LowerToMaxwell(invalid);
        if (mode == 0x0021e900U) {
            REQUIRE(lowered);
            REQUIRE(lowered.instructions.size() == 2);
            CHECK(lowered.instructions.front() == 0x5cb0118000371915ULL);
            invalid[0].words[2] |= 1;
            REQUIRE_FALSE(LowerToMaxwell(invalid));
        } else {
            REQUIRE_FALSE(lowered);
        }
    }
    for (const u32 source : {0x10003U}) {
        auto invalid = code;
        invalid[0].words[1] = source;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    for (const u32 source : {64U, 216U}) {
        auto invalid = code;
        invalid[0].words = {0x00157d05, source, 0x0821e100, 0x0001e400};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    for (const u32 source : {0x00400440U, 0x04800400U}) {
        auto invalid = code;
        invalid[0].words = {0x00157b05, source, 0x0021e100, 0x0001e400};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto missing_uniform_mode = code;
    missing_uniform_mode[0].words[0] = 0x00157d05;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(missing_uniform_mode)));
    auto invalid = code;
    invalid[0].words[0] |= 1U << 24;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    invalid = code;
    invalid[0].words[3] |= 1;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    invalid = code;
    invalid[0].words[0] = 0x00f67305;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
}
TEST_CASE("Ampere F2I applies abs and neg to constant-buffer sources", "[video_core][ampere]") {
    // nvdisasm: bit 62 prints |c[..]| and bit 63 -c[..] for the byte, U16 and TRUNC word forms.
    for (const u32 mode : {0x0021e100U, 0x00212800U, 0x0021f100U}) {
        for (const u32 modifiers : {1U, 2U, 3U}) {
            const std::array<Instruction, 2> code{{
                {{0x00157b05, 0x00400400 | (modifiers << 30), mode, 0x0001e400}},
                {{0x0000794d, 0, 0x03800000, 0}},
            }};
            const auto lowered = LowerToMaxwell(code);
            INFO(lowered.error);
            REQUIRE(lowered);
            const auto& out = lowered.instructions;
            const auto has = [&](u64 instruction) {
                return std::find(out.begin(), out.end(), instruction) != out.end();
            };
            CHECK(has(0x0400000000070000ULL | 252 | (252ULL << 8) | (0x7fffffffULL << 20)) ==
                  ((modifiers & 1) != 0));
            CHECK(has(0x0400000000070000ULL | 252 | (252ULL << 8) | (0x80000000ULL << 20) |
                      (2ULL << 53)) == ((modifiers & 2) != 0));
        }
    }
}

TEST_CASE("Ampere ceiling F2I preserves rounding, signedness and unsigned saturation",
          "[video_core][ampere]") {
    for (const bool is_signed : {false, true}) {
        for (const u32 ftz : {0U, 0x10000U}) {
            for (const u32 predicate : {2U, 7U, 10U}) {
                const std::array<Instruction, 2> code{{
                    {{0x00020305U | (predicate << 12), 2,
                      0x0020b000U | ftz | (is_signed ? 0x100U : 0U), 0x000e3000}},
                    {{0x0000794d, 0, 0x03800000, 0}},
                }};
                const auto lowered = LowerToMaxwell(code);
                REQUIRE(static_cast<bool>(lowered));
                REQUIRE(lowered.instructions.size() == (is_signed ? 2U : 9U));
                const u64 guard = u64{predicate} << 16;
                const size_t conversion = lowered.instructions.size() - (is_signed ? 2 : 3);
                REQUIRE(lowered.instructions[conversion] ==
                        (0x5cb0000000000000ULL | guard | 2 | (2ULL << 8) | (2ULL << 10) |
                         (u64{is_signed} << 12) | (u64{is_signed ? 2U : 250U} << 20) |
                         (2ULL << 39) | (u64{ftz != 0} << 44)));
                for (size_t index = 0; index + 1 < lowered.instructions.size(); ++index)
                    REQUIRE(((lowered.instructions[index] >> 16) & 15) == predicate);
                if (!is_signed) {
                    REQUIRE(lowered.instructions[2] ==
                            (0x0100000000000000ULL | guard | 252 | (15ULL << 12) |
                             (0x4f800000ULL << 20)));
                    REQUIRE(lowered.instructions[4] ==
                            (0x0100000000000000ULL | guard | 252 | (15ULL << 12) |
                             (0x4f7fffffULL << 20)));
                    REQUIRE(lowered.instructions[7] ==
                            (0x5c40000000000000ULL | guard | 2 | (2ULL << 8) | (251ULL << 20) |
                             (1ULL << 41) | (7ULL << 48)));
                }
            }
        }
    }
}

TEST_CASE("Ampere ceiling F2I rejects non-NTZ, other widths and unknown modifiers",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x00027305, 2, 0x0020b000, 0x000e3000}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    for (const u32 mode : {0x00209000U, 0x0020a000U, 0x0020b800U, 0x0020b200U,
                           0x0060b000U, 0x0820b000U}) {
        auto invalid = code;
        invalid[0].words[2] = mode;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto invalid = code;
    invalid[0].words[1] = 0x40000002;
    const auto absolute = LowerToMaxwell(invalid);
    REQUIRE(absolute);
    REQUIRE(absolute.instructions.size() == LowerToMaxwell(code).instructions.size() + 2);
    CHECK(absolute.instructions[0] == 0x5c980780002700f9ULL);
    CHECK(absolute.instructions[1] == 0x0407fffffff7f9f9ULL);
    invalid[0].words[1] |= 1U << 31;
    const auto negated_absolute = LowerToMaxwell(invalid);
    REQUIRE(negated_absolute);
    REQUIRE(negated_absolute.instructions.size() == absolute.instructions.size() + 1);
    CHECK(negated_absolute.instructions[1] == 0x0407fffffff7f9f9ULL);
    CHECK(negated_absolute.instructions[2] == 0x044800000007f9f9ULL);
    invalid[0].words[1] |= 1U << 29;
    REQUIRE_FALSE(LowerToMaxwell(invalid));
    invalid = code;
    invalid[0].words[3] |= 1;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
}

TEST_CASE("Ampere byte and ceiling F2I bound registers and extended instruction modes",
          "[video_core][ampere]") {
    for (const u32 mode : {0x0021e100U, 0x0020b000U}) {
        const std::array<Instruction, 2> code{{
            {{0x00027305, 2, mode, 0x000e3000}},
            {{0x0000794d, 0, 0x03800000, 0}},
        }};
        for (const u32 source : {0U, 245U, 255U}) {
            auto valid = code;
            valid[0].words[1] = source;
            REQUIRE(static_cast<bool>(LowerToMaxwell(valid)));
        }
        for (const u32 bit : {0U, 8U, 30U, 31U}) {
            auto invalid = code;
            invalid[0].words[3] ^= 1U << bit;
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
        }
        for (u32 bit = 9; bit < 30; ++bit) {
            auto scheduling = code;
            scheduling[0].words[3] ^= 1U << bit;
            REQUIRE(static_cast<bool>(LowerToMaxwell(scheduling)));
        }
    }
}
TEST_CASE("Ampere indexed constants preserve narrow widths and signed byte offsets",
          "[video_core][ampere]") {
    for (u32 width = 0; width <= 5; ++width) {
        for (const u32 mode : {0U, 2U}) {
            for (const u32 offset : {0U, 1U, 0x20U, 0x8000U, 0xffffU}) {
                for (const u32 source : {3U, 255U}) {
                    const std::array<Instruction, 2> code{{
                        {{0x00049b82U | (source << 24), (15U << 22) | (offset << 6),
                          (width << 9) | (mode << 14), 0x000e6200}},
                        {{0x0000794d, 0, 0x03800000, 0}},
                    }};
                    const auto lowered = LowerToMaxwell(code);
                    REQUIRE(static_cast<bool>(lowered));
                    REQUIRE(lowered.instructions.size() == 2);
                    REQUIRE(lowered.instructions.front() ==
                            (0xef90000000090000ULL | 4 | (u64{source} << 8) |
                             (u64{offset} << 20) | (15ULL << 36) | (u64{mode} << 44) |
                             (u64{width} << 48)));
                }
            }
        }
    }
}

TEST_CASE("Ampere indexed constants reject unsupported widths modes and operand fields",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0xff059b82, 0x03c00800, 0, 0x000e6200}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    REQUIRE(static_cast<bool>(LowerToMaxwell(code)));
    for (const u32 word : {0x0c00U, 0x0e00U, 0x4000U, 0xc000U, 0xa00U}) {
        auto invalid = code;
        invalid[0].words[2] = word;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    for (const u32 bit : {32U, 37U, 59U, 63U, 64U, 72U, 76U, 80U, 91U}) {
        auto invalid = code;
        invalid[0].words[bit / 32] ^= 1U << (bit % 32);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto invalid = code;
    invalid[0].words[1] = (18U << 22) | (0x20U << 6);
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    invalid = code;
    invalid[0].words[0] = 0xf6059b82;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
}

} // namespace
