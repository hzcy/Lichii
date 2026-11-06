// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/submission_policy.h"

namespace {

using Vulkan::SubmissionPolicy::Action;
using Vulkan::SubmissionPolicy::Advance;
using Vulkan::SubmissionPolicy::ParseBudget;

} // namespace

TEST_CASE("Native submission overrides do not round to a dispatch boundary",
          "[video_core][submission_policy]") {
    for (const u32 budget : {1U, 2U, 31U, 32U, 33U, 127U, 128U, 129U, 4096U}) {
        CAPTURE(budget);
        u32 counter{};
        for (u32 work = 1; work <= budget * 3; ++work) {
            const auto action = Advance(counter, true, false, budget);
            if (work % budget == 0) {
                REQUIRE(action == Action::Flush);
                REQUIRE(counter == 0);
            } else {
                REQUIRE(action == ((work % budget) % 32 == 0 ? Action::Dispatch : Action::None));
                REQUIRE(counter == work % budget);
            }
        }
    }
}

TEST_CASE("Legacy and Android submission schedules remain unchanged",
          "[video_core][submission_policy]") {
    for (const bool android : {false, true}) {
        for (const bool native : {false, true}) {
            if (native && !android) {
                continue;
            }
            for (const u32 budget : {1U, 128U, 129U, 4096U}) {
                CAPTURE(android, native, budget);
                u32 counter{};
                u32 original_counter{};
                u32 submissions{};
                const u32 check_mask = android ? 3 : 31;
                const u32 threshold = android ? 512 : 4096;
                const u32 cadence = android ? 515 : 4127;
                for (u32 work = 1; work <= cadence * 3; ++work) {
                    Action expected = Action::None;
                    if ((++original_counter & check_mask) == check_mask) {
                        if (original_counter < threshold) {
                            expected = Action::Dispatch;
                        } else {
                            expected = Action::Flush;
                            original_counter = 0;
                        }
                    }
                    const auto action = Advance(counter, native, android, budget);
                    REQUIRE(action == expected);
                    REQUIRE(counter == original_counter);
                    REQUIRE((action == Action::Flush) == (work % cadence == 0));
                    submissions += action == Action::Flush;
                }
                REQUIRE(submissions == 3);
            }
        }
    }
}

TEST_CASE("Submission budget parsing accepts only positive complete u32 decimals",
          "[video_core][submission_policy]") {
    REQUIRE(ParseBudget("1") == 1U);
    REQUIRE(ParseBudget("31") == 31U);
    REQUIRE(ParseBudget("128") == 128U);
    REQUIRE(ParseBudget("000129") == 129U);
    REQUIRE(ParseBudget("4294967295") == std::numeric_limits<u32>::max());
    for (const auto value : {"", "0", "00", "-1", "+128", " 128", "128 ", "128\n", "1.5", "1e2",
                             "0x80", "128garbage", "4294967296", "999999999999999999999999999"}) {
        CAPTURE(value);
        REQUIRE_FALSE(ParseBudget(value));
    }
    REQUIRE_FALSE(ParseBudget(std::string_view{"128\0junk", 8}));
}

TEST_CASE("Resetting the work counter restarts the submission budget",
          "[video_core][submission_policy]") {
    u32 counter{};
    for (u32 work = 1; work < 128; ++work) {
        REQUIRE(Advance(counter, true, false, 128) != Action::Flush);
    }
    counter = 0;
    for (u32 work = 1; work < 128; ++work) {
        REQUIRE(Advance(counter, true, false, 128) != Action::Flush);
    }
    REQUIRE(Advance(counter, true, false, 128) == Action::Flush);
    REQUIRE(counter == 0);
    REQUIRE(Advance(counter, true, false, 128) == Action::None);
    REQUIRE(counter == 1);
}

TEST_CASE("A maximum native budget flushes before the counter wraps",
          "[video_core][submission_policy]") {
    constexpr u32 maximum = std::numeric_limits<u32>::max();
    u32 counter = maximum - 1;
    REQUIRE(Advance(counter, true, false, maximum) == Action::Flush);
    REQUIRE(counter == 0);
    REQUIRE(Advance(counter, true, false, maximum) == Action::None);
    REQUIRE(counter == 1);
}

TEST_CASE("Changing the active native policy preserves recorded work",
          "[video_core][submission_policy]") {
    u32 counter = 200;
    REQUIRE(Advance(counter, true, false, 128) == Action::Flush);
    REQUIRE(counter == 0);
    for (u32 work = 1; work < 31; ++work) {
        REQUIRE(Advance(counter, false, false) == Action::None);
    }
    REQUIRE(Advance(counter, false, false) == Action::Dispatch);
    REQUIRE(counter == 31);
    REQUIRE(Advance(counter, true, false, 128) == Action::Dispatch);
    REQUIRE(counter == 32);
}
