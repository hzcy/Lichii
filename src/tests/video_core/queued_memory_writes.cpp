// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_split.h"

namespace {
// Two address spaces (only their addresses matter).
int space_a = 0;
int space_b = 1;
} // Anonymous namespace

TEST_CASE("Queued memory writes wait only for work on the range read",
          "[video_core][gpu_split][queued_memory_writes]") {
    VideoCommon::QueuedMemoryWrites writes;
    CHECK(writes.Target(&space_a, 0x10000, 64) == 0);

    // A constant buffer update at position 10, a semaphore release far away at 20.
    writes.Note(&space_a, 0x10000, 0x1000, 10, 0);
    writes.Note(&space_a, 0x900000, 16, 20, 0);
    CHECK(writes.Target(&space_a, 0x10800, 4) == 10);
    CHECK(writes.Target(&space_a, 0x900008, 4) == 20);
    CHECK(writes.Target(&space_a, 0x400000, 64) == 0);

    // The same buffer again only moves its position.
    writes.Note(&space_a, 0x10000, 0x1000, 30, 0);
    CHECK(writes.Target(&space_a, 0x10000, 4) == 30);

    // Another address space may alias anything.
    CHECK(writes.Target(&space_b, 0x400000, 64) == 30);

    // Work without a known range bounds every read.
    writes.NoteUnbounded(40);
    CHECK(writes.Target(&space_a, 0x400000, 64) == 40);
    writes.NoteUnbounded(35); // never moves back
    CHECK(writes.Target(&space_a, 0x400000, 64) == 40);
}

TEST_CASE("Queued memory writes merge neighbors and drop what the back passed",
          "[video_core][gpu_split][queued_memory_writes]") {
    VideoCommon::QueuedMemoryWrites writes;
    // Consecutive ring pieces merge into one range.
    writes.Note(&space_a, 0x20000, 0x100, 5, 0);
    writes.Note(&space_a, 0x20100, 0x100, 6, 0);
    CHECK(writes.Target(&space_a, 0x20000, 4) == 6);
    CHECK(writes.Target(&space_a, 0x201fc, 4) == 6);

    // More ranges than slots: the oldest becomes unbounded, nothing is lost.
    for (u64 i = 0; i < VideoCommon::QueuedMemoryWrites::MaxRanges + 2; ++i) {
        writes.Note(&space_a, 0x1000000 * (i + 1), 16, 10 + i, 0);
    }
    for (u64 i = 0; i < VideoCommon::QueuedMemoryWrites::MaxRanges + 2; ++i) {
        CHECK(writes.Target(&space_a, 0x1000000 * (i + 1), 4) >= 10 + i);
    }
    CHECK(writes.Target(&space_a, 0x20000, 4) >= 6);

    // Once the back passed everything, a new write starts clean.
    VideoCommon::QueuedMemoryWrites fresh;
    fresh.Note(&space_a, 0x30000, 16, 3, 0);
    fresh.Note(&space_a, 0x50000, 16, 8, 3);
    CHECK(fresh.Target(&space_a, 0x30000, 4) == 0);
    CHECK(fresh.Target(&space_a, 0x50000, 4) == 8);
}
