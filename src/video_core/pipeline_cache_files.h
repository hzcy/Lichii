// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <system_error>
#include <span>
#include <tuple>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"

namespace VideoCommon {

inline constexpr u32 OpenGLPipelineCacheVersion = 48;
inline constexpr u32 VulkanPipelineCacheVersion = 52;

// Official Eglantine (cache version 18 at c0cb3325) and this fork (a different version) both
// used <shader dir>/<title>/vulkan.bin and vulkan_pipelines.bin. Each build deletes a file
// with another version at boot, so alternating builds made every run start with no pipelines
// (MKW log: "Deleting old pipeline cache", "Total Pipeline Count: 0", then hundreds of
// pipelines built on the GPU thread). Version-suffixed names let the builds coexist.
[[nodiscard]] inline std::filesystem::path VersionedPipelineCachePath(
    const std::filesystem::path& directory, std::string_view stem, u32 version) {
    return directory / fmt::format("{}.fork-v{}.bin", stem, version);
}

/// True when the file starts with this magic number followed by this cache version.
[[nodiscard]] inline bool PipelineCacheHeaderMatches(const std::filesystem::path& file,
                                                     const std::array<char, 8>& magic,
                                                     u32 version) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    std::array<char, 8> file_magic{};
    u32 file_version{};
    stream.read(file_magic.data(), file_magic.size());
    stream.read(reinterpret_cast<char*>(&file_version), sizeof(file_version));
    return stream.good() && file_magic == magic && file_version == version;
}

/// Moves a legacy-named cache that this exact cache version wrote to its versioned name, so an
/// existing warm cache survives the rename. A file of any other version (another build's) is
/// never touched; neither is an existing versioned file.
inline bool AdoptLegacyPipelineCache(const std::filesystem::path& legacy,
                                     const std::filesystem::path& versioned,
                                     const std::array<char, 8>& magic, u32 version) {
    std::error_code error;
    if (std::filesystem::exists(versioned, error) || error) {
        return false;
    }
    if (!PipelineCacheHeaderMatches(legacy, magic, version)) {
        return false;
    }
    std::filesystem::rename(legacy, versioned, error);
    return !error;
}

/// Keeps only bounded record metadata; smaller usage ages are more recent.
class PipelinePreloadSelection {
public:
    struct Record {
        u8 age;
        u64 hash;
        std::streampos position;
    };

    explicit PipelinePreloadSelection(size_t limit_) : limit{limit_} {}

    void Add(u8 age, u64 hash, std::streampos position) {
        if (limit == 0) {
            return;
        }
        const Record record{age, hash, position};
        if (records.size() == limit) {
            if (!MoreRecent(record, records.front())) {
                return;
            }
            std::pop_heap(records.begin(), records.end(), MoreRecent);
            records.back() = record;
        } else {
            records.push_back(record);
        }
        std::push_heap(records.begin(), records.end(), MoreRecent);
    }

    /// Consumes the heap ordering after selection, returning newest records first.
    std::span<const Record> OrderedRecords() {
        std::sort_heap(records.begin(), records.end(), MoreRecent);
        return records;
    }

private:
    static bool MoreRecent(const Record& left, const Record& right) {
        return std::tie(left.age, left.hash) < std::tie(right.age, right.hash);
    }

    size_t limit;
    std::vector<Record> records;
};

} // namespace VideoCommon
