// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>

#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"

namespace {

constexpr std::array<char, 8> Magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
constexpr std::array<char, 8> OtherMagic{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("eden-pipeline-cache-files-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void WriteCache(const std::filesystem::path& file, const std::array<char, 8>& magic, u32 version,
                std::string_view body) {
    std::ofstream stream(file, std::ios::binary);
    stream.write(magic.data(), magic.size());
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    stream.write(body.data(), static_cast<std::streamsize>(body.size()));
}

std::string ReadAll(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class CapturedEnvironment final : public VideoCommon::GenericEnvironment {
public:
    CapturedEnvironment(Shader::Stage stage_, u64 instruction) {
        stage = stage_;
        cached_lowest = cached_highest = read_lowest = read_highest = 0;
        code = {instruction};
    }

    u32 ReadCbufValue(u32 bank, u32 offset) override {
        return cbuf_values.at((static_cast<u64>(bank) << 32) | offset);
    }
    Shader::TextureType ReadTextureType(u32 handle) override {
        return texture_types.at(handle);
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override {
        return texture_pixel_formats.at(handle);
    }
    bool IsTexturePixelFormatInteger(u32) override {
        throw std::logic_error("Serialization fixture does not query pixel formats");
    }
    u32 ReadViewportTransformState() override {
        return viewport_transform_state;
    }
    bool HasHLEMacroState() const override {
        return !cbuf_replacements.empty();
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        const auto it = cbuf_replacements.find((static_cast<u64>(bank) << 32) | offset);
        return it == cbuf_replacements.end() ? std::nullopt : std::optional{it->second};
    }
};

} // Anonymous namespace

TEST_CASE("Pipeline cache names are version-suffixed per build", "[pipeline_cache_files]") {
    const std::filesystem::path dir{"shader"};
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39).filename() ==
          "vulkan.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan_pipelines", 39).filename() ==
          "vulkan_pipelines.fork-v39.bin");
    CHECK(VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 18) !=
          VideoCommon::VersionedPipelineCachePath(dir, "vulkan", 39));
}

TEST_CASE("A legacy cache of this version is adopted with its contents",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "pipelines");
    const std::string before = ReadAll(legacy);
    REQUIRE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK_FALSE(std::filesystem::exists(legacy));
    CHECK(ReadAll(versioned) == before);
    // Idempotent: nothing left to adopt.
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
}

TEST_CASE("Another build's legacy cache is never touched", "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 18, "official");
    const std::string before = ReadAll(legacy);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(legacy) == before);
    CHECK_FALSE(std::filesystem::exists(versioned));

    WriteCache(legacy, OtherMagic, 39, "wrong magic");
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));

    // Truncated header.
    {
        std::ofstream stream(legacy, std::ios::binary | std::ios::trunc);
        stream.write(Magic.data(), 4);
    }
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(std::filesystem::exists(legacy));
}

TEST_CASE("An existing versioned cache is never overwritten by a legacy one",
          "[pipeline_cache_files]") {
    TempDir temp;
    const auto legacy = temp.path / "vulkan.bin";
    const auto versioned = VideoCommon::VersionedPipelineCachePath(temp.path, "vulkan", 39);
    WriteCache(legacy, Magic, 39, "older");
    WriteCache(versioned, Magic, 39, "current");
    const std::string current = ReadAll(versioned);
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(legacy, versioned, Magic, 39));
    CHECK(ReadAll(versioned) == current);
    CHECK(std::filesystem::exists(legacy));
    CHECK_FALSE(VideoCommon::AdoptLegacyPipelineCache(temp.path / "missing.bin", versioned,
                                                      Magic, 39));
}

TEST_CASE("Cached compute failures retain their cause and the current dispatch address",
          "[pipeline_cache][diagnostics]") {
    const Vulkan::ComputePipelineCacheKey key{
        .unique_hash = 0x1234,
        .shared_memory_size = 1024,
        .workgroup_size = {32, 1, 1},
    };
    std::unordered_map<Vulkan::ComputePipelineCacheKey, Vulkan::ComputePipelineCacheEntry> cache;
    const std::string cause =
        "SPIR-V emission failed at IR opcode UndefU8: SPIR-V Instruction is not implemented";
    cache[key].failure = cause;
    for (const GPUVAddr address : {0x211ddd9000ULL, 0x21ebdfa000ULL}) {
        const auto [entry, inserted] = cache.try_emplace(key);
        REQUIRE_FALSE(inserted);
        CHECK(entry->second.Get(false, address, key) == nullptr);
        try {
            static_cast<void>(entry->second.Get(true, address, key));
            FAIL("Native compute failure was not propagated");
        } catch (const Shader::RuntimeError& exception) {
            const std::string message{exception.what()};
            CHECK(message.find(cause) != std::string::npos);
            CHECK(message.find(fmt::format("address={:#x}", address)) != std::string::npos);
            CHECK(message.find(fmt::format("shader={:#016x}", key.unique_hash)) != std::string::npos);
            CHECK(message.find(fmt::format("pipeline={:#016x}", key.Hash())) != std::string::npos);
        }
        CHECK(entry->second.failure == cause);
        CHECK_FALSE(entry->second.pipeline);
    }
    auto other = key;
    other.workgroup_size[0] = 64;
    const auto [entry, inserted] = cache.try_emplace(other);
    REQUIRE(inserted);
    CHECK(entry->second.failure.empty());
    CHECK(entry->second.Get(false, 0x2120633c00, other) == nullptr);
    REQUIRE_THROWS_AS(entry->second.Get(true, 0x2120633c00, other), Shader::RuntimeError);
}

TEST_CASE("Pipeline preload selection ranks eligible records before imposing its bound",
          "[pipeline_cache][preload]") {
    VideoCommon::PipelinePreloadSelection selection{3000};
    for (u64 hash = 0; hash < 4000; ++hash) {
        selection.Add(15, hash, std::streampos{static_cast<std::streamoff>(hash)});
    }
    for (u64 hash = 4000; hash < 7000; ++hash) {
        selection.Add(0, hash, std::streampos{static_cast<std::streamoff>(hash)});
    }
    const auto records = selection.OrderedRecords();
    REQUIRE(records.size() == 3000);
    for (size_t i = 0; i < records.size(); ++i) {
        CHECK(records[i].age == 0);
        CHECK(records[i].hash == 4000 + i);
        CHECK(records[i].position == std::streampos{static_cast<std::streamoff>(4000 + i)});
    }
    VideoCommon::PipelinePreloadSelection disabled{0};
    disabled.Add(0, 1, std::streampos{12});
    CHECK(disabled.OrderedRecords().empty());
}

TEST_CASE("Pipeline record positions reread only selected environments in recency order",
          "[pipeline_cache][preload]") {
    TempDir temp;
    const auto cache = temp.path / "pipelines.bin";
    const CapturedEnvironment vertex{Shader::Stage::VertexA, 0x123456789abcdef0};
    const CapturedEnvironment fragment{Shader::Stage::Fragment, 0x1122334455667788};
    const CapturedEnvironment compute{Shader::Stage::Compute, 0x8877665544332211};
    const CapturedEnvironment task{Shader::Stage::Task, 0x1111};
    const std::array<const VideoCommon::GenericEnvironment*, 2> graphics{&vertex, &fragment};
    const std::array<const VideoCommon::GenericEnvironment*, 1> compute_envs{&compute};
    const std::array<const VideoCommon::GenericEnvironment*, 1> task_envs{&task};
    VideoCommon::SerializePipeline(u64{10}, graphics, cache, 7);
    VideoCommon::SerializePipeline(u64{99}, compute_envs, cache, 7);
    VideoCommon::SerializePipeline(u64{11}, graphics, cache, 7);
    VideoCommon::SerializePipeline(u64{12}, task_envs, cache, 7);
    VideoCommon::SerializePipeline(u64{13}, graphics, cache, 7);

    VideoCommon::PipelinePreloadSelection selection{2};
    const std::unordered_map<u64, u8> usage{{10, 9}, {11, 2}, {12, 0}, {13, 1}};
    size_t compute_count{};
    size_t graphics_count{};
    VideoCommon::LoadPipelines(
        {}, cache, 7,
        [&](std::ifstream& file, VideoCommon::FileEnvironment env) {
            u64 key{};
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            CHECK(key == 99);
            CHECK(env.ReadInstruction(0) == 0x8877665544332211);
            ++compute_count;
        },
        [&](std::ifstream& file, std::vector<VideoCommon::FileEnvironment> envs,
            std::streampos position) {
            u64 key{};
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            ++graphics_count;
            if (envs.front().ShaderStage() != Shader::Stage::Task) {
                selection.Add(usage.at(key), key, position);
            }
        });
    REQUIRE(compute_count == 1);
    REQUIRE(graphics_count == 4);
    const auto records = selection.OrderedRecords();
    REQUIRE(records.size() == 2);
    CHECK(records[0].hash == 13);
    CHECK(records[1].hash == 11);
    std::ifstream file(cache, std::ios::binary);
    file.exceptions(std::ifstream::failbit);
    for (const auto& record : records) {
        file.setstate(std::ifstream::eofbit);
        file.clear();
        file.seekg(record.position);
        auto envs = VideoCommon::DeserializePipelineEnvironments(file);
        u64 key{};
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        REQUIRE(key == record.hash);
        REQUIRE(envs.size() == 2);
        CHECK(envs[0].ShaderStage() == Shader::Stage::VertexA);
        CHECK(envs[1].ShaderStage() == Shader::Stage::Fragment);
        CHECK(envs[0].ReadInstruction(0) == 0x123456789abcdef0);
        CHECK(envs[1].ReadInstruction(0) == 0x1122334455667788);
    }
}

TEST_CASE("Pipeline loading retains cancellation and truncated-record handling",
          "[pipeline_cache][preload]") {
    TempDir temp;
    const auto cache = temp.path / "pipelines.bin";
    const CapturedEnvironment vertex{Shader::Stage::VertexA, 0x1234};
    const std::array<const VideoCommon::GenericEnvironment*, 1> envs{&vertex};
    VideoCommon::SerializePipeline(u64{1}, envs, cache, 7);
    VideoCommon::SerializePipeline(u64{2}, envs, cache, 7);
    std::stop_source stop;
    size_t count{};
    VideoCommon::LoadPipelines(
        stop.get_token(), cache, 7,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [&](std::ifstream& file, std::vector<VideoCommon::FileEnvironment>, std::streampos) {
            u64 key{};
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            ++count;
            stop.request_stop();
        });
    CHECK(count == 1);
    CHECK(std::filesystem::exists(cache));
    std::filesystem::resize_file(cache, std::filesystem::file_size(cache) - 1);
    VideoCommon::LoadPipelines(
        {}, cache, 7,
        [](std::ifstream&, VideoCommon::FileEnvironment) { FAIL("Unexpected compute record"); },
        [](std::ifstream& file, std::vector<VideoCommon::FileEnvironment>, std::streampos) {
            u64 key{};
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
        });
    CHECK_FALSE(std::filesystem::exists(cache));
}
