// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>
#include <bit>
#include <cstring>
#include <numeric>
#include <optional>
#include <unordered_set>
#include "common/cityhash.h"
#include "common/cpu_features.h"
#include "common/scope_exit.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_perf.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_shader_emission.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vertex_repack.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = VideoCommon::VulkanPipelineCacheVersion;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        if (device.IsMoltenVK()) {
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

std::array<std::atomic<std::chrono::nanoseconds::rep>, 2> runtime_spirv_ns{};

void AddRuntimeSpirvTime(size_t thread, std::chrono::steady_clock::time_point begin) {
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    runtime_spirv_ns[thread].fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
        std::memory_order::relaxed);
}

vk::ShaderModule EmitShaderModule(const Device& device, const Shader::Profile& profile,
                                 const Shader::RuntimeInfo& runtime_info,
                                 Shader::IR::Program& program, Shader::Backend::Bindings& binding,
                                 u64 shader_hash, size_t stage_index,
                                 std::span<const u32> reference = {}) {
    const auto code =
        EmitPipelineShader(profile, runtime_info, program, binding, shader_hash, reference);
    device.SaveShader(code);
    vk::ShaderModule module{BuildShader(device, code)};

    // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
    // is independent and gated only by gpu_log_shader_dumps.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry",
                                               "fragment"};
        const std::string shader_name =
            fmt::format("shader_{:016x}_{}", shader_hash, stage_names[stage_index]);
        if (should_log) {
            const std::string shader_info = fmt::format(
                "SPIR-V size: {} bytes, hash: {:016x}", code.size() * sizeof(u32), shader_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(shader_hash,
                                         std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const std::string name{fmt::format("Shader {:016x}", shader_hash)};
        module.SetObjectNameEXT(name.c_str());
    }
    return module;
}

struct DeferredStages {
    std::unique_ptr<EmissionBudget::Reservation> reservation;
    std::shared_ptr<ShaderPools> pools;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;
};

size_t GetTotalPipelineWorkers() {
    const size_t logical_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL);
    const size_t max_core_threads = logical_threads - 1ULL;
    if (const char* const env = std::getenv("EDEN_NX2_PIPELINE_WORKERS"); env && *env) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0) {
            return std::min<size_t>(static_cast<size_t>(parsed), max_core_threads);
        }
    }
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    if (const auto physical_cores = Common::GetProcessorCount();
        physical_cores && static_cast<size_t>(*physical_cores) >= logical_threads &&
        logical_threads <= 8ULL) {
        return std::max<size_t>(logical_threads / 2ULL, 2ULL);
    }
    return max_core_threads;
#endif
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    if (unique_hashes != rhs.unique_hashes) {
        return false;
    }
    return std::memcmp(reinterpret_cast<const char*>(this) + sizeof(unique_hashes),
                       reinterpret_cast<const char*>(&rhs) + sizeof(unique_hashes),
                       state.Size()) == 0;
}

namespace {
// NVIDIA's and AMD's drivers keep a disk cache of built pipelines of their own. Eden's copy on top
// only kept every pipeline the title ever built in host memory (title 1: a 245 MB file, ~0.35 GB
// of RAM for the session) and copied it whole at each save, which stalled frames; pipelines built
// no slower without it.
bool DriverCachesPipelines(const Device& device) {
    switch (device.GetDriverID()) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return true;
    default:
        return false;
    }
}

/// EDEN_NX2_PIPELINE_ASYNC=1 (default off; Vulkan, and only with asynchronous shaders on): a
/// graphics pipeline missing from the cache is translated (TranslateProgram, EmitSPIRV) and built
/// on the pipeline workers instead of on the GPU thread, where one took 100-3,500 ms. Its draws
/// are skipped meanwhile where CanDiscardPendingDraw allows it; a draw that may not be skipped
/// gets the pipeline on the GPU thread as before (the queued build run there, or waited for if a
/// worker has it). What the translation reads besides the program code (constant buffers, the
/// texture header pool, a few registers) is copied when the build is queued
/// (GraphicsEnvironment::SnapshotForAsync), so workers never read Maxwell3D or the memory
/// manager; a read the copy cannot answer sends the pipeline back to the GPU thread at its next
/// draw. EDEN_NX2_PIPELINE_ASYNC=2: as 1, and native (Switch 2) draws, which CanDiscardPendingDraw
/// never skips (they can leave resources later draws use), follow the rule of the others: without
/// it a native title's pipelines are all still built on the GPU thread.
u32 AsyncPipelinesMode() {
    static const u32 mode = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_ASYNC");
        if (value == nullptr || value[0] == '\0' || value[0] == '0') {
            return 0U;
        }
        return value[0] == '2' ? 2U : 1U;
    }();
    return mode;
}
} // Anonymous namespace

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue() &&
                                !DriverCachesPipelines(device_)},
      num_pipeline_workers{device.HasBrokenParallelShaderCompiling() ? 1ULL
                                                                   : GetTotalPipelineWorkers()},
      emission_budget{num_pipeline_workers},
      workers(num_pipeline_workers,
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {},
                           Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_byte_addressed_r16 = true,
    };
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();
    dynamic_features.has_static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();

    if (AsyncPipelinesMode() != 0) {
        if (use_asynchronous_shaders) {
            async_pipelines = AsyncPipelinesMode();
            async_stats.window = std::chrono::steady_clock::now();
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PIPELINE_ASYNC={}: graphics pipelines are translated and built on "
                     "the pipeline workers{}",
                     async_pipelines,
                     async_pipelines == 2 ? ", native draws skipped while theirs is pending" : "");
        } else {
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PIPELINE_ASYNC is ignored: asynchronous shaders are off");
        }
    }
}

namespace {
// <title>/vulkan_used.bin: the graphics pipelines recent sessions used, as key hash + sessions
// since the last use.
constexpr u32 UsedPipelinesMagic = 0x31505545; // "EUP1"
// Retain recent pipeline usage across launches to avoid rebuilding pipelines during gameplay.
// Each pipeline holds ~0.45 MB of driver memory, so startup compilation is capped separately.
// MoltenVK keeps the previous eight sessions: a pipeline built while playing goes through
// SPIR-V -> MSL -> Metal, far more than the 2-3 ms the shorter window assumes, so each one
// built at boot is a stutter avoided.
// Ocarina still built pipelines for 1-2 s of every 10 s early in play with two sessions kept:
// keep sixteen (#161).
constexpr u8 UsedPipelinesMaxAge = 16;
constexpr u8 UsedPipelinesMaxAgeMoltenVK = 8;

u8 MaxUsedPipelineAge(const Device& device) {
    if (device.IsMoltenVK()) {
        return UsedPipelinesMaxAgeMoltenVK;
    }
    return UsedPipelinesMaxAge;
}

// Cap startup compilation even with a used list; compile the rest on first use. A cache that grew
// over many builds held 48k pipelines, ~21 GB of driver memory before the game started.
constexpr size_t MaxPreloadedGraphicsPipelines = 3000;

/// Loads recent pipeline hashes retained for fewer than `max_age` sessions.
std::unordered_map<u64, u8> LoadUsedPipelines(const std::filesystem::path& path, u8 max_age) {
    std::unordered_map<u64, u8> used;
    std::ifstream file{path, std::ios::binary};
    u32 magic{}, count{};
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != UsedPipelinesMagic ||
        !file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return used;
    }
    used.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        if (!file.read(reinterpret_cast<char*>(&hash), sizeof(hash)) ||
            !file.read(reinterpret_cast<char*>(&age), sizeof(age))) {
            break;
        }
        if (age < max_age) { // lists saved with a longer window
            used.emplace(hash, age);
        }
    }
    return used;
}

// EDEN_NX2_PRELOAD_ALL_PIPELINES=1 builds every cached pipeline at boot, as before.
bool PreloadAllPipelines() {
    static const bool all = [] {
        const char* value = std::getenv("EDEN_NX2_PRELOAD_ALL_PIPELINES");
        return value != nullptr && value[0] == '1';
    }();
    return all;
}
} // Anonymous namespace

PipelineCache::~PipelineCache() {
    if (async_pipelines != 0) {
        // Queued builds return without starting; running ones finish before anything they use
        // is destroyed.
        std::unique_lock lock{async_mutex};
        async_stopping = true;
        async_idle.wait(lock, [this] { return async_running == 0; });
    }
    workers.WaitForRequests();
    serialization_thread.WaitForRequests();
    SaveUsedPipelines();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
    graphics_cache.clear();
    evicted_pipelines.clear();
    // EDEN_NX2_PIPELINE_ASYNC: pipelines of finished jobs not drained yet, likewise.
    async_pending.clear();
    async_completed.clear();
}

void PipelineCache::SaveUsedPipelines() {
    used_pipelines_saved = std::chrono::steady_clock::now();
    // A session too short to say what the game uses (a boot to the menu) keeps the old list.
    constexpr size_t MinimumUsed = 64;
    if (used_pipelines_filename.empty() ||
        used_this_session.size() + used_and_evicted.size() < MinimumUsed) {
        return;
    }
    std::unordered_map<u64, u8> next;
    next.reserve(used_pipelines.size() + used_this_session.size());
    for (const auto& [key, pipeline] : graphics_cache) {
        if (used_this_session.contains(pipeline.get())) {
            next.emplace(key.Hash(), u8{0});
        }
    }
    for (const u64 hash : used_and_evicted) {
        next.emplace(hash, u8{0});
    }
    for (const auto& [hash, age] : used_pipelines) {
        if (age + 1 < MaxUsedPipelineAge(device)) {
            next.try_emplace(hash, static_cast<u8>(age + 1));
        }
    }
    std::ofstream file{used_pipelines_filename, std::ios::binary | std::ios::trunc};
    const u32 count = static_cast<u32>(next.size());
    file.write(reinterpret_cast<const char*>(&UsedPipelinesMagic), sizeof(UsedPipelinesMagic));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [hash, age] : next) {
        file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.write(reinterpret_cast<const char*>(&age), sizeof(age));
    }
}

namespace {
// EDEN_NX2_PIPELINE_CLEAN=1 (default off): see CurrentGraphicsPipelineClean.
bool PipelineCleanEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_CLEAN");
        return value != nullptr && value[0] != '0';
    }();
    return enabled;
}
} // Anonymous namespace

void PipelineCache::BuildCleanWatch() {
    using Range = CleanLookup::Range;
    const Maxwell& regs = maxwell3d->regs;
    const DynamicFeatures& features = dynamic_features;
    const auto watch_into = [&regs](std::vector<Range>& ranges, const auto& field) {
        const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&field) -
                                                reinterpret_cast<const char*>(&regs));
        const size_t first = offset / sizeof(u32);
        const size_t last = (offset + sizeof(field) + sizeof(u32) - 1) / sizeof(u32);
        ranges.push_back({static_cast<u32>(first), static_cast<u32>(last - first)});
    };
    std::vector<Range> ranges;
    const auto watch = [&](const auto& field) { watch_into(ranges, field); };

    // Every register FixedPipelineState::Refresh (and what it calls) reads with these features.
    // Refresh itself, always:
    watch(regs.transform_feedback_enabled);
    watch(regs.depth_mode);
    watch(regs.polygon_mode_front);
    watch(regs.tessellation);
    watch(regs.patch_vertices);
    watch(regs.anti_alias_samples_mode);
    watch(regs.blend);                    // ComputeAttachment0DualSourceBlend
    watch(regs.blend_per_target_enabled); // ComputeAttachment0DualSourceBlend
    watch(regs.blend_per_target[0]);      // ComputeAttachment0DualSourceBlend
    watch(regs.alpha_test_enabled);
    watch(regs.alpha_test_func);
    watch(regs.alpha_test_ref);
    watch(regs.mandated_early_z);
    watch(regs.zeta_enable);
    watch(regs.zeta.format);
    watch(regs.window_origin);
    watch(regs.provoking_vertex);
    watch(regs.conservative_raster_enable);
    watch(regs.line_anti_alias_enable);
    watch(regs.anti_alias_alpha_control);
    watch(regs.depth_bounds);
    watch(regs.line_stipple_params);
    for (const auto& rt : regs.rt) {
        watch(rt.format);
    }
    watch(regs.point_size);
    // Dirty::VertexInput (both branches) and the static vertex repack:
    watch(regs.vertex_attrib_format);
    const bool repack = !features.has_dynamic_vertex_input && features.has_static_vertex_repack;
    if (!features.has_dynamic_vertex_input) {
        watch(regs.vertex_stream_instances);
        for (const auto& stream : regs.vertex_streams) {
            watch(stream.frequency);
        }
    }
    if (!features.has_extended_dynamic_state || repack) {
        // Strides (DynamicState::Refresh's vertex_strides, the repack): the first word of each
        // stream, VertexStream's leading union of stride and enable.
        for (const auto& stream : regs.vertex_streams) {
            const auto offset = static_cast<size_t>(reinterpret_cast<const char*>(&stream) -
                                                    reinterpret_cast<const char*>(&regs));
            ranges.push_back({static_cast<u32>(offset / sizeof(u32)), 1});
        }
    }
    // Dirty::ViewportSwizzles:
    for (const auto& viewport : regs.viewport_transform) {
        watch(viewport.swizzle);
    }
    // DynamicState::Refresh (no extended dynamic state):
    if (!features.has_extended_dynamic_state) {
        watch(regs.gl_front_face);
        watch(regs.stencil_front_op);
        watch(regs.stencil_two_side_enable);
        watch(regs.stencil_back_op);
        watch(regs.stencil_enable);
        watch(regs.depth_write_enabled);
        watch(regs.depth_bounds_enable);
        watch(regs.depth_test_enable);
        watch(regs.depth_test_func);
        watch(regs.gl_cull_face);
        watch(regs.gl_cull_test_enabled);
    }
    // DynamicState::Refresh2:
    if (!features.has_extended_dynamic_state_2_logic_op) {
        watch(regs.logic_op);
        if (!features.has_extended_dynamic_state_2) {
            watch(regs.polygon_offset_point_enable);
            watch(regs.polygon_offset_line_enable);
            watch(regs.polygon_offset_fill_enable);
            watch(regs.rasterize_enable);
            watch(regs.primitive_restart);
        }
    }
    // Dirty::Blending, BlendingAttachment::Refresh (no extended dynamic state 3 blending):
    if (!features.has_extended_dynamic_state_3_blend) {
        watch(regs.color_mask_common);
        watch(regs.color_mask);
        watch(regs.blend_per_target);
        watch(regs.iterated_blend);
    }
    // DynamicState::Refresh3:
    if (!features.has_dynamic_state3_logic_op_enable) {
        watch(regs.logic_op);
    }
    if (!features.has_dynamic_state3_depth_clamp_enable) {
        watch(regs.viewport_clip_control);
    }
    if (!features.has_dynamic_state3_line_stipple_enable) {
        watch(regs.line_stipple_enable);
    }

    // Sorted and merged: contiguous words compare as one block.
    const auto merge = [](std::vector<Range> input) {
        std::ranges::sort(input, {}, &Range::first);
        std::vector<Range> merged;
        for (const Range& range : input) {
            if (!merged.empty() && range.first <= merged.back().first + merged.back().count) {
                const u32 end = (std::max)(merged.back().first + merged.back().count,
                                           range.first + range.count);
                merged.back().count = end - merged.back().first;
            } else {
                merged.push_back(range);
            }
        }
        return merged;
    };
    clean.ranges = merge(std::move(ranges));
    // RefreshXfbState, while transform_feedback_enabled (watched above) is set:
    std::vector<Range> xfb_ranges;
    watch_into(xfb_ranges, regs.transform_feedback.controls);
    watch_into(xfb_ranges, regs.stream_out_layout);
    clean.xfb_ranges = merge(std::move(xfb_ranges));

    size_t words = 0;
    for (const Range& range : clean.ranges) {
        words += range.count;
    }
    clean.snapshot.assign(words, 0);
    size_t xfb_words = 0;
    for (const Range& range : clean.xfb_ranges) {
        xfb_words += range.count;
    }
    clean.xfb_snapshot.assign(xfb_words, 0);
    clean.built = true;
    LOG_INFO(Render_Vulkan, "EDEN_NX2_PIPELINE_CLEAN: watching {} register words in {} ranges",
             words, clean.ranges.size());
}

bool PipelineCache::CleanInputsUnchanged() const {
    const u32* const words = maxwell3d->regs.reg_array.data();
    const u32* saved = clean.snapshot.data();
    for (const CleanLookup::Range& range : clean.ranges) {
        if (std::memcmp(words + range.first, saved, range.count * sizeof(u32)) != 0) {
            return false;
        }
        saved += range.count;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        for (const CleanLookup::Range& range : clean.xfb_ranges) {
            if (std::memcmp(words + range.first, saved, range.count * sizeof(u32)) != 0) {
                return false;
            }
            saved += range.count;
        }
    }
    return true;
}

void PipelineCache::TakeCleanSnapshot() {
    const u32* const words = maxwell3d->regs.reg_array.data();
    u32* saved = clean.snapshot.data();
    for (const CleanLookup::Range& range : clean.ranges) {
        std::memcpy(saved, words + range.first, range.count * sizeof(u32));
        saved += range.count;
    }
    if (maxwell3d->regs.transform_feedback_enabled != 0) {
        saved = clean.xfb_snapshot.data();
        for (const CleanLookup::Range& range : clean.xfb_ranges) {
            std::memcpy(saved, words + range.first, range.count * sizeof(u32));
            saved += range.count;
        }
    }
    clean.hashes = graphics_key.unique_hashes;
    clean.maxwell3d = maxwell3d;
    clean.topology = maxwell3d->draw_manager.draw_state.topology;
    clean.engine_state = maxwell3d->engine_state;
}

// The key is (unique_hashes, FixedPipelineState). Refresh's output depends only on: the
// registers in clean.ranges (and clean.xfb_ranges while transform feedback is on), the dirty
// flags VertexInput, ViewportSwizzles and Blending (it recomputes those groups only while set,
// and clears some), dynamic_features (set once at construction), the draw's topology and the
// engine hint. When all of these equal their values at the last Refresh and the flags are
// clear, Refresh would rewrite the same key and Next(graphics_key) would return
// current_pipeline (whose key it is): the lookup returns it as that fast path does.
GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineClean() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        clean.pipeline = nullptr;
        return nullptr;
    }
    if (!clean.built) {
        BuildCleanWatch();
    }
    const auto& flags = maxwell3d->dirty.flags;
    const bool blending_dirty =
        !dynamic_features.has_extended_dynamic_state_3_blend && flags[Dirty::Blending];
    if (clean.pipeline != nullptr && clean.pipeline == current_pipeline &&
        clean.maxwell3d == maxwell3d && !flags[Dirty::VertexInput] &&
        !flags[Dirty::ViewportSwizzles] && !blending_dirty &&
        clean.hashes == graphics_key.unique_hashes &&
        clean.topology == maxwell3d->draw_manager.draw_state.topology &&
        clean.engine_state == maxwell3d->engine_state && CleanInputsUnchanged()) {
        ++clean.hits;
    } else {
        ++clean.misses;
        clean.pipeline = nullptr;
    }
    if (((clean.hits + clean.misses) & 0xfff) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (clean.last_log == std::chrono::steady_clock::time_point{}) {
            clean.last_log = now;
        } else if (now - clean.last_log >= std::chrono::seconds{10}) {
            const double seconds = std::chrono::duration<double>(now - clean.last_log).count();
            LOG_INFO(Render_Vulkan,
                     "EDEN_NX2_PIPELINE_CLEAN per second: lookups without Refresh {}, "
                     "with Refresh {}",
                     static_cast<u64>(static_cast<double>(clean.hits) / seconds),
                     static_cast<u64>(static_cast<double>(clean.misses) / seconds));
            clean.last_log = now;
            clean.hits = 0;
            clean.misses = 0;
        }
    }
    if (clean.pipeline != nullptr) {
        current_pipeline->MarkUsed(frame_count);
        return BuiltPipeline(current_pipeline);
    }

    TakeCleanSnapshot();
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            clean.pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    // The slow path makes graphics_cache[graphics_key] current, unless it has no pipeline.
    GraphicsPipeline* const previous = current_pipeline;
    GraphicsPipeline* const result = CurrentGraphicsPipelineSlowPath();
    if (current_pipeline != previous) {
        clean.pipeline = current_pipeline;
    }
    return result;
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (async_pipelines != 0 && async_completed_count.load(std::memory_order_acquire) != 0) {
        const auto begin = std::chrono::steady_clock::now();
        DrainAsyncPipelines();
        async_stats.gpu_time += std::chrono::steady_clock::now() - begin;
    }
    if (PipelineCleanEnabled()) {
        return CurrentGraphicsPipelineClean();
    }

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            current_pipeline->MarkUsed(frame_count);
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* ComputePipelineCacheEntry::Get(bool native, GPUVAddr address,
                                                const ComputePipelineCacheKey& key) const {
    if (!pipeline && native) {
        throw Shader::RuntimeError(
            "Native compute pipeline unavailable: address={:#x} shader={:#016x} pipeline={:#016x}: {}",
            address, key.unique_hash, key.Hash(),
            failure.empty() ? "No compiler diagnostic was recorded" : failure);
    }
    return pipeline.get();
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& entry{pair->second};
    if (is_new) {
        entry = CreateComputePipeline(key, shader);
    }
    return entry.Get(kepler_compute->IsNativeMode(),
                     kepler_compute->regs.code_loc.Address() + qmd.program_start, key);
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan", CACHE_VERSION);
    used_pipelines_filename =
        VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_used", CACHE_VERSION);
    used_pipelines = LoadUsedPipelines(used_pipelines_filename, MaxUsedPipelineAge(device));
    used_pipelines_saved = std::chrono::steady_clock::now();
    size_t skipped_pipelines = 0;
    size_t queued_graphics = 0;
    const bool preload_all = PreloadAllPipelines();
    const bool rank_by_usage = !preload_all && !used_pipelines.empty();
    VideoCommon::PipelinePreloadSelection selection{rank_by_usage
                                                      ? MaxPreloadedGraphicsPipelines
                                                      : 0};

    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename =
            VideoCommon::VersionedPipelineCachePath(base_dir, "vulkan_pipelines", CACHE_VERSION);
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
    } state;

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        workers.QueueWork([this, key, env_ = std::move(env), &state, &callback]() mutable {
            ShaderPools pools;
            auto entry{CreateComputePipeline(pools, key, env_, state.statistics.get(), false)};
            std::scoped_lock lock{state.mutex};
            if (entry.pipeline) {
                compute_cache.emplace(key, std::move(entry));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto queue_graphics{[&](GraphicsPipelineCacheKey key, std::vector<FileEnvironment> envs) {
        workers.QueueWork([this, key, envs_ = std::move(envs), &state, &callback]() mutable {
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false)};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                graphics_cache.emplace(key, std::move(pipeline));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs,
                                std::streampos record_position) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        const u64 hash = key.Hash();
        disk_pipelines.insert(hash);
        if (rank_by_usage) {
            if (const auto used = used_pipelines.find(hash); used != used_pipelines.end()) {
                selection.Add(used->second, hash, record_position);
            }
            ++skipped_pipelines;
            return;
        }
        if (!preload_all && queued_graphics >= MaxPreloadedGraphicsPipelines) {
            ++skipped_pipelines;
            return;
        }
        ++queued_graphics;
        queue_graphics(key, std::move(envs));
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);

    if (rank_by_usage && !stop_loading.stop_requested()) {
        try {
            std::ifstream file(pipeline_cache_filename, std::ios::binary);
            if (file.is_open()) {
                file.exceptions(std::ifstream::failbit);
                for (const auto& record : selection.OrderedRecords()) {
                    if (stop_loading.stop_requested()) {
                        break;
                    }
                    file.clear();
                    file.seekg(record.position);
                    auto envs = VideoCommon::DeserializePipelineEnvironments(file);
                    GraphicsPipelineCacheKey key;
                    file.read(reinterpret_cast<char*>(&key), sizeof(key));
                    queue_graphics(key, std::move(envs));
                    ++queued_graphics;
                    --skipped_pipelines;
                }
            }
        } catch (const std::ios_base::failure& e) {
            LOG_ERROR(Common_Filesystem, "{}", e.what());
            if (!Common::FS::RemoveFile(pipeline_cache_filename)) {
                LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                          Common::FS::PathToUTF8String(pipeline_cache_filename));
            }
        }
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    if (skipped_pipelines != 0) {
        LOG_INFO(Render_Vulkan,
                 "{} cached graphics pipelines are built when first needed",
                 skipped_pipelines);
    }

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

namespace {
// EDEN_NX2_PIPELINE_DIFF=1: for a new graphics pipeline whose shaders an earlier one uses too,
// log which parts of the fixed state differ from that one (finds state that splits the same
// draws into many pipelines, each translated on the GPU thread).
void ReportNewPipelineKey(const GraphicsPipelineCacheKey& key) {
    static const bool enabled = std::getenv("EDEN_NX2_PIPELINE_DIFF") != nullptr;
    if (!enabled) {
        return;
    }
    static std::unordered_map<u64, GraphicsPipelineCacheKey> last_by_shaders;
    static u32 logged{};
    const u64 shaders = Common::CityHash64(reinterpret_cast<const char*>(key.unique_hashes.data()),
                                           sizeof(key.unique_hashes));
    const auto [it, first] = last_by_shaders.try_emplace(shaders, key);
    if (first) {
        return;
    }
    const FixedPipelineState& now = key.state;
    const FixedPipelineState& before = it->second.state;
    std::string diffs;
    if (now.raw1 != before.raw1) diffs += fmt::format(" raw1^{:#x}", now.raw1 ^ before.raw1);
    if (now.raw2 != before.raw2) diffs += fmt::format(" raw2^{:#x}", now.raw2 ^ before.raw2);
    if (now.color_formats != before.color_formats) {
        diffs += fmt::format(" color_formats {:02x} -> {:02x}", fmt::join(before.color_formats, ""),
                             fmt::join(now.color_formats, ""));
    }
    if (now.alpha_test_ref != before.alpha_test_ref) {
        diffs += fmt::format(" alpha_test_ref {:#x} -> {:#x} (func {})", before.alpha_test_ref,
                             now.alpha_test_ref, now.alpha_test_func.Value());
    }
    if (now.point_size != before.point_size) {
        diffs += fmt::format(" point_size {:#x} -> {:#x}", before.point_size, now.point_size);
    }
    if (now.viewport_swizzles != before.viewport_swizzles) diffs += " viewport_swizzles";
    if (now.attribute_types != before.attribute_types) {
        diffs += fmt::format(" attribute_types^{:#x}", now.attribute_types ^ before.attribute_types);
    }
    if (std::memcmp(&now.dynamic_state, &before.dynamic_state, sizeof(now.dynamic_state)) != 0) {
        diffs += " dynamic_state";
    }
    if (now.driver_id != before.driver_id || now.driver_version != before.driver_version) {
        diffs += " driver";
    }
    const size_t size = now.Size();
    if (diffs.empty() && size > offsetof(FixedPipelineState, attachments)) {
        diffs = fmt::format(" past attachments (key size {})", size);
    }
    it->second = key;
    if (logged++ < 400) {
        LOG_INFO(Render_Vulkan, "New pipeline for known shaders {:016x} differs in:{}", shaders,
                 diffs.empty() ? std::string{" nothing in the key prefix"} : diffs);
    }
}
} // Anonymous namespace

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    if (async_pipelines != 0) {
        return CurrentGraphicsPipelineAsync();
    }
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        ReportNewPipelineKey(graphics_key);
        pipeline = CreateGraphicsPipeline();
    }
    if (!pipeline) {
        return nullptr;
    }
    return MakeCurrent(pipeline.get());
}

GraphicsPipeline* PipelineCache::MakeCurrent(GraphicsPipeline* pipeline) {
    constexpr auto save_interval = std::chrono::seconds{15};
    if (used_this_session.insert(pipeline).second &&
        std::chrono::steady_clock::now() - used_pipelines_saved > save_interval) {
        SaveUsedPipelines();
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline);
    }
    current_pipeline = pipeline;
    current_pipeline->MarkUsed(frame_count);
    return BuiltPipeline(current_pipeline);
}

void PipelineCache::TickFrame() {
    ++frame_count;
    while (!evicted_pipelines.empty() && scheduler.IsFree(evicted_pipelines.front().first)) {
        evicted_pipelines.pop_front();
    }
    constexpr u64 EvictCheckFrames = 600;
    if (frame_count % EvictCheckFrames == 0) {
        EvictUnusedPipelines();
    }
}

void PipelineCache::EvictUnusedPipelines() {
    // Building a pipeline again on MoltenVK means a Metal compile mid-game, far dearer than the
    // 2-3 ms below, so none are evicted there.
    if (device.IsMoltenVK()) {
        return;
    }
    // Each built pipeline holds ~0.45 MB of driver memory and none was ever destroyed, so it grew
    // with every place a session went (title 1: 1,500 on its title screen alone). One unused for
    // this long is destroyed; it takes 2-3 ms to build again from the driver's pipeline cache.
    constexpr u64 EvictAfterFrames = 36000;
    if (frame_count <= EvictAfterFrames) {
        return;
    }
    const u64 cutoff = frame_count - EvictAfterFrames;
    const u64 tick = scheduler.CurrentTick();
    size_t evicted = 0;
    for (auto it = graphics_cache.begin(); it != graphics_cache.end();) {
        GraphicsPipeline* const pipeline = it->second.get();
        if (pipeline == nullptr || pipeline == current_pipeline || !pipeline->IsBuilt() ||
            pipeline->LastUse() >= cutoff) {
            ++it;
            continue;
        }
        if (used_this_session.erase(pipeline) != 0) {
            used_and_evicted.insert(it->first.Hash());
        }
        evicted_pipelines.emplace_back(tick, std::move(it->second));
        it = graphics_cache.erase(it);
        ++evicted;
    }
    if (evicted == 0) {
        return;
    }
    for (auto& [key, pipeline] : graphics_cache) {
        if (pipeline) {
            pipeline->ClearTransitions();
        }
    }
    LOG_INFO(Render_Vulkan, "Evicted {} graphics pipelines unused for {} frames, {} left", evicted,
             EvictAfterFrames, graphics_cache.size());
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool discard = CanDiscardPendingDraw(maxwell3d->GetNativeMacroEngine().native_turing,
                                              maxwell3d->regs.zeta_enable != 0,
                                              draw_state.draw_element_count,
                                              draw_state.draw_element_count);
    return discard ? nullptr : pipeline;
}

// EDEN_NX2_PIPELINE_ASYNC: a build off the GPU thread reads the engine's mode from here, and its
// environments say whether a read missed their snapshot (the result is then thrown away).
struct PipelineCache::AsyncBuildContext {
    bool native_turing{};
    const GraphicsEnvironments* environments{};

    [[nodiscard]] bool Missed() const {
        return std::ranges::any_of(environments->envs,
                                   [](const VideoCommon::GraphicsEnvironment& env) {
                                       return env.AsyncSnapshotMissed();
                                   });
    }
};

struct PipelineCache::AsyncGraphicsJob {
    enum : u8 { Queued, Running, Done };

    GraphicsPipelineCacheKey key{};
    GraphicsEnvironments environments;
    AsyncBuildContext context;
    std::atomic<u8> state{Queued}; ///< Queued -> Running by whoever builds it (worker or GPU thread)
    std::mutex mutex;
    std::condition_variable done;
    std::unique_ptr<GraphicsPipeline> pipeline; ///< Set when Done; null if the build failed
    bool fallback{}; ///< Set when Done: build it again on the GPU thread
};

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, std::shared_ptr<ShaderPools> emit_pools,
    std::unique_ptr<EmissionBudget::Reservation> reservation, const AsyncBuildContext* async) try {
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        if (index == static_cast<size_t>(Maxwell::ShaderType::Geometry) &&
            !device.IsGeometryShaderSupported() &&
            (async != nullptr ? async->native_turing
                              : maxwell3d != nullptr &&
                                    maxwell3d->GetNativeMacroEngine().native_turing)) {
            // A native (Switch 2) geometry program on a device without geometry shaders: the
            // pipeline cannot run it, so its draws are skipped.
            static std::atomic<bool> logged{};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "Skipping native geometry shader pipelines: the device has no geometry shaders");
            }
            return nullptr;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer).
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            return nullptr;
        }

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
        }
        if (async != nullptr && async->Missed()) {
            return nullptr; // Translated from a guess: the GPU thread builds it again
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    // A tessellation pipeline the driver would refuse is skipped like a missing mesh pipeline:
    // the draw is dropped (see the pipeline check in the rasterizer) instead of failing the build.
    {
        const Shader::IR::Program* control{};
        const Shader::IR::Program* evaluation{};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] == 0) {
                continue;
            }
            if (programs[index].stage == Shader::Stage::TessellationControl) {
                control = &programs[index];
            } else if (programs[index].stage == Shader::Stage::TessellationEval) {
                evaluation = &programs[index];
            }
        }
        if (control != nullptr || evaluation != nullptr) {
            const u32 max_patch_size = device.GetMaxTessellationPatchSize();
            const char* reason = nullptr;
            if (!device.IsTessellationSupported()) {
                reason = "the device has no tessellation shaders";
            } else if (control == nullptr || evaluation == nullptr) {
                reason = "it has only one of the control and evaluation stages";
            } else if (control->invocations == 0 || control->invocations > max_patch_size) {
                reason = "the control shader's output vertex count is out of range";
            }
            if (reason != nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping tessellation pipeline: {} (tcs={} tes={}, tcs output "
                            "vertices {}, patch control points {}, device max patch size {})",
                            reason, control != nullptr, evaluation != nullptr,
                            control != nullptr ? control->invocations : 0,
                            key.state.patch_control_points_minus_one.Value() + 1, max_patch_size);
                return nullptr;
            }
        }
    }
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    std::array<std::optional<Shader::RuntimeInfo>, Maxwell::MaxShaderStage> deferred_runtime_infos;
    std::array<std::vector<u32>, Maxwell::MaxShaderStage> reference_spirv;

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        const auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        ConvertLegacyToGeneric(program, runtime_info);
        if (emit_pools) {
            deferred_runtime_infos[stage_index] = runtime_info;
            if (VideoCommon::Nx2Perf::Verify()) {
                reference_spirv[stage_index] = EmitSPIRV(profile, runtime_info, program, binding);
            }
        } else {
            const auto begin = std::chrono::steady_clock::now();
            modules[stage_index] = EmitShaderModule(device, profile, runtime_info, program, binding,
                                                   key.unique_hashes[index], stage_index);
            if (build_in_parallel) {
                AddRuntimeSpirvTime(0, begin);
            }
        }
        previous_stage = &program;
    }
    std::function<std::array<vk::ShaderModule, Maxwell::MaxShaderStage>()> emit_stages;
    if (emit_pools) {
        auto job = std::make_shared<DeferredStages>();
        job->reservation = std::move(reservation);
        job->pools = emit_pools;
        job->programs = std::move(programs);
        job->runtime_infos = std::move(deferred_runtime_infos);
        job->reference_spirv = std::move(reference_spirv);
        for (size_t stage = 0; stage < infos.size(); ++stage) {
            if (infos[stage]) {
                infos[stage] = &job->programs[stage + 1].info;
            }
        }
        emit_stages = [this, job = std::move(job), hashes = key.unique_hashes]() mutable {
            const auto begin = std::chrono::steady_clock::now();
            std::array<vk::ShaderModule, Maxwell::MaxShaderStage> stage_modules;
            Shader::Backend::Bindings stage_binding;
            for (size_t stage = 0; stage < stage_modules.size(); ++stage) {
                if (job->runtime_infos[stage]) {
                    stage_modules[stage] =
                        EmitShaderModule(device, profile, *job->runtime_infos[stage],
                                         job->programs[stage + 1], stage_binding,
                                         hashes[stage + 1], stage, job->reference_spirv[stage]);
                }
            }
            auto pools = std::move(job->pools);
            job.reset();
            pools->ReleaseContents();
            {
                std::scoped_lock lock{spare_pools_mutex};
                if (spare_pools.size() < 4) {
                    spare_pools.push_back(std::move(pools));
                }
            }
            AddRuntimeSpirvTime(1, begin);
            return stage_modules;
        };
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos, std::move(emit_stages));

} catch (const Shader::Exception& exception) {
    if (async != nullptr && async->Missed()) {
        return nullptr; // A read the snapshot lacked; the GPU thread's build reports real errors
    }
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        try {
            // Walks the program for the dump; the one that failed may fail here again, and an
            // exception leaving this handler would end the process.
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        } catch (const Shader::Exception&) {
        }
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

namespace {
// Pipelines built on the GPU thread at run time (shader translation, and the SPIR-V unless a worker
// emits it; the driver compile is on workers): the first 400 over 2 ms and every one over 100 ms
// are logged, and the totals every 10 s, to tell how much of the stutter on new content is
// translation.
void ReportRuntimePipeline(const char* kind, std::chrono::steady_clock::time_point begin,
                           const std::string& shaders) {
    static std::chrono::steady_clock::duration total{};
    static u32 count{}, logged{};
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - begin;
    total += elapsed;
    ++count;
    const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    // Slow builds are always logged with their shaders, to find the programs that are slow to
    // translate.
    if (elapsed > std::chrono::milliseconds{100} ||
        (elapsed > std::chrono::milliseconds{2} && logged < 400)) {
        ++logged;
        LOG_INFO(Render_Vulkan, "{} pipeline built on the GPU thread in {:.1f} ms: {}", kind,
                 ms(elapsed), shaders);
    }
    if (now - window > std::chrono::seconds{10}) {
        const auto take = [](auto& counter) {
            return std::chrono::nanoseconds{counter.exchange(0, std::memory_order::relaxed)};
        };
        LOG_INFO(Render_Vulkan,
                 "Runtime pipelines in the last 10 s: {} built, {:.1f} ms on the GPU thread "
                 "({:.1f} ms of it graphics SPIR-V), {:.1f} ms graphics SPIR-V on workers, "
                 "{:.1f} ms of draws held for unbuilt pipelines",
                 count, ms(total), ms(take(runtime_spirv_ns[0])), ms(take(runtime_spirv_ns[1])),
                 ms(GraphicsPipeline::TakeHoldTime()));
        total = {};
        count = 0;
        window = now;
    }
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        std::string shaders = fmt::format("pipeline={:016x}", graphics_key.Hash());
        for (size_t stage = 0; stage < graphics_key.unique_hashes.size(); ++stage) {
            if (graphics_key.unique_hashes[stage] != 0) {
                shaders += fmt::format(" s{}:{:016x}", stage, graphics_key.unique_hashes[stage]);
            }
        }
        ReportRuntimePipeline("Graphics", begin, shaders);
    };
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    std::shared_ptr<ShaderPools> emit_pools;
    auto reservation = emission_budget.TryAcquire(maxwell3d->GetNativeMacroEngine().native_turing);
    if (reservation) {
        {
            std::scoped_lock lock{spare_pools_mutex};
            if (!spare_pools.empty()) {
                emit_pools = std::move(spare_pools.back());
                spare_pools.pop_back();
            }
        }
        if (!emit_pools) {
            emit_pools = std::make_shared<ShaderPools>();
        }
    } else {
        main_pools.ReleaseContents();
    }
    ShaderPools& pools = emit_pools ? *emit_pools : main_pools;
    auto pipeline{CreateGraphicsPipeline(pools, graphics_key, environments.Span(), nullptr, true,
                                         std::move(emit_pools), std::move(reservation))};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    StoreRuntimePipeline(graphics_key, *pipeline, environments);
    return pipeline;
}

void PipelineCache::StoreRuntimePipeline(const GraphicsPipelineCacheKey& key,
                                         const GraphicsPipeline& pipeline,
                                         GraphicsEnvironments& environments) {
    if (pipeline_cache_filename.empty()) {
        return;
    }
    if (pipeline.IsMeshPipeline()) {
        // Not stored: task and mesh programs are translated with the mesh registers, which the
        // cache format does not carry. The few of them rebuild quickly from the driver cache.
        QueueVulkanPipelineCacheFlush();
        return;
    }
    if (disk_pipelines.contains(key.Hash())) {
        // In the disk cache already, only not built at boot.
        QueueVulkanPipelineCacheFlush();
        return;
    }
    disk_pipelines.insert(key.Hash());
    serialization_thread.QueueWork([this, key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineAsync() {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        const auto end = std::chrono::steady_clock::now();
        async_stats.gpu_time += end - begin;
        ReportAsyncPipelines(end);
    };
    DrainAsyncPipelines();
    auto it{graphics_cache.find(graphics_key)};
    if (it == graphics_cache.end()) {
        const bool on_gpu_thread_only = async_sync_keys.erase(graphics_key.Hash()) != 0;
        if (!on_gpu_thread_only) {
            ReportNewPipelineKey(graphics_key);
        }
        // Null while it builds; if building throws, it stays null and its draws are skipped, as
        // the synchronous path leaves it.
        graphics_cache.try_emplace(graphics_key);
        if (!on_gpu_thread_only && CanDiscardAsyncDraw() && QueueAsyncGraphicsPipeline()) {
            ++async_stats.skipped;
            return nullptr;
        }
        if (!on_gpu_thread_only) {
            ++async_stats.sync_built;
        }
        auto pipeline{CreateGraphicsPipeline()};
        it = graphics_cache.find(graphics_key);
        it->second = std::move(pipeline);
    } else if (!it->second) {
        const auto pending{async_pending.find(graphics_key)};
        if (pending == async_pending.end()) {
            return nullptr; // Its build failed: skipped, as on the synchronous path
        }
        if (CanDiscardAsyncDraw()) {
            ++async_stats.skipped;
            return nullptr;
        }
        const std::shared_ptr<AsyncGraphicsJob> job{pending->second};
        FinishAsyncGraphicsPipeline(job);
        DrainAsyncPipelines();
        it = graphics_cache.find(graphics_key);
        if (it == graphics_cache.end()) {
            // The build missed its snapshot: built here, with the state of this draw.
            async_sync_keys.erase(graphics_key.Hash());
            graphics_cache.try_emplace(graphics_key);
            auto pipeline{CreateGraphicsPipeline()};
            it = graphics_cache.find(graphics_key);
            it->second = std::move(pipeline);
        }
    }
    if (!it->second) {
        return nullptr;
    }
    return MakeCurrent(it->second.get());
}

bool PipelineCache::CanDiscardAsyncDraw() const {
    // The rule BuiltPipeline applies to a pipeline whose driver compile is pending; mode 2 drops
    // its exception for native draws.
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing && async_pipelines < 2;
    return CanDiscardPendingDraw(native, maxwell3d->regs.zeta_enable != 0,
                                 draw_state.draw_element_count, draw_state.draw_element_count);
}

bool PipelineCache::QueueAsyncGraphicsPipeline() {
    auto job{std::make_shared<AsyncGraphicsJob>()};
    job->key = graphics_key;
    GetGraphicsEnvironments(job->environments, graphics_key.unique_hashes);
    boost::container::static_vector<VideoCommon::GraphicsEnvironment*, Maxwell::MaxShaderProgram>
        envs;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (graphics_key.unique_hashes[index] != 0) {
            envs.push_back(&job->environments.envs[index]);
        }
    }
    const auto snapshot{VideoCommon::GraphicsEnvironment::SnapshotForAsync(
        std::span<VideoCommon::GraphicsEnvironment* const>(envs.data(), envs.size()))};
    if (!snapshot) {
        return false;
    }
    async_stats.snapshot_bytes += snapshot->bytes;
    job->context.native_turing = maxwell3d->GetNativeMacroEngine().native_turing;
    job->context.environments = &job->environments;
    async_pending.emplace(graphics_key, job);
    ++async_stats.queued;
    workers.QueueWork([this, job] {
        {
            std::scoped_lock lock{async_mutex};
            if (async_stopping) {
                return;
            }
            ++async_running;
        }
        u8 expected = AsyncGraphicsJob::Queued;
        if (job->state.compare_exchange_strong(expected, AsyncGraphicsJob::Running)) {
            BuildAsyncGraphicsPipeline(job, false);
        }
        std::scoped_lock lock{async_mutex};
        --async_running;
        async_idle.notify_all();
    });
    return true;
}

void PipelineCache::BuildAsyncGraphicsPipeline(const std::shared_ptr<AsyncGraphicsJob>& job,
                                               bool on_gpu_thread) {
    // On a worker the driver compile runs right here, so the pipeline arrives built. On the GPU
    // thread (a draw that may not be skipped) it goes to the workers, as on the synchronous path.
    std::optional<ShaderPools> worker_pools;
    if (on_gpu_thread) {
        main_pools.ReleaseContents();
    } else {
        worker_pools.emplace();
    }
    ShaderPools& pools{on_gpu_thread ? main_pools : *worker_pools};
    std::unique_ptr<GraphicsPipeline> pipeline;
    bool fallback = false;
    try {
        pipeline = CreateGraphicsPipeline(pools, job->key, job->environments.Span(), nullptr,
                                          on_gpu_thread, nullptr, nullptr, &job->context);
        fallback = job->context.Missed();
    } catch (const std::exception& exception) {
        // Anything CreateGraphicsPipeline does not handle itself: the GPU thread builds it again
        // and meets the error where the synchronous path would.
        LOG_WARNING(Render_Vulkan, "Pipeline {:016x} build off the GPU thread failed: {}",
                    job->key.Hash(), exception.what());
        fallback = true;
    }
    if (fallback) {
        pipeline.reset();
    } else if (!on_gpu_thread) {
        async_stats.worker_built.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::scoped_lock lock{job->mutex};
        job->pipeline = std::move(pipeline);
        job->fallback = fallback;
        job->state.store(AsyncGraphicsJob::Done, std::memory_order_release);
    }
    job->done.notify_all();
    std::scoped_lock lock{async_mutex};
    async_completed.push_back(job);
    async_completed_count.fetch_add(1, std::memory_order_release);
}

void PipelineCache::FinishAsyncGraphicsPipeline(const std::shared_ptr<AsyncGraphicsJob>& job) {
    u8 expected = AsyncGraphicsJob::Queued;
    if (job->state.compare_exchange_strong(expected, AsyncGraphicsJob::Running)) {
        ++async_stats.inline_built;
        BuildAsyncGraphicsPipeline(job, true);
        return;
    }
    ++async_stats.waited;
    std::unique_lock lock{job->mutex};
    job->done.wait(lock, [&job] {
        return job->state.load(std::memory_order_acquire) == AsyncGraphicsJob::Done;
    });
}

void PipelineCache::DrainAsyncPipelines() {
    if (async_completed_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::vector<std::shared_ptr<AsyncGraphicsJob>> done;
    {
        std::scoped_lock lock{async_mutex};
        done.swap(async_completed);
        async_completed_count.store(0, std::memory_order_relaxed);
    }
    static std::unordered_set<u64> logged_fallbacks; // GPU thread only
    for (const std::shared_ptr<AsyncGraphicsJob>& job : done) {
        async_pending.erase(job->key);
        auto it{graphics_cache.find(job->key)};
        if (job->fallback) {
            ++async_stats.fallback;
            if (it != graphics_cache.end() && !it->second) {
                graphics_cache.erase(it);
            }
            const u64 hash = job->key.Hash();
            async_sync_keys.insert(hash);
            if (logged_fallbacks.insert(hash).second) {
                LOG_INFO(Render_Vulkan,
                         "Pipeline {:016x} needs a read its snapshot lacks: built on the GPU "
                         "thread at its next draw",
                         hash);
            }
            continue;
        }
        if (it == graphics_cache.end()) {
            it = graphics_cache.try_emplace(job->key).first;
        }
        if (it->second) {
            // Not expected (the entry stays null while pending): keep the cached one, and
            // destroy this one once the GPU is done with whatever it recorded.
            if (job->pipeline) {
                evicted_pipelines.emplace_back(scheduler.CurrentTick(), std::move(job->pipeline));
            }
            continue;
        }
        GraphicsPipeline* const pipeline{job->pipeline.get()};
        it->second = std::move(job->pipeline);
        if (pipeline != nullptr) {
            // The environments hold what the translation read, as the GPU thread's do.
            StoreRuntimePipeline(job->key, *pipeline, job->environments);
        }
    }
}

void PipelineCache::ReportAsyncPipelines(std::chrono::steady_clock::time_point now) {
    auto& stats{async_stats};
    if (now - stats.window < std::chrono::seconds{10}) {
        return;
    }
    const u32 worker_built = stats.worker_built.exchange(0, std::memory_order_relaxed);
    if (stats.queued != 0 || worker_built != 0 || stats.inline_built != 0 || stats.waited != 0 ||
        stats.sync_built != 0 || stats.fallback != 0 || stats.skipped != 0) {
        LOG_INFO(Render_Vulkan,
                 "Async pipelines in the last 10 s: {} queued, {} completed on workers, {} inline, "
                 "{} waited for, {} built on the GPU thread (draw not skippable), {} "
                 "failed/fallback, draws skipped {}, snapshot {} KiB per job, GPU-thread ms {:.1f}",
                 stats.queued, worker_built, stats.inline_built, stats.waited, stats.sync_built,
                 stats.fallback, stats.skipped,
                 stats.queued != 0 ? stats.snapshot_bytes / stats.queued / 1024 : 0,
                 std::chrono::duration<double, std::milli>(stats.gpu_time).count());
    }
    stats.queued = 0;
    stats.inline_built = 0;
    stats.waited = 0;
    stats.sync_built = 0;
    stats.fallback = 0;
    stats.skipped = 0;
    stats.snapshot_bytes = 0;
    stats.gpu_time = {};
    stats.window = now;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {
    const auto begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        ReportRuntimePipeline("Compute", begin,
                              fmt::format("pipeline={:016x} s6:{:016x} address={:#x}", key.Hash(),
                                          key.unique_hash,
                                          kepler_compute->regs.code_loc.Address() +
                                              kepler_compute->launch_description.program_start));
    };
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);

    main_pools.ReleaseContents();
    auto entry{CreateComputePipeline(main_pools, key, env, nullptr, true)};
    if (const char* directory = std::getenv("EDEN_NX2_SHADER_DUMP");
        directory && !entry.pipeline && env.CanBeSerialized()) {
        const auto path = std::filesystem::path{directory} /
                          fmt::format("failed-compute-{:016x}-v{}.env", key.unique_hash, CACHE_VERSION);
        std::ofstream file{path, std::ios::binary};
        if (file) {
            env.Serialize(file);
            LOG_INFO(Render_Vulkan, "Captured failed compute environment: {} written={}",
                     path.string(), file.good());
        }
    }
    if (!entry.pipeline || pipeline_cache_filename.empty()) {
        return entry;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return entry;
}

ComputePipelineCacheEntry PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return {.failure = "Device reports broken compute support"};
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY || driver_id == VK_DRIVER_ID_MOLTENVK;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    const std::vector<u32> code{EmitSPIRV(profile, program)};
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return {.pipeline = std::make_unique<ComputePipeline>(
                device, scheduler, vulkan_pipeline_cache, descriptor_pool, guest_descriptor_queue,
                descriptor_buffer_ring, thread_worker, statistics, &shader_notify, program.info,
                std::move(spv_module), key.unique_hash)};

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Compute pipeline compilation failed shader={:#016x} pipeline={:#016x}: {}",
              key.unique_hash, key.Hash(), exception.what());
    return {.failure = exception.what()};
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    // Written beside the cache and renamed over it: the emulator stopped mid-write left the cache
    // empty, and every later boot failed to read it.
    auto temporary = filename;
    temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    size_t cache_size = 0;
    std::vector<char> cache_data;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    file.write(cache_data.data(), cache_size);
    file.close();
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}: {}",
                  Common::FS::PathToUTF8String(filename), error.message());
        std::filesystem::remove(temporary, error);
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        // A file cut short (the emulator stopped while writing it) would throw from the reads
        // below, and macOS's libc++ throws an ios_base::failure the handler doesn't match.
        if (static_cast<size_t>(end) < magic_number.size() + sizeof(cache_version)) {
            file.close();
            LOG_ERROR(Common_Filesystem, "Truncated Vulkan driver pipeline cache file, deleting it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: ", Common::FS::PathToUTF8String(filename));

        return create_pipeline_cache(cache_size, cache_data.data());

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
