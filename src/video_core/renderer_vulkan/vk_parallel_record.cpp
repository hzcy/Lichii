// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <source_location>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_parallel_record.h"

namespace Vulkan {

namespace {

using Kind = ParallelRecord::Kind;
using Fallback = ParallelRecord::Fallback;
using SlotRange = ParallelRecord::SlotRange;
using ReplayFn = ParallelRecord::ReplayFn;

thread_local ParallelRecord* t_record{};

constexpr u32 MaxViewports = 16;
constexpr u32 MaxVertexBuffers = 32;
constexpr u32 MaxSets = 8;
constexpr u32 PushWords = 128;
constexpr u32 MaxAttachments = 8;
constexpr u32 MaxDynamicOffsets = 64;
constexpr u32 MaxVertexInputs = 32;
constexpr u32 MaxClears = 16;

/// Bound state, by the call that sets it. Graphics and compute keep their own pipeline,
/// descriptor sets and push constants.
enum Slot : u16 {
    Pipeline = 0,
    IndexBuffer = Pipeline + 2,
    Viewport,
    Scissor = Viewport + MaxViewports,
    VertexBuffer = Scissor + MaxViewports,
    DescriptorSet = VertexBuffer + MaxVertexBuffers,
    PushConstant = DescriptorSet + MaxSets * 2,
    ColorWriteMask = PushConstant + PushWords * 2,
    ColorBlendEnable = ColorWriteMask + MaxAttachments,
    ColorBlendEquation = ColorBlendEnable + MaxAttachments,
    ColorWriteEnable = ColorBlendEquation + MaxAttachments,
    StencilCompareMask,
    StencilWriteMask = StencilCompareMask + 2,
    StencilReference = StencilWriteMask + 2,
    StencilOp = StencilReference + 2,
    DescriptorBuffers = StencilOp + 2,
    VertexInput,
    BlendConstants,
    DepthBias,
    CullMode,
    FrontFace,
    DepthBiasEnable,
    DepthBounds,
    DepthBoundsTestEnable,
    DepthCompareOp,
    DepthTestEnable,
    DepthWriteEnable,
    PrimitiveRestartEnable,
    RasterizerDiscardEnable,
    AlphaToCoverageEnable,
    AlphaToOneEnable,
    ConservativeRasterizationMode,
    LineRasterizationMode,
    LineStippleEnable,
    LineStipple,
    LogicOpEnable,
    DepthClampEnable,
    PatchControlPoints,
    LogicOp,
    LineWidth,
    PrimitiveTopology,
    StencilTestEnable,
    SlotCount,
};

constexpr bool IsComputeSlot(u32 slot) {
    return slot == Pipeline + 1 ||
           (slot >= DescriptorSet + MaxSets && slot < DescriptorSet + MaxSets * 2) ||
           (slot >= PushConstant + PushWords && slot < PushConstant + PushWords * 2);
}

constexpr const char* FallbackName(Fallback reason) {
    switch (reason) {
    case Fallback::Small:
        return "small";
    case Fallback::Query:
        return "query";
    case Fallback::Barrier:
        return "barrier";
    case Fallback::TransformFeedback:
        return "transform feedback";
    case Fallback::ConditionalRendering:
        return "conditional rendering";
    case Fallback::Timestamp:
        return "timestamp";
    case Fallback::DebugLabel:
        return "debug label";
    case Fallback::Event:
        return "event";
    case Fallback::OutsideCommand:
        return "outside command";
    case Fallback::UnloggedState:
        return "unlogged state";
    case Fallback::RenderPassInfo:
        return "render pass info";
    case Fallback::Count:
        break;
    }
    return "?";
}

/// 0 for graphics, 1 for compute; nullopt for other bind points.
std::optional<u32> BindIndex(VkPipelineBindPoint bind_point) {
    switch (bind_point) {
    case VK_PIPELINE_BIND_POINT_GRAPHICS:
        return 0;
    case VK_PIPELINE_BIND_POINT_COMPUTE:
        return 1;
    default:
        return std::nullopt;
    }
}

std::optional<SlotRange> Range(u32 base, u32 limit, u32 first, u32 count) {
    if (count == 0 || first >= limit || count > limit - first) {
        return std::nullopt;
    }
    return SlotRange{static_cast<u16>(base + first), static_cast<u16>(count)};
}

constexpr size_t Align16(size_t value) {
    return (value + 15) & ~size_t{15};
}

/// Payload layout: scalars, then arrays that each start 16-byte aligned.
class Out {
public:
    explicit Out(u8* p_) : base{p_}, p{p_} {}
    template <typename T>
    void Put(const T& value) {
        std::memcpy(p, &value, sizeof(T));
        p += sizeof(T);
    }
    template <typename T>
    void Array(const T* values, u32 count) {
        p = base + Align16(static_cast<size_t>(p - base));
        if (count != 0) {
            std::memcpy(p, values, sizeof(T) * count);
        }
        p += sizeof(T) * count;
    }

private:
    u8* base;
    u8* p;
};

class In {
public:
    explicit In(const u8* p_) : base{p_}, p{p_} {}
    template <typename T>
    T Get() {
        T value;
        std::memcpy(&value, p, sizeof(T));
        p += sizeof(T);
        return value;
    }
    template <typename T>
    const T* Array(u32 count) {
        p = base + Align16(static_cast<size_t>(p - base));
        const T* const values = reinterpret_cast<const T*>(p);
        p += sizeof(T) * count;
        return values;
    }

private:
    const u8* base;
    const u8* p;
};

template <typename... A>
constexpr size_t ScalarSize() {
    return (size_t{0} + ... + sizeof(A));
}

/// Logs the call when it sets state or a pass is captured, and forwards it unless captured.
template <typename Write, typename Forward>
void Handle(Kind kind, std::optional<SlotRange> slots, size_t size, ReplayFn replay,
            Write&& write, Forward&& forward,
            std::source_location location = std::source_location::current()) {
    ParallelRecord& r = *t_record;
    if (!slots) {
        static std::set<u32> logged;
        if (logged.size() < 16 && logged.insert(location.line()).second) {
            LOG_INFO(Render_Vulkan, "Parallel record: {} can't be logged ({}:{})",
                     location.function_name(), location.file_name(), location.line());
        }
        if (kind == Kind::State) {
            r.Unlogged(Fallback::UnloggedState);
        } else {
            r.Serial(Fallback::UnloggedState);
        }
        if (kind == Kind::Draw) {
            r.NoteInlineDraw();
        }
        forward();
        return;
    }
    if (kind != Kind::State && !r.Capturing()) {
        if (kind == Kind::Draw) {
            r.NoteInlineDraw();
        }
        forward();
        return;
    }
    r.Log(replay, kind, *slots, size, write);
    if (!r.Capturing()) {
        forward();
    }
}

template <auto Member>
using Pfn = std::remove_cvref_t<decltype(std::declval<const vk::DeviceDispatch&>().*Member)>;

template <auto Member, typename F = Pfn<Member>>
struct Hook;

/// Calls whose arguments are all values.
template <auto Member, typename... A>
struct Hook<Member, void(VKAPI_PTR*)(VkCommandBuffer, A...)> {
    static void Replay(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
        In in{payload};
        std::tuple<A...> args{in.Get<A>()...};
        std::apply([&](const A&... values) { (dld.*Member)(cmdbuf, values...); }, args);
    }

    static void Call(VkCommandBuffer cmdbuf, Kind kind, std::optional<SlotRange> slots,
                     A... args) {
        static_assert((!(std::is_pointer_v<A> && std::is_const_v<std::remove_pointer_t<A>>) && ...),
                      "array arguments need their own hook");
        Handle(
            kind, slots, ScalarSize<A...>(), &Replay,
            [&](u8* p) {
                Out out{p};
                (out.Put(args), ...);
            },
            [&] { (t_record->Real().*Member)(cmdbuf, args...); });
    }

    template <u16 slot>
    static void VKAPI_CALL State(VkCommandBuffer cmdbuf, A... args) {
        Call(cmdbuf, Kind::State, SlotRange{slot, 1}, args...);
    }

    static void VKAPI_CALL Draw(VkCommandBuffer cmdbuf, A... args) {
        Call(cmdbuf, Kind::Draw, SlotRange{0, 0}, args...);
    }

    template <Fallback reason>
    static void VKAPI_CALL Serial(VkCommandBuffer cmdbuf, A... args) {
        t_record->Serial(reason);
        (t_record->Real().*Member)(cmdbuf, args...);
    }
};

/// Stencil calls set the front, back or both faces.
template <auto Member, u16 base, typename F = Pfn<Member>>
struct StencilHook;

template <auto Member, u16 base, typename... A>
struct StencilHook<Member, base, void(VKAPI_PTR*)(VkCommandBuffer, VkStencilFaceFlags, A...)> {
    static void VKAPI_CALL Fn(VkCommandBuffer cmdbuf, VkStencilFaceFlags faces, A... args) {
        const bool front = (faces & VK_STENCIL_FACE_FRONT_BIT) != 0;
        const bool back = (faces & VK_STENCIL_FACE_BACK_BIT) != 0;
        std::optional<SlotRange> slots;
        if (front || back) {
            slots = SlotRange{static_cast<u16>(front ? base : base + 1),
                              static_cast<u16>(front && back ? 2 : 1)};
        }
        Hook<Member>::Call(cmdbuf, Kind::State, slots, faces, args...);
    }
};

void VKAPI_CALL BindPipeline(VkCommandBuffer cmdbuf, VkPipelineBindPoint bind_point,
                             VkPipeline pipeline) {
    const auto index = BindIndex(bind_point);
    Hook<&vk::DeviceDispatch::vkCmdBindPipeline>::Call(
        cmdbuf, Kind::State,
        index ? std::optional{SlotRange{static_cast<u16>(Pipeline + *index), 1}} : std::nullopt,
        bind_point, pipeline);
}

void ReplayViewport(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
    In in{payload};
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    dld.vkCmdSetViewport(cmdbuf, first, count, in.Array<VkViewport>(count));
}

void VKAPI_CALL SetViewport(VkCommandBuffer cmdbuf, u32 first, u32 count,
                            const VkViewport* viewports) {
    Handle(
        Kind::State, Range(Viewport, MaxViewports, first, count),
        Align16(8) + sizeof(VkViewport) * count, &ReplayViewport,
        [&](u8* p) {
            Out out{p};
            out.Put(first);
            out.Put(count);
            out.Array(viewports, count);
        },
        [&] { t_record->Real().vkCmdSetViewport(cmdbuf, first, count, viewports); });
}

void ReplayScissor(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
    In in{payload};
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    dld.vkCmdSetScissor(cmdbuf, first, count, in.Array<VkRect2D>(count));
}

void VKAPI_CALL SetScissor(VkCommandBuffer cmdbuf, u32 first, u32 count,
                           const VkRect2D* scissors) {
    Handle(
        Kind::State, Range(Scissor, MaxViewports, first, count),
        Align16(8) + sizeof(VkRect2D) * count, &ReplayScissor,
        [&](u8* p) {
            Out out{p};
            out.Put(first);
            out.Put(count);
            out.Array(scissors, count);
        },
        [&] { t_record->Real().vkCmdSetScissor(cmdbuf, first, count, scissors); });
}

void ReplayDescriptorSets(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                          const u8* payload) {
    In in{payload};
    const auto bind_point = in.Get<VkPipelineBindPoint>();
    const auto layout = in.Get<VkPipelineLayout>();
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    const u32 dynamic_count = in.Get<u32>();
    const VkDescriptorSet* const sets = in.Array<VkDescriptorSet>(count);
    const u32* const dynamic_offsets = in.Array<u32>(dynamic_count);
    dld.vkCmdBindDescriptorSets(cmdbuf, bind_point, layout, first, count, sets, dynamic_count,
                                dynamic_count != 0 ? dynamic_offsets : nullptr);
}

void VKAPI_CALL BindDescriptorSets(VkCommandBuffer cmdbuf, VkPipelineBindPoint bind_point,
                                   VkPipelineLayout layout, u32 first, u32 count,
                                   const VkDescriptorSet* sets, u32 dynamic_count,
                                   const u32* dynamic_offsets) {
    const auto index = BindIndex(bind_point);
    std::optional<SlotRange> slots;
    if (index && dynamic_count <= MaxDynamicOffsets) {
        slots = Range(DescriptorSet + *index * MaxSets, MaxSets, first, count);
    }
    Handle(
        Kind::State, slots,
        Align16(sizeof(bind_point) + sizeof(layout) + 12) + Align16(sizeof(VkDescriptorSet) * count) +
            sizeof(u32) * dynamic_count,
        &ReplayDescriptorSets,
        [&](u8* p) {
            Out out{p};
            out.Put(bind_point);
            out.Put(layout);
            out.Put(first);
            out.Put(count);
            out.Put(dynamic_count);
            out.Array(sets, count);
            out.Array(dynamic_offsets, dynamic_count);
        },
        [&] {
            t_record->Real().vkCmdBindDescriptorSets(cmdbuf, bind_point, layout, first, count,
                                                     sets, dynamic_count, dynamic_offsets);
        });
}

void ReplayVertexBuffers(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                         const u8* payload) {
    In in{payload};
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    const VkBuffer* const buffers = in.Array<VkBuffer>(count);
    dld.vkCmdBindVertexBuffers(cmdbuf, first, count, buffers, in.Array<VkDeviceSize>(count));
}

void VKAPI_CALL BindVertexBuffers(VkCommandBuffer cmdbuf, u32 first, u32 count,
                                  const VkBuffer* buffers, const VkDeviceSize* offsets) {
    Handle(
        Kind::State, Range(VertexBuffer, MaxVertexBuffers, first, count),
        Align16(8) + Align16(sizeof(VkBuffer) * count) + sizeof(VkDeviceSize) * count,
        &ReplayVertexBuffers,
        [&](u8* p) {
            Out out{p};
            out.Put(first);
            out.Put(count);
            out.Array(buffers, count);
            out.Array(offsets, count);
        },
        [&] { t_record->Real().vkCmdBindVertexBuffers(cmdbuf, first, count, buffers, offsets); });
}

void ReplayVertexBuffers2(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                          const u8* payload) {
    In in{payload};
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    const u32 has_sizes = in.Get<u32>();
    const u32 has_strides = in.Get<u32>();
    const VkBuffer* const buffers = in.Array<VkBuffer>(count);
    const VkDeviceSize* const offsets = in.Array<VkDeviceSize>(count);
    const VkDeviceSize* const sizes = in.Array<VkDeviceSize>(has_sizes ? count : 0);
    const VkDeviceSize* const strides = in.Array<VkDeviceSize>(has_strides ? count : 0);
    dld.vkCmdBindVertexBuffers2EXT(cmdbuf, first, count, buffers, offsets,
                                   has_sizes ? sizes : nullptr, has_strides ? strides : nullptr);
}

void VKAPI_CALL BindVertexBuffers2(VkCommandBuffer cmdbuf, u32 first, u32 count,
                                   const VkBuffer* buffers, const VkDeviceSize* offsets,
                                   const VkDeviceSize* sizes, const VkDeviceSize* strides) {
    const u32 has_sizes = sizes != nullptr ? 1 : 0;
    const u32 has_strides = strides != nullptr ? 1 : 0;
    const size_t array = Align16(sizeof(VkDeviceSize) * count);
    Handle(
        Kind::State, Range(VertexBuffer, MaxVertexBuffers, first, count),
        Align16(16) + array * (2 + has_sizes + has_strides), &ReplayVertexBuffers2,
        [&](u8* p) {
            Out out{p};
            out.Put(first);
            out.Put(count);
            out.Put(has_sizes);
            out.Put(has_strides);
            out.Array(buffers, count);
            out.Array(offsets, count);
            out.Array(sizes, has_sizes ? count : 0);
            out.Array(strides, has_strides ? count : 0);
        },
        [&] {
            t_record->Real().vkCmdBindVertexBuffers2EXT(cmdbuf, first, count, buffers, offsets,
                                                        sizes, strides);
        });
}

void ReplayPushConstants(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                         const u8* payload) {
    In in{payload};
    const auto layout = in.Get<VkPipelineLayout>();
    const auto stages = in.Get<VkShaderStageFlags>();
    const u32 offset = in.Get<u32>();
    const u32 size = in.Get<u32>();
    dld.vkCmdPushConstants(cmdbuf, layout, stages, offset, size, in.Array<u8>(size));
}

void VKAPI_CALL PushConstants(VkCommandBuffer cmdbuf, VkPipelineLayout layout,
                              VkShaderStageFlags stages, u32 offset, u32 size,
                              const void* values) {
    std::optional<SlotRange> slots;
    const bool compute = stages == VK_SHADER_STAGE_COMPUTE_BIT;
    if (compute || (stages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        slots = Range(PushConstant + (compute ? PushWords : 0), PushWords, offset / 4,
                      (offset % 4 + size + 3) / 4);
    }
    Handle(
        Kind::State, slots, Align16(sizeof(layout) + 12) + size, &ReplayPushConstants,
        [&](u8* p) {
            Out out{p};
            out.Put(layout);
            out.Put(stages);
            out.Put(offset);
            out.Put(size);
            out.Array(static_cast<const u8*>(values), size);
        },
        [&] {
            t_record->Real().vkCmdPushConstants(cmdbuf, layout, stages, offset, size, values);
        });
}

void ReplayClearAttachments(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                            const u8* payload) {
    In in{payload};
    const u32 count = in.Get<u32>();
    const u32 rect_count = in.Get<u32>();
    const VkClearAttachment* const attachments = in.Array<VkClearAttachment>(count);
    dld.vkCmdClearAttachments(cmdbuf, count, attachments, rect_count,
                              in.Array<VkClearRect>(rect_count));
}

void VKAPI_CALL ClearAttachments(VkCommandBuffer cmdbuf, u32 count,
                                 const VkClearAttachment* attachments, u32 rect_count,
                                 const VkClearRect* rects) {
    Handle(
        Kind::Action,
        count <= MaxClears && rect_count <= MaxClears ? std::optional{SlotRange{0, 0}}
                                                      : std::nullopt,
        Align16(8) + Align16(sizeof(VkClearAttachment) * count) + sizeof(VkClearRect) * rect_count,
        &ReplayClearAttachments,
        [&](u8* p) {
            Out out{p};
            out.Put(count);
            out.Put(rect_count);
            out.Array(attachments, count);
            out.Array(rects, rect_count);
        },
        [&] {
            t_record->Real().vkCmdClearAttachments(cmdbuf, count, attachments, rect_count, rects);
        });
}

void ReplayBlendConstants(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                          const u8* payload) {
    In in{payload};
    dld.vkCmdSetBlendConstants(cmdbuf, in.Array<float>(4));
}

void VKAPI_CALL SetBlendConstants(VkCommandBuffer cmdbuf, const float constants[4]) {
    Handle(
        Kind::State, SlotRange{BlendConstants, 1}, sizeof(float) * 4, &ReplayBlendConstants,
        [&](u8* p) { Out{p}.Array(constants, 4); },
        [&] { t_record->Real().vkCmdSetBlendConstants(cmdbuf, constants); });
}

void ReplayVertexInput(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
    In in{payload};
    const u32 binding_count = in.Get<u32>();
    const u32 attribute_count = in.Get<u32>();
    const auto* const bindings = in.Array<VkVertexInputBindingDescription2EXT>(binding_count);
    dld.vkCmdSetVertexInputEXT(
        cmdbuf, binding_count, bindings, attribute_count,
        in.Array<VkVertexInputAttributeDescription2EXT>(attribute_count));
}

void VKAPI_CALL SetVertexInput(VkCommandBuffer cmdbuf, u32 binding_count,
                               const VkVertexInputBindingDescription2EXT* bindings,
                               u32 attribute_count,
                               const VkVertexInputAttributeDescription2EXT* attributes) {
    bool loggable = binding_count <= MaxVertexInputs && attribute_count <= MaxVertexInputs;
    for (u32 i = 0; loggable && i < binding_count; ++i) {
        loggable = bindings[i].pNext == nullptr;
    }
    for (u32 i = 0; loggable && i < attribute_count; ++i) {
        loggable = attributes[i].pNext == nullptr;
    }
    Handle(
        Kind::State, loggable ? std::optional{SlotRange{VertexInput, 1}} : std::nullopt,
        Align16(8) + Align16(sizeof(*bindings) * binding_count) +
            sizeof(*attributes) * attribute_count,
        &ReplayVertexInput,
        [&](u8* p) {
            Out out{p};
            out.Put(binding_count);
            out.Put(attribute_count);
            out.Array(bindings, binding_count);
            out.Array(attributes, attribute_count);
        },
        [&] {
            t_record->Real().vkCmdSetVertexInputEXT(cmdbuf, binding_count, bindings,
                                                    attribute_count, attributes);
        });
}

/// SetColorWriteMaskEXT, SetColorBlendEnableEXT and SetColorBlendEquationEXT.
template <auto Member, u16 base, typename T>
struct AttachmentArray {
    static void Replay(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
        In in{payload};
        const u32 first = in.Get<u32>();
        const u32 count = in.Get<u32>();
        (dld.*Member)(cmdbuf, first, count, in.Array<T>(count));
    }

    static void VKAPI_CALL Fn(VkCommandBuffer cmdbuf, u32 first, u32 count, const T* values) {
        Handle(
            Kind::State, Range(base, MaxAttachments, first, count),
            Align16(8) + sizeof(T) * count, &Replay,
            [&](u8* p) {
                Out out{p};
                out.Put(first);
                out.Put(count);
                out.Array(values, count);
            },
            [&] { (t_record->Real().*Member)(cmdbuf, first, count, values); });
    }
};

void ReplayColorWriteEnable(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                            const u8* payload) {
    In in{payload};
    const u32 count = in.Get<u32>();
    dld.vkCmdSetColorWriteEnableEXT(cmdbuf, count, in.Array<VkBool32>(count));
}

void VKAPI_CALL SetColorWriteEnable(VkCommandBuffer cmdbuf, u32 count, const VkBool32* enables) {
    Handle(
        Kind::State,
        count <= MaxAttachments ? std::optional{SlotRange{ColorWriteEnable, 1}} : std::nullopt,
        Align16(4) + sizeof(VkBool32) * count, &ReplayColorWriteEnable,
        [&](u8* p) {
            Out out{p};
            out.Put(count);
            out.Array(enables, count);
        },
        [&] { t_record->Real().vkCmdSetColorWriteEnableEXT(cmdbuf, count, enables); });
}

void ReplayDescriptorBuffers(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                             const u8* payload) {
    In in{payload};
    const u32 count = in.Get<u32>();
    dld.vkCmdBindDescriptorBuffersEXT(cmdbuf, count,
                                      in.Array<VkDescriptorBufferBindingInfoEXT>(count));
}

void VKAPI_CALL BindDescriptorBuffers(VkCommandBuffer cmdbuf, u32 count,
                                      const VkDescriptorBufferBindingInfoEXT* infos) {
    bool loggable = count <= MaxSets;
    for (u32 i = 0; loggable && i < count; ++i) {
        loggable = infos[i].pNext == nullptr;
    }
    Handle(
        Kind::State, loggable ? std::optional{SlotRange{DescriptorBuffers, 1}} : std::nullopt,
        Align16(4) + sizeof(*infos) * count, &ReplayDescriptorBuffers,
        [&](u8* p) {
            Out out{p};
            out.Put(count);
            out.Array(infos, count);
        },
        [&] { t_record->Real().vkCmdBindDescriptorBuffersEXT(cmdbuf, count, infos); });
}

void ReplayDescriptorBufferOffsets(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                                   const u8* payload) {
    In in{payload};
    const auto bind_point = in.Get<VkPipelineBindPoint>();
    const auto layout = in.Get<VkPipelineLayout>();
    const u32 first = in.Get<u32>();
    const u32 count = in.Get<u32>();
    const u32* const indices = in.Array<u32>(count);
    dld.vkCmdSetDescriptorBufferOffsetsEXT(cmdbuf, bind_point, layout, first, count, indices,
                                           in.Array<VkDeviceSize>(count));
}

void VKAPI_CALL SetDescriptorBufferOffsets(VkCommandBuffer cmdbuf, VkPipelineBindPoint bind_point,
                                           VkPipelineLayout layout, u32 first, u32 count,
                                           const u32* indices, const VkDeviceSize* offsets) {
    const auto index = BindIndex(bind_point);
    Handle(
        Kind::State,
        index ? Range(DescriptorSet + *index * MaxSets, MaxSets, first, count) : std::nullopt,
        Align16(sizeof(bind_point) + sizeof(layout) + 8) + Align16(sizeof(u32) * count) +
            sizeof(VkDeviceSize) * count,
        &ReplayDescriptorBufferOffsets,
        [&](u8* p) {
            Out out{p};
            out.Put(bind_point);
            out.Put(layout);
            out.Put(first);
            out.Put(count);
            out.Array(indices, count);
            out.Array(offsets, count);
        },
        [&] {
            t_record->Real().vkCmdSetDescriptorBufferOffsetsEXT(cmdbuf, bind_point, layout, first,
                                                                count, indices, offsets);
        });
}

void ReplayPushDescriptor(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf,
                          const u8* payload) {
    In in{payload};
    const auto update_template = in.Get<VkDescriptorUpdateTemplate>();
    const auto layout = in.Get<VkPipelineLayout>();
    const u32 set = in.Get<u32>();
    const u32 size = in.Get<u32>();
    dld.vkCmdPushDescriptorSetWithTemplateKHR(cmdbuf, update_template, layout, set,
                                              in.Array<u8>(size));
}

void VKAPI_CALL PushDescriptorSet(VkCommandBuffer cmdbuf, VkDescriptorUpdateTemplate update_template,
                                  VkPipelineLayout layout, u32 set, const void* data) {
    VkPipelineBindPoint bind_point{};
    const size_t size = vk::DescriptorUpdateTemplateDataSize(update_template, bind_point);
    const auto index = BindIndex(bind_point);
    std::optional<SlotRange> slots;
    if (size != 0 && size <= 0x10000 && index) {
        slots = Range(DescriptorSet + *index * MaxSets, MaxSets, set, 1);
    }
    Handle(
        Kind::State, slots, Align16(sizeof(update_template) + sizeof(layout) + 8) + size,
        &ReplayPushDescriptor,
        [&](u8* p) {
            Out out{p};
            out.Put(update_template);
            out.Put(layout);
            out.Put(set);
            out.Put(static_cast<u32>(size));
            out.Array(static_cast<const u8*>(data), static_cast<u32>(size));
        },
        [&] {
            t_record->Real().vkCmdPushDescriptorSetWithTemplateKHR(cmdbuf, update_template,
                                                                   layout, set, data);
        });
}

void ReplayDepthBias2(const vk::DeviceDispatch& dld, VkCommandBuffer cmdbuf, const u8* payload) {
    In in{payload};
    auto info = in.Get<VkDepthBiasInfoEXT>();
    auto representation = in.Get<VkDepthBiasRepresentationInfoEXT>();
    info.pNext = representation.sType == VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT
                     ? &representation
                     : nullptr;
    dld.vkCmdSetDepthBias2EXT(cmdbuf, &info);
}

/// The chain may hold one VkDepthBiasRepresentationInfoEXT (the rasterizer's D24 handling).
void VKAPI_CALL SetDepthBias2(VkCommandBuffer cmdbuf, const VkDepthBiasInfoEXT* info) {
    const auto* const representation =
        static_cast<const VkDepthBiasRepresentationInfoEXT*>(info->pNext);
    const bool loggable =
        representation == nullptr ||
        (representation->sType == VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT &&
         representation->pNext == nullptr);
    Handle(
        Kind::State, loggable ? std::optional{SlotRange{DepthBias, 1}} : std::nullopt,
        sizeof(*info) + sizeof(VkDepthBiasRepresentationInfoEXT), &ReplayDepthBias2,
        [&](u8* p) {
            Out out{p};
            out.Put(*info);
            out.Put(representation != nullptr ? *representation : VkDepthBiasRepresentationInfoEXT{});
        },
        [&] { t_record->Real().vkCmdSetDepthBias2EXT(cmdbuf, info); });
}

void VKAPI_CALL BeginQuery(VkCommandBuffer cmdbuf, VkQueryPool pool, u32 query,
                           VkQueryControlFlags flags) {
    t_record->Serial(Fallback::Query);
    t_record->NoteScope(Fallback::Query, 1);
    t_record->Real().vkCmdBeginQuery(cmdbuf, pool, query, flags);
}

void VKAPI_CALL EndQuery(VkCommandBuffer cmdbuf, VkQueryPool pool, u32 query) {
    t_record->Serial(Fallback::Query);
    t_record->NoteScope(Fallback::Query, -1);
    t_record->Real().vkCmdEndQuery(cmdbuf, pool, query);
}

void VKAPI_CALL BeginConditionalRendering(VkCommandBuffer cmdbuf,
                                          const VkConditionalRenderingBeginInfoEXT* info) {
    t_record->Serial(Fallback::ConditionalRendering);
    t_record->NoteScope(Fallback::ConditionalRendering, 1);
    t_record->Real().vkCmdBeginConditionalRenderingEXT(cmdbuf, info);
}

void VKAPI_CALL EndConditionalRendering(VkCommandBuffer cmdbuf) {
    t_record->Serial(Fallback::ConditionalRendering);
    t_record->NoteScope(Fallback::ConditionalRendering, -1);
    t_record->Real().vkCmdEndConditionalRenderingEXT(cmdbuf);
}

void VKAPI_CALL BeginRenderPass(VkCommandBuffer cmdbuf, const VkRenderPassBeginInfo* info,
                                VkSubpassContents contents) {
    if (!t_record->BeginRenderPass(*info, contents)) {
        t_record->Real().vkCmdBeginRenderPass(cmdbuf, info, contents);
    }
}

void VKAPI_CALL EndRenderPass(VkCommandBuffer cmdbuf) {
    if (!t_record->EndRenderPass()) {
        t_record->Real().vkCmdEndRenderPass(cmdbuf);
    }
}

void InstallHooks(vk::DeviceDispatch& d) {
    using D = vk::DeviceDispatch;
#define HOOK(name, fn)                                                                             \
    if (d.name != nullptr) {                                                                       \
        d.name = fn;                                                                               \
    }
#define STATE(name, slot) HOOK(name, &Hook<&D::name>::State<slot>)
#define DRAW(name) HOOK(name, &Hook<&D::name>::Draw)
#define SERIAL(name, reason) HOOK(name, &Hook<&D::name>::Serial<Fallback::reason>)
    STATE(vkCmdBindIndexBuffer, IndexBuffer);
    STATE(vkCmdSetCullModeEXT, CullMode);
    STATE(vkCmdSetDepthBias, DepthBias);
    STATE(vkCmdSetDepthBiasEnableEXT, DepthBiasEnable);
    STATE(vkCmdSetDepthBounds, DepthBounds);
    STATE(vkCmdSetDepthBoundsTestEnableEXT, DepthBoundsTestEnable);
    STATE(vkCmdSetDepthCompareOpEXT, DepthCompareOp);
    STATE(vkCmdSetDepthTestEnableEXT, DepthTestEnable);
    STATE(vkCmdSetDepthWriteEnableEXT, DepthWriteEnable);
    STATE(vkCmdSetPrimitiveRestartEnableEXT, PrimitiveRestartEnable);
    STATE(vkCmdSetRasterizerDiscardEnableEXT, RasterizerDiscardEnable);
    STATE(vkCmdSetAlphaToCoverageEnableEXT, AlphaToCoverageEnable);
    STATE(vkCmdSetAlphaToOneEnableEXT, AlphaToOneEnable);
    STATE(vkCmdSetConservativeRasterizationModeEXT, ConservativeRasterizationMode);
    STATE(vkCmdSetLineRasterizationModeEXT, LineRasterizationMode);
    STATE(vkCmdSetLineStippleEnableEXT, LineStippleEnable);
    STATE(vkCmdSetLineStippleEXT, LineStipple);
    STATE(vkCmdSetLogicOpEnableEXT, LogicOpEnable);
    STATE(vkCmdSetDepthClampEnableEXT, DepthClampEnable);
    STATE(vkCmdSetFrontFaceEXT, FrontFace);
    STATE(vkCmdSetPatchControlPointsEXT, PatchControlPoints);
    STATE(vkCmdSetLogicOpEXT, LogicOp);
    STATE(vkCmdSetLineWidth, LineWidth);
    STATE(vkCmdSetPrimitiveTopologyEXT, PrimitiveTopology);
    STATE(vkCmdSetStencilTestEnableEXT, StencilTestEnable);
    HOOK(vkCmdSetStencilCompareMask, (&StencilHook<&D::vkCmdSetStencilCompareMask, StencilCompareMask>::Fn));
    HOOK(vkCmdSetStencilWriteMask, (&StencilHook<&D::vkCmdSetStencilWriteMask, StencilWriteMask>::Fn));
    HOOK(vkCmdSetStencilReference, (&StencilHook<&D::vkCmdSetStencilReference, StencilReference>::Fn));
    HOOK(vkCmdSetStencilOpEXT, (&StencilHook<&D::vkCmdSetStencilOpEXT, StencilOp>::Fn));
    HOOK(vkCmdBindPipeline, &BindPipeline);
    DRAW(vkCmdDraw);
    DRAW(vkCmdDrawIndexed);
    DRAW(vkCmdDrawIndirect);
    DRAW(vkCmdDrawIndexedIndirect);
    DRAW(vkCmdDrawIndirectCount);
    DRAW(vkCmdDrawIndexedIndirectCount);
    DRAW(vkCmdDrawMeshTasksEXT);
    HOOK(vkCmdSetViewport, &SetViewport);
    HOOK(vkCmdSetScissor, &SetScissor);
    HOOK(vkCmdBindDescriptorSets, &BindDescriptorSets);
    HOOK(vkCmdBindVertexBuffers, &BindVertexBuffers);
    HOOK(vkCmdBindVertexBuffers2EXT, &BindVertexBuffers2);
    HOOK(vkCmdPushConstants, &PushConstants);
    HOOK(vkCmdClearAttachments, &ClearAttachments);
    HOOK(vkCmdSetBlendConstants, &SetBlendConstants);
    HOOK(vkCmdSetVertexInputEXT, &SetVertexInput);
    HOOK(vkCmdSetColorWriteMaskEXT,
         (&AttachmentArray<&D::vkCmdSetColorWriteMaskEXT, ColorWriteMask, VkColorComponentFlags>::Fn));
    HOOK(vkCmdSetColorBlendEnableEXT,
         (&AttachmentArray<&D::vkCmdSetColorBlendEnableEXT, ColorBlendEnable, VkBool32>::Fn));
    HOOK(vkCmdSetColorBlendEquationEXT,
         (&AttachmentArray<&D::vkCmdSetColorBlendEquationEXT, ColorBlendEquation,
                           VkColorBlendEquationEXT>::Fn));
    HOOK(vkCmdSetColorWriteEnableEXT, &SetColorWriteEnable);
    HOOK(vkCmdBindDescriptorBuffersEXT, &BindDescriptorBuffers);
    HOOK(vkCmdSetDescriptorBufferOffsetsEXT, &SetDescriptorBufferOffsets);
    HOOK(vkCmdPushDescriptorSetWithTemplateKHR, &PushDescriptorSet);
    HOOK(vkCmdSetDepthBias2EXT, &SetDepthBias2);
    HOOK(vkCmdBeginRenderPass, &BeginRenderPass);
    HOOK(vkCmdEndRenderPass, &EndRenderPass);
    HOOK(vkCmdBeginConditionalRenderingEXT, &BeginConditionalRendering);
    HOOK(vkCmdEndConditionalRenderingEXT, &EndConditionalRendering);
    SERIAL(vkCmdBeginDebugUtilsLabelEXT, DebugLabel);
    SERIAL(vkCmdEndDebugUtilsLabelEXT, DebugLabel);
    HOOK(vkCmdBeginQuery, &BeginQuery);
    HOOK(vkCmdEndQuery, &EndQuery);
    SERIAL(vkCmdResetQueryPool, Query);
    SERIAL(vkCmdCopyQueryPoolResults, Query);
    SERIAL(vkCmdBeginTransformFeedbackEXT, TransformFeedback);
    SERIAL(vkCmdEndTransformFeedbackEXT, TransformFeedback);
    SERIAL(vkCmdBindTransformFeedbackBuffersEXT, TransformFeedback);
    SERIAL(vkCmdDrawIndirectByteCountEXT, TransformFeedback);
    SERIAL(vkCmdPipelineBarrier, Barrier);
    SERIAL(vkCmdPipelineBarrier2, Barrier);
    SERIAL(vkCmdWriteTimestamp, Timestamp);
    SERIAL(vkCmdSetEvent, Event);
    SERIAL(vkCmdWaitEvents, Event);
    SERIAL(vkCmdBlitImage, OutsideCommand);
    SERIAL(vkCmdClearColorImage, OutsideCommand);
    SERIAL(vkCmdCopyBuffer, OutsideCommand);
    SERIAL(vkCmdCopyBufferToImage, OutsideCommand);
    SERIAL(vkCmdCopyImage, OutsideCommand);
    SERIAL(vkCmdCopyImageToBuffer, OutsideCommand);
    SERIAL(vkCmdDispatch, OutsideCommand);
    SERIAL(vkCmdDispatchIndirect, OutsideCommand);
    SERIAL(vkCmdFillBuffer, OutsideCommand);
    SERIAL(vkCmdResolveImage, OutsideCommand);
    SERIAL(vkCmdExecuteCommands, OutsideCommand);
#undef SERIAL
#undef DRAW
#undef STATE
#undef HOOK
}

} // Anonymous namespace

u32 ParallelRecord::ThreadsFromEnv() {
    const char* const value = std::getenv("EDEN_NX2_PARALLEL_RECORD");
    if (value == nullptr) {
        return 0;
    }
    const u32 threads = static_cast<u32>(std::strtoul(value, nullptr, 10));
    return threads < 2 ? 0 : std::min(threads, 8U);
}

ParallelRecord::ParallelRecord(const vk::DeviceDispatch& dld, u32 threads_, SecondarySource& source_,
                               u32 min_pass_draws_, u32 min_range_draws_)
    : real{dld}, capture{dld}, source{source_}, threads{threads_},
      min_pass_draws{std::max(min_pass_draws_, min_range_draws_ * 2)},
      min_range_draws{std::max(min_range_draws_, 1U)}, last_writer(SlotCount, None),
      jobs(threads_) {
    InstallHooks(capture);
    stats.record_ns.resize(threads);
    helpers.reserve(threads - 1);
    for (u32 index = 0; index + 1 < threads; ++index) {
        helpers.push_back(std::make_unique<Helper>());
    }
    for (u32 index = 0; index + 1 < threads; ++index) {
        helpers[index]->thread = std::thread([this, index] { HelperLoop(index); });
    }
    LOG_INFO(Render_Vulkan,
             "Parallel render pass recording on {} threads (passes of {}+ draws, {}+ draws per "
             "secondary)",
             threads, min_pass_draws, min_range_draws);
}

ParallelRecord::~ParallelRecord() {
    stopping.store(true);
    for (const auto& helper : helpers) {
        helper->generation.fetch_add(1, std::memory_order_release);
        helper->generation.notify_one();
    }
    for (const auto& helper : helpers) {
        helper->thread.join();
    }
}

vk::CommandBuffer ParallelRecord::Wrap(VkCommandBuffer primary_) {
    t_record = this;
    primary = primary_;
    return vk::CommandBuffer(primary_, capture);
}

void ParallelRecord::BeginPrimary() {
    if (mode != Mode::Passthrough) {
        LOG_ERROR(Render_Vulkan, "Parallel record: a command buffer ended inside a render pass");
        mode = Mode::Passthrough;
    }
    ops.clear();
    bytes.clear();
    std::ranges::fill(last_writer, None);
    state_unknown = false;
    open_queries = 0;
    open_conditionals = 0;
}

void ParallelRecord::Serial(Fallback reason) {
    if (mode == Mode::Capturing) {
        FallBack(reason);
    }
}

void ParallelRecord::Unlogged(Fallback reason) {
    Serial(reason);
    state_unknown = true;
}

bool ParallelRecord::BeginRenderPass(const VkRenderPassBeginInfo& info,
                                     VkSubpassContents contents) {
    if (mode != Mode::Passthrough) {
        return false;
    }
    MaybeCompact();
    std::optional<Fallback> reason;
    if (state_unknown) {
        reason = Fallback::UnloggedState;
    } else if (open_queries > 0) {
        reason = Fallback::Query;
    } else if (open_conditionals > 0) {
        reason = Fallback::ConditionalRendering;
    } else if (info.pNext != nullptr || info.clearValueCount > pass_clear_values.size() ||
               contents != VK_SUBPASS_CONTENTS_INLINE) {
        reason = Fallback::RenderPassInfo;
    }
    if (reason) {
        ++stats.fallbacks[static_cast<size_t>(*reason)];
        mode = Mode::SerialPass;
        return false;
    }
    pass_info = info;
    if (info.clearValueCount != 0) {
        std::copy_n(info.pClearValues, info.clearValueCount, pass_clear_values.begin());
        pass_info.pClearValues = pass_clear_values.data();
    }
    pass_first_op = static_cast<u32>(ops.size());
    pass_draws = 0;
    pass_writers = last_writer;
    mode = Mode::Capturing;
    return true;
}

bool ParallelRecord::EndRenderPass() {
    if (mode == Mode::Passthrough) {
        return false;
    }
    ++stats.passes;
    if (mode == Mode::Capturing) {
        if (pass_draws < min_pass_draws) {
            FallBack(Fallback::Small);
        } else {
            FinishParallel();
        }
    }
    if (mode == Mode::SerialPass) {
        real.vkCmdEndRenderPass(primary);
    }
    mode = Mode::Passthrough;
    Report();
    return true;
}

void ParallelRecord::FallBack(Fallback reason) {
    ++stats.fallbacks[static_cast<size_t>(reason)];
    stats.inline_draws += pass_draws;
    real.vkCmdBeginRenderPass(primary, &pass_info, VK_SUBPASS_CONTENTS_INLINE);
    Replay(primary, {}, pass_first_op, static_cast<u32>(ops.size()));
    mode = Mode::SerialPass;
}

void ParallelRecord::Prefix(const std::vector<u32>& writers, bool graphics_only,
                            std::vector<u32>& out) const {
    out.clear();
    for (u32 slot = 0; slot < SlotCount; ++slot) {
        if (writers[slot] != None && !(graphics_only && IsComputeSlot(slot))) {
            out.push_back(writers[slot]);
        }
    }
    std::ranges::sort(out);
    const auto [first, last] = std::ranges::unique(out);
    out.erase(first, last);
}

void ParallelRecord::Replay(VkCommandBuffer cmdbuf, const std::vector<u32>& prefix, u32 begin,
                            u32 end) const {
    const u8* const base = bytes.data();
    for (const u32 index : prefix) {
        ops[index].replay(real, cmdbuf, base + ops[index].payload);
    }
    for (u32 index = begin; index < end; ++index) {
        ops[index].replay(real, cmdbuf, base + ops[index].payload);
    }
}

void ParallelRecord::RecordJob(u32 thread, Job& job) {
    const u64 start = VideoCommon::Nx2Profile::Now();
    job.failed = false;
    job.result = source.Acquire(thread);
    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                 VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &inheritance,
    };
    if (real.vkBeginCommandBuffer(job.result, &begin_info) != VK_SUCCESS) {
        job.failed = true;
    } else {
        Replay(job.result, job.prefix, job.begin, job.end);
        job.failed = real.vkEndCommandBuffer(job.result) != VK_SUCCESS;
    }
    job.record_ns = VideoCommon::Nx2Profile::Now() - start;
}

void ParallelRecord::FinishParallel() {
    const u64 split_start = VideoCommon::Nx2Profile::Now();
    const u32 ranges = std::min(threads, pass_draws / min_range_draws);
    const u32 end = static_cast<u32>(ops.size());
    std::vector<u32>& writers = scratch;
    writers = pass_writers;
    Prefix(writers, true, jobs[0].prefix);
    jobs[0].begin = pass_first_op;
    u32 range = 0;
    u32 draws = 0;
    for (u32 index = pass_first_op; index < end && range + 1 < ranges; ++index) {
        const Op& op = ops[index];
        for (u32 slot = op.slots.first; slot < u32{op.slots.first} + op.slots.count; ++slot) {
            writers[slot] = index;
        }
        if (op.kind != Kind::Draw ||
            ++draws != static_cast<u32>(u64{range + 1} * pass_draws / ranges)) {
            continue;
        }
        jobs[range].end = index + 1;
        ++range;
        jobs[range].begin = index + 1;
        Prefix(writers, true, jobs[range].prefix);
    }
    jobs[range].end = end;
    const u32 used = range + 1;
    inheritance = VkCommandBufferInheritanceInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .pNext = nullptr,
        .renderPass = pass_info.renderPass,
        .subpass = 0,
        .framebuffer = pass_info.framebuffer,
        .occlusionQueryEnable = VK_FALSE,
        .queryFlags = 0,
        .pipelineStatistics = 0,
    };
    pending.store(used - 1, std::memory_order_relaxed);
    for (u32 index = 1; index < used; ++index) {
        Helper& helper = *helpers[index - 1];
        helper.generation.fetch_add(1, std::memory_order_release);
        helper.generation.notify_one();
    }
    const u64 record_start = VideoCommon::Nx2Profile::Now();
    stats.split_ns += record_start - split_start;
    RecordJob(0, jobs[0]);
    const u64 wait_start = VideoCommon::Nx2Profile::Now();
    for (u32 left = pending.load(std::memory_order_acquire); left != 0;
         left = pending.load(std::memory_order_acquire)) {
        pending.wait(left, std::memory_order_acquire);
    }
    stats.wait_ns += VideoCommon::Nx2Profile::Now() - wait_start;
    bool failed = false;
    secondaries.clear();
    for (u32 index = 0; index < used; ++index) {
        failed |= jobs[index].failed;
        secondaries.push_back(jobs[index].result);
        stats.record_ns[index] += jobs[index].record_ns;
    }
    if (failed) {
        ++stats.failures;
        FallBack(Fallback::RenderPassInfo);
        return;
    }
    real.vkCmdBeginRenderPass(primary, &pass_info, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    real.vkCmdExecuteCommands(primary, used, secondaries.data());
    real.vkCmdEndRenderPass(primary);
    Prefix(last_writer, false, scratch);
    Replay(primary, scratch, 0, 0);
    ++stats.parallel_passes;
    stats.parallel_draws += pass_draws;
    stats.secondaries += used;
}

void ParallelRecord::MaybeCompact() {
    if (ops.size() < 8192 && bytes.size() < (4U << 20)) {
        return;
    }
    Prefix(last_writer, false, scratch);
    std::vector<Op> kept_ops;
    std::vector<u8> kept_bytes;
    kept_ops.reserve(scratch.size());
    for (const u32 index : scratch) {
        const Op& op = ops[index];
        const u32 next = index + 1 < ops.size() ? ops[index + 1].payload
                                                 : static_cast<u32>(bytes.size());
        const u32 offset = static_cast<u32>(kept_bytes.size());
        kept_bytes.insert(kept_bytes.end(), bytes.begin() + op.payload, bytes.begin() + next);
        kept_ops.push_back(Op{op.replay, offset, op.slots, op.kind});
    }
    for (u32& writer : last_writer) {
        if (writer != None) {
            writer = static_cast<u32>(std::ranges::lower_bound(scratch, writer) - scratch.begin());
        }
    }
    ops = std::move(kept_ops);
    bytes = std::move(kept_bytes);
}

void ParallelRecord::HelperLoop(u32 index) {
    Common::SetCurrentThreadName(fmt::format("VulkanRecord{}", index + 1).c_str());
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    Helper& helper = *helpers[index];
    u32 seen = 0;
    while (true) {
        helper.generation.wait(seen, std::memory_order_acquire);
        seen = helper.generation.load(std::memory_order_acquire);
        if (stopping.load()) {
            return;
        }
        RecordJob(index + 1, jobs[index + 1]);
        if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            pending.notify_one();
        }
    }
}

void ParallelRecord::Report() {
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    const u64 now = VideoCommon::Nx2Profile::Now();
    if (stats.window_start == 0) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - stats.window_start) / 1e9;
    const auto per_second = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto ms_per_second = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    std::string record;
    for (size_t index = 0; index < stats.record_ns.size(); ++index) {
        record += fmt::format(" t{} {:.1f}", index, ms_per_second(stats.record_ns[index]));
    }
    std::string fallbacks;
    for (size_t index = 0; index < stats.fallbacks.size(); ++index) {
        if (stats.fallbacks[index] != 0) {
            fallbacks += fmt::format(" {} {:.0f}", FallbackName(static_cast<Fallback>(index)),
                                     per_second(stats.fallbacks[index]));
        }
    }
    LOG_INFO(Render_Vulkan,
             "Parallel record per second: {:.0f} render passes, {:.0f} split into {:.0f} "
             "secondaries | draws (packets): {:.0f} parallel, {:.0f} inline | worker split "
             "{:.1f} ms, wait {:.1f} ms | record ms by thread:{} | inline passes by reason:{} | "
             "failed {}",
             per_second(stats.passes), per_second(stats.parallel_passes),
             per_second(stats.secondaries), per_second(stats.parallel_draws),
             per_second(stats.inline_draws), ms_per_second(stats.split_ns),
             ms_per_second(stats.wait_ns), record, fallbacks, stats.failures);
    const std::vector<u64> zero(stats.record_ns.size());
    stats = Stats{};
    stats.record_ns = zero;
    stats.window_start = now;
}

} // namespace Vulkan
