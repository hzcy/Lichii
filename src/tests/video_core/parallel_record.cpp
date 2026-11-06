// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_parallel_record.h"

namespace {

using namespace Vulkan;

enum class Type { Set, Draw, Dispatch, Action, BeginPass, EndPass, Execute };

struct Event {
    Type type;
    std::string key;
    std::string value;
    std::vector<VkCommandBuffer> secondaries;
};

std::mutex g_mutex;
std::map<VkCommandBuffer, std::vector<Event>> g_log;
std::map<VkCommandBuffer, VkSubpassContents> g_contents;

void Push(VkCommandBuffer cmdbuf, Event event) {
    std::scoped_lock lock{g_mutex};
    g_log[cmdbuf].push_back(std::move(event));
}

void Set(VkCommandBuffer cmdbuf, std::string key, std::string value) {
    Push(cmdbuf, {Type::Set, std::move(key), std::move(value), {}});
}

template <typename T>
u64 Bits(T handle) {
    return static_cast<u64>(reinterpret_cast<std::uintptr_t>(handle));
}

template <typename T>
T Handle(u64 value) {
    return reinterpret_cast<T>(static_cast<std::uintptr_t>(value));
}

VkResult VKAPI_CALL MockBegin(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
}
VkResult VKAPI_CALL MockEnd(VkCommandBuffer) {
    return VK_SUCCESS;
}
void VKAPI_CALL MockBeginPass(VkCommandBuffer cmdbuf, const VkRenderPassBeginInfo* info,
                              VkSubpassContents contents) {
    Push(cmdbuf, {Type::BeginPass, "", fmt::format("{}", Bits(info->renderPass)), {}});
    std::scoped_lock lock{g_mutex};
    g_contents[cmdbuf] = contents;
}
void VKAPI_CALL MockEndPass(VkCommandBuffer cmdbuf) {
    Push(cmdbuf, {Type::EndPass, "", "", {}});
}
void VKAPI_CALL MockExecute(VkCommandBuffer cmdbuf, u32 count, const VkCommandBuffer* buffers) {
    Push(cmdbuf, {Type::Execute, "", "", {buffers, buffers + count}});
}
void VKAPI_CALL MockBindPipeline(VkCommandBuffer cmdbuf, VkPipelineBindPoint bind_point,
                                 VkPipeline pipeline) {
    Set(cmdbuf, fmt::format("pipeline{}", static_cast<int>(bind_point)),
        fmt::format("{}", Bits(pipeline)));
}
void VKAPI_CALL MockViewport(VkCommandBuffer cmdbuf, u32 first, u32 count, const VkViewport* v) {
    for (u32 i = 0; i < count; ++i) {
        Set(cmdbuf, fmt::format("viewport{}", first + i), fmt::format("{}", v[i].x));
    }
}
void VKAPI_CALL MockSets(VkCommandBuffer cmdbuf, VkPipelineBindPoint bind_point, VkPipelineLayout,
                         u32 first, u32 count, const VkDescriptorSet* sets, u32 dynamic_count,
                         const u32* offsets) {
    std::string dynamic;
    for (u32 i = 0; i < dynamic_count; ++i) {
        dynamic += fmt::format(",{}", offsets[i]);
    }
    for (u32 i = 0; i < count; ++i) {
        Set(cmdbuf, fmt::format("set{}:{}", static_cast<int>(bind_point), first + i),
            fmt::format("{}{}", Bits(sets[i]), dynamic));
    }
}
void VKAPI_CALL MockPush(VkCommandBuffer cmdbuf, VkPipelineLayout, VkShaderStageFlags stages,
                         u32 offset, u32 size, const void* values) {
    const auto* const bytes = static_cast<const u8*>(values);
    const bool compute = stages == VK_SHADER_STAGE_COMPUTE_BIT;
    for (u32 i = 0; i < size; ++i) {
        Set(cmdbuf, fmt::format("push{}:{}", compute ? "c" : "g", offset + i),
            fmt::format("{}", bytes[i]));
    }
}
void VKAPI_CALL MockVertexBuffers(VkCommandBuffer cmdbuf, u32 first, u32 count,
                                  const VkBuffer* buffers, const VkDeviceSize* offsets) {
    for (u32 i = 0; i < count; ++i) {
        Set(cmdbuf, fmt::format("vb{}", first + i),
            fmt::format("{}+{}", Bits(buffers[i]), offsets[i]));
    }
}
void VKAPI_CALL MockDepthBias2(VkCommandBuffer cmdbuf, const VkDepthBiasInfoEXT* info) {
    const auto* const representation =
        static_cast<const VkDepthBiasRepresentationInfoEXT*>(info->pNext);
    Set(cmdbuf, "depth_bias",
        fmt::format("{} {}", info->depthBiasConstantFactor,
                    representation != nullptr ? static_cast<int>(representation->depthBiasRepresentation)
                                              : -1));
}
void VKAPI_CALL MockCull(VkCommandBuffer cmdbuf, VkCullModeFlags mode) {
    Set(cmdbuf, "cull", fmt::format("{}", mode));
}
void VKAPI_CALL MockStencilReference(VkCommandBuffer cmdbuf, VkStencilFaceFlags faces, u32 ref) {
    if (faces & VK_STENCIL_FACE_FRONT_BIT) {
        Set(cmdbuf, "sref_front", fmt::format("{}", ref));
    }
    if (faces & VK_STENCIL_FACE_BACK_BIT) {
        Set(cmdbuf, "sref_back", fmt::format("{}", ref));
    }
}
void VKAPI_CALL MockDraw(VkCommandBuffer cmdbuf, u32 vertices, u32 instances, u32 first_vertex,
                         u32 first_instance) {
    Push(cmdbuf, {Type::Draw, "",
                  fmt::format("{} {} {} {}", vertices, instances, first_vertex, first_instance),
                  {}});
}
void VKAPI_CALL MockDispatch(VkCommandBuffer cmdbuf, u32 x, u32 y, u32 z) {
    Push(cmdbuf, {Type::Dispatch, "", fmt::format("{} {} {}", x, y, z), {}});
}
void VKAPI_CALL MockBeginQuery(VkCommandBuffer cmdbuf, VkQueryPool, u32 query,
                               VkQueryControlFlags) {
    Push(cmdbuf, {Type::Action, "", fmt::format("begin query {}", query), {}});
}
void VKAPI_CALL MockEndQuery(VkCommandBuffer cmdbuf, VkQueryPool, u32 query) {
    Push(cmdbuf, {Type::Action, "", fmt::format("end query {}", query), {}});
}

vk::DeviceDispatch MockDispatchTable() {
    vk::DeviceDispatch dld{};
    dld.vkBeginCommandBuffer = &MockBegin;
    dld.vkEndCommandBuffer = &MockEnd;
    dld.vkCmdBeginRenderPass = &MockBeginPass;
    dld.vkCmdEndRenderPass = &MockEndPass;
    dld.vkCmdExecuteCommands = &MockExecute;
    dld.vkCmdBindPipeline = &MockBindPipeline;
    dld.vkCmdSetViewport = &MockViewport;
    dld.vkCmdBindDescriptorSets = &MockSets;
    dld.vkCmdPushConstants = &MockPush;
    dld.vkCmdBindVertexBuffers = &MockVertexBuffers;
    dld.vkCmdSetCullModeEXT = &MockCull;
    dld.vkCmdSetDepthBias2EXT = &MockDepthBias2;
    dld.vkCmdSetStencilReference = &MockStencilReference;
    dld.vkCmdDraw = &MockDraw;
    dld.vkCmdDispatch = &MockDispatch;
    dld.vkCmdBeginQuery = &MockBeginQuery;
    dld.vkCmdEndQuery = &MockEndQuery;
    return dld;
}

class FakeSecondaries final : public ParallelRecord::SecondarySource {
public:
    VkCommandBuffer Acquire(u32) override {
        return Handle<VkCommandBuffer>(next.fetch_add(1));
    }

private:
    std::atomic<u64> next{0x100000};
};

std::string Snapshot(const std::map<std::string, std::string>& state, bool compute) {
    std::string out;
    for (const auto& [key, value] : state) {
        const bool compute_key = key == "pipeline1" || key.starts_with("set1:") ||
                                 key.starts_with("pushc:");
        if (compute_key == compute) {
            out += fmt::format("{}={};", key, value);
        }
    }
    return out;
}

/// What each draw, dispatch and query of a primary saw, in GPU order.
std::vector<std::string> Evaluate(VkCommandBuffer primary, u32& executes) {
    std::vector<std::string> results;
    std::map<std::string, std::string> state;
    s32 open_queries = 0;
    const auto apply = [&](const Event& event, std::map<std::string, std::string>& bound) {
        switch (event.type) {
        case Type::Set:
            bound[event.key] = event.value;
            break;
        case Type::Draw:
            results.push_back("draw " + event.value + " | " + Snapshot(bound, false));
            break;
        case Type::Dispatch:
            results.push_back("dispatch " + event.value + " | " + Snapshot(bound, true));
            break;
        case Type::Action:
            open_queries += event.value.starts_with("begin query") ? 1 : -1;
            results.push_back(event.value);
            break;
        case Type::BeginPass:
            results.push_back("begin pass " + event.value);
            break;
        case Type::EndPass:
            results.push_back("end pass");
            break;
        case Type::Execute:
            break;
        }
    };
    for (const Event& event : g_log[primary]) {
        if (event.type != Type::Execute) {
            apply(event, state);
            continue;
        }
        ++executes;
        // Secondaries don't inherit active queries.
        REQUIRE(open_queries == 0);
        for (const VkCommandBuffer secondary : event.secondaries) {
            std::map<std::string, std::string> secondary_state;
            for (const Event& inner : g_log[secondary]) {
                REQUIRE(inner.type != Type::BeginPass);
                REQUIRE(inner.type != Type::Execute);
                apply(inner, secondary_state);
            }
        }
        // Bound state is undefined after vkCmdExecuteCommands.
        state.clear();
    }
    return results;
}

void EmitStream(vk::CommandBuffer cmdbuf, u32 seed) {
    std::mt19937 rng{seed};
    const auto pick = [&](u32 count) { return static_cast<u32>(rng() % count); };
    const auto state_change = [&] {
        switch (pick(8)) {
        case 7: {
            VkDepthBiasRepresentationInfoEXT representation{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation = static_cast<VkDepthBiasRepresentationEXT>(pick(2)),
                .depthBiasExact = VK_FALSE,
            };
            const float constant = static_cast<float>(pick(50));
            if (pick(2) == 0) {
                cmdbuf.SetDepthBias(constant, 0.0f, 1.0f, &representation);
            } else {
                cmdbuf.SetDepthBias(constant, 0.0f, 1.0f, nullptr);
            }
            break;
        }
        case 0:
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, Handle<VkPipeline>(1 + pick(4)));
            break;
        case 1: {
            std::array<VkViewport, 2> viewports{};
            viewports[0].x = static_cast<float>(pick(100));
            viewports[1].x = static_cast<float>(pick(100));
            cmdbuf.SetViewport(pick(4), vk::Span<VkViewport>(viewports.data(), 1 + pick(2)));
            break;
        }
        case 2: {
            const std::array sets{Handle<VkDescriptorSet>(1 + pick(50)),
                                  Handle<VkDescriptorSet>(1 + pick(50))};
            const std::array<u32, 2> offsets{pick(256), pick(256)};
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, Handle<VkPipelineLayout>(7),
                                      pick(3), vk::Span<VkDescriptorSet>(sets.data(), 1 + pick(2)),
                                      vk::Span<u32>(offsets.data(), pick(3)));
            break;
        }
        case 3: {
            std::array<u8, 16> data{};
            for (u8& byte : data) {
                byte = static_cast<u8>(pick(256));
            }
            cmdbuf.PushConstants(Handle<VkPipelineLayout>(7), VK_SHADER_STAGE_ALL_GRAPHICS,
                                 pick(16) * 4, 4 * (1 + pick(4)), data.data());
            break;
        }
        case 4: {
            const std::array buffers{Handle<VkBuffer>(1 + pick(9)), Handle<VkBuffer>(1 + pick(9)),
                                     Handle<VkBuffer>(1 + pick(9))};
            const std::array<VkDeviceSize, 3> offsets{pick(64), pick(64), pick(64)};
            cmdbuf.BindVertexBuffers(pick(5), 1 + pick(3), buffers.data(), offsets.data());
            break;
        }
        case 5:
            cmdbuf.SetCullModeEXT(pick(4));
            break;
        default:
            cmdbuf.SetStencilReference(1 + pick(3), pick(255));
            break;
        }
    };
    constexpr std::array<u32, 6> pass_draws{0, 5, 40, 150, 700, 97};
    for (u32 pass = 0; pass < 40; ++pass) {
        for (u32 i = pick(6); i != 0; --i) {
            state_change();
        }
        if (pick(3) == 0) {
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, Handle<VkPipeline>(100 + pick(3)));
            const u32 value = pick(1000);
            cmdbuf.PushConstants(Handle<VkPipelineLayout>(8), VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                                 &value);
            cmdbuf.Dispatch(pass, 1, 1);
        }
        const VkRenderPassBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = Handle<VkRenderPass>(500 + pass),
            .framebuffer = Handle<VkFramebuffer>(600 + pass),
            .renderArea = {},
            .clearValueCount = 0,
            .pClearValues = nullptr,
        };
        const bool spanning_query = pick(5) == 0;
        if (spanning_query) {
            cmdbuf.BeginQuery(Handle<VkQueryPool>(9), 1000 + pass, 0);
        }
        cmdbuf.BeginRenderPass(info, VK_SUBPASS_CONTENTS_INLINE);
        const u32 draws = pass_draws[pick(pass_draws.size())];
        const bool query = pick(4) == 0;
        for (u32 draw = 0; draw < draws; ++draw) {
            for (u32 i = pick(4); i != 0; --i) {
                state_change();
            }
            if (query && draw == draws / 2) {
                cmdbuf.BeginQuery(Handle<VkQueryPool>(9), draw, 0);
                cmdbuf.EndQuery(Handle<VkQueryPool>(9), draw);
            }
            cmdbuf.Draw(3, 1, draw, pass);
        }
        cmdbuf.EndRenderPass();
        if (spanning_query) {
            cmdbuf.EndQuery(Handle<VkQueryPool>(9), 1000 + pass);
        }
        cmdbuf.Dispatch(pass, 2, 1);
    }
}

} // Anonymous namespace

TEST_CASE("ParallelRecord gives every draw the state it has inline", "[video_core][vulkan]") {
    const vk::DeviceDispatch dld = MockDispatchTable();
    for (const u32 threads : {2U, 3U, 4U, 8U}) {
        for (u32 seed = 1; seed <= 6; ++seed) {
            {
                std::scoped_lock lock{g_mutex};
                g_log.clear();
                g_contents.clear();
            }
            const VkCommandBuffer serial = Handle<VkCommandBuffer>(0x10);
            const VkCommandBuffer parallel = Handle<VkCommandBuffer>(0x20);
            EmitStream(vk::CommandBuffer(serial, dld), seed);

            FakeSecondaries secondaries;
            {
                ParallelRecord record{dld, threads, secondaries};
                record.BeginPrimary();
                EmitStream(record.Wrap(parallel), seed);
            }

            u32 serial_executes = 0;
            u32 parallel_executes = 0;
            const auto expected = Evaluate(serial, serial_executes);
            const auto actual = Evaluate(parallel, parallel_executes);
            REQUIRE(serial_executes == 0);
            REQUIRE(parallel_executes > 0);
            // Every pass large enough is split unless a query is open in it.
            REQUIRE(parallel_executes >= 8);
            REQUIRE(actual.size() == expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                INFO("threads " << threads << " seed " << seed << " result " << i);
                REQUIRE(actual[i] == expected[i]);
            }
        }
    }
}
