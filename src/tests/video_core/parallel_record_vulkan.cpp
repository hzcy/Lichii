// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/dynamic_library.h"
#include "video_core/renderer_vulkan/vk_parallel_record.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

// EDEN_TEST_VULKAN=1 runs this on the first Vulkan device, with VK_LAYER_KHRONOS_validation when
// it is installed. For lavapipe: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json.

namespace {

using namespace Vulkan;

// q.vert: a rect from push constants as a 4-vertex strip; q.frag: the push constant color.
constexpr u32 VERT_SPV[] = {
0x07230203,0x00010000,0x0008000b,0x00000032,0x00000000,0x00020011,0x00000001,0x0006000b,
0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x0000000c,0x0000001c,0x00030003,
0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00030005,0x00000009,
0x00007675,0x00060005,0x0000000c,0x565f6c67,0x65747265,0x646e4978,0x00007865,0x00060005,
0x0000001a,0x505f6c67,0x65567265,0x78657472,0x00000000,0x00060006,0x0000001a,0x00000000,
0x505f6c67,0x7469736f,0x006e6f69,0x00070006,0x0000001a,0x00000001,0x505f6c67,0x746e696f,
0x657a6953,0x00000000,0x00070006,0x0000001a,0x00000002,0x435f6c67,0x4470696c,0x61747369,
0x0065636e,0x00070006,0x0000001a,0x00000003,0x435f6c67,0x446c6c75,0x61747369,0x0065636e,
0x00030005,0x0000001c,0x00000000,0x00030005,0x0000001e,0x00000050,0x00050006,0x0000001e,
0x00000000,0x74636572,0x00000000,0x00050006,0x0000001e,0x00000001,0x6f6c6f63,0x00000072,
0x00030005,0x00000020,0x00000070,0x00040047,0x0000000c,0x0000000b,0x0000002a,0x00030047,
0x0000001a,0x00000002,0x00050048,0x0000001a,0x00000000,0x0000000b,0x00000000,0x00050048,
0x0000001a,0x00000001,0x0000000b,0x00000001,0x00050048,0x0000001a,0x00000002,0x0000000b,
0x00000003,0x00050048,0x0000001a,0x00000003,0x0000000b,0x00000004,0x00030047,0x0000001e,
0x00000002,0x00050048,0x0000001e,0x00000000,0x00000023,0x00000000,0x00050048,0x0000001e,
0x00000001,0x00000023,0x00000010,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,
0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000002,0x00040020,
0x00000008,0x00000007,0x00000007,0x00040015,0x0000000a,0x00000020,0x00000001,0x00040020,
0x0000000b,0x00000001,0x0000000a,0x0004003b,0x0000000b,0x0000000c,0x00000001,0x0004002b,
0x0000000a,0x0000000e,0x00000001,0x00040017,0x00000016,0x00000006,0x00000004,0x00040015,
0x00000017,0x00000020,0x00000000,0x0004002b,0x00000017,0x00000018,0x00000001,0x0004001c,
0x00000019,0x00000006,0x00000018,0x0006001e,0x0000001a,0x00000016,0x00000006,0x00000019,
0x00000019,0x00040020,0x0000001b,0x00000003,0x0000001a,0x0004003b,0x0000001b,0x0000001c,
0x00000003,0x0004002b,0x0000000a,0x0000001d,0x00000000,0x0004001e,0x0000001e,0x00000016,
0x00000016,0x00040020,0x0000001f,0x00000009,0x0000001e,0x0004003b,0x0000001f,0x00000020,
0x00000009,0x00040020,0x00000021,0x00000009,0x00000016,0x0004002b,0x00000006,0x0000002b,
0x00000000,0x0004002b,0x00000006,0x0000002c,0x3f800000,0x00040020,0x00000030,0x00000003,
0x00000016,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,
0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003d,0x0000000a,0x0000000d,0x0000000c,
0x000500c7,0x0000000a,0x0000000f,0x0000000d,0x0000000e,0x0004006f,0x00000006,0x00000010,
0x0000000f,0x0004003d,0x0000000a,0x00000011,0x0000000c,0x000500c3,0x0000000a,0x00000012,
0x00000011,0x0000000e,0x000500c7,0x0000000a,0x00000013,0x00000012,0x0000000e,0x0004006f,
0x00000006,0x00000014,0x00000013,0x00050050,0x00000007,0x00000015,0x00000010,0x00000014,
0x0003003e,0x00000009,0x00000015,0x00050041,0x00000021,0x00000022,0x00000020,0x0000001d,
0x0004003d,0x00000016,0x00000023,0x00000022,0x0007004f,0x00000007,0x00000024,0x00000023,
0x00000023,0x00000000,0x00000001,0x0004003d,0x00000007,0x00000025,0x00000009,0x00050041,
0x00000021,0x00000026,0x00000020,0x0000001d,0x0004003d,0x00000016,0x00000027,0x00000026,
0x0007004f,0x00000007,0x00000028,0x00000027,0x00000027,0x00000002,0x00000003,0x00050085,
0x00000007,0x00000029,0x00000025,0x00000028,0x00050081,0x00000007,0x0000002a,0x00000024,
0x00000029,0x00050051,0x00000006,0x0000002d,0x0000002a,0x00000000,0x00050051,0x00000006,
0x0000002e,0x0000002a,0x00000001,0x00070050,0x00000016,0x0000002f,0x0000002d,0x0000002e,
0x0000002b,0x0000002c,0x00050041,0x00000030,0x00000031,0x0000001c,0x0000001d,0x0003003e,
0x00000031,0x0000002f,0x000100fd,0x00010038
};

constexpr u32 FRAG_SPV[] = {
0x07230203,0x00010000,0x0008000b,0x00000012,0x00000000,0x00020011,0x00000001,0x0006000b,
0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
0x0006000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000009,0x00030010,0x00000004,
0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,0x00000000,
0x00030005,0x00000009,0x0000006f,0x00030005,0x0000000a,0x00000050,0x00050006,0x0000000a,
0x00000000,0x74636572,0x00000000,0x00050006,0x0000000a,0x00000001,0x6f6c6f63,0x00000072,
0x00030005,0x0000000c,0x00000070,0x00040047,0x00000009,0x0000001e,0x00000000,0x00030047,
0x0000000a,0x00000002,0x00050048,0x0000000a,0x00000000,0x00000023,0x00000000,0x00050048,
0x0000000a,0x00000001,0x00000023,0x00000010,0x00020013,0x00000002,0x00030021,0x00000003,
0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000004,
0x00040020,0x00000008,0x00000003,0x00000007,0x0004003b,0x00000008,0x00000009,0x00000003,
0x0004001e,0x0000000a,0x00000007,0x00000007,0x00040020,0x0000000b,0x00000009,0x0000000a,
0x0004003b,0x0000000b,0x0000000c,0x00000009,0x00040015,0x0000000d,0x00000020,0x00000001,
0x0004002b,0x0000000d,0x0000000e,0x00000001,0x00040020,0x0000000f,0x00000009,0x00000007,
0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x00050041,
0x0000000f,0x00000010,0x0000000c,0x0000000e,0x0004003d,0x00000007,0x00000011,0x00000010,
0x0003003e,0x00000009,0x00000011,0x000100fd,0x00010038
};

u32 g_validation_messages = 0;
std::vector<std::string> g_validation_log;

VkBool32 VKAPI_CALL OnMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                              VkDebugUtilsMessageTypeFlagsEXT,
                              const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        ++g_validation_messages;
        if (g_validation_log.size() < 16) {
            g_validation_log.emplace_back(data->pMessage);
        }
    }
    return VK_FALSE;
}

class PoolSource final : public ParallelRecord::SecondarySource {
public:
    PoolSource(const vk::Device& device, u32 threads) {
        for (u32 i = 0; i < threads; ++i) {
            pools.push_back(device.CreateCommandPool({
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = 0,
            }));
            buffers.emplace_back();
        }
    }

    VkCommandBuffer Acquire(u32 thread) override {
        buffers[thread].push_back(pools[thread].Allocate(1, VK_COMMAND_BUFFER_LEVEL_SECONDARY));
        return buffers[thread].back()[0];
    }

private:
    std::vector<vk::CommandPool> pools;
    std::vector<std::vector<vk::CommandBuffers>> buffers;
};

struct Push {
    std::array<float, 4> rect;
    std::array<float, 4> color;
};

constexpr u32 Size = 64;

} // Anonymous namespace

TEST_CASE("ParallelRecord renders the same image as inline recording", "[video_core][vulkan_device]") {
    if (std::getenv("EDEN_TEST_VULKAN") == nullptr) {
        SKIP("EDEN_TEST_VULKAN is unset");
    }
    const auto library = OpenLibrary();
    REQUIRE(library->IsOpen());
    vk::InstanceDispatch idld{};
    REQUIRE(library->GetSymbol("vkGetInstanceProcAddr", &idld.vkGetInstanceProcAddr));
    REQUIRE(vk::Load(idld));
    const std::array<const char*, 1> layers{"VK_LAYER_KHRONOS_validation"};
    const std::array<const char*, 1> extensions{VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    const vk::Instance instance = vk::Instance::Create(VK_API_VERSION_1_1, layers, extensions, idld);
    REQUIRE(vk::Load(*instance, idld));
    const vk::DebugUtilsMessenger messenger = instance.CreateDebugUtilsMessenger({
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        .pfnUserCallback = &OnMessage,
        .pUserData = nullptr,
    });
    const auto physicals = instance.EnumeratePhysicalDevices();
    REQUIRE(!physicals.empty());
    const vk::PhysicalDevice physical(physicals[0], idld);

    vk::DeviceDispatch dld{};
    static_cast<vk::InstanceDispatch&>(dld) = idld;
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = 0,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const vk::Device device = vk::Device::Create(physicals[0], queue_ci, {}, nullptr, dld);
    const vk::Queue queue = device.GetQueue(0);

    const auto memory_type = [&](u32 bits, VkMemoryPropertyFlags flags) {
        const auto properties = physical.GetMemoryProperties().memoryProperties;
        for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
            if ((bits & (1U << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        FAIL("no memory type");
        return 0U;
    };

    const VkImageCreateInfo image_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {Size, Size, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image{};
    REQUIRE(dld.vkCreateImage(*device, &image_ci, nullptr, &image) == VK_SUCCESS);
    const std::unique_ptr<VkImage_T, std::function<void(VkImage)>> image_owner{
        image, [&](VkImage handle) { dld.vkDestroyImage(*device, handle, nullptr); }};
    const auto image_requirements = device.GetImageMemoryRequirements(image);
    vk::DeviceMemory image_memory = device.AllocateMemory({
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = image_requirements.size,
        .memoryTypeIndex = memory_type(image_requirements.memoryTypeBits, 0),
    });
    REQUIRE(dld.vkBindImageMemory(*device, image, *image_memory, 0) == VK_SUCCESS);
    const vk::ImageView view = device.CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .components = {},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    });

    const VkDeviceSize readback_size = Size * Size * 4;
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = readback_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer readback{};
    REQUIRE(dld.vkCreateBuffer(*device, &buffer_ci, nullptr, &readback) == VK_SUCCESS);
    const std::unique_ptr<VkBuffer_T, std::function<void(VkBuffer)>> readback_owner{
        readback, [&](VkBuffer handle) { dld.vkDestroyBuffer(*device, handle, nullptr); }};
    const auto buffer_requirements = device.GetBufferMemoryRequirements(readback);
    vk::DeviceMemory buffer_memory = device.AllocateMemory({
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = buffer_requirements.size,
        .memoryTypeIndex = memory_type(buffer_requirements.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    });
    REQUIRE(dld.vkBindBufferMemory(*device, readback, *buffer_memory, 0) == VK_SUCCESS);

    const VkAttachmentDescription attachment{
        .flags = 0,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    };
    const VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_ref,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
    const VkSubpassDependency dependency{
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dependencyFlags = 0,
    };
    const vk::RenderPass render_pass = device.CreateRenderPass({
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    });
    const VkImageView view_handle = *view;
    const vk::Framebuffer framebuffer = device.CreateFramebuffer({
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = *render_pass,
        .attachmentCount = 1,
        .pAttachments = &view_handle,
        .width = Size,
        .height = Size,
        .layers = 1,
    });

    const VkPushConstantRange push_range{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push)};
    const vk::PipelineLayout layout = device.CreatePipelineLayout({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    const auto shader = [&](const u32* code, size_t size) {
        return device.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = size,
            .pCode = code,
        });
    };
    const vk::ShaderModule vertex = shader(VERT_SPV, sizeof(VERT_SPV));
    const vk::ShaderModule fragment = shader(FRAG_SPV, sizeof(FRAG_SPV));
    const std::array stages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = *vertex,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = *fragment,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        .primitiveRestartEnable = VK_FALSE,
    };
    const VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };
    const VkPipelineRasterizationStateCreateInfo raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };
    const VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
        .blendConstants = {},
    };
    const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const vk::Pipeline pipeline = device.CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<u32>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = *layout,
        .renderPass = *render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });

    const vk::CommandPool primary_pool = device.CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = 0,
    });

    // Overlapping quads in draw order, with the viewport and scissor changing mid-pass.
    const auto record = [&](vk::CommandBuffer cmdbuf) {
        const VkViewport full{0.0f, 0.0f, float{Size}, float{Size}, 0.0f, 1.0f};
        const VkRect2D full_scissor{{0, 0}, {Size, Size}};
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
        cmdbuf.SetViewport(0, full);
        cmdbuf.SetScissor(0, full_scissor);
        const VkClearValue clear{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
        cmdbuf.BeginRenderPass(
            {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = *render_pass,
                .framebuffer = *framebuffer,
                .renderArea = {{0, 0}, {Size, Size}},
                .clearValueCount = 1,
                .pClearValues = &clear,
            },
            VK_SUBPASS_CONTENTS_INLINE);
        for (u32 draw = 0; draw < 600; ++draw) {
            if (draw % 97 == 50) {
                const VkRect2D scissor{{static_cast<s32>(draw % 16), 0}, {Size / 2, Size}};
                cmdbuf.SetScissor(0, scissor);
            } else if (draw % 97 == 80) {
                cmdbuf.SetScissor(0, full_scissor);
            }
            const float x = -1.0f + static_cast<float>(draw % 37) / 20.0f;
            const float y = -1.0f + static_cast<float>(draw % 23) / 12.0f;
            const Push push{
                .rect = {x, y, 0.4f, 0.4f},
                .color = {static_cast<float>(draw % 7) / 6.0f, static_cast<float>(draw % 11) / 10.0f,
                          static_cast<float>(draw % 13) / 12.0f, 1.0f},
            };
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                 0, sizeof(push), &push);
            cmdbuf.Draw(4, 1, 0, 0);
        }
        cmdbuf.EndRenderPass();
        const VkBufferImageCopy copy{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {Size, Size, 1},
        };
        cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, copy);
    };

    const auto render = [&](u32 threads) {
        vk::CommandBuffers primaries = primary_pool.Allocate(1);
        vk::CommandBuffer primary(primaries[0], dld);
        primary.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        PoolSource source(device, threads == 0 ? 1 : threads);
        if (threads == 0) {
            record(primary);
        } else {
            ParallelRecord parallel(dld, threads, source);
            record(parallel.Wrap(*primary));
        }
        primary.End();
        const VkCommandBuffer handle = *primary;
        const VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &handle,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        REQUIRE(queue.Submit(submit) == VK_SUCCESS);
        device.WaitIdle();
        std::vector<u8> pixels(readback_size);
        std::memcpy(pixels.data(), buffer_memory.Map(0, readback_size), readback_size);
        buffer_memory.Unmap();
        return pixels;
    };

    const std::vector<u8> expected = render(0);
    u32 lit = 0;
    for (size_t i = 0; i < expected.size(); i += 4) {
        lit += expected[i] != 0 || expected[i + 1] != 0 || expected[i + 2] != 0 ? 1 : 0;
    }
    REQUIRE(lit > Size * Size / 2);
    for (const u32 threads : {2U, 4U}) {
        INFO("threads " << threads);
        REQUIRE(render(threads) == expected);
    }
    for (const std::string& message : g_validation_log) {
        UNSCOPED_INFO(message);
    }
    CHECK(g_validation_messages == 0);
}
