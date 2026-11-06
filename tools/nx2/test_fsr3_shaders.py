#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the production FSR3 shaders on Vulkan, including lavapipe.

Requires glslangValidator, spirv-val, the Vulkan validation layer, and Python
packages vulkan==1.3.275.1, numpy and pillow. Run with --out <evidence directory>.
Add --build <configured CMake directory> to test production image-copy commands.
The build must have compile_commands.json and the two generated FSR3 shader headers.
"""

import argparse
import ctypes
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from vulkan import *


class VulkanTest:
    def __init__(self):
        self.errors = []
        self.cleanup = []
        self.discard = None

        def report(severity, kind, data, user):
            if severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                self.errors.append(ffi.string(data.pMessage).decode())
            return VK_FALSE

        self.report = report
        debug = VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            messageType=7, pfnUserCallback=report)
        validation = VkValidationFeaturesEXT(
            pEnabledValidationFeatures=[VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT],
            pNext=debug)
        self.instance = vkCreateInstance(VkInstanceCreateInfo(
            pApplicationInfo=VkApplicationInfo(apiVersion=VK_MAKE_VERSION(1, 1, 0)),
            ppEnabledLayerNames=["VK_LAYER_KHRONOS_validation"],
            ppEnabledExtensionNames=["VK_EXT_debug_utils", "VK_EXT_validation_features"],
            pNext=validation), None)
        self.debug = vkGetInstanceProcAddr(self.instance, "vkCreateDebugUtilsMessengerEXT")(
            self.instance, debug, None)
        physical = vkEnumeratePhysicalDevices(self.instance)[0]
        print("Vulkan device:", vkGetPhysicalDeviceProperties(physical).deviceName)
        family = next(i for i, p in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(physical))
                      if p.queueFlags & VK_QUEUE_COMPUTE_BIT)
        self.memory = vkGetPhysicalDeviceMemoryProperties(physical)
        self.device = vkCreateDevice(physical, VkDeviceCreateInfo(
            pQueueCreateInfos=[VkDeviceQueueCreateInfo(queueFamilyIndex=family,
                                                     pQueuePriorities=[1.0])]), None)
        self.queue = vkGetDeviceQueue(self.device, family, 0)
        self.pool = vkCreateCommandPool(self.device, VkCommandPoolCreateInfo(
            queueFamilyIndex=family), None)
        self.sampler = vkCreateSampler(self.device, VkSamplerCreateInfo(
            magFilter=VK_FILTER_LINEAR, minFilter=VK_FILTER_LINEAR,
            addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE), None)

    def allocate(self, req, flags):
        index = next(i for i in range(self.memory.memoryTypeCount)
                     if req.memoryTypeBits & (1 << i)
                     and self.memory.memoryTypes[i].propertyFlags & flags == flags)
        memory = vkAllocateMemory(self.device, VkMemoryAllocateInfo(
            allocationSize=req.size, memoryTypeIndex=index), None)
        self.cleanup.append(lambda: vkFreeMemory(self.device, memory, None))
        return memory

    def command(self, record):
        cmd = vkAllocateCommandBuffers(self.device, VkCommandBufferAllocateInfo(
            commandPool=self.pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1))[0]
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo())
        record(cmd)
        vkEndCommandBuffer(cmd)
        vkQueueSubmit(self.queue, 1, [VkSubmitInfo(pCommandBuffers=[cmd])], VK_NULL_HANDLE)
        vkQueueWaitIdle(self.queue)
        vkFreeCommandBuffers(self.device, self.pool, 1, [cmd])
        if self.errors:
            raise AssertionError("\n".join(self.errors))

    def barrier(self, cmd):
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                             [VkMemoryBarrier(srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT,
                                              dstAccessMask=VK_ACCESS_MEMORY_READ_BIT |
                                                            VK_ACCESS_MEMORY_WRITE_BIT)],
                             0, None, 0, None)

    def image(self, data, fmt, initialize=True):
        h, w = data.shape[:2]
        image = vkCreateImage(self.device, VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D, format=fmt, extent=VkExtent3D(w, h, 1),
            mipLevels=1, arrayLayers=1, samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
        memory = self.allocate(vkGetImageMemoryRequirements(self.device, image), 0)
        vkBindImageMemory(self.device, image, memory, 0)
        self.cleanup.append(lambda: vkDestroyImage(self.device, image, None))
        sub = VkImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        view = vkCreateImageView(self.device, VkImageViewCreateInfo(
            image=image, viewType=VK_IMAGE_VIEW_TYPE_2D, format=fmt,
            subresourceRange=sub), None)
        self.cleanup.append(lambda: vkDestroyImageView(self.device, view, None))
        result = (image, view, data.shape, data.dtype)
        if not initialize:
            return result
        self.command(lambda cmd: vkCmdPipelineBarrier(
            cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 0, None, 0, None, 1, [VkImageMemoryBarrier(
                dstAccessMask=VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                oldLayout=VK_IMAGE_LAYOUT_UNDEFINED, newLayout=VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, image=image, subresourceRange=sub)]))
        self.transfer(result, data)
        return result

    def transfer(self, image, data=None):
        handle, view, shape, dtype = image
        size = int(np.prod(shape)) * dtype.itemsize
        buffer = vkCreateBuffer(self.device, VkBufferCreateInfo(
            size=size, usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
        memory = self.allocate(vkGetBufferMemoryRequirements(self.device, buffer),
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        vkBindBufferMemory(self.device, buffer, memory, 0)
        self.cleanup.append(lambda: vkDestroyBuffer(self.device, buffer, None))
        mapped = vkMapMemory(self.device, memory, 0, size, 0)
        if data is not None:
            mapped[:] = data.tobytes()
        region = VkBufferImageCopy(
            imageSubresource=VkImageSubresourceLayers(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            imageExtent=VkExtent3D(shape[1], shape[0], 1))

        def record(cmd):
            self.barrier(cmd)
            if data is None:
                vkCmdCopyImageToBuffer(cmd, handle, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, [region])
            else:
                vkCmdCopyBufferToImage(cmd, buffer, handle, VK_IMAGE_LAYOUT_GENERAL, 1, [region])
            self.barrier(cmd)

        self.command(record)
        result = np.frombuffer(mapped, dtype=dtype).reshape(shape).copy()
        vkUnmapMemory(self.device, memory)
        return result

    def dispatch(self, shader, inputs, output, constants, defer=False):
        images = inputs + [output]
        types = [VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] * len(inputs) + [VK_DESCRIPTOR_TYPE_STORAGE_IMAGE]
        layout = vkCreateDescriptorSetLayout(self.device, VkDescriptorSetLayoutCreateInfo(
            pBindings=[VkDescriptorSetLayoutBinding(i, t, 1, VK_SHADER_STAGE_COMPUTE_BIT)
                       for i, t in enumerate(types)]), None)
        self.cleanup.append(lambda: vkDestroyDescriptorSetLayout(self.device, layout, None))
        pool = vkCreateDescriptorPool(self.device, VkDescriptorPoolCreateInfo(
            maxSets=1, pPoolSizes=[VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, len(inputs)),
                                  VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1)]), None)
        self.cleanup.append(lambda: vkDestroyDescriptorPool(self.device, pool, None))
        descriptor = vkAllocateDescriptorSets(self.device, VkDescriptorSetAllocateInfo(
            descriptorPool=pool, pSetLayouts=[layout]))[0]
        writes = [VkWriteDescriptorSet(
            dstSet=descriptor, dstBinding=i, descriptorCount=1, descriptorType=t,
            pImageInfo=[VkDescriptorImageInfo(self.sampler if i < len(inputs) else VK_NULL_HANDLE,
                                             image[1], VK_IMAGE_LAYOUT_GENERAL)])
                  for i, (image, t) in enumerate(zip(images, types))]
        vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        pipe_layout = vkCreatePipelineLayout(self.device, VkPipelineLayoutCreateInfo(
            pSetLayouts=[layout], pPushConstantRanges=[VkPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, 24)]), None)
        self.cleanup.append(lambda: vkDestroyPipelineLayout(self.device, pipe_layout, None))
        code = shader.read_bytes()
        module = vkCreateShaderModule(self.device, VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self.cleanup.append(lambda: vkDestroyShaderModule(self.device, module, None))
        stage = VkPipelineShaderStageCreateInfo(stage=VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main")
        pipeline = vkCreateComputePipelines(self.device, VK_NULL_HANDLE, 1, [VkComputePipelineCreateInfo(
            stage=stage, layout=pipe_layout)], None)[0]
        self.cleanup.append(lambda: vkDestroyPipeline(self.device, pipeline, None))

        def record(cmd):
            self.barrier(cmd)
            if self.discard:
                self.discard(cmd, output[0])
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, [descriptor], 0, None)
            values = ffi.new("float[]", constants)
            vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, values)
            h, w = output[2][:2]
            vkCmdDispatch(cmd, (w + 15) // 16, (h + 15) // 16, 1)
            self.barrier(cmd)

        if defer:
            return record
        self.command(record)
        result = self.transfer(output)
        return result

    def close(self):
        vkDeviceWaitIdle(self.device)
        for cleanup in reversed(self.cleanup):
            cleanup()
        vkDestroySampler(self.device, self.sampler, None)
        vkDestroyCommandPool(self.device, self.pool, None)
        vkDestroyDevice(self.device, None)
        vkGetInstanceProcAddr(self.instance, "vkDestroyDebugUtilsMessengerEXT")(
            self.instance, self.debug, None)
        vkDestroyInstance(self.instance, None)
        assert not self.errors, "\n".join(self.errors)


def build_copy_probe(build, out, sanitize):
    commands = json.loads((build / "compile_commands.json").read_text())
    entry = next(c for c in commands if c["file"].endswith("present/fsr3_frame_gen.cpp"))
    command = shlex.split(entry["command"])
    index = command.index("-o")
    del command[index:index + 2]
    command.remove("-c")
    command.remove(entry["file"])
    command = [arg for arg in command if not arg.startswith("-mtls-dialect=")]
    library = out.resolve() / "copy-probe.so"
    command += ["-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                "-ffunction-sections", "-fdata-sections", "-shared",
                "-Wl,--gc-sections,--no-undefined", str(Path(__file__).with_name("fsr3_copy_probe.cpp").resolve()),
                "-o", str(library)]
    if sanitize:
        command += ["-fsanitize=address,undefined", "-shared-libsan", "-fno-sanitize-recover=all"]
    subprocess.run(command, cwd=entry["directory"], check=True)
    return ctypes.CDLL(str(library))


def check_copies(vk, probe):
    copy = probe.Fsr3CopyProbe
    copy.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_bool]
    native = ctypes.CDLL("libvulkan.so.1")

    def address(handle):
        return int(ffi.cast("uintptr_t", handle))

    discard = probe.Fsr3DiscardProbe
    discard.argtypes = [ctypes.c_void_p] * 3
    vk.discard = lambda cmd, image: discard(
        address(cmd), ctypes.cast(native.vkCmdPipelineBarrier, ctypes.c_void_p), address(image))

    for w, h in ((64, 48), (65, 49), (1, 1)):
        pixels = np.full((h, w, 4), 32, np.uint8)
        source = vk.image(pixels, VK_FORMAT_R8G8B8A8_UNORM)
        history = vk.image(pixels, VK_FORMAT_R8G8B8A8_UNORM, initialize=False)
        current = vk.image(pixels, VK_FORMAT_R8G8B8A8_UNORM, initialize=False)

        def record(cmd, first):
            copy(address(cmd), ctypes.cast(native.vkCmdPipelineBarrier, ctypes.c_void_p),
                 ctypes.cast(native.vkCmdCopyImage, ctypes.c_void_p),
                 address(source[0]), address(history[0]), address(current[0]), w, h, first)

        vk.command(lambda cmd: record(cmd, True))
        np.testing.assert_array_equal(vk.transfer(history), pixels)
        np.testing.assert_array_equal(vk.transfer(current), pixels)
        for value in (96, 192):
            previous = pixels.copy()
            pixels[:] = value
            vk.transfer(source, pixels)
            vk.command(lambda cmd: record(cmd, False))
            np.testing.assert_array_equal(vk.transfer(history), previous)
            np.testing.assert_array_equal(vk.transfer(current), pixels)
        print(f"PASS production history init and two rollovers, fresh extent {w}x{h}")


def run(out, build, shaders, sanitize):
    out.mkdir(parents=True, exist_ok=True)
    compiled = []
    for name in ("fsr3_optical_flow", "fsr3_frame_interpolation"):
        binary = out / (name + ".spv")
        subprocess.run(["glslangValidator", "-V", str(shaders / (name + ".comp")), "-o", str(binary)], check=True)
        subprocess.run(["spirv-val", str(binary)], check=True)
        compiled.append(binary)
    flow_shader, interp_shader = compiled
    vk = VulkanTest()
    try:
        if build:
            check_copies(vk, build_copy_probe(build, out, sanitize))
        for w, h in ((64, 48), (65, 49), (1, 1)):
            fw, fh = max(1, w // 2), max(1, h // 2)
            blank = np.zeros((h, w, 4), np.uint8)
            blank[:, :, 3] = 255
            previous = vk.image(blank, VK_FORMAT_R8G8B8A8_UNORM)
            newer = blank.copy()
            newer[:, :, :3] = 128
            current = vk.image(newer, VK_FORMAT_R8G8B8A8_UNORM)
            motion = vk.image(np.zeros((fh, fw, 2), np.float16), VK_FORMAT_R16G16_SFLOAT)
            dest = vk.image(blank, VK_FORMAT_R8G8B8A8_UNORM)
            for t in (0.25, 0.5, 0.75):
                result = vk.dispatch(interp_shader, [current, previous, motion], dest,
                                     [1 / w, 1 / h, w, h, t, 0.5])
                occlusion = min(1, (128 / 255) * 0.999 * 2)
                alpha = t * (1 - occlusion * 0.4) + 0.5 * occlusion * 0.4
                expected = 128 * (alpha * 0.7 + t * 0.3)
                assert np.max(abs(result[:, :, :3].astype(float) - expected)) < 1, (w, h, t, result[0, 0], expected)
            print(f"PASS distinct constant frames, three generation times, extent {w}x{h}")
            targets = [vk.image(blank, VK_FORMAT_R8G8B8A8_UNORM, initialize=not bool(build))
                       for _ in range(3)]
            for _ in range(2):
                commands = [vk.dispatch(interp_shader, [current, previous, motion], target,
                                        [1 / w, 1 / h, w, h, t, 0.5], defer=True)
                            for target, t in zip(targets, (0.25, 0.5, 0.75))]
                vk.command(lambda cmd: [record(cmd) for record in commands])
                values = [int(vk.transfer(target)[0, 0, 0]) for target in targets]
                assert values == [41, 64, 87], values
            print(f"PASS three generated targets in one submission and target reuse, {w}x{h}")

        w, h = 64, 48
        fw, fh = w // 2, h // 2
        y, x = np.mgrid[:h, :w]
        for axis in (0, 1):
            for direction in (-1, 1):
                coord = x if axis == 0 else y
                pixels = np.zeros((h, w, 4), np.uint8)
                pixels[:, :, :3] = (((coord // 4) % 2) * 255)[:, :, None]
                pixels[:, :, 3] = 255
                translated = np.roll(pixels, 4 * direction, axis=1 - axis)
                previous = vk.image(pixels, VK_FORMAT_R8G8B8A8_UNORM)
                current = vk.image(translated, VK_FORMAT_R8G8B8A8_UNORM)
                field = np.zeros((fh, fw, 2), np.float16)
                field[:, :, axis] = -2 * direction
                motion = vk.image(field, VK_FORMAT_R16G16_SFLOAT)
                dest = vk.image(np.zeros_like(pixels), VK_FORMAT_R8G8B8A8_UNORM)
                for t in (0.25, 0.5, 0.75):
                    result = vk.dispatch(interp_shader, [current, previous, motion], dest,
                                         [1 / w, 1 / h, w, h, t, 0.5])
                    expected = np.roll(pixels, int(4 * direction * t), axis=1 - axis)
                    assert np.max(abs(result[8:-8, 8:-8, :3].astype(float) - expected[8:-8, 8:-8, :3])) <= 11
                    if axis == 0 and direction == 1 and t == 0.5:
                        strip = np.concatenate([pixels, result, translated], axis=1)
                        evidence = Image.new("RGB", (768, 216), "#252525")
                        evidence.paste(Image.fromarray(strip).resize((768, 192)), (0, 24))
                        labels = ImageDraw.Draw(evidence)
                        for i, label in enumerate(("Previous", "Generated 50%", "Current (+4 pixels)")):
                            labels.text((i * 256 + 8, 5), label, fill="white")
                        evidence.save(out / "translation.png")
                print(f"PASS interpolation direction/half-resolution scale axis={axis} sign={direction}")
                if axis == 0 and direction == 1:
                    field[10:12, 12:14] = 0
                    vk.transfer(motion, field)
                    filtered = vk.dispatch(interp_shader, [current, previous, motion], dest,
                                           [1 / w, 1 / h, w, h, 0.5, 0.5])
                    assert np.all(filtered[21, 25, :3] >= 244), filtered[21, 25]
                    print("PASS median neighborhood uses motion-field texels")

                ramp = np.zeros_like(pixels)
                ramp[:, :, :3] = np.clip((coord - 16) * 25, 0, 255)[:, :, None]
                ramp[:, :, 3] = 255
                translated = np.roll(ramp, 4 * direction, axis=1 - axis)
                vk.transfer(previous, ramp)
                vk.transfer(current, translated)
                measured = vk.dispatch(flow_shader, [current, previous], motion,
                                       [1 / fw, 1 / fh, fw, fh, 2, 2])
                at = 12 if direction == 1 else 9
                vector = measured[10, at] if axis == 0 else measured[at, 10]
                assert vector[axis] == -2 * direction and vector[1 - axis] == 0, vector
                print(f"PASS production optical flow current-to-previous texels: {vector}")
                vk.transfer(previous, translated)
                static = vk.dispatch(flow_shader, [current, previous], motion,
                                     [1 / fw, 1 / fh, fw, fh, 2, 2])
                assert not np.any(static)
        print("PASS identical frames produce zero flow; no Vulkan validation errors")
    finally:
        vk.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--shaders", type=Path,
                        default=Path(__file__).resolve().parents[2] / "src/video_core/host_shaders")
    parser.add_argument("--sanitize", action="store_true",
                        help="Instrument the C++ probe; preload clang's ASan runtime")
    args = parser.parse_args()
    run(args.out, args.build, args.shaders, args.sanitize)
