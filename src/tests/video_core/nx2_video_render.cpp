// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/nx2_video_decoder.h"

extern "C" {
#include <libswscale/swscale.h>
}

namespace {
using Tegra::Host1x::Nx2Picture;
using Tegra::Host1x::Nx2PictureFormat;
using Tegra::Host1x::Nx2VideoDecoder;

Nx2Picture Picture(u32 width, u32 height, AVPixelFormat format) {
    Nx2Picture picture{width, height, 0, std::make_shared<FFmpeg::Frame>()};
    AVFrame* frame = picture.frame->GetFrame();
    REQUIRE(frame != nullptr);
    frame->width = static_cast<int>(width);
    frame->height = static_cast<int>(height);
    frame->format = format;
    frame->color_range = AVCOL_RANGE_MPEG;
    frame->colorspace = AVCOL_SPC_BT709;
    REQUIRE(av_frame_get_buffer(frame, 32) == 0);
    for (u32 y = 0; y < height; ++y) {
        u8* row = frame->data[0] + y * frame->linesize[0];
        std::fill_n(row, frame->linesize[0], 0xee);
        for (u32 x = 0; x < width; ++x) {
            row[x] = static_cast<u8>(16 + (y * width + x) % 220);
        }
    }
    const u32 chroma_width = (width + 1) / 2;
    for (u32 y = 0; y < (height + 1) / 2; ++y) {
        std::fill_n(frame->data[1] + y * frame->linesize[1], frame->linesize[1], 0xee);
        if (format != AV_PIX_FMT_NV12) {
            std::fill_n(frame->data[2] + y * frame->linesize[2], frame->linesize[2], 0xee);
        }
        for (u32 x = 0; x < chroma_width; ++x) {
            const u8 u = static_cast<u8>(64 + (y * chroma_width + x) % 64);
            const u8 v = static_cast<u8>(160 + (y * chroma_width + x) % 64);
            if (format == AV_PIX_FMT_NV12) {
                frame->data[1][y * frame->linesize[1] + x * 2] = u;
                frame->data[1][y * frame->linesize[1] + x * 2 + 1] = v;
            } else {
                frame->data[1][y * frame->linesize[1] + x] = u;
                frame->data[2][y * frame->linesize[2] + x] = v;
            }
        }
    }
    return picture;
}

std::vector<u8> Reference(const Nx2Picture& picture, u32 width, u32 height, u32 pitch) {
    const AVFrame* frame = picture.frame->GetFrame();
    std::unique_ptr<SwsContext, decltype(&sws_freeContext)> scaler{
        sws_getContext(frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                       static_cast<int>(width), static_cast<int>(height), AV_PIX_FMT_NV12,
                       SWS_BILINEAR, nullptr, nullptr, nullptr),
        sws_freeContext};
    REQUIRE(scaler != nullptr);
    const int* coefficients = sws_getCoefficients(SWS_CS_ITU709);
    REQUIRE(sws_setColorspaceDetails(scaler.get(), coefficients,
                                     frame->color_range == AVCOL_RANGE_JPEG ||
                                         frame->format == AV_PIX_FMT_YUVJ420P, coefficients,
                                     0, 0, 1 << 16, 1 << 16) == 0);
    const u32 chroma_offset = pitch * height;
    std::vector<u8> bytes(chroma_offset + pitch * ((height + 1) / 2), 0xcd);
    std::array<u8*, 4> planes{bytes.data(), bytes.data() + chroma_offset, nullptr, nullptr};
    std::array<int, 4> strides{static_cast<int>(pitch), static_cast<int>(pitch), 0, 0};
    REQUIRE(sws_scale(scaler.get(), frame->data, frame->linesize, 0, frame->height,
                      planes.data(), strides.data()) == static_cast<int>(height));
    return bytes;
}

// swscale defines only the visible width of each output row: the bytes up to the stride are
// spill from its vector stores and can differ between two identical conversions. Compare the
// visible pixels only.
bool SameVisiblePixels(const std::vector<u8>& actual, const std::vector<u8>& expected, u32 width,
                       u32 height, u32 pitch) {
    const u32 chroma_offset = pitch * height;
    const u32 chroma_width = ((width + 1) / 2) * 2;
    for (u32 y = 0; y < height; ++y) {
        const u32 row = y * pitch;
        if (!std::equal(actual.begin() + row, actual.begin() + row + width,
                        expected.begin() + row)) {
            return false;
        }
    }
    for (u32 y = 0; y < (height + 1) / 2; ++y) {
        const u32 row = chroma_offset + y * pitch;
        if (!std::equal(actual.begin() + row, actual.begin() + row + chroma_width,
                        expected.begin() + row)) {
            return false;
        }
    }
    return true;
}
} // namespace

TEST_CASE("NX2 video interleaves planar chroma and copies complete odd NV12 rows",
          "[video_core][nx2_video_render]") {
    Nx2VideoDecoder decoder{"video/avc"};
    REQUIRE(decoder.IsOpen());
    for (const AVPixelFormat format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12}) {
        for (const auto [width, height] : std::array{std::array{4U, 2U}, std::array{3U, 3U}}) {
            const auto picture = Picture(width, height, format);
            REQUIRE(picture.frame->GetStride(0) > static_cast<int>(width));
            constexpr u32 pitch = 8;
            const u32 chroma_offset = pitch * height + 8;
            const u32 size = chroma_offset + pitch * ((height + 1) / 2);
            std::vector<u8> actual(size + 16, 0xcd);
            auto expected = actual;
            for (u32 y = 0; y < height; ++y) {
                for (u32 x = 0; x < width; ++x) {
                    expected[y * pitch + x] = static_cast<u8>(16 + y * width + x);
                }
            }
            const u32 chroma_width = (width + 1) / 2;
            for (u32 y = 0; y < (height + 1) / 2; ++y) {
                for (u32 x = 0; x < chroma_width; ++x) {
                    expected[chroma_offset + y * pitch + x * 2] =
                        static_cast<u8>(64 + y * chroma_width + x);
                    expected[chroma_offset + y * pitch + x * 2 + 1] =
                        static_cast<u8>(160 + y * chroma_width + x);
                }
            }
            decoder.Render(picture, Nx2PictureFormat::Nv12, width, height, pitch, chroma_offset,
                           std::span<u8>{actual}.first(size));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("NX2 video rejects insufficient NV12 pitch and storage before writing",
          "[video_core][nx2_video_render]") {
    Nx2VideoDecoder decoder{"video/avc"};
    REQUIRE(decoder.IsOpen());
    for (const AVPixelFormat format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12}) {
        const auto picture = Picture(3, 3, format);
        for (const auto [width, height, pitch, offset, size] : std::array{
                 std::array{3U, 3U, 3U, 9U, 15U},
                 std::array{3U, 3U, 2U, 6U, 10U},
                 std::array{3U, 3U, 4U, 12U, 19U},
                 std::array{0U, 3U, 4U, 12U, 20U},
                 std::array{3U, 0U, 4U, 12U, 12U}}) {
            std::vector<u8> bytes(size + 16, 0xcd);
            const auto expected = bytes;
            decoder.Render(picture, Nx2PictureFormat::Nv12, width, height, pitch, offset,
                           std::span<u8>{bytes}.first(size));
            REQUIRE(bytes == expected);
        }
    }
}

TEST_CASE("NX2 video refreshes cached scaler ranges from full to limited and back",
          "[video_core][nx2_video_render]") {
    Nx2VideoDecoder decoder{"video/avc"};
    REQUIRE(decoder.IsOpen());
    auto picture = Picture(16, 16, AV_PIX_FMT_YUV420P);
    constexpr u32 width = 8, height = 8, pitch = 32;
    std::vector<u8> first_full;
    for (const AVColorRange range : {AVCOL_RANGE_JPEG, AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG}) {
        picture.frame->GetFrame()->color_range = range;
        const auto expected = Reference(picture, width, height, pitch);
        std::vector<u8> actual(expected.size(), 0xcd);
        decoder.Render(picture, Nx2PictureFormat::Nv12, width, height, pitch, pitch * height,
                       actual);
        REQUIRE(SameVisiblePixels(actual, expected, width, height, pitch));
        if (first_full.empty()) {
            first_full = actual;
        } else if (range == AVCOL_RANGE_MPEG) {
            REQUIRE_FALSE(SameVisiblePixels(actual, first_full, width, height, pitch));
        } else {
            REQUIRE(SameVisiblePixels(actual, first_full, width, height, pitch));
        }
    }
}

TEST_CASE("NX2 video keeps full-range planar frames on the scaler path",
          "[video_core][nx2_video_render]") {
    Nx2VideoDecoder decoder{"video/avc"};
    REQUIRE(decoder.IsOpen());
    for (const AVPixelFormat format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P}) {
        auto picture = Picture(16, 16, format);
        picture.frame->GetFrame()->color_range = AVCOL_RANGE_JPEG;
        const auto expected = Reference(picture, 16, 16, 32);
        std::vector<u8> actual(expected.size(), 0xcd);
        decoder.Render(picture, Nx2PictureFormat::Nv12, 16, 16, 32, 32 * 16, actual);
        REQUIRE(SameVisiblePixels(actual, expected, 16, 16, 32));
        if (format == AV_PIX_FMT_YUVJ420P) {
            picture.frame->GetFrame()->color_range = AVCOL_RANGE_UNSPECIFIED;
            std::fill(actual.begin(), actual.end(), 0xcd);
            decoder.Render(picture, Nx2PictureFormat::Nv12, 16, 16, 32, 32 * 16, actual);
            REQUIRE(SameVisiblePixels(actual, expected, 16, 16, 32));
        }
    }
}

TEST_CASE("NX2 full-range NV12 preserves same-size chroma and converts resized frames",
          "[video_core][nx2_video_render]") {
    Nx2VideoDecoder decoder{"video/avc"};
    REQUIRE(decoder.IsOpen());
    auto picture = Picture(16, 16, AV_PIX_FMT_NV12);
    AVFrame* frame = picture.frame->GetFrame();
    frame->color_range = AVCOL_RANGE_JPEG;
    constexpr u32 pitch = 32, size = pitch * 24;
    std::vector<u8> actual(size + 16, 0xcd);
    auto expected = actual;
    for (u32 y = 0; y < 16; ++y) {
        std::copy_n(frame->data[0] + y * frame->linesize[0], 16,
                    expected.data() + y * pitch);
    }
    for (u32 y = 0; y < 8; ++y) {
        std::copy_n(frame->data[1] + y * frame->linesize[1], 16,
                    expected.data() + pitch * 16 + y * pitch);
    }
    decoder.Render(picture, Nx2PictureFormat::Nv12, 16, 16, pitch, pitch * 16,
                   std::span<u8>{actual}.first(size));
    REQUIRE(actual == expected);

    std::vector<u8> first_full;
    for (const AVColorRange range : {AVCOL_RANGE_JPEG, AVCOL_RANGE_MPEG, AVCOL_RANGE_JPEG}) {
        frame->color_range = range;
        expected = Reference(picture, 8, 8, pitch);
        actual.assign(expected.size(), 0xcd);
        decoder.Render(picture, Nx2PictureFormat::Nv12, 8, 8, pitch, pitch * 8, actual);
        REQUIRE(SameVisiblePixels(actual, expected, 8, 8, pitch));
        if (first_full.empty()) {
            first_full = actual;
        } else if (range == AVCOL_RANGE_MPEG) {
            REQUIRE_FALSE(SameVisiblePixels(actual, first_full, 8, 8, pitch));
        } else {
            REQUIRE(SameVisiblePixels(actual, first_full, 8, 8, pitch));
        }
    }
}
