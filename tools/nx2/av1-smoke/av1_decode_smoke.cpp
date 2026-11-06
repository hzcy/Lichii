// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
}

namespace {

// Original, lossless 16x16 black YUV420P frame. See README.md for its generation command.
constexpr std::array<std::uint8_t, 22> BlackFrame{
    0x12, 0x00, 0x0a, 0x06, 0x18, 0x0c, 0xff, 0xdb, 0x00, 0x80, 0x32,
    0x0a, 0x10, 0x00, 0x00, 0x00, 0x14, 0xb9, 0xb8, 0x75, 0xde, 0xe0,
};

struct Resources {
    AVCodecContext* context{};
    AVPacket* packet{};
    AVFrame* frame{};

    ~Resources() {
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&context);
    }
};

bool Check(int result, const char* operation) {
    if (result >= 0) {
        return true;
    }
    std::array<char, AV_ERROR_MAX_STRING_SIZE> error{};
    av_strerror(result, error.data(), error.size());
    std::fprintf(stderr, "AV1 smoke: %s: %s (%d)\n", operation, error.data(), result);
    return false;
}

bool CheckFrame(const AVFrame& frame) {
    if (frame.width != 16 || frame.height != 16 || frame.format != AV_PIX_FMT_YUV420P) {
        std::fprintf(stderr, "AV1 smoke: unexpected frame %dx%d, format %d\n", frame.width,
                     frame.height, frame.format);
        return false;
    }
    for (int plane = 0; plane < 3; ++plane) {
        const int size = plane == 0 ? 16 : 8;
        const std::uint8_t expected = plane == 0 ? 16 : 128;
        if (!frame.data[plane] || frame.linesize[plane] < size) {
            std::fprintf(stderr, "AV1 smoke: invalid plane %d\n", plane);
            return false;
        }
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                if (frame.data[plane][y * frame.linesize[plane] + x] != expected) {
                    std::fprintf(stderr, "AV1 smoke: wrong pixel at plane %d, (%d,%d)\n", plane,
                                 x, y);
                    return false;
                }
            }
        }
    }
    return true;
}

bool Decode() {
    // Match the emulator's codec-ID lookup; do not substitute an encoder or an external ffmpeg.
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_AV1);
    if (!codec) {
        std::fprintf(stderr, "AV1 smoke: linked FFmpeg has no AV1 decoder\n");
        return false;
    }
    Resources resources;
    resources.context = avcodec_alloc_context3(codec);
    resources.packet = av_packet_alloc();
    resources.frame = av_frame_alloc();
    if (!resources.context || !resources.packet || !resources.frame) {
        std::fprintf(stderr, "AV1 smoke: allocation failed\n");
        return false;
    }
    resources.context->thread_count = 1;
    if (!Check(avcodec_open2(resources.context, codec, nullptr), "open decoder") ||
        !Check(av_new_packet(resources.packet, static_cast<int>(BlackFrame.size())), "allocate packet")) {
        return false;
    }
    // av_new_packet supplies the zeroed trailing padding required by libavcodec.
    std::memcpy(resources.packet->data, BlackFrame.data(), BlackFrame.size());
    if (!Check(avcodec_send_packet(resources.context, resources.packet), "submit frame")) {
        return false;
    }

    int frames = 0;
    bool draining = false;
    for (;;) {
        const int result = avcodec_receive_frame(resources.context, resources.frame);
        if (result == AVERROR_EOF) {
            break;
        }
        if (result == AVERROR(EAGAIN) && !draining) {
            if (!Check(avcodec_send_packet(resources.context, nullptr), "drain decoder")) {
                return false;
            }
            draining = true;
            continue;
        }
        if (!Check(result, "receive frame") || !CheckFrame(*resources.frame)) {
            return false;
        }
        ++frames;
        av_frame_unref(resources.frame);
    }
    if (frames != 1) {
        std::fprintf(stderr, "AV1 smoke: expected one decoded frame, got %d\n", frames);
        return false;
    }
    std::printf("AV1 smoke: %s decoded one 16x16 frame with exact YUV pixels\n", codec->name);
    return true;
}

} // namespace

int main() {
    return Decode() ? 0 : 1;
}
