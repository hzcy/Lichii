// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <string_view>
#include <vector>

#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/memory.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/memory_manager.h"

extern "C" {
#ifdef LIBVA_FOUND
// for querying VAAPI driver information
#include <libavutil/hwcontext_vaapi.h>
#endif

#include <libavutil/hwcontext.h>
}

namespace FFmpeg {

namespace {

constexpr AVPixelFormat PreferredGpuFormat = AV_PIX_FMT_NV12;
constexpr AVPixelFormat PreferredCpuFormat = AV_PIX_FMT_YUV420P;
constexpr std::array PreferredGpuDecoders = {
#if defined(_WIN32)
    AV_HWDEVICE_TYPE_CUDA,
    AV_HWDEVICE_TYPE_D3D11VA,
    AV_HWDEVICE_TYPE_DXVA2,
    AV_HWDEVICE_TYPE_D3D12VA,
#elif defined(__FreeBSD__)
    AV_HWDEVICE_TYPE_VAAPI,
    AV_HWDEVICE_TYPE_VDPAU,
    AV_HWDEVICE_TYPE_DRM,
#elif defined(__APPLE__)
    AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
#elif defined(__ANDROID__)
    AV_HWDEVICE_TYPE_MEDIACODEC,
#elif defined(__unix__)
    AV_HWDEVICE_TYPE_CUDA,
    AV_HWDEVICE_TYPE_VAAPI,
    AV_HWDEVICE_TYPE_VDPAU,
#endif
    AV_HWDEVICE_TYPE_VULKAN,
};

AVPixelFormat GetGpuFormat(AVCodecContext* codec_context, const AVPixelFormat* pix_fmts) {
    if (codec_context->hw_device_ctx) {
        const auto* hw_ctx =
            reinterpret_cast<const AVHWDeviceContext*>(codec_context->hw_device_ctx->data);
        for (int i = 0;; i++) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec_context->codec, i);
            if (!config) {
                break;
            }
            if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) &&
                config->device_type == hw_ctx->type) {
                for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
                    if (*p == config->pix_fmt) {
                        return *p;
                    }
                }
            }
        }
    }

    LOG_INFO(HW_GPU, "Could not find supported GPU pixel format, falling back to CPU decoder");
    av_buffer_unref(&codec_context->hw_device_ctx);
    for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == PreferredCpuFormat) {
            return *p;
        }
    }
    return pix_fmts[0];
}

std::string AVError(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_make_error_string(errbuf, sizeof(errbuf) - 1, errnum);
    return errbuf;
}

#if defined(__ANDROID__)
size_t FindNalStartCode(std::span<const u8> data, size_t i) {
    const size_t n = data.size();
    if (i + 3 < n && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1) {
        return 4;
    }
    if (i + 2 < n && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
        return 3;
    }
    return 0;
}

std::vector<u8> ExtractH264ParameterSetExtradata(std::span<const u8> packet) {
    std::vector<u8> extradata;
    const size_t size = packet.size();
    size_t i = 0;
    while (i < size) {
        const size_t sc = FindNalStartCode(packet, i);
        if (sc == 0) {
            ++i;
            continue;
        }
        const size_t nal_start = i + sc;
        if (nal_start >= size) {
            break;
        }
        const u8 nal_type = packet[nal_start] & 0x1F;

        size_t j = nal_start + 1;
        while (j < size && FindNalStartCode(packet, j) == 0) {
            ++j;
        }

        if (nal_type == 7 || nal_type == 8) {
            constexpr u8 start[4] = {0, 0, 0, 1};
            extradata.insert(extradata.end(), start, start + sizeof(start));
            extradata.insert(extradata.end(), packet.begin() + nal_start, packet.begin() + j);
        } else if (nal_type == 1 || nal_type == 5) {
            break;
        }
        i = j;
    }
    return extradata;
}
#endif

}

Packet::Packet(std::span<const u8> data) {
    m_packet = av_packet_alloc();
    m_packet->data = const_cast<u8*>(data.data());
    m_packet->size = static_cast<s32>(data.size());
}

Packet::~Packet() {
    av_packet_free(&m_packet);
}

Frame::Frame() {
    m_frame = av_frame_alloc();
}

Frame::~Frame() {
    av_frame_free(&m_frame);
}

Decoder::Decoder(Tegra::Host1x::NvdecCommon::VideoCodec codec, bool prefer_hw) {
    const AVCodecID av_codec = [&] {
        switch (codec) {
        case Tegra::Host1x::NvdecCommon::VideoCodec::H264:
            return AV_CODEC_ID_H264;
        case Tegra::Host1x::NvdecCommon::VideoCodec::VP8:
            return AV_CODEC_ID_VP8;
        case Tegra::Host1x::NvdecCommon::VideoCodec::VP9:
            return AV_CODEC_ID_VP9;
        case Tegra::Host1x::NvdecCommon::VideoCodec::AV1:
            return AV_CODEC_ID_AV1;
        default:
            UNIMPLEMENTED_MSG("Unknown codec {}", codec);
            return AV_CODEC_ID_NONE;
        }
    }();

#if defined(__ANDROID__)
    if (prefer_hw) {
        const char* mc_name = nullptr;
        switch (av_codec) {
        case AV_CODEC_ID_H264: mc_name = "h264_mediacodec"; break;
        // Disabled: causes crashes in Diablo II
        // case AV_CODEC_ID_VP8:  mc_name = "vp8_mediacodec";  break;
        case AV_CODEC_ID_VP9:  mc_name = "vp9_mediacodec";  break;
        default: break;
        }
        if (mc_name) {
            m_codec = avcodec_find_decoder_by_name(mc_name);
        }
    }
#else
    if (prefer_hw && av_codec == AV_CODEC_ID_AV1) {
        m_codec = avcodec_find_decoder_by_name("av1");
    }
#endif
    if (!m_codec) {
        m_codec = avcodec_find_decoder(av_codec);
    }
}

bool Decoder::SupportsDecodingOnDevice(AVPixelFormat* out_pix_fmt, AVHWDeviceType type) const {
    for (int i = 0;; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(m_codec, i);
        if (!config) {
            LOG_DEBUG(HW_GPU, "{} decoder does not support device type {}", m_codec->name, av_hwdevice_get_type_name(type));
            break;
        }

        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == type) {
            LOG_INFO(HW_GPU, "Using {} GPU decoder", av_hwdevice_get_type_name(type));
            *out_pix_fmt = config->pix_fmt;
            return true;
        }
    }

    return false;
}

std::vector<AVHWDeviceType> HardwareContext::GetSupportedDeviceTypes() {
    std::vector<AVHWDeviceType> types;
    AVHWDeviceType current_device_type = AV_HWDEVICE_TYPE_NONE;

    while (true) {
        current_device_type = av_hwdevice_iterate_types(current_device_type);
        if (current_device_type == AV_HWDEVICE_TYPE_NONE) {
            return types;
        }

        types.push_back(current_device_type);
    }
}

HardwareContext::~HardwareContext() {
    av_buffer_unref(&m_gpu_decoder);
}

bool HardwareContext::InitializeForDecoder(DecoderContext& decoder_context, const Decoder& decoder) {
    const auto supported_types = GetSupportedDeviceTypes();
    for (const auto type : PreferredGpuDecoders) {
        AVPixelFormat hw_pix_fmt;

        if (std::ranges::find(supported_types, type) == supported_types.end()) {
            LOG_DEBUG(HW_GPU, "{} explicitly unsupported", av_hwdevice_get_type_name(type));
            continue;
        }

        if (!this->InitializeWithType(type)) {
            continue;
        }

        if (decoder.SupportsDecodingOnDevice(&hw_pix_fmt, type)) {
            decoder_context.InitializeHardwareDecoder(*this, hw_pix_fmt);
            return true;
        }
    }

    return false;
}

bool HardwareContext::InitializeWithType(AVHWDeviceType type) {
    av_buffer_unref(&m_gpu_decoder);

    if (const int ret = av_hwdevice_ctx_create(&m_gpu_decoder, type, nullptr, nullptr, 0); ret < 0) {
        LOG_DEBUG(HW_GPU, "av_hwdevice_ctx_create({}) failed: {}", av_hwdevice_get_type_name(type), AVError(ret));
        return false;
    }

#ifdef LIBVA_FOUND
    if (type == AV_HWDEVICE_TYPE_VAAPI) {
        // We need to determine if this is an impersonated VAAPI driver.
        auto* hwctx = reinterpret_cast<AVHWDeviceContext*>(m_gpu_decoder->data);
        auto* vactx = static_cast<AVVAAPIDeviceContext*>(hwctx->hwctx);
        const char* vendor_name = vaQueryVendorString(vactx->display);
        if (strstr(vendor_name, "VDPAU backend")) {
            // VDPAU impersonated VAAPI impls are super buggy, we need to skip them.
            LOG_DEBUG(HW_GPU, "Skipping VDPAU impersonated VAAPI driver");
            return false;
        } else {
            // According to some user testing, certain VAAPI drivers (Intel?) could be buggy.
            // Log the driver name just in case.
            LOG_DEBUG(HW_GPU, "Using VAAPI driver: {}", vendor_name);
        }
    }
#endif

    return true;
}

DecoderContext::DecoderContext(const Decoder& decoder) : m_decoder{decoder} {
    m_codec_context = avcodec_alloc_context3(m_decoder.GetCodec());
    av_opt_set(m_codec_context->priv_data, "tune", "zerolatency", 0);
    m_codec_context->thread_count = 0;
    m_codec_context->thread_type &= ~FF_THREAD_FRAME;
#if defined(__ANDROID__)
    m_codec_context->flags |= AV_CODEC_FLAG_LOW_DELAY;
    m_codec_context->flags2 |= AV_CODEC_FLAG2_FAST;
#endif
}

DecoderContext::~DecoderContext() {
    av_buffer_unref(&m_codec_context->hw_device_ctx);
    avcodec_free_context(&m_codec_context);
}

void DecoderContext::InitializeHardwareDecoder(const HardwareContext& context,
                                               [[maybe_unused]] AVPixelFormat hw_pix_fmt) {
    m_codec_context->hw_device_ctx = av_buffer_ref(context.GetBufferRef());
    m_codec_context->get_format = GetGpuFormat;
}

bool DecoderContext::OpenContext(const Decoder& decoder, std::span<const u8> extradata) {
    if (!extradata.empty()) {
        av_freep(&m_codec_context->extradata);
        m_codec_context->extradata = static_cast<u8*>(
            av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!m_codec_context->extradata) {
            LOG_ERROR(HW_GPU, "Failed to allocate extradata");
            return false;
        }
        std::memcpy(m_codec_context->extradata, extradata.data(), extradata.size());
        m_codec_context->extradata_size = static_cast<int>(extradata.size());
    }

    if (const int ret = avcodec_open2(m_codec_context, decoder.GetCodec(), nullptr); ret < 0) {
        LOG_ERROR(HW_GPU, "avcodec_open2 error: {}", AVError(ret));
        return false;
    }

    LOG_INFO(HW_GPU, "Using decoder {}", decoder.GetCodec()->name);

    return true;
}

bool DecoderContext::SendPacket(const Packet& packet) {
    m_had_decode_error = false;
    if (const int ret = avcodec_send_packet(m_codec_context, packet.GetPacket());
        ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
        LOG_ERROR(HW_GPU, "avcodec_send_packet error: {}", AVError(ret));
        m_had_decode_error = true;
        return false;
    }

    return true;
}

std::shared_ptr<Frame> DecoderContext::ReceiveFrame() {
    m_had_decode_error = false;
    auto ReceiveImpl = [&](AVFrame* frame) -> int {
        const int ret = avcodec_receive_frame(m_codec_context, frame);
        if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
            LOG_ERROR(HW_GPU, "avcodec_receive_frame error: {}", AVError(ret));
            m_had_decode_error = true;
        }
        return ret;
    };

    if (!m_temp_frame) {
        m_temp_frame = std::make_shared<Frame>();
    } else {
        av_frame_unref(m_temp_frame->GetFrame());
    }
    if (ReceiveImpl(m_temp_frame->GetFrame()) < 0) {
        return {};
    }

    if (m_codec_context->hw_device_ctx) {
        m_final_frame = std::make_shared<Frame>();
        m_final_frame->SetFormat(PreferredGpuFormat);
        if (const int ret = av_hwframe_transfer_data(m_final_frame->GetFrame(), m_temp_frame->GetFrame(), 0); ret < 0) {
            LOG_ERROR(HW_GPU, "av_hwframe_transfer_data error: {}", AVError(ret));
            m_had_decode_error = true;
            return {};
        }
        av_frame_copy_props(m_final_frame->GetFrame(), m_temp_frame->GetFrame());
    } else {
        m_final_frame = std::exchange(m_temp_frame, std::make_shared<Frame>());
    }

    return std::move(m_final_frame);
}

void DecodeApi::Reset() {
    m_hardware_context.reset();
    m_decoder_context.reset();
    m_decoder.reset();
    m_opened = false;
    m_defer_android_mediacodec_open = false;
    m_needs_h264_extradata = false;
    m_next_pts = 0;
    while (!m_pending_offsets.empty()) {
        m_pending_offsets.pop();
    }
}

bool DecodeApi::Initialize(Tegra::Host1x::NvdecCommon::VideoCodec codec) {
    this->Reset();
    m_decoder.emplace(codec,
                      Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu);
    m_decoder_context.emplace(*m_decoder);

    bool is_mediacodec = false;
#if defined(__ANDROID__)
    const std::string_view decoder_name = m_decoder->GetCodec() ? m_decoder->GetCodec()->name : "";
    is_mediacodec = decoder_name == "h264_mediacodec" ||
                    decoder_name == "vp8_mediacodec" ||
                    decoder_name == "vp9_mediacodec";
#endif

    // Enable GPU decoding if requested.
    if (!is_mediacodec &&
        Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu) {
        m_hardware_context.emplace();
        if (!m_hardware_context->InitializeForDecoder(*m_decoder_context, *m_decoder)) {
            m_decoder_context.reset();
            m_hardware_context.reset();
            m_decoder.emplace(codec, false);
            m_decoder_context.emplace(*m_decoder);
        }
    }

#if defined(__ANDROID__)
    m_defer_android_mediacodec_open = is_mediacodec;
    m_needs_h264_extradata = decoder_name == "h264_mediacodec";
    if (m_defer_android_mediacodec_open) {
        return true;
    }
#endif

    // Open the decoder context.
    if (!m_decoder_context->OpenContext(*m_decoder)) {
        this->Reset();
        return false;
    }
    m_opened = true;

    return true;
}

bool DecodeApi::SendPacket(std::span<const u8> packet_data, const FrameOffsets& offsets,
                           std::optional<FrameDimensions> dimensions) {
    if (!m_opened) {
        std::vector<u8> extradata;
#if defined(__ANDROID__)
        if (m_defer_android_mediacodec_open) {
            if (!dimensions) {
                return true;
            }

            auto* ctx = m_decoder_context->GetCodecContext();
            ctx->width = dimensions->width;
            ctx->height = dimensions->height;
            ctx->coded_width = dimensions->width;
            ctx->coded_height = dimensions->height;
        }

        if (m_needs_h264_extradata) {
            extradata = ExtractH264ParameterSetExtradata(packet_data);
            if (extradata.empty()) {
                return true;
            }
        }
#endif
        if (!m_decoder_context->OpenContext(*m_decoder, extradata)) {
            this->Reset();
            return false;
        }
        m_opened = true;
    }
    if (!offsets.hidden) {
        m_pending_offsets.push(offsets);
    }
    FFmpeg::Packet packet(packet_data);
    packet.GetPacket()->pts = m_next_pts;
    packet.GetPacket()->dts = m_next_pts;
    ++m_next_pts;
    return m_decoder_context->SendPacket(packet);
}

std::optional<DecodeApi::DecodedFrame> DecodeApi::ReceiveFrame() {
    auto frame = m_decoder_context->ReceiveFrame();
    if (!frame) {
        return std::nullopt;
    }
    FrameOffsets offsets{};
    if (!m_pending_offsets.empty()) {
        offsets = m_pending_offsets.front();
        m_pending_offsets.pop();
    }
    return DecodedFrame{std::move(frame), offsets};
}

}
