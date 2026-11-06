// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "common/nx2_sample_profile.h"

#ifdef _WIN32
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include <windows.h>
#include <tlhelp32.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#endif

namespace Common::Nx2SampleProfile {

namespace {

struct Config {
    unsigned hz{};
    unsigned delay_seconds{};
    bool delay_until_file{};
    unsigned seconds{20};
    std::vector<std::string> threads;
};

unsigned ReadUnsigned(const char* variable, unsigned fallback) {
    const char* const value = std::getenv(variable);
    return value != nullptr ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : fallback;
}

const Config& GetConfig() {
    static const Config config = [] {
        const char* const delay = std::getenv("EDEN_NX2_SAMPLE_DELAY");
        Config result{
            .hz = ReadUnsigned("EDEN_NX2_SAMPLE_PROFILE", 0),
            .delay_seconds = ReadUnsigned("EDEN_NX2_SAMPLE_DELAY", 0),
            .delay_until_file = delay != nullptr && std::string_view{delay} == "file",
            .seconds = ReadUnsigned("EDEN_NX2_SAMPLE_SECONDS", 20),
        };
        const char* const list = std::getenv("EDEN_NX2_SAMPLE_THREADS");
        std::string_view rest = list != nullptr ? list : "GPU,GPUBack,VulkanWorker";
        while (!rest.empty()) {
            const size_t comma = rest.find(',');
            if (const std::string_view item = rest.substr(0, comma); !item.empty()) {
                result.threads.emplace_back(item);
            }
            rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        }
        return result;
    }();
    return config;
}

bool Selected(std::string_view name) {
    for (const std::string& pattern : GetConfig().threads) {
        if (!pattern.empty() && pattern.back() == '*'
                ? name.starts_with(std::string_view{pattern}.substr(0, pattern.size() - 1))
                : name == pattern) {
            return true;
        }
    }
    return false;
}

#ifdef _WIN32

// Record kinds of the sample file (tools/nx2/sample_report.py parses it).
constexpr std::uint8_t KindSample = 1;
constexpr std::uint8_t KindThread = 2;
constexpr std::uint8_t KindModule = 3;
constexpr std::uint8_t KindEnd = 0xff;
constexpr size_t MaxThreads = 32;
constexpr size_t MaxDepth = 32;

struct SampledThread {
    HANDLE handle{};
    std::string name;
};

std::mutex threads_mutex;
std::array<SampledThread, MaxThreads> sampled_threads;
std::atomic<size_t> num_threads{};

template <typename T>
void Put(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void PutString(std::ofstream& file, std::string_view text) {
    Put(file, static_cast<std::uint16_t>(text.size()));
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string Utf8(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string result(size > 0 ? size - 1 : 0, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    return result;
}

void WriteModules(std::ofstream& file) {
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    MODULEENTRY32W entry{.dwSize = sizeof(MODULEENTRY32W)};
    for (BOOL more = Module32FirstW(snapshot, &entry); more; more = Module32NextW(snapshot, &entry)) {
        Put(file, KindModule);
        Put(file, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(entry.modBaseAddr)));
        Put(file, static_cast<std::uint32_t>(entry.modBaseSize));
        PutString(file, Utf8(entry.szExePath));
    }
    CloseHandle(snapshot);
}

/// Unwinds a suspended thread's stack. Allocates nothing: the thread may hold the heap lock.
size_t Walk(CONTEXT& context, std::array<std::uint64_t, MaxDepth>& frames) {
    size_t depth = 0;
    while (depth < MaxDepth && context.Rip != 0) {
        frames[depth++] = context.Rip;
        DWORD64 image_base{};
        const PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (function == nullptr) {
            // A leaf without unwind data has its return address on top of the stack. Past the
            // first frame this is code without unwind data (JIT): stop there.
            if (depth > 1) {
                break;
            }
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += sizeof(DWORD64);
            continue;
        }
        PVOID handler_data{};
        DWORD64 establisher_frame{};
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
                         &handler_data, &establisher_frame, nullptr);
    }
    return depth;
}

void SamplerMain() {
    const Config& config = GetConfig();
    const auto log_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    if (config.delay_until_file) {
        const auto trigger = log_dir / "sample.now";
        std::error_code error;
        std::filesystem::remove(trigger, error);
        LOG_INFO(Common, "Sample profile: starts once {} exists", trigger.string());
        while (!std::filesystem::exists(trigger, error)) {
            Sleep(250);
        }
        std::filesystem::remove(trigger, error);
        LOG_INFO(Common, "Sample profile: started by {}", trigger.string());
    } else {
        Sleep(config.delay_seconds * 1000);
    }

    const auto path =
        log_dir / ("nx2_samples_" + std::to_string(GetCurrentProcessId()) + ".bin");
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file) {
        LOG_ERROR(Common, "Sample profile: can't open {}", path.string());
        return;
    }
    LARGE_INTEGER frequency{};
    QueryPerformanceFrequency(&frequency);
    file.write("NX2SMPL1", 8);
    Put(file, std::uint32_t{1});
    Put(file, static_cast<std::uint32_t>(config.hz));
    Put(file, static_cast<std::uint64_t>(frequency.QuadPart));
    WriteModules(file);

    // A high resolution timer reaches 2 kHz; without it (before Windows 10 1803) Sleep(1) runs
    // at the system timer resolution.
    constexpr DWORD HighResolution = 0x2; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
    const HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, HighResolution, TIMER_ALL_ACCESS);
    const LONGLONG period_100ns = -static_cast<LONGLONG>(10'000'000 / config.hz);

    LARGE_INTEGER start{};
    QueryPerformanceCounter(&start);
    const LONGLONG end = start.QuadPart + static_cast<LONGLONG>(config.seconds) * frequency.QuadPart;
    size_t threads_written = 0;
    std::uint64_t samples = 0;
    std::array<std::uint64_t, MaxDepth> frames{};
    for (LARGE_INTEGER now = start; now.QuadPart < end; QueryPerformanceCounter(&now)) {
        const size_t count = num_threads.load(std::memory_order_acquire);
        for (; threads_written < count; ++threads_written) {
            Put(file, KindThread);
            Put(file, static_cast<std::uint8_t>(threads_written));
            PutString(file, sampled_threads[threads_written].name);
        }
        for (size_t index = 0; index < count; ++index) {
            const HANDLE thread = sampled_threads[index].handle;
            if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
                continue;
            }
            alignas(16) CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            const size_t depth = GetThreadContext(thread, &context) ? Walk(context, frames) : 0;
            ResumeThread(thread);
            if (depth == 0) {
                continue;
            }
            Put(file, KindSample);
            Put(file, static_cast<std::uint8_t>(index));
            Put(file, static_cast<std::uint8_t>(depth));
            Put(file, static_cast<std::uint8_t>(0));
            Put(file, static_cast<std::uint64_t>(now.QuadPart));
            file.write(reinterpret_cast<const char*>(frames.data()),
                       static_cast<std::streamsize>(depth * sizeof(std::uint64_t)));
            ++samples;
        }
        if (timer != nullptr) {
            LARGE_INTEGER due{.QuadPart = period_100ns};
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, INFINITE);
        } else {
            Sleep(1);
        }
    }
    Put(file, KindEnd);
    file.close();
    if (timer != nullptr) {
        CloseHandle(timer);
    }
    LOG_INFO(Common, "Sample profile: {} samples of {} threads at {} Hz written to {}", samples,
             threads_written, config.hz, path.string());
}

#endif

} // namespace

bool Requested() {
    return GetConfig().hz != 0;
}

void OnThreadNamed(const char* name) {
    if (!Requested() || !Selected(name)) {
        return;
    }
#ifdef _WIN32
    HANDLE handle{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                         FALSE, 0)) {
        return;
    }
    {
        std::scoped_lock lock{threads_mutex};
        const size_t index = num_threads.load(std::memory_order_relaxed);
        if (index == MaxThreads) {
            CloseHandle(handle);
            return;
        }
        sampled_threads[index] = {handle, name};
        num_threads.store(index + 1, std::memory_order_release);
    }
    static std::once_flag started;
    std::call_once(started, [] { std::thread{SamplerMain}.detach(); });
#endif
}

} // namespace Common::Nx2SampleProfile
