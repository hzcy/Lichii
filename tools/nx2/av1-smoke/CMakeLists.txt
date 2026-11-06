# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(av1_decode_smoke av1_decode_smoke.cpp)
target_compile_features(av1_decode_smoke PRIVATE cxx_std_17)
target_include_directories(av1_decode_smoke PRIVATE ${FFmpeg_INCLUDE_DIR})
target_link_libraries(av1_decode_smoke PRIVATE ${FFmpeg_LIBRARIES})
target_link_options(av1_decode_smoke PRIVATE ${FFmpeg_LDFLAGS})

if (TARGET ffmpeg-build)
    add_dependencies(av1_decode_smoke ffmpeg-build)
endif()
