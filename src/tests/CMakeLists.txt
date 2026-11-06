# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    video_core/unswizzle_workers.cpp
    video_core/aftermath_dump.cpp
    video_core/frame_dump_schedule.cpp
    video_core/gpu_timestamp.cpp
    core/nx2_streamplay.cpp
    core/nx2_video_ready.cpp
    core/nx2_debug_server.cpp
    core/nx2_present_pacer.cpp
    common/dkb_cpu_trace.cpp
    common/dkb_frame_trace.cpp
    audio_core/nx2_sysmix.cpp
    audio_core/sysmix_sink.cpp
    audio_core/sysmix_test_context.h
    video_core/ampere_starfox_test10.cpp
    video_core/ampere_yoshi_title.cpp
    video_core/ampere_yoshi_compute.cpp
    video_core/texture_rect_descriptor.cpp
    audio_core/batch_decoder.cpp
    audio_core/hardware_audio_codec.cpp
    core/gpu_dirty_memory.cpp
    core/memory_acquire.cpp
    core/builtin_patches.cpp
    core/system_test_context.h
    video_core/ampere_branch_modifiers.cpp
    video_core/ampere_fast_fusion_logged.cpp
    video_core/fast_fusion_surface_store.cpp
    video_core/fast_fusion_byte_store.cpp
    video_core/vk_byte_addressed_shadow.cpp
    video_core/global_memory_coherence.cpp
    video_core/storage_buffer_pointer_tracking.cpp
    video_core/storage_buffer_control_flow.cpp
    core/nup_validation.cpp
    input_common/calibration_lifetime.cpp
    video_core/ampere_logged_numeric.cpp
    video_core/ampere_logged_textures.cpp
    video_core/native_texture_header_index.cpp
    video_core/texture_handles.cpp
    video_core/ampere_selevet_modes.cpp
    video_core/ampere_selevet_spills.cpp
    video_core/blit_dither.cpp
    video_core/maxwell_numeric.cpp
    video_core/maxwell_permute_lod.cpp
    video_core/submission_policy.cpp
    video_core/framebuffer_clear_layers.cpp
    video_core/fire_emblem_upscale.cpp
    video_core/dlss_contract.cpp
    video_core/ampere_fast_fusion.cpp
    video_core/ampere_async_copy.cpp
    video_core/ampere_trap_guards.cpp
    video_core/async_draw_policy.cpp
    video_core/async_shader_emission.cpp
    video_core/pipeline_build.cpp
    video_core/multi_range_refresh.cpp
    video_core/command_fetch_order.cpp
    video_core/command_snapshot_suspensions.cpp
    video_core/pending_maps.cpp
    video_core/maxwell_fast_method.cpp
    video_core/queued_memory_writes.cpp
    hid_core/sensor_info.cpp
    common/paged_scalar.cpp
    common/queue_lifetime.cpp
    common/queue_timed_wait.cpp
    common/range_set_move.cpp
    video_core/uniform_cache_history.cpp
    video_core/x8d24_conversion.cpp
    video_core/ampere_ported_half_compare.cpp
    video_core/ampere_geometry_capture.cpp
    video_core/ampere_geometry_exit.cpp
    video_core/ampere_geometry_lowering.cpp
    video_core/ampere_ported_conversions.cpp
    video_core/ampere_ported_uniform_vote.cpp
    video_core/ampere_ported_textures.cpp
    video_core/ampere_tennis_i2f.cpp
    video_core/ampere_tennis_tessellation.cpp
    video_core/pipeline_cache_files.cpp
    video_core/storage_binding_capacity.cpp
    video_core/native_indirect_reset.cpp
    video_core/gpu_command_safety.cpp
    video_core/syncpoint_actions.cpp
    video_core/nx2_video_render.cpp
    video_core/ampere_matrix.cpp
    video_core/spirv_software_matrix.cpp
    video_core/spirv_builtin_capabilities.cpp
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/event.cpp
    common/fibers.cpp
    common/game_presets.cpp
    common/host_memory.cpp
    common/page_table.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/slot_vector.cpp
    common/undefined_fix.cpp
    common/unique_function.cpp
    common/zstd_compression.cpp
    core/core_timing.cpp
    core/thread_startup_policy.cpp
    core/nup_writer.cpp
    core/nup_metadata.cpp
    core/nx2_image_layout.cpp
    core/nx2_video_input_queue.cpp
    core/ldn.cpp
    core/internal_network/socket_proxy.cpp
    core/priority_queue.cpp
    core/internal_network/network.cpp
    video_core/memory_tracker.cpp
    video_core/memory_manager_runs.cpp
    video_core/nx2_perf_draw_binds.cpp
    video_core/nx2_perf_texture_writes.cpp
    video_core/h264_packet.cpp
    video_core/nx2_gpu_mapping.cpp
    video_core/vertex_repack.cpp
    video_core/ampere_lowering.cpp
    video_core/ampere_geometry.cpp
    video_core/ampere_i2f.cpp
    video_core/ampere_returns.cpp
    video_core/ampere_guarded_ulea.cpp
    video_core/ampere_compute_lowering.cpp
    video_core/ampere_uniform_immediate_shift.cpp
    video_core/ampere_bindless_surfaces.cpp
    video_core/native_macro_fifo.cpp
    video_core/uniform_stream_reuse.cpp
    video_core/shader_invalidation.cpp
    video_core/descriptor_table.cpp
    video_core/global_address_tracking.cpp
    video_core/nvn_storage_descriptor.cpp
    video_core/ampere_f2i.cpp
    video_core/ampere_ldc.cpp
    video_core/ampere_vote.cpp
    video_core/shader_attributes.cpp
    video_core/indirect_reset.cpp
    input_common/calibration_configuration_job.cpp
)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core audio_core input_common video_core)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

if (TARGET dynarmic::dynarmic)
    target_sources(tests PRIVATE core/window_trace.cpp)
    target_link_libraries(tests PRIVATE dynarmic::dynarmic)
endif()

add_test(NAME tests COMMAND tests)

if (WIN32 AND ARCHITECTURE_x86_64)
    add_executable(event_wait_probe ../../tools/nx2/event_wait_probe.cpp)
    target_link_libraries(event_wait_probe PRIVATE common)
endif()

add_executable(update_checker_tests
    common/update_checker.cpp
    ../frontend_common/update_checker.cpp)
target_link_libraries(update_checker_tests PRIVATE common Catch2::Catch2WithMain)
add_test(NAME update_checker_tests COMMAND update_checker_tests)

# needed for vma
if (NOT MSVC)
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
