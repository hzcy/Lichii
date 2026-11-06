# NX2 parallel render pass recording (prototype)

`EDEN_NX2_PARALLEL_RECORD=<threads>` (2 to 8) records large render passes into Vulkan secondary
command buffers on several threads. It is off by default. Unset, the scheduler passes the real
command buffer to recorded commands, exactly as before.

## Where the time goes

In Donkey Kong Bananza (Ingot Isle, 7429fba04e, `DKB cpu through_frame`), the GPU back thread is
busy for 920-1031 ms of each ~1.06 s window. The Vulkan worker, which runs every `vkCmd*` call
today, is busy for 187-265 ms. The back's ~7.3 us per draw is emulator-side resolve: buffer and
texture caches, method processing, descriptors. So moving `vkCmd*` recording off the worker
removes worker time, not back time. This prototype proves the recording mechanism. It is not
expected to raise the frame rate on its own.

## Design

The back thread is unchanged. It still resolves every draw and records closures into the
scheduler's command chunks. Each draw's closures are its record packet: they carry the pipeline,
descriptor set handles, vertex and index bindings, dynamic state, push constants and draw
parameters.

On the worker (`ParallelRecord`, `vk_parallel_record.cpp`):

1. The worker runs the closures in order, as before. It passes them a `vk::CommandBuffer` that
   holds the real primary handle with a copy of the dispatch table whose `vkCmd*` entries are
   hooks. Host-side work inside closures (descriptor set allocation, pipeline build waits) still
   happens in the original order on the worker.
2. Outside a render pass, a hook forwards the call to the primary. Calls that set state are also
   logged: arguments are copied into an arena, and a last-writer table records which call last
   set each state slot (pipeline, viewport *n*, vertex buffer *n*, descriptor set *n*, push
   constant word *n*, each dynamic state, ...).
3. `vkCmdBeginRenderPass` starts a capture. Calls inside the pass are only logged.
4. At `vkCmdEndRenderPass`, a pass with at least 96 draws is split into up to *threads* ranges of
   32 or more draws. Each range is recorded into a secondary command buffer
   (`RENDER_PASS_CONTINUE`, inheriting the render pass and framebuffer). A secondary first
   replays, in original order, the last writer of every graphics slot as of the range start, then
   its own calls. The worker records the first range. The helper threads, each with its own
   command pool, record the rest. The primary then records
   `BeginRenderPass(SECONDARY_COMMAND_BUFFERS)`, one `vkCmdExecuteCommands` with the
   secondaries in order, and `EndRenderPass`. After that it replays every last writer, because
   the primary's bound state is undefined after `vkCmdExecuteCommands`.
5. Replaying a subsequence that keeps the original order, and that contains the last writer of
   every slot, leaves each slot with the value it has inline.
6. A new primary command buffer clears the log. The back already re-emits all state after a
   submit (`InvalidateCommandBufferState`).

### Serial fallback

The pass replays inline (`VK_SUBPASS_CONTENTS_INLINE`), and the rest of it records directly, when
it meets:

- queries (`vkCmdBeginQuery`/`EndQuery`/reset/copy), including a query or conditional rendering
  that began before the pass, since secondaries don't inherit either;
- barriers;
- transform feedback;
- conditional rendering;
- timestamps (`EDEN_DKB_FRAME_TRACE` GPU sampling);
- debug labels;
- events;
- any transfer or compute command;
- a render pass begin with a `pNext` chain or more than 16 clear values;
- a pass with fewer than 96 draws.

A state call the log can't copy makes the rest of that primary record inline. Examples are a
push descriptor template of unknown size, a `pNext` chain, or more than 16 viewports.

Clears are deferred into render pass begins (`RealizeDeferredClear`), so the captured begin info
carries them. `vkCmdClearAttachments` is recorded into the secondaries.

### Profiling

With `EDEN_NX2_GPU_PROFILE=1`, every 10 s the log has a `Parallel record per second` line. It
shows render passes, split passes, secondaries, draws recorded in parallel and inline, worker
split and wait ms, record ms per thread, and inline passes by reason. The back's remaining
resolve cost is in the existing `GPU back` zone report.

## Tests

- `src/tests/video_core/parallel_record.cpp` runs randomized streams through a mock dispatch
  table. Every draw, dispatch and query must see the same bound state as the inline stream, with
  2, 3, 4 and 8 threads. Removing the prefix replay, or the replay after
  `vkCmdExecuteCommands`, makes it fail.
- `src/tests/video_core/parallel_record_vulkan.cpp`, with `EDEN_TEST_VULKAN=1`, renders 600
  overlapping draws on a real device with `VK_LAYER_KHRONOS_validation`. It requires a
  byte-identical image and no validation warnings. On lavapipe, set
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`.

## Prototype limits

- **Hook coverage.** Every `vkCmd*` entry in `vk::DeviceDispatch` must have a hook. A new entry
  without one bypasses the capture and lands in the primary out of order inside a captured pass.
- **Synchronous wait.** The worker waits for the helpers at the end of each split pass, so it
  can't overlap recording with the next pass.
- **Queries.** Occlusion queries inside a pass force the inline path. Supporting them needs
  inherited queries or queries split at range boundaries.
- **Driver risk.** Secondaries cost more on some drivers:
  - Tilers and Qualcomm.
  - MoltenVK, which emulates secondaries.
  - NVIDIA and RADV handle them well, but per-secondary overhead can exceed the recording saved
    on passes near the 96-draw threshold.
- **No fast path.** There is no unchanged-draw fast path yet. Cutting the back's resolve cost is
  the step that can move the frame rate.
