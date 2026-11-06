# FSR3 temporal regression checks

This port adapts Frizti/Lichii `a605fac744b2e20202165d4b8c100d735cd39987`
from the pinned `oot-gather-fix` tip `765659af4ccb7201d1145c57f3b45ce84653cea0`.
The original change was authored by chuppatricky as
`e725488a1e0c69efe31780d9d97f5b67bdefdf82`.

The old path copied the newest frame into both interpolation inputs. It also
treated half-resolution, current-to-previous flow as full-resolution forward
motion, and reversed the unwarped blend. The port preserves the previous image
before copying the new frame and uses the motion texture's dimensions for both
the median neighborhood and UV conversion. The optical-flow shader itself is
unchanged. Its search compares the current pixel against previous pixels at
`uv + offset / flow_extent`, which establishes the direction and units.

The Vulkan integration needs more than those arithmetic changes:

- The current copy needs transfer-source usage for its history rollover.
- New history/current images begin undefined. Motion writes and reads must use
  the layout declared by their descriptors, with compute write/read barriers.
- Newly allocated generated targets also begin undefined. The caller now passes
  the image as well as its view, so generation can discard and transition it.
  `PresentManager::GetRenderFrame` waits for the target's presentation fence
  before reuse, so no old target contents or in-flight uses need preserving.
- A single rewritten interpolation descriptor can still be in use by an older
  generation. The existing `DescriptorAllocator` retires each committed set at
  the scheduler's GPU completion tick. Each dispatch captures its own set.
- Flow descriptors reference the stable internal current/history copies rather
  than a changing caller view. Reset/resize still destroys the chain only after
  `FrameGen::Rebuild` finishes the scheduler; disabling also finishes first.

## Run

Install `glslang-tools`, `spirv-tools`, `mesa-vulkan-drivers`,
`vulkan-validationlayers`, and Python packages `vulkan==1.3.275.1`, `numpy`,
and `pillow`. Configure the project with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`.
Generate its two `fsr3_*_comp_spv.h` headers before running the C++ probe.

```sh
python tools/nx2/test_fsr3_shaders.py --out /tmp/fsr3-tests --build build
```

The test compiles and dispatches the real production shaders. With `--build`,
it also compiles the actual C++ copy/layout helpers through a small exported
probe, using the configured translation unit's includes and defines. The probe
does not instantiate the emulator's scheduler or the complete `Fsr3FrameGen`
class; descriptor retirement and reset orchestration are reviewed at their
production call sites, not claimed as end-to-end emulator coverage.

Checks cover distinct constants at three interpolation times, both translation
directions on each axis, flow-texel scale, a nonuniform median neighborhood,
identical-frame flow, history initialization and successive rollovers, fresh
even/odd/one-pixel extents, three generated targets in one submission, target
reuse, and Vulkan object teardown. Synchronization validation is enabled.
`translation.png` shows a synthetic previous/generated/current triptych.

`--shaders <directory>` runs the same checks against old or mutated shaders.
`--sanitize` instruments the production C++ probe with ASan and UBSan; preload
the matching Clang ASan runtime and add its directory to `LD_LIBRARY_PATH`.
Disable leak detection for the Python/driver host with
`ASAN_OPTIONS=detect_leaks=0`. No game, emulator launch, or PC is needed.
