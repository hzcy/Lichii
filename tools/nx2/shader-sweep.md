# Offline shader sweep

`shader_sweep.py` runs native captures and serialized Vulkan shader environments through the
shader compiler and `spirv-val`. It does not start an emulator, access a GPU, or change input files.

## Run

Build the `shader-sweep` C++ target under `tools/shader-sweep`, then run:

```sh
python3 tools/nx2/shader_sweep.py \
    /path/to/native-dumps /path/to/cache/shader \
    --compiler /path/to/shader-sweep \
    --output /path/to/sweep-before
```

Install SPIRV-Tools so `spirv-val` is on `PATH`, or pass `--spirv-val /path/to/spirv-val`.
The script validates against Vulkan 1.2. `--jobs` controls concurrent compiler subprocesses and
`--timeout` bounds each compiler and validator invocation. Every shader runs in its own process,
so an assertion, crash, or hang becomes a result rather than stopping the rest of the sweep.

The output directory contains `results.json`, a grouped `summary.md`, and the individual
environment records, pipeline keys, lowered code, and SPIR-V modules in `records/`.
The command exits nonzero if any input, context, compiler, or validation failure remains.

After changing the lowering, preserve the original native-to-cache matches:

```sh
python3 tools/nx2/shader_sweep.py \
    /path/to/native-dumps /path/to/cache/shader \
    --compiler /path/to/shader-sweep \
    --contexts-from /path/to/sweep-before/results.json \
    --output /path/to/sweep-after
```

Keep the earlier output directory. Its extracted environments and keys supply context when the
new lowering no longer produces the exact bytes stored in the older cache.

## Inputs and coverage

The loader reads little-endian `yuzucach` Vulkan caches, fork versions 46 through 53. It verifies each
record boundary and rejects unknown versions or truncated input. Version 47 includes captured
storage-pointer classifications; versions 48 through 50 retain that layout and version 46 omits them.
Version 49 invalidates pipelines after tessellation patch-size clamping. Version 50 captures the
additional texture-table context needed by dynamic texture/static sampler composition. Neither
changes the serialized layout.
Version 53 stores the three mesh registers and the 80-byte mesh primitive header
for task/mesh environments. It uses a separate Vulkan environment-cache filename;
the runtime must support v53 to load those entries.
The offline reader preserves that context, but standalone task/mesh compilation
still reports an explicit context limitation rather than emitting a guessed module.
Graphics keys are the 1,056-byte layout
from these versions; compute keys contain 24 bytes. Driver pipeline blobs and `vulkan_used`
indexes do not contain replayable shader environments and are not treated as shader inputs.

An environment preserves its actual shader stage, header, constant-buffer reads, texture types,
formats, compute workgroup size, and shared/local memory sizes. Cache entries contain already
lowered Maxwell code, not the original native instructions. Replaying them tests translation,
optimization, and SPIR-V emission, but cannot recover native programs rejected before caching.

Normal `hash-gfx-address.bin` captures contain only native instructions. Their filename does not
identify a graphics stage. The sweep lowers these and matches the result against cached code,
excluding headers, scheduling words, padding, and the native address map. A match supplies the
captured stage and environment for a fresh lowering and compilation. It never labels every
graphics program as a vertex shader. `rejected-stage-N-address.bin` captures include a 128-byte
header; `N` is the Maxwell engine shader slot, not the `Shader::Stage` enumeration. Compute
captures use `hash-cs-address.bin`.

Missing context is reported separately from compiler failures. A missing stage, texture format,
constant-buffer value, or storage-pointer classification is not evidence of an unsupported GPU
instruction. Fixed-pipeline assumptions are included in the compiler's JSON diagnostics.
Validation establishes that the emitted module is structurally valid under the tested profile;
it does not establish correct pixels, synchronization, or behavior on every driver.

Native captures default to the game label `Mario Tennis Fever`; use `--dump-game` for another
game. Cache game labels come from their parent title-ID directory; unknown IDs remain visible
as IDs. Counts are distinct compile jobs, including different captured environments and pipeline
keys, not draw counts. Identical native files are deduplicated by content.

If a capture log establishes the stage of an uncached program, pass
`--native-stage 0123456789abcdef=tess-eval`, using its filename hash. This supplies only the
stage, not missing headers or resource data; full compilation still reports that context gap.

## Parser tests

```sh
python3 -m unittest discover -s tools/nx2 -p test_shader_sweep.py -v
```

Keep bulk game captures and generated modules outside the repository. The small regression
environments in `tools/shader-sweep/fixtures` preserve captured programs that exposed backend
defects. Each records its source cache hash and byte offset. Run them with:

```sh
python3 tools/shader-sweep/test_captured.py /path/to/shader-sweep
```

## Playthrough hints

Use `nup-shader-export` and `extract_static_shaders.py` to produce the local
archive manifests. BAHS v11 and BFSHA are supported. For archives compressed
with a game dictionary, pass `--zstd-dictionary-pack` to the extractor
(Python 3.14 or `backports.zstd`).

Close the emulator before exporting. The hint file contains pipeline keys,
captured translation metadata, and shader fingerprints. It excludes shader
instruction bytes, saves, and driver cache blobs. Specify the exact game update
for both commands. Use hints from a trusted source: the pipeline metadata
comes from the recording even though the shader code is rebuilt locally.

```sh
python tools/nx2/playthrough_hints.py export \
    --title 0400486024dec000 --game-version 131072 \
    --cache /path/to/0400486024dec000/vulkan.fork-v52.bin \
    --output /path/to/zelda.hints.json

python tools/nx2/playthrough_hints.py import \
    --title 0400486024dec000 --game-version 131072 \
    --hints /path/to/zelda.hints.json --manifest /path/to/static/manifest.json \
    --compiler /path/to/shader-sweep --output /path/to/rebuilt-cache
```

Zelda's NVN2 `.vsetb.byml.zs` effect bundles may contain additional embedded
BNSH graphics and compute programs. Include the shared effects bundle and
relevant scene bundles in the local extraction inputs.

Pass each local extraction manifest with a separate `--manifest`. The importer
requires an exact packed shader match for every stage and reports incomplete
pipelines separately. NVN2 BNSH and BAHS task and mesh records can supply
the local code for v53 hints. The importer keeps the recorded mesh context and
requires an exact packed-code match for every stage, including the fragment
shader. The result is a new cache directory. For testing,
copy the resulting title directory into an empty portable profile's
`user/cache/shader` and launch with `EDEN_NX2_PRELOAD_ALL_PIPELINES=1`. The
cache format and pipeline state must be compatible with the recipient's build
and Vulkan features.
