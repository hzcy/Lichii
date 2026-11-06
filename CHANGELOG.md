# Changelog

All notable changes to Eglantine are documented here. Each release's notes are taken from its
`## [X.Y.Z]` section (release candidates use the section of the version they are a candidate for).
New entries go under `## [Unreleased]`; `tools/ci/bump-version.sh` turns them into a version section.

## [Unreleased]

- Reduce Ocarina of Time GPU submission and timing overhead, apply inherited title settings before
  initialization, and preserve explicit per-game choices across global configuration reloads.
- Fix split-GPU startup waits, syncpoint callback registration races, and deferred fragment-barrier
  lifetime; prioritize recent compatible pipelines within the bounded startup preload budget.
- Wait for all queued GPU writers before inspecting command/payload page-written flags, including
  writers queued after the last host sync.
- Recover from hardware video-decoder startup failures by replaying buffered input through the CPU
  decoder, including receive/transfer failures and stream resets.
- Recreate the legacy video decoder and context for CPU decoding when hardware initialization fails.

## [0.1.0]

- Initial release candidate
