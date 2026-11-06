#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from types import SimpleNamespace
import json
import struct
import tempfile
import unittest
from unittest.mock import patch

import shader_sweep as sweep
from playthrough_hints import export_hints, import_hints, restore_pipeline


def environment(version, stage):
    """Create a cache environment with version-specific metadata tails."""
    code = bytes(88)
    counts = [len(code), 1, 1, 1]
    if version >= 47:
        counts.append(1)
    counts.append(1)
    header = struct.pack("<" + "Q" * len(counts) + "7I", *counts, 0, 2, 0, 0, 80, 1, stage)
    context = bytes(8 + 8 + 12 + (9 if version >= 47 else 0) + 12)
    tail = bytes(16 if stage == 5 else 80 + (32 if stage == 3 else 0))
    if version >= 53 and stage in (7, 8):
        tail += struct.pack("<3I", 32, 64, 128) + bytes(range(80))
    return header + code + context + tail


def pipeline_cache(version):
    data = b"yuzucach" + struct.pack("<I", version)
    data += struct.pack("<I", 3)
    data += environment(version, 0) + environment(version, 3) + environment(version, 4)
    data += bytes(sweep.GRAPHICS_KEY_SIZE)
    data += struct.pack("<I", 1) + environment(version, 5) + bytes(24)
    return data


class CacheTests(unittest.TestCase):
    def test_supported_versions_preserve_records_and_stage(self):
        """Verify ordinary record layouts stay compatible across supported versions."""
        for version in (46, 47, 48, 49, 50, 51, 52, 53):
            records = list(sweep.parse_cache(pipeline_cache(version)))
            self.assertEqual([r["stage"] for r in records], ["vertex", "geometry", "fragment", "compute"])
            for record, stage in zip(records, (0, 3, 4, 5)):
                self.assertEqual(record["bytes"], environment(version, stage))
                self.assertEqual(record["version"], version)
                self.assertEqual(len(record["key"]), 24 if stage == 5 else 1056)

    def test_mesh_context_keeps_following_stages_and_pipeline_keys_aligned(self):
        """Verify mesh metadata leaves following cache fields aligned."""
        data = b"yuzucach" + struct.pack("<II", 53, 3)
        data += environment(53, 7) + environment(53, 8) + environment(53, 4)
        key = bytes(range(256)) * 4 + bytes(32)
        data += key
        data += struct.pack("<I", 1) + environment(53, 5) + bytes(24)
        records = list(sweep.parse_cache(data))
        self.assertEqual([r["stage"] for r in records], ["task", "mesh", "fragment", "compute"])
        for record, stage in zip(records, (7, 8, 4, 5)):
            self.assertEqual(record["bytes"], environment(53, stage))
            self.assertEqual(record["key"], bytes(24) if stage == 5 else key)
        truncated = b"yuzucach" + struct.pack("<II", 53, 1) + environment(53, 8)[:-1]
        with self.assertRaises(sweep.CacheError):
            list(sweep.parse_cache(truncated))

    def test_legacy_mesh_records_without_context_are_rejected(self):
        """Reject legacy mesh records that cannot supply mesh context."""
        data = b"yuzucach" + struct.pack("<II", 52, 1) + environment(52, 8) + bytes(1056)
        with self.assertRaisesRegex(sweep.CacheError, "require cache version 53"):
            list(sweep.parse_cache(data))

    def test_truncation_is_not_an_empty_success(self):
        data = pipeline_cache(47)
        for length in (0, 7, 11, 13, 50, len(data) - 1):
            with self.subTest(length=length), self.assertRaises(sweep.CacheError):
                list(sweep.parse_cache(data[:length]))

    def test_unknown_version_and_environment_count_rejected(self):
        """Reject unsupported versions and out-of-range stage counts."""
        for data in (b"yuzucach" + struct.pack("<I", 54),
                     b"yuzucach" + struct.pack("<II", 47, 7)):
            with self.assertRaises(sweep.CacheError):
                list(sweep.parse_cache(data))

    def test_playthrough_hints_round_trip_requires_local_code(self):
        """Rebuild an exact task, mesh and fragment pipeline using only local code."""
        key = bytes(range(256)) * 4 + bytes(32)
        payload = struct.pack("<I", 3)
        payload += environment(53, 7) + environment(53, 8) + environment(53, 4) + key
        cache = b"yuzucach" + struct.pack("<I", 53) + payload
        hint = export_hints(cache, "0400486024dec000", "131072")
        record, = hint["pipelines"]
        programs = {}
        for stage, env in zip(record["stages"], sweep.parse_cache(cache), strict=True):
            self.assertEqual(stage["shader_sha256"], sweep.digest(env["code"]))
            self.assertEqual(
                bytes.fromhex(stage["prefix"]) + env["code"] + bytes.fromhex(stage["context"]),
                env["bytes"])
            self.assertEqual(set(stage), {"stage", "shader_sha256", "header_sha256",
                                          "prefix", "context"})
            self.assertIsNone(restore_pipeline(record, 53, programs))
            programs[env["stage"], stage["shader_sha256"]] = env["code"]
        self.assertEqual(restore_pipeline(record, 53, programs), (key, payload))
        record["stages"][1]["context"] = record["stages"][1]["context"][:-2]
        with self.assertRaises(sweep.CacheError):
            restore_pipeline(record, 53, programs)

    def test_import_skips_malformed_records_and_restores_valid_ones(self):
        """Check malformed metadata before and after shader lookup."""
        data = pipeline_cache(53)
        hint = export_hints(data, "0400486024dec000", "131072")
        compute = hint["pipelines"][1]
        invalid = {**compute, "stages": [{**compute["stages"][0], "context": ""}]}
        hint["pipelines"] = [None, invalid, compute]
        code = list(sweep.parse_cache(data))[-1]["code"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"programs": [{
                "title_id": hint["title_id"], "stage": "compute",
                "format": None, "code_sha256": sweep.digest(code),
            }]}), encoding="utf-8")
            args = SimpleNamespace(manifest=[manifest], title=hint["title_id"],
                                   output=root / "out", compiler=root / "compiler", jobs=1)
            with patch("playthrough_hints.pack_archive_shader", return_value=("compute", code)):
                summary = import_hints(args, hint)
            self.assertEqual(summary["invalid_record"], 2)
            self.assertEqual(summary["restored"], 1)
            output = args.output / args.title / "vulkan.fork-v53.bin"
            record, = sweep.parse_cache(output.read_bytes())
            self.assertEqual(record["code"], code)

    def test_native_code_matching_excludes_header_scheduling_padding_and_map(self):
        words = [0] * 10 + [11, 12, 0, 13, 14, 15, 0, 16, 0, 80, 88, 2, sweep.MAP_MAGIC]
        code = struct.pack("<" + "Q" * len(words), *words)
        self.assertEqual(sweep.unpack_native_code(code), struct.pack("<6Q", 11, 12, 13, 14, 15, 16))

    def test_maxwell_without_address_map_does_not_match_native(self):
        self.assertIsNone(sweep.unpack_native_code(bytes(128)))


if __name__ == "__main__":
    unittest.main()
