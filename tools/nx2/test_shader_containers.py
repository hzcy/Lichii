# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from shader_archive import ArchiveError
from shader_containers import dcx_decompress, lz4_block, parse_bnd4, parse_cgsb, parse_sharcb, parse_shn


def graphics(stage):
    return struct.pack("<I", 0xc0081 | ((stage + 1) << 10)) + bytes(124) + struct.pack("<4I", 0x794d, 0, 0x03800000, 0)


def cgsb(stage=0):
    code = graphics(stage)
    control = bytes.fromhex("34127698c9000000")
    header = b"CGSB\x33\x22\x11\0" + struct.pack("<HHIQ", 1, stage, 0, 24)
    record = b"NVN \1\0\0\0" + bytes(8)
    record += struct.pack("<8Q", 104, len(code), 104 + len(code), len(control), 0, 0, 0, 0)
    return header + record + code + control


def literals(data):
    size = len(data)
    if size < 15:
        return bytes([size << 4]) + data
    remaining = size - 15
    return b"\xf0" + b"\xff" * (remaining // 255) + bytes([remaining % 255]) + data



def sharcb(slot=0):
    """Two records reference one control and one code allocation."""
    section, pool = 0x200, 0x600
    table = section + 0x150
    first = table + 8
    second = first + 0x100
    end = second + 0x80
    control = first + 0x50
    allocation = 0x100
    data = bytearray(pool + allocation)
    data[:4] = b"BAHS"
    struct.pack_into("<7I", data, 4, 11, len(data), 5, 0, section, allocation, pool)
    struct.pack_into("<II", data, table, end - table, 2)
    for start, length in [(first, second - first), (second, end - second)]:
        struct.pack_into("<4IQIi", data, start, length, slot, 0, length - 16,
                         0, 104, control - (start + 16))
    data[control:control + 8] = bytes.fromhex("34127698c9000000")
    payload = bytes(range(16)) if slot == 3 else graphics(0)
    data[pool:pool + len(payload)] = payload
    struct.pack_into("<4I", data, control + 56, len(payload), 0, allocation, allocation)
    struct.pack_into("<I", data, control + 84, (5 if slot == 3 else 0) << 16)
    return data, first, second, control


class SharcbTests(unittest.TestCase):
    def test_graphics_alias_uses_signed_control_offset(self):
        """Verify two records can point backwards to the same control block."""
        data, _, _, control = sharcb()
        records = parse_sharcb(data)
        self.assertEqual([item["stage"] for item in records], ["vertex", "vertex"])
        self.assertEqual([item["control_offset"] for item in records], [control, control])
        self.assertEqual(records[0]["code"], records[1]["code"])
        self.assertEqual(records[0]["code"][:144], graphics(0))

    def test_compute_uses_declared_extent_without_graphics_header(self):
        """Check compute instructions exclude allocation padding."""
        data, _, _, _ = sharcb(3)
        for item in parse_sharcb(data):
            self.assertEqual(item["stage"], "compute")
            self.assertEqual(item["code"], bytes(range(16)))

    def test_rejects_cross_pool_or_invalid_control(self):
        """Reject descriptors that leave their pools or disagree with the stage."""
        data, first, second, control = sharcb()
        for offset, value in [(8, len(data) + 1), (first, 0), (first + 16, 1),
                              (second + 28, 0x80000000), (control + 56, 0),
                              (control + 84, 5 << 16), (first - 4, 1)]:
            with self.subTest(offset=offset):
                invalid = bytearray(data)
                struct.pack_into("<I", invalid, offset, value)
                with self.assertRaises(ArchiveError):
                    parse_sharcb(invalid)


class ContainerTests(unittest.TestCase):
    def test_lz4_literals_and_overlapping_matches(self):
        for data in (b"", b"abc", b"x" * 600):
            self.assertEqual(lz4_block(literals(data), len(data)), data)
        self.assertEqual(lz4_block(b"\x13a\x01\x00", 8), b"a" * 8)
        for data, size in ((b"\x00\x00\x00", 4), (b"\xf0", 15), (b"\x20a", 2), (b"\x10a", 0)):
            with self.assertRaises(ArchiveError):
                lz4_block(data, size)

    def test_cgsb_vertex_and_fragment_have_independent_stage_bits(self):
        for stage, name in ((0, "vertex"), (4, "fragment")):
            records = parse_cgsb(cgsb(stage))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["stage"], name)
            self.assertEqual(records[0]["code"], graphics(stage))

    def test_cgsb_rejects_bad_extents_and_stage_conflicts(self):
        source = cgsb()
        for size in (0, 23, 103, len(source) - 1):
            with self.assertRaises(ArchiveError):
                parse_cgsb(source[:size])
        data = bytearray(source)
        struct.pack_into("<H", data, 10, 4)
        with self.assertRaises(ArchiveError):
            parse_cgsb(data)

    def test_cgsb_preserves_mesh_allocation_without_assuming_vertex_header(self):
        data = bytearray(cgsb())
        struct.pack_into("<H", data, 10, 7)
        record, = parse_cgsb(data)
        self.assertEqual(record["stage"], "mesh")
        self.assertEqual(record["envelope"], "unverified-mesh")
        self.assertEqual(record["code"], graphics(0))
        self.assertEqual(record["code_offset"], 104)
        self.assertEqual(record["control_offset"], 104 + len(graphics(0)))

    def test_dcx_declared_size_and_stream_bounds(self):
        raw = b"bounded shader fixture" * 20
        compressed = zlib.compress(raw)
        header = bytearray(76)
        header[:4] = b"DCX\0"
        struct.pack_into(">I", header, 20, 76)
        header[24:28] = b"DCS\0"
        struct.pack_into(">II", header, 28, len(raw), len(compressed))
        header[36:44] = b"DCP\0DFLT"
        self.assertEqual(dcx_decompress(header + compressed), raw)
        struct.pack_into(">I", header, 28, len(raw) - 1)
        with self.assertRaises(ArchiveError):
            dcx_decompress(header + compressed)

    def test_binder_preserves_names_and_payloads(self):
        payload = cgsb(4)
        name = "fragment.ppo".encode("utf-16-le") + bytes(2)
        offset = 100 + len(name)
        header = bytearray(64)
        header[:4] = b"BND4"
        header[10] = 1
        struct.pack_into("<I", header, 12, 1)
        struct.pack_into("<Q", header, 16, 64)
        struct.pack_into("<Q", header, 32, 36)
        header[48:52] = b"\1t\0\0"
        record = b"\x40\0\0\0\xff\xff\xff\xff" + struct.pack("<QQIII", len(payload), len(payload), offset, 0, 100)
        self.assertEqual(parse_bnd4(header + record + name + payload), [("fragment.ppo", payload)])

    def test_shn_separates_metadata_from_gpu_allocation(self):
        code = graphics(0)
        metadata = bytearray(128 + 9 * 80 + 8)
        metadata[:8] = b"SHDROUNC"
        struct.pack_into("<II", metadata, 8, 20241128, 1)
        struct.pack_into("<Q", metadata, 48, 128)
        struct.pack_into("<QQ", metadata, 128, 0, 128 + 9 * 80)
        struct.pack_into("<II", metadata, 192, len(code), 8)
        metadata[-8:] = bytes.fromhex("34127698c9000000")
        packed_meta, packed_code = literals(metadata), literals(code)
        header = bytearray(68)
        header[:4] = b"SHN!"
        struct.pack_into("<III", header, 4, 68, len(packed_meta), len(metadata))
        struct.pack_into("<III", header, 36, 68 + len(packed_meta), len(packed_code), len(code))
        records = parse_shn(header + packed_meta + packed_code)
        self.assertEqual(records[0]["code"], code)
        self.assertEqual(records[0]["stage"], "vertex")

    def test_shn_reads_every_variation_and_compute_slot(self):
        code = graphics(0) + bytes(16)
        metadata = bytearray(176 + 18 * 80 + 16)
        metadata[:8] = b"SHDROUNC"
        struct.pack_into("<II", metadata, 8, 20241128, 2)
        control = 176 + 18 * 80
        struct.pack_into("<Q", metadata, 48, 176)
        struct.pack_into("<Q", metadata, 128, 176 + 9 * 80)
        struct.pack_into("<QQ", metadata, 176, 0, control)
        struct.pack_into("<II", metadata, 176 + 64, len(graphics(0)), 8)
        compute = 176 + 11 * 80
        struct.pack_into("<QQ", metadata, compute, len(graphics(0)), control + 8)
        struct.pack_into("<II", metadata, compute + 64, 16, 8)
        metadata[control:] = bytes.fromhex("34127698c9000000") * 2
        packed_meta, packed_code = literals(metadata), literals(code)
        header = bytearray(68)
        header[:4] = b"SHN!"
        struct.pack_into("<III", header, 4, 68, len(packed_meta), len(metadata))
        struct.pack_into("<III", header, 36, 68 + len(packed_meta), len(packed_code), len(code))
        records = parse_shn(header + packed_meta + packed_code)
        self.assertEqual([(r["variation"], r["stage"]) for r in records], [(0, "vertex"), (1, "compute")])
        self.assertEqual(records[1]["envelope"], "unverified-compute")
        metadata[compute + 80:compute + 160] = metadata[compute:compute + 80]
        metadata[compute:compute + 80] = bytes(80)
        records = parse_shn(header + literals(metadata) + packed_code)
        self.assertEqual(records[0]["stage"], "vertex")
        self.assertEqual(records[1]["stage"], "unknown-shdrounc-slot-3")
        self.assertEqual(records[1]["envelope"], "unverified-shdrounc-slot")

    def test_inventory_omissions_are_reported_instead_of_empty_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            import json
            root = Path(temporary)
            library = root / "library"
            library.mkdir()
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps([
                {"name": "missing game", "directory": str(library), "programs": [], "candidates": [],
                 "errors": [{"path": str(library), "error": "read failed"}]},
                {"name": "metadata only", "directory": str(library), "errors": [],
                 "programs": [{"title_id": "0400000000000000"}],
                 "candidates": [{"path": str(library / "shader.json")}]}]))
            output = root / "output"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("extract_static_shaders.py")),
                                     "--inventory", str(inventory), "--read-only-root", str(library),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["programs"])
            self.assertEqual({error["phase"] for error in manifest["errors"]}, {"inventory", "coverage"})
            self.assertTrue(any(error["game"] == "metadata only" and error["phase"] == "coverage"
                                for error in manifest["errors"]))
            self.assertFalse(list(library.iterdir()))

    def test_extraction_refuses_writing_inside_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "shader.cgsb"
            source.write_bytes(cgsb())
            before = source.read_bytes()
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("extract_static_shaders.py")),
                                     "--source", f"0400000000000000={source}",
                                     "--read-only-root", str(root), "--output", str(root / "output")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "output").exists())
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
