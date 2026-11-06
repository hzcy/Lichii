import struct
import unittest

from shader_archive import ArchiveError, find_bnsh, parse_bnsh, parse_sarc
from extract_static_shaders import walk_container


def bnsh(stage=0):
    data = bytearray(0x400)
    data[:8] = b"BNSH\0\0\0\0"
    data[8:12] = bytes((0, 4, 2, 0))
    data[12:14] = b"\xff\xfe"
    struct.pack_into("<I", data, 0x1c, len(data))
    data[0x60:0x64] = b"grsc"
    struct.pack_into("<HH", data, 0x70, 4, 200)
    struct.pack_into("<IQ", data, 0x7c, 1, 0xc0)
    struct.pack_into("<Q", data, 0xd0, 0x100)
    struct.pack_into("<Q", data, 0x108 + stage * 8, 0x140)
    struct.pack_into("<QQII", data, 0x148, 0x180, 0x200, 0x100, 4)
    data[0x180:0x184] = b"ctrl"
    struct.pack_into("<I", data, 0x200, (stage + 1) << 10)
    return bytes(data)


def sarc():
    data = bytearray(0x54)
    data[:4] = b"SARC"
    struct.pack_into("<H", data, 4, 0x14)
    data[6:8] = b"\xff\xfe"
    struct.pack_into("<II", data, 8, len(data), 0x50)
    data[0x14:0x18] = b"SFAT"
    struct.pack_into("<HH", data, 0x18, 0xc, 1)
    struct.pack_into("<IIII", data, 0x20, 0, 0x1000000, 0, 4)
    data[0x30:0x34] = b"SFNT"
    struct.pack_into("<H", data, 0x34, 8)
    data[0x38:0x3d] = b"file\0"
    data[0x50:] = b"data"
    return bytes(data)


class ShaderArchiveTests(unittest.TestCase):
    def test_bnsh_graphics_and_compute(self):
        for stage, name in ((0, "vertex"), (3, "geometry"), (4, "fragment"), (5, "compute")):
            row, = parse_bnsh(bnsh(stage))
            self.assertEqual(row["stage"], name)
            self.assertEqual(row["code_offset"], 0x200)
            self.assertEqual(len(row["code"]), 0x100)
            self.assertEqual(row["control"], b"ctrl")
            self.assertEqual(row["version"], (0, 4, 2, 0))

    def test_v142_task_and_mesh_use_extended_slots_and_control_stage(self):
        """Check both mesh header variants and reject mismatched control stages."""
        for slot, control_stage, header_type, expected in (
                (0x40, 6, 1, "task"), (0x48, 7, 1, "mesh"), (0x48, 7, 3, "mesh")):
            with self.subTest(stage=expected, header=header_type):
                data = bytearray(bnsh())
                data[8:12] = bytes((1, 4, 2, 0))
                struct.pack_into("<QQ", data, 0x100, 2, 0)
                data[0x140:0x150] = bytes(16)
                struct.pack_into("<Q", data, 0x100 + slot, 0x150)
                struct.pack_into("<QQII", data, 0x158, 0x300, 0x200, 0x100, 104)
                data[0x300:0x308] = bytes.fromhex("34127698c9000000")
                struct.pack_into("<I", data, 0x354, control_stage << 16)
                struct.pack_into("<I", data, 0x200, header_type << 10)
                row, = parse_bnsh(bytes(data))
                self.assertEqual(row["stage"], expected)
                struct.pack_into("<I", data, 0x354, 5 << 16)
                with self.assertRaises(ArchiveError):
                    parse_bnsh(bytes(data))

    def test_effect_bundle_extracts_embedded_graphics_and_compute(self):
        """Check extraction stays scoped to effects bundles and rejects truncation."""
        header = b"YB\x05\x00" + bytes(12)
        data = header + bnsh() + bytes(48) + bnsh(5)
        records = list(walk_container(data, "effects.vsetb.byml"))
        self.assertEqual([record["stage"] for record in records], ["vertex", "compute"])
        self.assertTrue(records[0]["member"].endswith("!BNSH@10"))
        self.assertEqual(list(walk_container(data, "unrelated.byml")), [])
        with self.assertRaises(ArchiveError):
            list(walk_container(data[:-1], "effects.vsetb.byml"))

    def test_bad_bnsh(self):
        for offset, value, fmt in ((0x7c, 0xffffffff, "I"), (0xd0, 0x3ff, "Q"),
                                   (0x150, 0x3ff, "Q"), (0x158, 0xffffffff, "I"),
                                   (0x200, 5 << 10, "I"), (0x72, 201, "H")):
            data = bytearray(bnsh())
            struct.pack_into("<" + fmt, data, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ArchiveError):
                parse_bnsh(bytes(data))
        with self.assertRaises(ArchiveError):
            parse_bnsh(bnsh()[:-1])

    def test_supported_versions(self):
        for version in ((0, 4, 2, 0), (1, 3, 2, 0), (1, 4, 2, 0)):
            data = bytearray(bnsh())
            data[8:12] = bytes(version)
            with self.subTest(version=version):
                row, = parse_bnsh(bytes(data))
                self.assertEqual(row["version"], version)
                self.assertEqual(row["control"], b"ctrl")
                self.assertEqual(len(row["code"]), 0x100)
        data[8:12] = bytes((2, 4, 2, 0))
        with self.assertRaisesRegex(ArchiveError, "unsupported BNSH version"):
            parse_bnsh(bytes(data))

    def test_sarc_and_errors(self):
        self.assertEqual(parse_sarc(sarc()), [("file", b"data")])
        for offset, value in ((0x2c, 99), (0x24, 0x1ffffff)):
            data = bytearray(sarc())
            struct.pack_into("<I", data, offset, value)
            with self.assertRaises(ArchiveError):
                parse_sarc(bytes(data))
        with self.assertRaises(ArchiveError):
            parse_sarc(sarc()[:-1])

    def test_find_bnsh(self):
        data = bytearray(0x1000)
        data[:4] = b"FSHA"
        data[12:14] = b"\xff\xfe"
        struct.pack_into("<I", data, 0x1c, len(data))
        data[0x200:0x600] = bnsh()
        found = find_bnsh(bytes(data))
        self.assertEqual(found, [(0x200, bnsh())])
        struct.pack_into("<I", data, 0x21c, 0xffffffff)
        with self.assertRaises(ArchiveError):
            find_bnsh(bytes(data))


if __name__ == "__main__":
    unittest.main()
