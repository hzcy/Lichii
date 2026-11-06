# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import struct
import tempfile
import os
import unittest
import xml.dom.minidom

import sample_report

BASE = 0x140000000
DLL = 0x7FF800000000


def header(hz=2000):
    return b"NX2SMPL1" + struct.pack("<IIQ", 1, hz, 10_000_000)


def module(base, size, path):
    name = path.encode()
    return bytes([3]) + struct.pack("<QIH", base, size, len(name)) + name


def thread(tag, name):
    raw = name.encode()
    return bytes([2]) + struct.pack("<BH", tag, len(raw)) + raw


def sample(tag, frames, qpc=0):
    return bytes([1]) + struct.pack("<BBB", tag, len(frames), 0) + struct.pack("<Q", qpc) + \
        struct.pack("<%dQ" % len(frames), *frames)


class FakeSymbolizer:
    def __init__(self, names):
        self.names = names
        self.asked = set()

    def resolve(self, rvas):
        self.asked |= set(rvas)
        return {rva: self.names.get(rva, ["lichii.exe+%#x" % rva]) for rva in rvas}


def file_bytes(end=True):
    data = header() + module(BASE, 0x1000000, r"C:\lx\build\bin\lichii.exe") + \
        module(DLL, 0x10000, r"C:\Windows\System32\ntdll.dll") + thread(0, "GPUBack") + thread(1, "GPU")
    # GPUBack: CallMethod <- Run (twice), Draw <- Run (once, with Configure inlined into Draw)
    data += sample(0, [BASE + 0x100, BASE + 0x501])
    data += sample(0, [BASE + 0x100, BASE + 0x501])
    data += sample(0, [BASE + 0x200, BASE + 0x501])
    # GPU: inside ntdll, called from JIT code
    data += sample(1, [DLL + 0x40, 0x12345678])
    return data + (bytes([0xFF]) if end else b"")


NAMES = {
    0x100: ["Maxwell3D::CallMethod"],
    0x200: ["GraphicsPipeline::Configure", "RasterizerVulkan::Draw"],
    0x500: ["GpuSplit::Run"],
}


class SampleReportTest(unittest.TestCase):
    def test_parse(self):
        parsed = sample_report.parse(file_bytes())
        self.assertTrue(parsed.complete)
        self.assertEqual(parsed.hz, 2000)
        self.assertEqual(parsed.threads, {0: "GPUBack", 1: "GPU"})
        self.assertEqual(len(parsed.modules), 2)
        self.assertEqual(len(parsed.samples), 4)
        self.assertEqual(parsed.samples[0][2], [BASE + 0x100, BASE + 0x501])

    def test_cut_short_file_keeps_whole_records(self):
        data = file_bytes(end=False)
        parsed = sample_report.parse(data[:-5])
        self.assertFalse(parsed.complete)
        self.assertEqual(len(parsed.samples), 3)

    def test_rejects_other_files(self):
        with self.assertRaises(ValueError):
            sample_report.parse(b"NOTSAMPL" + bytes(16))

    def test_symbolizes_return_addresses_at_the_call(self):
        parsed = sample_report.parse(file_bytes())
        symbolizer = FakeSymbolizer(NAMES)
        stacks = sample_report.symbolize_samples(parsed, "lichii.exe", symbolizer)
        self.assertIn(0x500, symbolizer.asked)  # 0x501 - 1
        self.assertNotIn(0x501, symbolizer.asked)
        self.assertEqual(stacks[0], ("GPUBack", ["GpuSplit::Run", "Maxwell3D::CallMethod"]))
        self.assertEqual(stacks[2], ("GPUBack", ["GpuSplit::Run", "RasterizerVulkan::Draw",
                                                 "GraphicsPipeline::Configure"]))
        self.assertEqual(stacks[3], ("GPU", ["[jit]", "ntdll.dll+0x40"]))

    def test_summary(self):
        parsed = sample_report.parse(file_bytes())
        stacks = sample_report.symbolize_samples(parsed, "lichii.exe", FakeSymbolizer(NAMES))
        threads = sample_report.summarize(stacks)
        back = threads["GPUBack"]
        self.assertEqual(back["total"], 3)
        self.assertEqual(back["self"]["Maxwell3D::CallMethod"], 2)
        self.assertEqual(back["self"]["GraphicsPipeline::Configure"], 1)
        self.assertEqual(back["inclusive"]["GpuSplit::Run"], 3)
        self.assertEqual(back["inclusive"]["RasterizerVulkan::Draw"], 1)
        self.assertEqual(back["paths"].most_common(1)[0],
                         ("GpuSplit::Run;Maxwell3D::CallMethod", 2))
        out = io.StringIO()
        sample_report.report(threads, parsed.hz, out=out)
        self.assertIn("66.67%       2  Maxwell3D::CallMethod", out.getvalue())

    def test_folded_and_svg(self):
        parsed = sample_report.parse(file_bytes())
        threads = sample_report.summarize(
            sample_report.symbolize_samples(parsed, "lichii.exe", FakeSymbolizer(NAMES)))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "folded.txt")
            sample_report.write_folded(threads, path)
            with open(path) as file:
                lines = file.read().splitlines()
        self.assertIn("GPUBack;GpuSplit::Run;Maxwell3D::CallMethod 2", lines)
        self.assertIn("GPU;[jit];ntdll.dll+0x40 1", lines)
        svg = sample_report.flamegraph_svg(threads)
        xml.dom.minidom.parseString(svg)
        self.assertIn("Maxwell3D::CallMethod (2 samples", svg)

    def test_symbolizer_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = os.path.join(directory, "lichii.exe")
            with open(exe, "wb") as file:
                file.write(b"MZ")
            fake = os.path.join(directory, "fake-symbolizer")
            with open(fake, "w") as file:
                file.write("#!/bin/sh\nwhile read a; do echo '{\"Symbol\":[{\"FunctionName\":\"F'$a'\"}]}'; done\n")
            os.chmod(fake, 0o755)
            cache = os.path.join(directory, "samples.bin.symcache.json")
            first = sample_report.Symbolizer(exe, cache, fake).resolve({0x10, 0x20})
            self.assertEqual(first[0x10], ["F0x10"])
            again = sample_report.Symbolizer(exe, cache, "/nonexistent").resolve({0x10})
            self.assertEqual(again[0x10], ["F0x10"])


if __name__ == "__main__":
    unittest.main()
