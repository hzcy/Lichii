#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reports an EDEN_NX2_SAMPLE_PROFILE file (src/common/nx2_sample_profile.h).

    python3 -I tools/nx2/sample_report.py nx2_samples_<pid>.bin --exe lichii.exe [--folded out.txt] [--svg out.svg]

Symbolizes lichii.exe frames with llvm-symbolizer (lichii.pdb next to the exe), caching the
result next to the sample file, and prints per thread: self and inclusive time by function and
the top call paths. Other modules show as module+offset, code outside any module as [jit].
"""

import argparse
import collections
import html
import json
import os
import struct
import subprocess
import sys

KIND_SAMPLE = 1
KIND_THREAD = 2
KIND_MODULE = 3
KIND_END = 0xFF


class SampleFile:
    def __init__(self):
        self.hz = 0
        self.qpc_frequency = 0
        self.modules = []  # (base, size, path)
        self.threads = {}  # tag -> name
        self.samples = []  # (tag, qpc, frames innermost first)
        self.complete = False


def parse(data):
    if data[:8] != b"NX2SMPL1":
        raise ValueError("not an EDEN_NX2_SAMPLE_PROFILE file")
    result = SampleFile()
    _version, result.hz, result.qpc_frequency = struct.unpack_from("<IIQ", data, 8)
    offset = 24
    while offset < len(data):
        kind = data[offset]
        offset += 1
        try:
            if kind == KIND_SAMPLE:
                tag, depth, _ = struct.unpack_from("<BBB", data, offset)
                (qpc,) = struct.unpack_from("<Q", data, offset + 3)
                frames = list(struct.unpack_from("<%dQ" % depth, data, offset + 11))
                offset += 11 + 8 * depth
                result.samples.append((tag, qpc, frames))
            elif kind == KIND_THREAD:
                tag, length = struct.unpack_from("<BH", data, offset)
                result.threads[tag] = data[offset + 3:offset + 3 + length].decode("utf-8", "replace")
                offset += 3 + length
            elif kind == KIND_MODULE:
                base, size, length = struct.unpack_from("<QIH", data, offset)
                path = data[offset + 14:offset + 14 + length].decode("utf-8", "replace")
                offset += 14 + length
                result.modules.append((base, size, path))
            elif kind == KIND_END:
                result.complete = True
                break
            else:
                raise ValueError("unknown record kind %#x at %#x" % (kind, offset - 1))
        except struct.error:
            break  # cut short (the process ended while sampling)
    return result


def find_module(modules, address):
    for base, size, path in modules:
        if base <= address < base + size:
            return base, path
    return None, None


def is_exe(path, exe_name):
    return os.path.basename(path.replace("\\", "/")).lower() == exe_name.lower()


class Symbolizer:
    """Maps lichii.exe RVAs to function names (innermost inline frame first)."""

    def __init__(self, exe, cache_path, symbolizer="llvm-symbolizer"):
        self.exe = exe
        self.cache_path = cache_path
        self.symbolizer = symbolizer
        self.cache = {}
        stamp = self._stamp()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as file:
                cached = json.load(file)
            if cached.get("stamp") == stamp:
                self.cache = {int(key): value for key, value in cached["symbols"].items()}
        self.stamp = stamp

    def _stamp(self):
        if not self.exe or not os.path.exists(self.exe):
            return None
        info = os.stat(self.exe)
        return [os.path.abspath(self.exe), info.st_size, int(info.st_mtime)]

    def resolve(self, rvas):
        missing = sorted({rva for rva in rvas if rva not in self.cache})
        if missing and self.exe:
            for start in range(0, len(missing), 4096):
                batch = missing[start:start + 4096]
                output = subprocess.run(
                    [self.symbolizer, "--obj=" + self.exe, "--relative-address", "--inlines",
                     "--output-style=JSON"],
                    input="\n".join("%#x" % rva for rva in batch), capture_output=True, text=True,
                    check=True).stdout
                lines = [line for line in output.splitlines() if line.strip()]
                for rva, line in zip(batch, lines):
                    symbols = json.loads(line).get("Symbol", [])
                    names = [symbol.get("FunctionName") or "??" for symbol in symbols]
                    self.cache[rva] = names or ["??"]
            if self.cache_path:
                with open(self.cache_path, "w") as file:
                    json.dump({"stamp": self.stamp, "symbols": self.cache}, file)
        return {rva: self.cache.get(rva, ["lichii.exe+%#x" % rva]) for rva in rvas}


def symbolize_samples(sample_file, exe_name, symbolizer):
    """Returns (thread name, stack outermost first) per sample, inline frames expanded."""
    keys = []
    rvas = set()
    for tag, _qpc, frames in sample_file.samples:
        sample_keys = []
        for depth, address in enumerate(frames):
            # Past the innermost frame these are return addresses: attribute the call.
            lookup = address if depth == 0 else address - 1
            base, path = find_module(sample_file.modules, lookup)
            if base is None:
                sample_keys.append(("label", "[jit]"))
            elif is_exe(path, exe_name):
                sample_keys.append(("rva", lookup - base))
                rvas.add(lookup - base)
            else:
                name = os.path.basename(path.replace("\\", "/"))
                sample_keys.append(("label", "%s+%#x" % (name, lookup - base)))
        keys.append((tag, sample_keys))
    names = symbolizer.resolve(rvas) if rvas else {}
    stacks = []
    for tag, sample_keys in keys:
        stack = []
        for kind, value in sample_keys:
            stack.extend([value] if kind == "label" else names[value])
        stack.reverse()
        thread = sample_file.threads.get(tag, "thread %d" % tag)
        stacks.append((thread, stack))
    return stacks


def summarize(stacks):
    """Per thread: total, self counts, inclusive counts and folded paths."""
    threads = collections.OrderedDict()
    for thread, stack in stacks:
        entry = threads.setdefault(thread, {
            "total": 0, "self": collections.Counter(), "inclusive": collections.Counter(),
            "paths": collections.Counter()})
        entry["total"] += 1
        if stack:
            entry["self"][stack[-1]] += 1
        for function in set(stack):
            entry["inclusive"][function] += 1
        entry["paths"][";".join(stack)] += 1
    return threads


def write_folded(threads, path):
    with open(path, "w") as file:
        for thread, entry in threads.items():
            for stack, count in entry["paths"].most_common():
                file.write("%s;%s %d\n" % (thread, stack, count))


def flamegraph_svg(threads, width=1600, row=16):
    """A plain flame graph: one root per thread, width proportional to samples."""
    root = {"name": "all", "count": 0, "children": collections.OrderedDict()}
    for thread, entry in threads.items():
        for stack, count in entry["paths"].items():
            node = root
            node["count"] += count
            for name in [thread] + (stack.split(";") if stack else []):
                node = node["children"].setdefault(
                    name, {"name": name, "count": 0, "children": collections.OrderedDict()})
                node["count"] += count
    rects = []
    depth_max = [0]

    def emit(node, x, depth):
        depth_max[0] = max(depth_max[0], depth)
        w = width * node["count"] / max(root["count"], 1)
        if w >= 0.5:
            hue = 30 + (hash(node["name"]) % 40)
            rects.append((x, depth, w, node["name"], node["count"], hue))
        child_x = x
        for child in node["children"].values():
            emit(child, child_x, depth + 1)
            child_x += width * child["count"] / max(root["count"], 1)

    emit(root, 0.0, 0)
    height = (depth_max[0] + 1) * row
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace" font-size="11">'
             % (width, height)]
    for x, depth, w, name, count, hue in rects:
        y = height - (depth + 1) * row
        label = html.escape(name)
        percent = 100.0 * count / max(root["count"], 1)
        parts.append('<g><title>%s (%d samples, %.2f%%)</title><rect x="%.1f" y="%d" width="%.1f" height="%d" '
                     'fill="hsl(%d,80%%,60%%)" stroke="white" stroke-width="0.5"/>'
                     % (label, count, percent, x, y, w, row - 1, hue))
        if w > 40:
            text = label[:int(w / 7)]
            parts.append('<text x="%.1f" y="%d">%s</text>' % (x + 2, y + row - 4, text))
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def report(threads, hz, top=25, paths=30, out=sys.stdout):
    for thread, entry in threads.items():
        total = entry["total"]
        seconds = total / hz if hz else 0
        out.write("== %s: %d samples (%.1f s at %d Hz)\n" % (thread, total, seconds, hz))
        out.write("-- self\n")
        for function, count in entry["self"].most_common(top):
            out.write("%6.2f%% %7d  %s\n" % (100.0 * count / total, count, function))
        out.write("-- inclusive\n")
        for function, count in entry["inclusive"].most_common(top):
            out.write("%6.2f%% %7d  %s\n" % (100.0 * count / total, count, function))
        out.write("-- top call paths (outermost first)\n")
        for stack, count in entry["paths"].most_common(paths):
            out.write("%6.2f%% %7d  %s\n" % (100.0 * count / total, count, stack.replace(";", " > ")))
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples")
    parser.add_argument("--exe", default="lichii.exe", help="lichii.exe of the build (lichii.pdb next to it)")
    parser.add_argument("--symbolizer", default="llvm-symbolizer")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--paths", type=int, default=30)
    parser.add_argument("--folded", help="write folded stacks (flamegraph.pl input)")
    parser.add_argument("--svg", help="write a flame graph")
    args = parser.parse_args()

    with open(args.samples, "rb") as file:
        sample_file = parse(file.read())
    if not sample_file.complete:
        print("note: the file ends without its end record (cut short)", file=sys.stderr)
    symbolizer = Symbolizer(args.exe if os.path.exists(args.exe) else None,
                            args.samples + ".symcache.json", args.symbolizer)
    stacks = symbolize_samples(sample_file, os.path.basename(args.exe), symbolizer)
    threads = summarize(stacks)
    report(threads, sample_file.hz, args.top, args.paths)
    if args.folded:
        write_folded(threads, args.folded)
    if args.svg:
        with open(args.svg, "w") as file:
            file.write(flamegraph_svg(threads))


if __name__ == "__main__":
    main()
