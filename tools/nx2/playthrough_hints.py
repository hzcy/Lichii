#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export code-free playthrough hints and reconstruct a local Vulkan cache from game archives."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import re
import struct
import tempfile

from shader_sweep import GRAPHICS_KEY_SIZE, STAGES, digest, invoke, parse_cache
from sweep_static_shaders import artifact_path, compute_format, output_path

SCHEMA = "lichii-playthrough-hints-1"
MAGIC = b"yuzucach"
SUPPORTED_STAGES = {"vertex", "fragment", "geometry", "tess-control", "tess-eval", "compute", "task", "mesh"}


ENVIRONMENT_HEADER_SIZE = struct.calcsize("<6Q7I")
GRAPHICS_HEADER_SIZE = 80


def pack_archive_shader(program, directory, compiler, scratch, header_hashes):
    """Lower a verified local allocation into packed shader code."""
    stage = program["stage"]
    code_hash = program["code_sha256"]
    source = artifact_path(directory, program["code"])
    code = source.read_bytes()
    if digest(code) != code_hash:
        raise ValueError(f"archive shader changed: {source}")
    if stage == "compute":
        control = artifact_path(directory, program["control"]).read_bytes()
        if digest(control) != program["control_sha256"] or not compute_format(code, control):
            return None
        size, = struct.unpack_from("<I", control, 56)
        source = scratch / f"{code_hash}.instructions"
        source.write_bytes(code[:size])
        input_format = "instructions"
    else:
        if (stage, digest(code[:GRAPHICS_HEADER_SIZE])) not in header_hashes:
            return None
        input_format = "native-header"
    destination = scratch / f"{stage}-{code_hash}.packed"
    result = invoke([str(compiler), str(source), "--stage", stage, "--capture-extent",
                     "--format", input_format, "--packed-code", str(destination)], 60)
    if result["status"] == "ok":
        return stage, destination.read_bytes()
    return None


def export_hints(cache, title, game_version):
    """Replace shader instructions with fingerprints while preserving pipeline metadata."""
    records = []
    for _, group in groupby(parse_cache(cache), key=lambda env: env["entry"]):
        envs = list(group)
        stages = []
        for env in envs:
            code = env["code"]
            context = env["bytes"]
            stages.append({
                "stage": env["stage"],
                "shader_sha256": digest(code),
                "header_sha256": digest(code[:GRAPHICS_HEADER_SIZE]),
                "prefix": context[:ENVIRONMENT_HEADER_SIZE].hex(),
                "context": context[ENVIRONMENT_HEADER_SIZE + len(code):].hex(),
            })
        records.append({"key": envs[0]["key"].hex(), "stages": stages})
    return {"schema": SCHEMA, "title_id": title, "game_version": game_version,
            "cache_version": struct.unpack_from("<I", cache, 8)[0], "pipelines": records}


def restore_pipeline(record, version, programs):
    """Rebuild one cache record if every shader matches, or return None."""
    stages = record["stages"]
    if not 1 <= len(stages) <= 6:
        raise ValueError("invalid stage count")
    key = bytes.fromhex(record["key"])
    compute = stages[0]["stage"] == "compute"
    if len(key) != (24 if compute else GRAPHICS_KEY_SIZE) or (compute and len(stages) != 1):
        raise ValueError("invalid pipeline key")
    parts = [struct.pack("<I", len(stages))]
    for stage in stages:
        name = stage["stage"]
        code = programs.get((name, stage["shader_sha256"]))
        if code is None:
            return None
        prefix = bytes.fromhex(stage["prefix"])
        if (len(prefix) != ENVIRONMENT_HEADER_SIZE or
                struct.unpack_from("<Q", prefix)[0] != len(code) or
                struct.unpack_from("<I", prefix, len(prefix) - 4)[0] != STAGES.index(name)):
            raise ValueError("invalid shader environment header")
        parts.extend((prefix, code, bytes.fromhex(stage["context"])))
    parts.append(key)
    payload = b"".join(parts)
    parsed = list(parse_cache(MAGIC + struct.pack("<I", version) + payload))
    if (len(parsed) != len(stages) or any(e["entry"] != 0 or e["key"] != key or
            digest(e["code"]) != s["shader_sha256"] for e, s in zip(parsed, stages))):
        raise ValueError("invalid reconstructed pipeline layout")
    return key, payload


def import_hints(args, hint):
    """Rebuild complete pipelines and report why other records were skipped."""
    records = hint["pipelines"]
    version = hint["cache_version"]
    headers = set()
    wanted = set()
    valid = []
    invalid = 0
    unsupported = 0
    for record in records:
        try:
            stages = record["stages"]
            if not isinstance(stages, list) or not 1 <= len(stages) <= 6:
                raise ValueError("invalid stage count")
            names = [stage["stage"] for stage in stages]
            if any(name not in STAGES for name in names):
                raise ValueError("invalid shader stage")
            fingerprints = [stage[field] for stage in stages
                            for field in ("header_sha256", "shader_sha256")]
            if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in fingerprints):
                raise ValueError("invalid shader fingerprint")
            bytes.fromhex(record["key"])
            for stage in stages:
                bytes.fromhex(stage["prefix"])
                bytes.fromhex(stage["context"])
        except (TypeError, KeyError, ValueError):
            invalid += 1
            continue
        if any(name not in SUPPORTED_STAGES for name in names):
            unsupported += 1
            continue
        valid.append(record)
        headers.update((s["stage"], s["header_sha256"]) for s in stages)
        wanted.update((s["stage"], s["shader_sha256"]) for s in stages)
    candidates = {}
    for path in args.manifest:
        for program in json.loads(path.read_bytes())["programs"]:
            stage = program["stage"]
            if (program["title_id"].lower() == args.title and stage in SUPPORTED_STAGES and
                    (program["format"] == "native-header" or stage == "compute")):
                candidates[stage, program["code_sha256"]] = program, path.parent

    programs = {}
    with tempfile.TemporaryDirectory(dir=args.output.parent) as folder:
        scratch = Path(folder)
        with ThreadPoolExecutor(max_workers=args.jobs) as workers:
            results = workers.map(lambda item: pack_archive_shader(
                *item, args.compiler.resolve(), scratch, headers),
                candidates.values())
            for result in results:
                if result:
                    stage, code = result
                    identity = stage, digest(code)
                    if identity in wanted:
                        programs[identity] = code

    restored = {}
    missing = 0
    for record in valid:
        try:
            result = restore_pipeline(record, version, programs)
        except ValueError:  # Includes CacheError from the serialized cache parser.
            invalid += 1
            continue
        if result is None:
            missing += 1
        else:
            key, payload = result
            restored[key] = payload
    args.output.mkdir()
    if restored:
        directory = args.output / args.title
        directory.mkdir()
        (directory / f"vulkan.fork-v{version}.bin").write_bytes(
            MAGIC + struct.pack("<I", version) + b"".join(restored.values()))
    return {"pipelines_in_hints": len(records), "restored": len(restored),
            "missing_local_shader_match": missing, "unsupported_stage": unsupported,
            "invalid_record": invalid}


def main():
    """Run the hint export or import command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "import"):
        command = commands.add_parser(name)
        command.add_argument("--title", required=True, type=str.lower)
        command.add_argument("--game-version", required=True)
        command.add_argument("--output", required=True, type=Path)
    commands.choices["export"].add_argument("--cache", required=True, type=Path)
    importer = commands.choices["import"]
    importer.add_argument("--hints", required=True, type=Path)
    importer.add_argument("--manifest", required=True, action="append", type=Path)
    importer.add_argument("--compiler", required=True, type=Path)
    importer.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{16}", args.title):
        parser.error("expected a 16-digit hexadecimal title ID")
    if args.command == "export":
        data = args.cache.read_bytes()
        if data[:12] not in (MAGIC + struct.pack("<I", 52), MAGIC + struct.pack("<I", 53)):
            parser.error("expected a v52 or v53 Vulkan cache")
        if args.cache.parent.name.lower() != args.title:
            parser.error("cache directory does not match the title ID")
        hint = export_hints(data, args.title, args.game_version)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(hint, stream, separators=(",", ":"))
            stream.write("\n")
        print(f"Exported {len(hint['pipelines'])} pipeline hints.")
        return 0
    if args.jobs < 1 or not args.compiler.is_file():
        parser.error("positive jobs and a valid offline compiler are required")
    args.output = output_path(args.output, [p.parent for p in args.manifest])
    if args.hints.stat().st_size > 128 * 1024 * 1024:
        parser.error("hint file exceeds 128 MiB")
    hint = json.loads(args.hints.read_bytes())
    if (not isinstance(hint, dict) or not isinstance(hint.get("pipelines"), list) or
            hint.get("schema") != SCHEMA or hint.get("cache_version") not in (52, 53) or
            hint.get("title_id") != args.title or hint.get("game_version") != args.game_version):
        parser.error("hint format, title, or game version does not match")
    summary = import_hints(args, hint)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
