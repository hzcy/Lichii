#!/usr/bin/env python3
"""Replay native shader captures and v46-v53 pipeline caches without a GPU."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


STAGES = ("vertex", "tess-control", "tess-eval", "geometry", "fragment", "compute",
          "vertex-a", "task", "mesh")
GAMES = {
    "040074a01bf12000": "Donkey Kong Bananza",
    "040077201be50000": "Kirby Air Riders",
    "04007c6028aa8000": "Yoshi demo",
    "04007ef01f8ae000": "Fire Emblem: Fortune's Weave",
    "0400c3f00006e000": "Mario Kart World",
    "04008e001f098000": "Nintendo Switch Sports Resort",
    "0400a3a01ee4c000": "PokÃ©mon Pokopia",
    "0400a4802437e000": "Drag x Drive Global Jam",
    "040041e0212ae000": "Fast Fusion",
    "04005b902517c000": "Kirby Air Riders Global Test Ride",
    "040057c02159a000": "Nintendo Switch 2 Welcome Tour",
    "040004f027560000": "Orbitals Global Friends Pass",
    "0400380020056000": "Raiders",
    "0400d1d02399a000": "The Duskbloods network test",
    "0400d4201c6ea000": "Mario Tennis Fever",
    "0400e88025b1a000": "Star Fox",
}
MAP_MAGIC = 0x50414D4E4544454E
GRAPHICS_KEY_SIZE = 1056


class CacheError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_cache(data):
    """Yield shader environments with their original pipeline keys."""
    if len(data) < 12 or data[:8] != b"yuzucach":
        raise CacheError("not a serialized shader-environment cache")
    version, = struct.unpack_from("<I", data, 8)
    if version not in (46, 47, 48, 49, 50, 51, 52, 53):
        raise CacheError(f"unsupported cache version {version}")
    cursor = 12
    entry = 0
    while cursor < len(data):
        if cursor + 4 > len(data):
            raise CacheError(f"truncated pipeline count at {cursor}")
        count, = struct.unpack_from("<I", data, cursor)
        cursor += 4
        if not 1 <= count <= 6:
            raise CacheError(f"invalid environment count {count} at {cursor - 4}")
        environments = []
        for index in range(count):
            begin = cursor
            header = struct.Struct("<" + "Q" * (6 if version >= 47 else 5) + "7I")
            if cursor + header.size > len(data):
                raise CacheError(f"truncated environment header at {cursor}")
            fields = header.unpack_from(data, cursor)
            sizes, params = fields[:-7], fields[-7:]
            code_size, textures, formats, constants = sizes[:4]
            pointers, replacements = sizes[4:] if version >= 47 else (0, sizes[4])
            local, bound, start, low, high, viewport, stage = params
            if stage >= len(STAGES) or code_size > 64 * 1024 * 1024 or code_size % 8:
                raise CacheError(f"invalid stage/code size at {cursor}")
            if any(n > 1_000_000 for n in sizes[1:]):
                raise CacheError(f"excessive context count at {cursor}")
            cursor += header.size
            code = data[cursor:cursor + code_size]
            cursor += code_size + textures * 8 + formats * 8 + constants * 12
            cursor += pointers * 9 + replacements * 12
            if stage in (7, 8) and version < 53:
                raise CacheError("task/mesh environments require cache version 53")
            cursor += 16 if stage == 5 else 80 + (32 if stage == 3 else 0)
            if stage in (7, 8):
                cursor += 92  # Three mesh registers and the primitive program header.
            if cursor > len(data):
                raise CacheError(f"truncated environment at {begin}")
            environments.append({"version": version, "entry": entry, "index": index,
                                 "stage": STAGES[stage], "offset": begin,
                                 "bytes": data[begin:cursor], "code": code})
        compute = environments[0]["stage"] == "compute"
        if compute and count != 1:
            raise CacheError(f"compute pipeline has {count} environments")
        key_size = 24 if compute else GRAPHICS_KEY_SIZE
        if cursor + key_size > len(data):
            raise CacheError(f"truncated pipeline key at {cursor}")
        key = data[cursor:cursor + key_size]
        cursor += key_size
        for env in environments:
            env["key"] = key
            yield env
        entry += 1


def unpack_native_code(code):
    if len(code) < 96 or len(code) % 8:
        return None
    words = struct.unpack("<" + "Q" * (len(code) // 8), code)
    if words[-1] != MAP_MAGIC:
        return None
    count = words[-2]
    if not 0 < count <= 65536 or count + 2 >= len(words):
        raise CacheError("invalid native address map")
    end = len(words) - count - 3
    return b"".join(struct.pack("<Q", words[i]) for i in range(10, end) if i % 4)


def invoke(command, timeout):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "phase": "timeout", "error": {"message": "compiler timeout"}}
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"status": "error", "phase": "process", "error": {
            "message": f"compiler exited {proc.returncode}: {proc.stderr[-2000:]}"}}
    result["returncode"] = proc.returncode
    if proc.returncode and result.get("status") == "ok":
        result.update(status="error", phase="process", error={"message": "nonzero compiler exit"})
    return result


def failure(phase, message):
    return {"status": "error", "phase": phase, "error": {"message": message}}


def summarize(results):
    counts = Counter()
    for row in results:
        message = row.get("error", {}).get("message", "ok")
        message = re.sub(r"%\d+", "%id", message)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", message)
        counts[(row["game"], row.get("stage") or "unknown", row["phase"],
                row.get("opcode", ""), message)] += 1
    lines = ["| Game | Stage | Phase | Opcode | Cause | Count |", "|---|---|---|---|---|---:|"]
    for (game, stage, phase, opcode, cause), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        cause = cause.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {game} | {stage} | {phase} | {opcode} | {cause} | {count} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dump-game", default="Mario Tennis Fever")
    parser.add_argument("--native-stage", action="append", default=[], metavar="KEY=STAGE",
                        help="stage established by a capture log for a native filename hash")
    parser.add_argument("--contexts-from", type=Path,
                        help="prior results.json preserving native/cache matches across lowering changes")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    if not args.compiler.is_file():
        parser.error("--compiler must name the built shader-sweep executable")
    for path in args.inputs:
        if not path.exists():
            parser.error(f"input does not exist: {path}")
    native_stages = {}
    for mapping in args.native_stage:
        key, separator, stage = mapping.partition("=")
        if not separator or not re.fullmatch(r"[0-9a-f]{16}", key) or stage not in STAGES[:6]:
            parser.error("--native-stage requires a 16-digit lowercase hash and a graphics/compute stage")
        native_stages[key] = stage
    compiler = str(args.compiler.resolve())
    validator = shutil.which(args.spirv_val)
    if not validator:
        parser.error("spirv-val is required, install SPIRV-Tools or pass --spirv-val")
    args.output.mkdir(parents=True, exist_ok=True)
    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)
    files = sorted({f.resolve() for p in args.inputs for f in
                    (p.rglob("*.bin") if p.is_dir() else [p])})
    rows, jobs, cache_index = [], {}, defaultdict(list)
    prior_contexts = defaultdict(list)
    if args.contexts_from:
        for row in json.loads(args.contexts_from.read_text()):
            if row.get("kind") == "native" and "env" in row:
                prior_contexts[row["native_id"]].append(row)
    for path in files:
        if not re.fullmatch(r"vulkan(?:\.fork-v\d+)?\.bin", path.name):
            continue
        game = GAMES.get(path.parent.name.lower(), path.parent.name)
        try:
            envs = list(parse_cache(path.read_bytes()))
            for env in envs:
                identity = digest(game.encode() + env["bytes"] + env["key"] + bytes([env["version"]]))
                if identity not in jobs:
                    env_path = records_dir / f"{identity}.env"
                    key_path = records_dir / f"{identity}.key"
                    env_path.write_bytes(env["bytes"])
                    key_path.write_bytes(env["key"])
                    jobs[identity] = {"id": identity, "game": game, "stage": env["stage"],
                                      "source": str(path), "offset": env["offset"],
                                      "kind": "cache", "env": str(env_path),
                                      "key": str(key_path), "version": env["version"]}
                lowered = unpack_native_code(env["code"])
                if lowered is not None:
                    match = cache_index[digest(lowered)]
                    if identity not in match:
                        match.append(identity)
        except (CacheError, OSError) as exc:
            rows.append({"game": game, "source": str(path), **failure("input", str(exc))})
    header_stages = {}
    for path in files:
        rejected = re.fullmatch(r"rejected-stage-([1-5])-[0-9a-f]+\.bin", path.name)
        if rejected:
            header_stages[digest(path.read_bytes()[128:])] = STAGES[int(rejected[1]) - 1]
    raw_seen = set()
    for path in files:
        match = re.fullmatch(r"(?:[0-9a-f]+-(gfx|cs)-[0-9a-f]+|rejected-stage-(\d+)-[0-9a-f]+)\.bin", path.name)
        if not match:
            continue
        source = path.read_bytes()
        identity = digest(source)
        if identity in raw_seen:
            continue
        raw_seen.add(identity)
        rejected_stage = int(match[2]) if match[2] is not None else None
        stage = STAGES[rejected_stage - 1] if rejected_stage is not None and 1 <= rejected_stage <= 5 else None
        if match[1] == "cs":
            stage = "compute"
        if stage is None:
            stage = header_stages.get(identity)
        explicit_stage = native_stages.get(path.name.split("-")[0])
        if explicit_stage:
            if stage is not None and explicit_stage != stage:
                parser.error(f"explicit stage conflicts with capture metadata: {path}")
            stage = explicit_stage
        fmt = "native-header" if rejected_stage is not None else "instructions"
        lower_path = records_dir / f"{identity}.lowered"
        cmd = [compiler, str(path), "--format", fmt, "--lower-only", str(lower_path)]
        if stage:
            cmd += ["--stage", stage]
        result = invoke(cmd, args.timeout)
        if (stage is None and result["status"] != "ok" and
                "outside geometry stage" in result.get("error", {}).get("message", "")):
            stage = "geometry"
            result = invoke(cmd + ["--stage", stage], args.timeout)
        base = {"game": args.dump_game, "id": identity, "source": str(path),
                "native_id": identity, "stage": stage or "unknown", "kind": "native"}
        if result["status"] != "ok":
            offset = result.get("error", {}).get("offset")
            if result.get("phase") == "lower" and isinstance(offset, int):
                offset += 128 if fmt == "native-header" else 0
                if 0 <= offset <= len(source) - 16:
                    words = struct.unpack_from("<4I", source, offset)
                    base["opcode"] = f"0x{words[0] & 0xfff:03x}"
                    base["instruction"] = [f"{word:08x}" for word in words]
            rows.append({**base, **result})
            continue
        matches = [jobs[target] for target in cache_index.get(digest(lower_path.read_bytes()), [])
                   if jobs[target]["game"] == args.dump_game]
        if not matches:
            matches = prior_contexts.get(identity, [])
        if matches:
            compatible = False
            for cached in matches:
                if stage is not None and cached["stage"] != stage:
                    continue
                compatible = True
                new_id = digest((identity + cached["env"] + cached["key"]).encode())
                jobs[new_id] = {**cached, **base, "id": new_id, "stage": cached["stage"], "format": fmt}
            if not compatible:
                rows.append({**base, **failure("context", "cached stage disagrees with captured stage")})
        elif stage is not None:
            jobs[identity] = {**base, "format": fmt}
        else:
            rows.append({**base, **failure("context", "instruction-only dump has no matching cached stage/header")})

    def compile_one(job):
        output = records_dir / f"{job['id']}.spv"
        command = [compiler, "--output", str(output)]
        if job["kind"] == "native":
            command += [job["source"], "--stage", job["stage"], "--format", job["format"]]
        if "env" in job:
            command += ["--environment", job["env"], "--cache-version", str(job["version"])]
            if job["stage"] != "compute":
                command += ["--pipeline-key", job["key"]]
        result = invoke(command, args.timeout)
        if result["status"] == "ok":
            try:
                validation = subprocess.run([validator, "--target-env", "vulkan1.2", str(output)],
                                            capture_output=True, text=True, timeout=args.timeout)
                if validation.returncode:
                    result.update(failure("validate", validation.stderr.strip() or validation.stdout.strip()))
                else:
                    result["phase"] = "validated"
            except subprocess.TimeoutExpired:
                result.update(failure("timeout", "spirv-val timeout"))
        return {**job, **result, "command": command}

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for i, row in enumerate(executor.map(compile_one, jobs.values()), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"compiled {i}/{len(jobs)}", file=sys.stderr)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = summarize(rows)
    (args.output / "summary.md").write_text(report)
    print(report)
    print(f"{len(files)} input files; {len(raw_seen)} unique native captures; {len(jobs)} compile jobs")
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
