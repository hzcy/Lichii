#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract verified NVN2 shader allocations without running games or accessing keys."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from shader_archive import ArchiveError, find_bnsh, find_embedded_bnsh, parse_bnsh, parse_sarc
from shader_containers import MAX_ALLOCATION, dcx_decompress, parse_bnd4, parse_cgsb, parse_sharcb, parse_shn, read_zstd_dictionaries, zstd_decompress


def digest(data):
    return hashlib.sha256(data).hexdigest()


def walk_container(data, name, depth=0, dictionaries=None):
    """Yield shader records and member errors from a supported container."""
    if depth > 12:
        raise ArchiveError("excessive container nesting")
    magic = data[:4]
    if magic == bytes.fromhex("28b52ffd"):
        yield from walk_container(zstd_decompress(data, dictionaries), name + "!zstd", depth + 1, dictionaries)
    elif magic == b"DCX\0":
        yield from walk_container(dcx_decompress(data), name + "!dcx", depth + 1, dictionaries)
    elif magic in (b"SARC", b"BND4"):
        members = parse_sarc(data) if magic == b"SARC" else parse_bnd4(data)
        for member, payload in members:
            try:
                yield from walk_container(payload, name + "!" + member, depth + 1, dictionaries)
            except (ArchiveError, ValueError) as exc:
                yield {"member": name + "!" + member, "error": str(exc)}
    elif magic == b"FSHA":
        for offset, bnsh in find_bnsh(data):
            yield from walk_container(bnsh, name + f"!BNSH@{offset:x}", depth + 1, dictionaries)
    elif magic == b"YB\x05\x00" and name.lower().endswith(
            (".vsetb.byml", ".vsetb.byml.zs!zstd")):
        for offset, bnsh in find_embedded_bnsh(data):
            yield from walk_container(bnsh, name + f"!BNSH@{offset:x}", depth + 1, dictionaries)
    elif magic in (b"BNSH", b"CGSB", b"SHN!", b"BAHS"):
        parser = {b"BNSH": parse_bnsh, b"CGSB": parse_cgsb, b"SHN!": parse_shn, b"BAHS": parse_sharcb}[magic]
        for record in parser(data):
            yield {"member": name, **record}
    elif name.lower().endswith((".json", ".presetb", ".meminfo", ".tex", ".byml", ".yaml", ".raw")):
        return
    else:
        raise ArchiveError(f"unsupported container magic {magic.hex()}")


def extract(sources, output, dictionaries=None):
    """Extract unique code and control allocations and return their provenance."""
    programs = {}
    archives = []
    errors = []
    for source in sources:
        path = Path(source["path"])
        game = source["game"]
        title_id = source["title_id"]
        print(f"Reading {game}: {path.name}", file=sys.stderr, flush=True)
        try:
            before = path.stat()
            if before.st_size > MAX_ALLOCATION:
                raise ArchiveError("archive exceeds in-memory limit; needs selective reader")
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ArchiveError("source changed during read")
            archive = {**source, "size": len(data), "sha256": digest(data), "records": 0}
            for record in walk_container(data, path.name, dictionaries=dictionaries):
                if "error" in record:
                    errors.append({**source, **record})
                    continue
                code = record.pop("code")
                control = record.pop("control")
                code_hash, control_hash = digest(code), digest(control)
                identity = (title_id, record["stage"], code_hash, control_hash)
                if identity not in programs:
                    directory = output / "programs" / title_id
                    directory.mkdir(parents=True, exist_ok=True)
                    controls = output / "controls"
                    controls.mkdir(exist_ok=True)
                    code_path = directory / f"{code_hash}-{record['stage']}.bin"
                    control_path = controls / f"{control_hash}.bin"
                    code_path.write_bytes(code)
                    control_path.write_bytes(control)
                    programs[identity] = {
                        "game": game, "title_id": title_id, "stage": record["stage"],
                        "code": code_path.relative_to(output).as_posix(),
                        "control": control_path.relative_to(output).as_posix(),
                        "code_sha256": code_hash, "control_sha256": control_hash,
                        "format": "native-header" if "envelope" not in record and record["stage"] in
                                  ("vertex", "tess-control", "tess-eval", "geometry", "fragment", "task", "mesh") else None,
                        "context": "archive header only; runtime resources unavailable",
                        "origins": [],
                    }
                programs[identity]["origins"].append({"source": str(path), **record})
                archive["records"] += 1
            archives.append(archive)
        except (OSError, ArchiveError, ValueError) as exc:
            errors.append({**source, "error": str(exc)})
    return {"schema_version": 1, "archives": archives, "programs": list(programs.values()), "errors": errors}


def main():
    """Validate local extraction paths and write the shader manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="TITLE_ID=PATH")
    parser.add_argument("--read-only-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zstd-dictionary-pack", type=Path,
                        help="ZsDic.pack.zs or its decompressed SARC from the same dump")
    args = parser.parse_args()
    library = args.read_only_root.resolve()
    output = args.output.resolve()
    if output == library or library in output.parents:
        parser.error("output must be outside the read-only library")
    if output.exists():
        parser.error("output must be a new scratch directory")
    sources = []
    coverage_errors = []
    if args.inventory:
        for game in json.loads(args.inventory.read_text(encoding="utf-8")):
            coverage_errors.extend({"game": game["name"], "phase": "inventory", **error}
                                   for error in game["errors"])
            ids = {p["title_id"] for p in game["programs"]}
            if len(ids) != 1:
                coverage_errors.append({"game": game["name"], "path": game["directory"],
                                        "phase": "coverage", "title_ids": sorted(ids),
                                        "error": "no unique loose-program title ID; select package sources separately"})
                continue
            title_id = next(iter(ids))
            candidates = [candidate for candidate in game["candidates"]
                          if Path(candidate["path"]).suffix.lower() not in
                          (".tex", ".json", ".presetb", ".meminfo")]
            if not candidates:
                coverage_errors.append({"game": game["name"], "path": game["directory"],
                                        "phase": "coverage", "title_id": title_id,
                                        "error": "no supported loose shader candidates; asset container needs another reader"})
            for candidate in candidates:
                sources.append({"game": game["name"], "title_id": title_id, "path": candidate["path"]})
    for specification in args.source:
        title_id, separator, path = specification.partition("=")
        if not separator or not re.fullmatch("[0-9a-fA-F]{16}", title_id):
            parser.error("--source requires TITLE_ID=PATH")
        sources.append({"game": title_id.upper(), "title_id": title_id.upper(), "path": path})
    if not sources and not coverage_errors:
        parser.error("no shader archive sources selected")
    for source in sources:
        path = Path(source["path"]).resolve()
        if path != library and library not in path.parents:
            parser.error(f"source is outside the declared read-only root: {path}")
        if not re.fullmatch("[0-9a-fA-F]{16}", source["title_id"]):
            parser.error("invalid inventory title id")
    dictionaries = None
    if args.zstd_dictionary_pack:
        dictionary_pack = args.zstd_dictionary_pack.resolve()
        if library not in dictionary_pack.parents:
            parser.error("dictionary pack must be inside the declared read-only root")
        try:
            if dictionary_pack.stat().st_size > MAX_ALLOCATION:
                raise ArchiveError("dictionary pack exceeds the in-memory limit")
            dictionaries = read_zstd_dictionaries(dictionary_pack.read_bytes())
        except (ArchiveError, OSError, ValueError) as exc:
            parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    result = extract(sources, output, dictionaries)
    result["errors"].extend(coverage_errors)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archives": len(result["archives"]), "programs": len(result["programs"]),
                      "stages": dict(Counter(p["stage"] for p in result["programs"])),
                      "errors": len(result["errors"])}))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    sys.exit(main())
