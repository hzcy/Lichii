# SPDX-License-Identifier: GPL-3.0-or-later
"""Read SARC and NVN2 BNSH shader archives.

BNSH layout reference: MIT-licensed ShaderLibrary, BnshLoader.cs and Structs.cs:
https://github.com/KillzXGaming/ShaderLibrary/tree/790dd2e501926e536a9193907d1256d446104162/ShaderLibrary
"""

import struct


class ArchiveError(ValueError):
    pass


def _range(data, offset, size):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ArchiveError(f"out-of-range data at {offset:#x}, size {size:#x}")
    return data[offset:offset + size]


def parse_sarc(data: bytes) -> list[tuple[str, bytes]]:
    _range(data, 0, 0x20)
    if data[:4] != b"SARC":
        raise ArchiveError("not a SARC archive")
    if data[6:8] == b"\xff\xfe":
        endian = "<"
    elif data[6:8] == b"\xfe\xff":
        endian = ">"
    else:
        raise ArchiveError("invalid SARC byte order")
    u16 = lambda off: struct.unpack(endian + "H", _range(data, off, 2))[0]
    u32 = lambda off: struct.unpack(endian + "I", _range(data, off, 4))[0]
    header_size, file_size, data_offset = u16(4), u32(8), u32(12)
    if header_size < 0x14 or file_size != len(data) or data_offset > file_size:
        raise ArchiveError("invalid SARC header or file size")
    sfat = header_size
    if _range(data, sfat, 4) != b"SFAT" or u16(sfat + 4) < 0x0c:
        raise ArchiveError("invalid SFAT header")
    count = u16(sfat + 6)
    nodes = sfat + u16(sfat + 4)
    _range(data, nodes, count * 0x10)
    sfnt = nodes + count * 0x10
    if _range(data, sfnt, 4) != b"SFNT" or u16(sfnt + 4) < 8:
        raise ArchiveError("invalid SFNT header")
    names = sfnt + u16(sfnt + 4)
    if names > data_offset:
        raise ArchiveError("SARC names overlap file data")
    result = []
    for i in range(count):
        node = nodes + i * 0x10
        flags, start, end = u32(node + 4), u32(node + 8), u32(node + 12)
        if flags >> 24 != 1:
            raise ArchiveError("SARC member has no supported name")
        name_offset = names + (flags & 0xffffff) * 4
        if name_offset >= data_offset:
            raise ArchiveError("SARC name outside string table")
        terminator = data.find(b"\0", name_offset, data_offset)
        if terminator < 0:
            raise ArchiveError("unterminated SARC name")
        try:
            name = data[name_offset:terminator].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArchiveError("invalid SARC name") from exc
        if end < start or end > file_size - data_offset:
            raise ArchiveError("invalid SARC member range")
        result.append((name, _range(data, data_offset + start, end - start)))
    return result


def parse_bnsh(data: bytes) -> list[dict]:
    """Read bounded NVN2 allocations and validate stage metadata."""
    _range(data, 0, 0x98)
    if data[:8] != b"BNSH\0\0\0\0" or data[0x60:0x64] != b"grsc":
        raise ArchiveError("invalid BNSH header")
    if data[0xc:0xe] != b"\xff\xfe":
        raise ArchiveError("unsupported BNSH byte order")
    u16 = lambda off: struct.unpack("<H", _range(data, off, 2))[0]
    u32 = lambda off: struct.unpack("<I", _range(data, off, 4))[0]
    u64 = lambda off: struct.unpack("<Q", _range(data, off, 8))[0]
    if u32(0x1c) != len(data):
        raise ArchiveError("invalid BNSH file size")
    version = tuple(data[8:12])
    if version not in ((0, 4, 2, 0), (1, 3, 2, 0), (1, 4, 2, 0)):
        raise ArchiveError(f"unsupported BNSH version {version}")
    api_type, api_version = u16(0x70), u16(0x72)
    if (api_type, api_version) != (4, 200):
        raise ArchiveError(f"unsupported BNSH API {api_type}/{api_version}")
    count, variations = u32(0x7c), u64(0x80)
    if count and not variations:
        raise ArchiveError("missing BNSH variation table")
    _range(data, variations, count * 0x40)
    stages = ("vertex", "tess-control", "tess-eval", "geometry", "fragment", "compute")
    result = []
    for variation in range(count):
        program = u64(variations + variation * 0x40 + 0x10)
        if not program:
            raise ArchiveError("missing BNSH variation program")
        _range(data, program, 0x38)
        slots = list(zip(range(8, 0x38, 8), stages))
        if version == (1, 4, 2, 0) and u64(program) == 2:
            _range(data, program, 0x50)
            slots.extend(((0x40, "task"), (0x48, "mesh")))
        for index, (slot, stage) in enumerate(slots):
            record = u64(program + slot)
            if not record:
                continue
            _range(data, record, 0x40)
            control_offset, code_offset = u64(record + 8), u64(record + 16)
            code_size, control_size = u32(record + 24), u32(record + 28)
            if not code_offset or not code_size or (control_size and not control_offset):
                raise ArchiveError("invalid BNSH stage data pointer or size")
            code = _range(data, code_offset, code_size)
            control = _range(data, control_offset, control_size)
            if stage != "compute":
                if len(code) <= 128 or (len(code) - 128) % 16:
                    raise ArchiveError("invalid native graphics code size")
                header_type = (struct.unpack_from("<I", code)[0] >> 10) & 15
                if stage in ("task", "mesh"):
                    expected = 6 if stage == "task" else 7
                    if (len(control) < 88 or control[:8] != bytes.fromhex("34127698c9000000")
                            or u32(control_offset + 84) >> 16 != expected
                            or header_type not in ((1,) if stage == "task" else (1, 3))):
                        raise ArchiveError("invalid BNSH task/mesh stage metadata")
                elif header_type != index + 1:
                    raise ArchiveError("BNSH graphics stage disagrees with native header")
            result.append(dict(variation=variation, stage=stage, code=code, control=control,
                               code_offset=code_offset, control_offset=control_offset,
                               api_type=api_type, api_version=api_version, version=version))
    return result


def find_bnsh(data: bytes) -> list[tuple[int, bytes]]:
    """Find bounded embedded BNSHs in FSHA, returning (FSHA offset, bytes)."""
    if len(data) < 0x20 or data[:4] != b"FSHA":
        raise ArchiveError("not a BFSHA archive")
    if data[0xc:0xe] != b"\xff\xfe" or struct.unpack_from("<I", data, 0x1c)[0] != len(data):
        raise ArchiveError("invalid BFSHA byte order or size")
    return find_embedded_bnsh(data)


def find_embedded_bnsh(data: bytes) -> list[tuple[int, bytes]]:
    """Return validated BNSH allocations embedded in a known container."""
    result = []
    offset = 0
    while True:
        offset = data.find(b"BNSH\0\0\0\0", offset)
        if offset < 0:
            break
        header = _range(data, offset, 0x20)
        size = struct.unpack_from("<I", header, 0x1c)[0]
        if size < 0x98:
            raise ArchiveError("invalid embedded BNSH size")
        bnsh = _range(data, offset, size)
        parse_bnsh(bnsh)
        result.append((offset, bnsh))
        offset += size
    if not result:
        raise ArchiveError("container contains no supported BNSH")
    return result
