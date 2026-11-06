# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded readers for shader containers observed in the owner's loose RomFS dumps.

BND4 reference: SoulsFormatsNEXT, BinderFileHeader.ReadBinder4FileHeader (MIT).
SHN/SHDROUNC and CGSB extents were checked against captured NVN2 files.
"""

import ctypes
import ctypes.util
import struct
import zlib

from shader_archive import ArchiveError, parse_sarc


MAX_ALLOCATION = 512 * 1024 * 1024
STAGES = ("vertex", "tess-control", "tess-eval", "geometry", "fragment")


def extent(data, offset, size):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ArchiveError(f"extent {offset:#x}+{size:#x} outside {len(data):#x} bytes")
    return data[offset:offset + size]


def zstd_module():
    """Find a Python Zstandard module with dictionary support."""
    try:
        from compression import zstd
    except ImportError:
        try:
            from backports import zstd
        except ImportError:
            return None
    return zstd


def read_zstd_dictionaries(data):
    """Index the game dictionary pack by Zstandard dictionary ID."""
    zstd = zstd_module()
    if zstd is None:
        raise ArchiveError("dictionary packs require Python 3.14 or backports.zstd")
    archive = zstd_decompress(data) if data.startswith(b"\x28\xb5\x2f\xfd") else data
    dictionaries = {}
    for name, member in parse_sarc(archive):
        if not member.startswith(b"\x37\xa4\x30\xec"):
            continue
        dictionary = zstd.ZstdDict(member)
        if dictionary.dict_id in dictionaries:
            raise ArchiveError(f"duplicate zstd dictionary ID {dictionary.dict_id} in {name}")
        dictionaries[dictionary.dict_id] = dictionary
    if not dictionaries:
        raise ArchiveError("no zstd dictionaries found in the pack")
    return dictionaries


def zstd_decompress(data, dictionaries=None):
    """Decode a bounded frame using its declared dictionary when needed."""
    zstd = zstd_module()
    if zstd is None:
        library = ctypes.CDLL(ctypes.util.find_library("zstd"))
        library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        size = library.ZSTD_getFrameContentSize(data, len(data))
        if size > MAX_ALLOCATION:
            raise ArchiveError("unknown or excessive zstd decompressed size")
        output = ctypes.create_string_buffer(size)
        library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        library.ZSTD_decompress.restype = ctypes.c_size_t
        actual = library.ZSTD_decompress(output, size, data, len(data))
        if actual != size:
            raise ArchiveError("zstd decompression failed; a dictionary may be required")
        return output.raw[:actual]
    try:
        info = zstd.get_frame_info(data)
        size = info.decompressed_size
        if size is None or size > MAX_ALLOCATION:
            raise ArchiveError("unknown or excessive zstd decompressed size")
        dictionary = (dictionaries or {}).get(info.dictionary_id)
        if info.dictionary_id and dictionary is None:
            raise ArchiveError(f"zstd dictionary {info.dictionary_id} is required")
        decoder = zstd.ZstdDecompressor(zstd_dict=dictionary)
        result = decoder.decompress(data, max_length=size + 1)
    except zstd.ZstdError as exc:
        raise ArchiveError(f"zstd decompression failed: {exc}") from exc
    if not decoder.eof or decoder.unused_data or len(result) != size:
        raise ArchiveError("zstd decompressed size mismatch or trailing input")
    return result


def lz4_block(data, expected):
    if expected > MAX_ALLOCATION:
        raise ArchiveError("excessive LZ4 allocation")
    output = bytearray()
    cursor = 0

    def length(initial):
        nonlocal cursor
        value = initial
        if initial == 15:
            while True:
                if cursor >= len(data):
                    raise ArchiveError("truncated LZ4 length")
                extra = data[cursor]
                cursor += 1
                value += extra
                if extra != 255:
                    break
        return value

    while cursor < len(data):
        token = data[cursor]
        cursor += 1
        literals = length(token >> 4)
        if len(output) + literals > expected:
            raise ArchiveError("LZ4 literals exceed allocation")
        output.extend(extent(data, cursor, literals))
        cursor += literals
        if cursor == len(data):
            break
        distance, = struct.unpack("<H", extent(data, cursor, 2))
        cursor += 2
        count = length(token & 15) + 4
        if not distance or distance > len(output) or len(output) + count > expected:
            raise ArchiveError("invalid LZ4 match")
        while count:
            size = min(distance, count)
            output.extend(output[-distance:][:size])
            count -= size
    if len(output) != expected:
        raise ArchiveError("LZ4 decompressed size mismatch")
    return bytes(output)


def dcx_decompress(data):
    extent(data, 0, 0x30)
    if data[:4] != b"DCX\0" or data[0x18:0x1c] != b"DCS\0" or data[0x24:0x28] != b"DCP\0":
        raise ArchiveError("unsupported DCX header")
    offset, = struct.unpack_from(">I", data, 0x14)
    size, compressed = struct.unpack_from(">II", data, 0x1c)
    if size > MAX_ALLOCATION:
        raise ArchiveError("excessive DCX allocation")
    payload = extent(data, offset, compressed)
    if data[0x28:0x2c] == b"ZSTD":
        result = zstd_decompress(payload)
    elif data[0x28:0x2c] == b"DFLT":
        decoder = zlib.decompressobj()
        result = decoder.decompress(payload, size + 1)
        if not decoder.eof or decoder.unused_data:
            raise ArchiveError("incomplete or trailing DCX zlib stream")
    else:
        raise ArchiveError(f"unsupported DCX compressor {data[0x28:0x2c]!r}")
    if len(result) != size:
        raise ArchiveError("DCX decompressed size mismatch")
    return result


def parse_bnd4(data):
    extent(data, 0, 64)
    if data[:4] != b"BND4" or data[9:12] != b"\0\1\0" or data[48:52] != b"\1t\0\0":
        raise ArchiveError("unsupported BND4 byte order or file-record format")
    count, = struct.unpack_from("<I", data, 12)
    header, record_size = struct.unpack_from("<Q8xQ", data, 16)
    if header != 64 or record_size != 36 or count > 100000:
        raise ArchiveError("unsupported BND4 record size or count")
    extent(data, 64, count * record_size)
    result = []
    for index in range(count):
        record = 64 + index * record_size
        if data[record:record + 8] not in (b"\x40\0\0\0\xff\xff\xff\xff", b"\xc0\0\0\0\xff\xff\xff\xff"):
            raise ArchiveError("unsupported BND4 member flags")
        size, unpacked, offset, ident, name = struct.unpack_from("<QQIII", data, record + 8)
        if size > MAX_ALLOCATION or unpacked > MAX_ALLOCATION or name % 2:
            raise ArchiveError("invalid BND4 member size or name")
        end = name
        while extent(data, end, 2) != b"\0\0":
            end += 2
            if end - name > 65536:
                raise ArchiveError("excessive BND4 filename")
        filename = data[name:end].decode("utf-16-le")
        payload = extent(data, offset, size)
        if data[record] == 0xc0:
            payload = dcx_decompress(payload)
        if len(payload) != unpacked:
            raise ArchiveError("BND4 member size mismatch")
        result.append((filename, payload))
    return result


def graphics_record(code, control, **metadata):
    if len(code) <= 128 or (len(code) - 128) % 16:
        raise ArchiveError("unsupported native graphics allocation")
    stage = (struct.unpack_from("<I", code)[0] >> 10) & 15
    if not 1 <= stage <= 5:
        raise ArchiveError("native envelope is not a verified graphics header")
    if len(control) < 8 or control[:8] != bytes.fromhex("34127698c9000000"):
        raise ArchiveError("unsupported native shader control version")
    return {"stage": STAGES[stage - 1], "code": code, "control": control, **metadata}


def parse_cgsb(data):
    extent(data, 0, 24)
    if data[:8] != b"CGSB\x33\x22\x11\0":
        raise ArchiveError("unsupported CGSB version")
    count, shader_stage = struct.unpack_from("<HH", data, 8)
    table, = struct.unpack_from("<Q", data, 16)
    if not 1 <= count <= 16:
        raise ArchiveError("invalid CGSB record count")
    if shader_stage > 7:
        raise ArchiveError(f"unsupported CGSB stage {shader_stage}")
    extent(data, table, count * 80)
    result = []
    for index in range(count):
        record = table + index * 80
        if data[record:record + 8] != b"NVN \1\0\0\0":
            raise ArchiveError("unsupported CGSB shader platform")
        code_offset, code_size, control_offset, control_size = struct.unpack_from("<4Q", data, record + 16)
        if shader_stage >= 5:
            stage = ("compute", "task", "mesh")[shader_stage - 5]
            result.append({"stage": stage, "code": extent(data, code_offset, code_size),
                           "control": extent(data, control_offset, control_size),
                           "variation": index, "envelope": f"unverified-{stage}",
                           "code_offset": code_offset, "control_offset": control_offset})
        else:
            item = graphics_record(extent(data, code_offset, code_size),
                                   extent(data, control_offset, control_size),
                                   variation=index, code_offset=code_offset, control_offset=control_offset)
            if item["stage"] != STAGES[shader_stage]:
                raise ArchiveError("CGSB stage disagrees with native header")
            result.append(item)
    return result


def parse_shn(data):
    extent(data, 0, 68)
    if data[:4] != b"SHN!":
        raise ArchiveError("not a SHN shader container")
    blocks = []
    for descriptor in (4, 36):
        offset, compressed, size = struct.unpack_from("<III", data, descriptor)
        blocks.append(lz4_block(extent(data, offset, compressed), size))
    metadata, code = blocks
    if metadata[:8] != b"SHDROUNC" or len(metadata) < 64:
        raise ArchiveError("unsupported SHN metadata format")
    version, variations = struct.unpack_from("<II", metadata, 8)
    if version != 20241128 or not 1 <= variations <= 65536:
        raise ArchiveError("unsupported SHDROUNC version or variation count")
    extent(metadata, 16, variations * 80)
    result = []
    for variation in range(variations):
        table, = struct.unpack_from("<Q", metadata, 48 + variation * 80)
        extent(metadata, table, 9 * 80)
        for index in range(9):
            record = table + index * 80
            code_offset, control_offset = struct.unpack_from("<QQ", metadata, record)
            code_size, control_size = struct.unpack_from("<II", metadata, record + 64)
            if not code_size and not control_size:
                continue
            payload = extent(code, code_offset, code_size)
            control = extent(metadata, control_offset, control_size)
            origin = {"variation": variation, "slot": index, "code_offset": code_offset,
                      "control_offset": control_offset}
            if index == 2:
                result.append({"stage": "compute", "code": payload, "control": control,
                               "envelope": "unverified-compute", **origin})
            elif index in (0, 1):
                item = graphics_record(payload, control, **origin)
                if item["stage"] != ("vertex", "fragment")[index]:
                    raise ArchiveError("SHDROUNC stage disagrees with native header")
                result.append(item)
            else:
                result.append({"stage": f"unknown-shdrounc-slot-{index}", "code": payload,
                               "control": control, "envelope": "unverified-shdrounc-slot", **origin})
    if not result:
        raise ArchiveError("SHDROUNC contains no supported shader records")
    return result


def parse_sharcb(data):
    """Read the NVN2 AGL BAHS v11 variant with its separate metadata and code pools."""
    extent(data, 0, 32)
    version, size, platform, flags, metadata_base, pool_size, pool = struct.unpack_from("<7I", data, 4)
    if (data[:4], version, platform, flags) != (b"BAHS", 11, 5, 0) or size != len(data):
        raise ArchiveError("unsupported BAHS version, platform or file size")
    table = metadata_base + 0x150
    if metadata_base < 32 or table + 8 > pool or pool + pool_size != size:
        raise ArchiveError("invalid BAHS metadata or code pool")
    table_size, count = struct.unpack_from("<II", data, table)
    end = table + table_size
    if not count or table_size < 8 or end > pool or count > (table_size - 8) // 32:
        raise ArchiveError("invalid BAHS record table")
    stages = {0: ("vertex", 0), 1: ("fragment", 1), 3: ("compute", 5),
              4: ("task", 6), 5: ("mesh", 7)}
    records = []
    allocations = {}
    cursor = table + 8
    for index in range(count):
        if cursor + 32 > end:
            raise ArchiveError("truncated BAHS record header")
        length, slot, _, body_size = struct.unpack_from("<4I", data, cursor)
        descriptor = cursor + 16
        code_relative, control_size, control_relative = struct.unpack_from("<QIi", data, descriptor)
        if length < 32 or body_size != length - 16 or cursor + length > end:
            raise ArchiveError("invalid BAHS record size")
        if slot not in stages:
            raise ArchiveError(f"unsupported BAHS shader slot {slot}")
        # A shared control can be referenced backwards from a later descriptor.
        control_offset = descriptor + control_relative
        if control_size < 104 or control_offset < table + 8 or control_offset + control_size > end:
            raise ArchiveError("invalid BAHS control extent")
        control = data[control_offset:control_offset + control_size]
        stage, expected_type = stages[slot]
        if (control[:8] != bytes.fromhex("34127698c9000000") or
                struct.unpack_from("<I", control, 84)[0] >> 16 != expected_type):
            raise ArchiveError("BAHS record disagrees with the NVN2 control stage or version")
        declared, _, instruction_allocation, allocation = struct.unpack_from("<4I", control, 56)
        if (not declared or declared % 16 or instruction_allocation % 16 or
                not declared <= instruction_allocation <= allocation or
                code_relative + allocation > pool_size):
            raise ArchiveError("invalid BAHS code extent")
        code_offset = pool + code_relative
        code_size = declared if stage == "compute" else instruction_allocation
        location = (code_offset, code_size)
        if location not in allocations:
            allocations[location] = data[code_offset:code_offset + code_size]
        code = allocations[location]
        origin = {"variation": index, "slot": slot, "record_offset": cursor,
                  "code_offset": code_offset, "control_offset": control_offset}
        if stage in ("task", "mesh"):
            header_type = (struct.unpack_from("<I", extent(code, 0, 4))[0] >> 10) & 15
            if (len(code) <= 128 or (len(code) - 128) % 16 or
                    header_type not in ((1,) if stage == "task" else (1, 3))):
                raise ArchiveError("invalid BAHS task/mesh graphics envelope")
            item = {"stage": stage, "code": code, "control": control, **origin}
        elif stage == "compute":
            item = {"stage": stage, "code": code, "control": control, **origin}
        else:
            item = graphics_record(code, control, **origin)
            if item["stage"] != stage:
                raise ArchiveError("BAHS record disagrees with the graphics header stage")
        records.append(item)
        cursor += length
    if cursor != end:
        raise ArchiveError("trailing bytes in BAHS record table")
    return records
