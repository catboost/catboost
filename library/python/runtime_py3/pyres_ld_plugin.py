"""
LD plugin (see LD_PLUGIN macro in build/ymake.core.conf) assembling the static resfs table
for python programs.

  1. tools/pyres_pack (one node per group of PY_SRCS files, see _PY3_RESFS_PACK) compiles .py files,
     compresses sources/bytecode/metadata with the library/cpp/resource codec and writes a *pack*
     `pyres_*.pyres`: a flat list of (key, value) records. Packs are global outputs of the modules:
     they are never passed to the linker, but every link of a program using them gets them as
     plugin inputs (`--start-plugin-inputs ... --end-plugin-inputs`, see build/conf/linkers).

  2. The link scripts run this plugin (`--start-plugins ... --end-plugins`) in front of every link
     that has library/python/runtime_py3 among its inputs. Records of all the packs are
     merged (conflicting values for one key fail the link), hashed with CityHash64 (the hash of
     library/cpp/resource TStore::ToK) into an open-addressing table. The table is written as an
     object of the target format (ELF, Mach-O or COFF, told by --target= or /MACHINE:) with one
     read-only data section defining `__pyres_static_blob`, and the object is put in front of
     all the inputs of the link. library/cpp/resource/registry.cpp references the symbol and
     consults the table after its own dynamic registry; without the plugin the reference resolves
     to the empty table of library/cpp/resource/static_blob_default.cpp. See
     library/cpp/resource/static_hash.h for the layout.

Pack layout (little-endian, 8-byte aligned):
    char[8] magic "YPYPACK1"; u64 size; u32 count; u32 flags;
    records: u32 klen, u32 vlen, key, value, zero padding to 8 bytes

Blob layout:
    header (64 bytes)
        0  char[8]  magic "YPYRES01"
        8  u64      count
       16  u64      nbuckets      power of two, >= 2 * count
       24  u64      buckets_off   nbuckets x (u64 hash, u64 key_off); key_off == 0 -> empty
       32  u64      order_off     count x u64 key_off, sorted by key bytes
       40  u64      keys_off      key records: u32 klen, u32 vlen, u64 val_off, key bytes (8-aligned)
       48  u64      total_size
       56  u64      values_off    raw values, referenced from key records by val_off

    Keys and values live in separate areas on purpose: enumerating the keys (which the python
    importer does at startup) must not touch the pages holding the compressed sources.

This module is self-contained on purpose: link_exe.py runs a copy of it from the build directory.
"""

import json
import os
import struct
import sys

PACK_MAGIC = b"YPYPACK1"
BLOB_MAGIC = b"YPYRES01"
BLOB_SYMBOL = "__pyres_static_blob"
PACK_SUFFIX = ".pyres"
OUT_OBJ_SUFFIX = {"elf": ".pyres_static.o", "macho": ".pyres_static.o", "coff": ".pyres_static.obj"}

SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHF_ALLOC = 0x2

MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------------------------
# CityHash64 (util/digest/city.cpp, CityHash v1) - must match NHashPrivate::ComputeStringHash
# --------------------------------------------------------------------------------------------

_K0 = 0xC3A5C85C97CB3127
_K1 = 0xB492B66FBE98F273
_K2 = 0x9AE16A3B2F90404F
_K3 = 0xC949D7C7509E6557
_KMUL = 0x9DDFEA08EB382D69


def _ld64(s, i):
    return struct.unpack_from("<Q", s, i)[0]


def _ld32(s, i):
    return struct.unpack_from("<I", s, i)[0]


def _rot(v, n):
    if n == 0:
        return v
    return ((v >> n) | (v << (64 - n))) & MASK64


def _shift_mix(v):
    return v ^ (v >> 47)


def _hash_len16(lo, hi):
    a = ((lo ^ hi) * _KMUL) & MASK64
    a ^= a >> 47
    b = ((hi ^ a) * _KMUL) & MASK64
    b ^= b >> 47
    return (b * _KMUL) & MASK64


def _hash_len0to16(s, n):
    if n > 8:
        a = _ld64(s, 0)
        b = _ld64(s, n - 8)
        return _hash_len16(a, _rot((b + n) & MASK64, n)) ^ b
    if n >= 4:
        a = _ld32(s, 0)
        return _hash_len16((n + (a << 3)) & MASK64, _ld32(s, n - 4))
    if n > 0:
        a, b, c = s[0], s[n >> 1], s[n - 1]
        y = (a + (b << 8)) & 0xFFFFFFFF
        z = (n + (c << 2)) & 0xFFFFFFFF
        return (_shift_mix(((y * _K2) ^ (z * _K3)) & MASK64) * _K2) & MASK64
    return _K2


def _hash_len17to32(s, n):
    a = (_ld64(s, 0) * _K1) & MASK64
    b = _ld64(s, 8)
    c = (_ld64(s, n - 8) * _K2) & MASK64
    d = (_ld64(s, n - 16) * _K0) & MASK64
    return _hash_len16(
        (_rot((a - b) & MASK64, 43) + _rot(c, 30) + d) & MASK64,
        (a + _rot(b ^ _K3, 20) - c + n) & MASK64,
    )


def _weak_hash_len32_with_seeds_raw(w, x, y, z, a, b):
    a = (a + w) & MASK64
    b = _rot((b + a + z) & MASK64, 21)
    c = a
    a = (a + x + y) & MASK64
    b = (b + _rot(a, 44)) & MASK64
    return (a + z) & MASK64, (b + c) & MASK64


def _weak_hash_len32_with_seeds(s, i, a, b):
    return _weak_hash_len32_with_seeds_raw(_ld64(s, i), _ld64(s, i + 8), _ld64(s, i + 16), _ld64(s, i + 24), a, b)


def _hash_len33to64(s, n):
    z = _ld64(s, 24)
    a = (_ld64(s, 0) + ((n + _ld64(s, n - 16)) & MASK64) * _K0) & MASK64
    b = _rot((a + z) & MASK64, 52)
    c = _rot(a, 37)
    a = (a + _ld64(s, 8)) & MASK64
    c = (c + _rot(a, 7)) & MASK64
    a = (a + _ld64(s, 16)) & MASK64
    vf = (a + z) & MASK64
    vs = (b + _rot(a, 31) + c) & MASK64
    a = (_ld64(s, 16) + _ld64(s, n - 32)) & MASK64
    z = _ld64(s, n - 8)
    b = _rot((a + z) & MASK64, 52)
    c = _rot(a, 37)
    a = (a + _ld64(s, n - 24)) & MASK64
    c = (c + _rot(a, 7)) & MASK64
    a = (a + _ld64(s, n - 16)) & MASK64
    wf = (a + z) & MASK64
    ws = (b + _rot(a, 31) + c) & MASK64
    r = _shift_mix(((vf + ws) * _K2 + (wf + vs) * _K0) & MASK64)
    return (_shift_mix((r * _K0 + vs) & MASK64) * _K2) & MASK64


def cityhash64(s):
    n = len(s)
    if n <= 32:
        if n <= 16:
            return _hash_len0to16(s, n)
        return _hash_len17to32(s, n)
    if n <= 64:
        return _hash_len33to64(s, n)

    x = _ld64(s, 0)
    y = _ld64(s, n - 16) ^ _K1
    z = _ld64(s, n - 56) ^ _K0
    v = _weak_hash_len32_with_seeds(s, n - 64, n, y)
    w = _weak_hash_len32_with_seeds(s, n - 32, (n * _K1) & MASK64, _K0)
    z = (z + _shift_mix(v[1]) * _K1) & MASK64
    x = (_rot((z + x) & MASK64, 39) * _K1) & MASK64
    y = (_rot(y, 33) * _K1) & MASK64

    n = (n - 1) & ~63
    i = 0
    while True:
        x = (_rot((x + y + v[0] + _ld64(s, i + 16)) & MASK64, 37) * _K1) & MASK64
        y = (_rot((y + v[1] + _ld64(s, i + 48)) & MASK64, 42) * _K1) & MASK64
        x ^= w[1]
        y ^= v[0]
        z = _rot(z ^ w[0], 33)
        v = _weak_hash_len32_with_seeds(s, i, (v[1] * _K1) & MASK64, (x + w[0]) & MASK64)
        w = _weak_hash_len32_with_seeds(s, i + 32, (z + w[1]) & MASK64, y)
        z, x = x, z
        i += 64
        n -= 64
        if n == 0:
            break
    return _hash_len16(
        (_hash_len16(v[0], w[0]) + _shift_mix(y) * _K1 + z) & MASK64,
        (_hash_len16(v[1], w[1]) + x) & MASK64,
    )


# --------------------------------------------------------------------------------------------
# pack reader
# --------------------------------------------------------------------------------------------


def unpack_records(data):
    if data[:8] != PACK_MAGIC:
        raise ValueError("bad pack magic")
    size, count, _flags = struct.unpack_from("<QII", data, 8)
    if size != len(data):
        raise ValueError("pack size mismatch: header says {}, section is {}".format(size, len(data)))
    pos = 24
    for _ in range(count):
        klen, vlen = struct.unpack_from("<II", data, pos)
        pos += 8
        key = bytes(data[pos : pos + klen])
        pos += klen
        value = bytes(data[pos : pos + vlen])
        pos += vlen
        pos = (pos + 7) & ~7
        yield key, value
    if pos != size:
        raise ValueError("truncated pack")


# --------------------------------------------------------------------------------------------
# blob: static hash table
# --------------------------------------------------------------------------------------------


def _align8(n):
    return (n + 7) & ~7


def build_blob(records):
    """records: dict key(bytes) -> value(bytes). Returns (blob bytes, max probe length)."""
    keys = sorted(records)
    count = len(keys)
    nbuckets = 1
    while nbuckets < 2 * count:
        nbuckets *= 2

    header_size = 64
    buckets_off = header_size
    order_off = buckets_off + 16 * nbuckets
    keys_off = _align8(order_off + 8 * count)

    # key records first, values after them
    key_offsets = []
    pos = keys_off
    for key in keys:
        key_offsets.append(pos)
        pos += _align8(16 + len(key))
    values_off = pos
    val_offsets = []
    for key in keys:
        val_offsets.append(pos)
        pos += len(records[key])
    total_size = _align8(pos)

    buckets = [(0, 0)] * nbuckets
    mask = nbuckets - 1
    max_probe = 0
    for key, off in zip(keys, key_offsets):
        h = cityhash64(key)
        i = h & mask
        probe = 0
        while buckets[i][1] != 0:
            i = (i + 1) & mask
            probe += 1
        buckets[i] = (h, off)
        max_probe = max(max_probe, probe)

    out = [
        BLOB_MAGIC,
        struct.pack("<QQQQQQQ", count, nbuckets, buckets_off, order_off, keys_off, total_size, values_off),
    ]
    out.append(b"".join(struct.pack("<QQ", h, off) for h, off in buckets))
    out.append(b"".join(struct.pack("<Q", off) for off in key_offsets))
    out.append(b"\0" * (keys_off - (order_off + 8 * count)))
    for key, voff in zip(keys, val_offsets):
        rec = struct.pack("<IIQ", len(key), len(records[key]), voff) + key
        out.append(rec + b"\0" * (_align8(len(rec)) - len(rec)))
    for key in keys:
        out.append(records[key])
    out.append(b"\0" * (total_size - pos))
    blob = b"".join(out)
    assert len(blob) == total_size
    return blob, max_probe


# --------------------------------------------------------------------------------------------
# minimal relocatable objects: one read-only data section defining one global symbol
# --------------------------------------------------------------------------------------------


def _align(n, a):
    return (n + a - 1) // a * a


_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_SHDR = struct.Struct("<IIQQQQIIQQ")
_SYM = struct.Struct("<IBBHQQ")
_ELF_MACHINE = {"x86_64": 62, "aarch64": 183}


def elf_object(arch, data, symbol):
    """ELF64 LE: .rodata.pyres + .symtab + .strtab + .shstrtab."""
    shstr = b"\0.rodata.pyres\0.symtab\0.strtab\0.shstrtab\0"
    strtab = b"\0" + symbol.encode() + b"\0"
    symtab = b"".join(
        [
            _SYM.pack(0, 0, 0, 0, 0, 0),
            _SYM.pack(0, 3, 0, 1, 0, 0),  # STB_LOCAL | STT_SECTION
            _SYM.pack(1, (1 << 4) | 1, 0, 1, 0, len(data)),  # STB_GLOBAL | STT_OBJECT
        ]
    )
    data_off = _align(_EHDR.size, 16)
    symtab_off = _align(data_off + len(data), 8)
    strtab_off = symtab_off + len(symtab)
    shstr_off = strtab_off + len(strtab)
    sh_off = _align(shstr_off + len(shstr), 8)
    shdrs = [
        _SHDR.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        _SHDR.pack(shstr.index(b".rodata.pyres"), SHT_PROGBITS, SHF_ALLOC, 0, data_off, len(data), 0, 0, 16, 0),
        _SHDR.pack(shstr.index(b".symtab"), SHT_SYMTAB, 0, 0, symtab_off, len(symtab), 3, 2, 8, _SYM.size),
        _SHDR.pack(shstr.index(b".strtab"), SHT_STRTAB, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0),
        _SHDR.pack(shstr.index(b".shstrtab"), SHT_STRTAB, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0),
    ]
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    ehdr = _EHDR.pack(ident, 1, _ELF_MACHINE[arch], 1, 0, 0, sh_off, 0, _EHDR.size, 0, 0, _SHDR.size, len(shdrs), 4)
    return [
        (0, ehdr),
        (data_off, data),
        (symtab_off, symtab),
        (strtab_off, strtab),
        (shstr_off, shstr),
        (sh_off, b"".join(shdrs)),
    ]


_MACHO_CPU = {"x86_64": (0x01000007, 3), "aarch64": (0x0100000C, 0)}  # (cputype, cpusubtype ALL)


def macho_object(arch, data, symbol):
    """Mach-O 64 MH_OBJECT: LC_SEGMENT_64 with __TEXT,__pyres + LC_SYMTAB."""
    cputype, cpusubtype = _MACHO_CPU[arch]
    seg_size = 72 + 80  # segment_command_64 + one section_64
    cmds_size = seg_size + 24  # + symtab_command
    data_off = _align(32 + cmds_size, 16)
    sym_off = _align(data_off + len(data), 8)
    strtab = b"\0" + symbol.encode() + b"\0"
    strtab += b"\0" * (_align(len(strtab), 8) - len(strtab))
    str_off = sym_off + 16
    cmds = b"".join(
        [
            # MH_MAGIC_64, MH_OBJECT, 2 commands, MH_SUBSECTIONS_VIA_SYMBOLS
            struct.pack("<IiiIIIII", 0xFEEDFACF, cputype, cpusubtype, 1, 2, cmds_size, 0x2000, 0),
            # LC_SEGMENT_64 (unnamed in objects), rwx protections as compilers emit them
            struct.pack("<II16sQQQQiiII", 0x19, seg_size, b"", 0, len(data), data_off, len(data), 7, 7, 1, 0),
            # section_64: 2^4 alignment, S_REGULAR
            struct.pack("<16s16sQQIIIIIIII", b"__pyres", b"__TEXT", 0, len(data), data_off, 4, 0, 0, 0, 0, 0, 0),
            # LC_SYMTAB
            struct.pack("<IIIIII", 0x2, 24, sym_off, 1, str_off, len(strtab)),
        ]
    )
    nlist = struct.pack("<IBBHQ", 1, 0x0F, 1, 0, 0)  # N_SECT | N_EXT in section 1
    return [(0, cmds), (data_off, data), (sym_off, nlist), (str_off, strtab)]


_COFF_MACHINE = {"x86_64": 0x8664, "aarch64": 0xAA64, "x86": 0x14C}


def coff_object(arch, data, symbol):
    """COFF: one .rdata section, @feat.00 and the symbol."""
    strtab = b""

    def name8(name):
        nonlocal strtab
        name = name.encode()
        if len(name) <= 8:
            return name.ljust(8, b"\0")
        off = 4 + len(strtab)
        strtab += name + b"\0"
        return struct.pack("<II", 0, off)

    data_off = _align(20 + 40, 16)
    sym_off = data_off + len(data)
    syms = b"".join(
        [
            # @feat.00 = 1 (safe SEH: there is no code here), IMAGE_SYM_ABSOLUTE, IMAGE_SYM_CLASS_STATIC
            struct.pack("<8sIhHBB", name8("@feat.00"), 1, -1, 0, 3, 0),
            # the symbol at the start of section 1, IMAGE_SYM_CLASS_EXTERNAL
            struct.pack("<8sIhHBB", name8(symbol), 0, 1, 0, 2, 0),
        ]
    )
    # IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_ALIGN_16BYTES | IMAGE_SCN_MEM_READ
    flags = 0x40 | 0x00500000 | 0x40000000
    hdrs = struct.pack("<HHIIIHH", _COFF_MACHINE[arch], 1, 0, sym_off, 2, 0, 0) + struct.pack(
        "<8sIIIIIIHHI", b".rdata", 0, 0, len(data), data_off, 0, 0, 0, 0, flags
    )
    return [(0, hdrs), (data_off, data), (sym_off, syms + struct.pack("<I", 4 + len(strtab)) + strtab)]


def write_object(path, fmt, arch, data, symbol):
    """symbol is the C name; platform decoration is added here."""
    if fmt == "macho":
        chunks = macho_object(arch, data, "_" + symbol)
    elif fmt == "coff":
        chunks = coff_object(arch, data, ("_" + symbol) if arch == "x86" else symbol)
    else:
        chunks = elf_object(arch, data, symbol)
    with open(path, "wb") as f:
        for off, chunk in chunks:
            f.write(b"\0" * (off - f.tell()))
            f.write(chunk)


# --------------------------------------------------------------------------------------------
# link command line
# --------------------------------------------------------------------------------------------


def iter_link_args(args):
    """Arguments with ya command files (@file, one argument per line) expanded."""
    for a in args:
        if a.startswith("@") and os.path.isfile(a[1:]):
            with open(a[1:]) as f:
                for line in f.read().splitlines():
                    if line:
                        yield line
        else:
            yield a


def iter_plugin_inputs(args):
    if "--start-plugin-inputs" in args:
        inputs = args[args.index("--start-plugin-inputs") + 1 : args.index("--end-plugin-inputs")]
        for a in iter_link_args(inputs):
            if a not in ("--ya-start-command-file", "--ya-end-command-file"):
                yield a


def collect_packs(args, root):
    """Returns [(path, pack bytes)] for the packs among the plugin inputs."""
    packs = []
    for a in iter_plugin_inputs(args):
        if a.endswith(PACK_SUFFIX):
            with open(a if os.path.isabs(a) else os.path.join(root, a), "rb") as f:
                packs.append((a, f.read()))
    return packs


def _norm_arch(arch):
    arch = arch.lower()
    if arch in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if arch in ("aarch64", "arm64", "arm64e"):
        return "aarch64"
    if arch in ("x86", "i386", "i686"):
        return "x86"
    return arch


def link_target(args):
    """(object format, arch): from the clang --target= triple, or /MACHINE: of link.exe."""
    for a in iter_link_args(args):
        if a.startswith("--target="):
            triple = a[len("--target=") :]
            fmt = "macho" if "-apple-" in triple else "coff" if "-windows" in triple else "elf"
            return fmt, _norm_arch(triple.split("-")[0])
        if a[:9].upper() in ("/MACHINE:", "-MACHINE:"):
            return "coff", _norm_arch(a[9:])
    return None


def output_position(args, fmt):
    """(index to insert the table object at: in front of every input, path of the linked target)."""
    if fmt == "coff":
        for i, a in enumerate(args):
            if a[:5].upper() in ("/OUT:", "-OUT:"):
                return i, a[5:]
    elif "-o" in args:
        i = args.index("-o")
        return i, args[i + 1]
    return None, None


# --------------------------------------------------------------------------------------------
# plugin entry point
# --------------------------------------------------------------------------------------------


def merge_packs(packs):
    records = {}
    origin = {}
    for src, sec in packs:
        for key, value in unpack_records(sec):
            prev = records.get(key)
            if prev is None:
                records[key] = value
                origin[key] = src
            elif prev != value:
                raise SystemExit(
                    "pyres: redefinition of resource {!r}: {} vs {}".format(
                        key.decode(errors="replace"), origin[key], src
                    )
                )
    return records


def main():
    # argv: [plugin, first link argument (passed through), link args...]
    first = sys.argv[1]
    args = sys.argv[2:]

    root = os.getcwd()
    if "--build-root" in args:
        root = args[args.index("--build-root") + 1]

    packs = collect_packs(args, root)
    if not packs:
        return
    target = link_target(args)
    if target is None:
        raise SystemExit("pyres: cannot tell the link target: neither --target= nor /MACHINE: on the command line")
    fmt, arch = target
    supported = {"elf": _ELF_MACHINE, "macho": _MACHO_CPU, "coff": _COFF_MACHINE}[fmt]
    if arch not in supported:
        raise SystemExit("pyres: unsupported target {} {}".format(fmt, arch))
    pos, output = output_position(args, fmt)
    if pos is None:
        raise SystemExit("pyres: no output on the link command line")

    records = merge_packs(packs)
    blob, max_probe = build_blob(records)

    # The table object goes in front of all the inputs: then the linker never pulls the empty
    # fallback table (library/cpp/resource/static_blob_default.cpp) out of its archive.
    # passed in the form of the output path: link.exe takes an absolute posix path for an option
    out_obj = output + OUT_OBJ_SUFFIX[fmt]
    write_object(out_obj if os.path.isabs(out_obj) else os.path.join(root, out_obj), fmt, arch, blob, BLOB_SYMBOL)

    if os.environ.get("PYRES_DIAG"):
        sys.stderr.write(
            "pyres: {} packs, {} keys, {} bytes, max probe {} -> {}\n".format(
                len(packs), len(records), len(blob), max_probe, out_obj
            )
        )

    out_args = list(args)
    out_args.insert(pos, out_obj)
    print(json.dumps([first] + out_args))


if __name__ == "__main__":
    main()
