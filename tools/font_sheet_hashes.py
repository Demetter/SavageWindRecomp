"""The game's font sheets by hash, for runtime/nvn/font_scale.c (SAVAGEWIND_FONT_SCALE).

Usage: font_sheet_hashes.py <romfs>/Font

Reads every Font_*.sbfarc there (Yaz0 SARC of .bffnt fonts) and prints one table row per sheet: its NVN format, size
and the hash of its texels as the game uploads them (block linear, as stored in the file). Rows are what k_sheets in
runtime/nvn/font_scale.c lists; run it again for a game version whose fonts changed. Only hashes are printed: nothing
of the fonts themselves.
"""
import os
import struct
import sys

NVN_FORMAT = {0x0C: 0x49, 0x08: 0x01}  # bffnt sheet format -> NVN format: BC4 -> RGTC1, A8 -> R8
SHEET_DATA = 0x2000                     # where the sheets start in BOTW's .bffnt files (aligned for the GPU)


def yaz0(d):
    n = struct.unpack('>I', d[4:8])[0]
    out = bytearray()
    i = 16
    while len(out) < n:
        code = d[i]
        i += 1
        for b in range(8):
            if len(out) >= n:
                break
            if code & (0x80 >> b):
                out.append(d[i])
                i += 1
                continue
            b1, b2 = d[i], d[i + 1]
            i += 2
            dist = ((b1 & 15) << 8 | b2) + 1
            count = b1 >> 4
            if count == 0:
                count = d[i] + 0x12
                i += 1
            else:
                count += 2
            s = len(out) - dist
            for k in range(count):
                out.append(out[s + k])
    return bytes(out)


def sarc(d):
    e = '<' if d[6:8] == b'\xff\xfe' else '>'
    data = struct.unpack(e + 'I', d[12:16])[0]
    count = struct.unpack(e + 'H', d[0x1A:0x1C])[0]
    names = 0x20 + count * 16 + 8
    for k in range(count):
        attr, start, end = struct.unpack(e + 'III', d[0x24 + 16 * k:0x30 + 16 * k])
        o = names + (attr & 0xFFFF) * 4
        yield d[o:d.index(b'\0', o)].decode(), d[data + start:data + end]


def layer_bytes(row, block_rows):  # Tegra block linear, NVN's default block height
    gobs = (block_rows + 7) // 8
    bh = 16
    while bh > 1 and gobs <= bh // 2:
        bh //= 2
    return (row + 63) // 64 * 64 * ((gobs + bh - 1) // bh * bh * 8)


def fnv64_words(b):  # (font_scale.c sheet_hash)
    h = 0xCBF29CE484222325
    for (w,) in struct.iter_unpack('<Q', b):
        h = ((h ^ w) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    seen = set()
    for arc in sorted(os.listdir(sys.argv[1])):
        if not arc.endswith('.sbfarc'):
            continue
        d = open(os.path.join(sys.argv[1], arc), 'rb').read()
        if d[:4] == b'Yaz0':
            d = yaz0(d)
        for name, f in sarc(d):
            if f[:4] != b'FFNT':
                continue
            t = struct.unpack('<H', f[6:8])[0]  # (TGLP follows the header and FINF)
            t += struct.unpack('<I', f[t + 4:t + 8])[0]
            if f[t:t + 4] != b'TGLP':
                continue
            tglp_end = t + struct.unpack('<I', f[t + 4:t + 8])[0]
            sheets = f[t + 10]
            fmt, = struct.unpack('<H', f[t + 18:t + 20])
            sw, sh = struct.unpack('<HH', f[t + 24:t + 28])
            if fmt not in NVN_FORMAT:
                print(f'    // {arc} {name}: sheet format {fmt:#x} left out', file=sys.stderr)
                continue
            row, rows = (sw // 4 * 8, sh // 4) if fmt == 0x0C else (sw, sh)
            size = layer_bytes(row, rows)
            assert SHEET_DATA + sheets * size <= tglp_end, (arc, name)
            for s in range(sheets):
                h = fnv64_words(f[SHEET_DATA + s * size:SHEET_DATA + (s + 1) * size])
                if h in seen:
                    continue
                seen.add(h)
                print(f'    {{0x{NVN_FORMAT[fmt]:02x}, {sw}, {sh}, 0x{h:016x}ull}},  // {arc[:-7]} {name[:-6]} {s}')


if __name__ == '__main__':
    main()
