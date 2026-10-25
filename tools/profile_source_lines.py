"""Attribute sampled native PCs to DWARF source lines, including inlined callees.

Use the exact executable saved with the capture. Symbol-only stack summaries hide
inlined work inside draw_/exec_list; this report preserves it. Counts are samples,
not measured milliseconds or proof that the line alone caused a stall.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import struct
import subprocess


def symbolize(exe, addresses, addr2line):
    addresses = sorted(addresses)
    if not addresses:
        return {}
    output = subprocess.run([addr2line, '-a', '-f', '-i', '-e', str(exe)],
                            input=''.join(f'{a:#x}\n' for a in addresses), text=True,
                            capture_output=True, check=True).stdout.splitlines()
    result, current, pending = {}, None, None
    for line in output:
        if re.fullmatch(r'0x[0-9a-fA-F]+', line):
            current, pending = int(line, 16), None
            result[current] = []
        elif current is not None:
            if pending is None:
                pending = line
            else:
                result[current].append({'function': pending, 'source': line})
                pending = None
    if set(result) != set(addresses):
        raise ValueError('Incomplete addr2line output; refusing partial attribution')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('samples', type=Path)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--addr2line', default='addr2line')
    ap.add_argument('--thread', type=lambda x: int(x, 0), default=0xf000)
    ap.add_argument('--frames', help='Comma-separated exact frame IDs; omit for whole sampled interval')
    ap.add_argument('--top', type=int, default=80)
    args = ap.parse_args()
    frames = {int(f) for f in args.frames.split(',')} if args.frames else None
    with args.exe.open('rb') as f:
        header = f.read(4096)
    pe = struct.unpack_from('<I', header, 0x3c)[0]
    if header[:2] != b'MZ' or struct.unpack_from('<H', header, pe + 24)[0] != 0x20b:
        raise ValueError('Expected a PE32+ executable')
    image_base = struct.unpack_from('<Q', header, pe + 24 + 24)[0]
    image_size = struct.unpack_from('<I', header, pe + 24 + 56)[0]
    load_base, frame, total = None, None, 0
    counts = {'native_leaf': collections.Counter(), 'native_caller_of_external_leaf': collections.Counter()}
    for line in args.samples.open():
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ['M', 'exe']:
            load_base = int(parts[2], 16)
        elif parts[0] == 'R':
            frame = int(parts[2])
        elif parts[0] == 'S' and int(parts[1]) == args.thread and (frames is None or frame in frames):
            if load_base is None:
                raise ValueError('Missing executable load base')
            total += 1
            pcs = [int(p, 16) for p in parts[2:]]
            for depth, pc in enumerate(pcs):
                if load_base <= pc < load_base + image_size:
                    # Return PCs refer to the instruction after a call; leaf PCs
                    # are the actual sampled instruction and must not be shifted.
                    address = pc - load_base + image_base - bool(depth)
                    key = 'native_leaf' if depth == 0 else 'native_caller_of_external_leaf'
                    counts[key][address] += 1
                    break
    addresses = set().union(*(c.keys() for c in counts.values()))
    symbols = symbolize(args.exe, addresses, args.addr2line)
    reports = {}
    for kind, counter in counts.items():
        by_line, example = collections.Counter(), {}
        for pc, count in counter.items():
            chain = symbols[pc]
            key = tuple((r['function'], r['source']) for r in chain)
            by_line[key] += count
            example[key] = pc
        reports[kind] = {'samples': sum(counter.values()), 'distinct_pcs': len(counter),
            'top': [{'samples': count, 'pct_of_all_thread_samples': count * 100 / total,
                     'example_pc': hex(example[key]),
                     'inline_chain': [{'function': f, 'source': s} for f, s in key]}
                    for key, count in by_line.most_common(args.top)]}
    print(json.dumps({'thread': hex(args.thread), 'samples': total, 'frames': sorted(frames) if frames else None,
        'executable': str(args.exe), 'profiles': reports,
        'note': 'Sampling attribution, not cycle counts. External-leaf rows locate the nearest runtime caller, not its self time. Inlined source positions follow compiler DWARF and can be approximate.'}, indent=2))


if __name__ == '__main__':
    main()
