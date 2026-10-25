#!/usr/bin/env python3
"""Self and inclusive time per function from build/samples.txt (runtime/sampler.c, SAVAGEWIND_SAMPLE).

Usage: profile_samples.py [samples.txt] [--top N] [--thread HANDLE] [--busy] [--exclude FUNCTION]...

Addresses in the game library become f_<guest address> (its exported symbols, from `nm build/swgame.dll`), those in
the runtime the runtime's functions (`nm build/savagewind.exe`). --busy leaves out samples whose top frames are a
wait (condition variables, sleeps), so the numbers are where the CPU goes.
"""
import argparse
import bisect
import collections
import functools
import mmap
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def symbols(path):
    out = subprocess.run(['nm', '--defined-only', path], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] in 'Tt':
            syms.append((int(p[0], 16), p[2]))
    syms.sort()
    return syms


def image_size(path):
    # SizeOfImage from the PE optional header: addresses past it belong to the next module loaded (the GPU driver
    # often lands right after swgame.dll and was named after the library's last function)
    with open(path, 'rb') as f:
        d = f.read(4096)
    pe = int.from_bytes(d[0x3c:0x40], 'little')
    return int.from_bytes(d[pe + 24 + 56:pe + 24 + 60], 'little')


class Map:
    def __init__(self, syms, image_base, load_base):
        self.addrs = [a for a, _ in syms]
        self.names = [n for _, n in syms]
        self.delta = load_base - image_base

    def name(self, addr):
        a = addr - self.delta
        i = bisect.bisect_right(self.addrs, a) - 1
        return self.names[i] if i >= 0 else None


# A sample is a wait when one of these is on its stack (the kernel's waits, sleeps, the GPU fence wait), and a
# wait for a core when core_acquire is (runnable, but the emulated core is taken).
WAITS = ('kwait', 'block_current', 'pthread_cond_wait', 'pthread_cond_timedwait', 'plat_sleep_until_ns',
         'kernel_sleep_until_ns', 'nanosleep', 'vkWaitSemaphores', 'vk_wait_seq')
CORE_WAIT = 'core_acquire'
SYSTEM_WAITS = {'NtWaitForSingleObject', 'NtWaitForMultipleObjects', 'NtWaitForAlertByThreadId',
                'NtDelayExecution', 'WaitForSingleObject', 'WaitForSingleObjectEx',
                'WaitForMultipleObjects', 'WaitForMultipleObjectsEx', 'WaitOnAddress',
                'SleepConditionVariableSRW', 'SleepConditionVariableCS', 'SleepEx'}


def pe_exports(path):
    """Named exports with actual x64 unwind ranges; sparse driver exports stay unknown."""
    try:
        with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as d:
            pe = struct.unpack_from('<I', d, 0x3c)[0]
            nsections = struct.unpack_from('<H', d, pe + 6)[0]
            optsize = struct.unpack_from('<H', d, pe + 20)[0]
            opt = pe + 24
            directories = opt + (112 if struct.unpack_from('<H', d, opt)[0] == 0x20b else 96)
            sections = [struct.unpack_from('<IIII', d, opt + optsize + i*40 + 8) for i in range(nsections)]

            def raw(rva):
                for virtual_size, virtual, size, offset in sections:
                    if virtual <= rva < virtual + max(virtual_size, size):
                        return offset + rva - virtual
                raise ValueError('Unmapped PE RVA')

            er, es = struct.unpack_from('<II', d, directories)
            if not er: return []
            e = raw(er)
            nf, nn, funcs, names, ords = struct.unpack_from('<IIIII', d, e + 20)
            funcs, names, ords = raw(funcs), raw(names), raw(ords)
            unwind = {}
            xr, xs = struct.unpack_from('<II', d, directories + 3*8)
            if xr:
                x = raw(xr)
                for i in range(xs // 12):
                    begin, end = struct.unpack_from('<II', d, x + i*12)
                    unwind[begin] = end
            result = []
            for i in range(nn):
                ordinal = struct.unpack_from('<H', d, ords + 2*i)[0]
                if ordinal >= nf: continue
                addr = struct.unpack_from('<I', d, funcs + ordinal*4)[0]
                if not addr or er <= addr < er + es: continue  # a forwarder, not code
                nr = raw(struct.unpack_from('<I', d, names + 4*i)[0])
                end = d.find(b'\0', nr, nr + 1024)
                if end < 0: continue
                name = d[nr:end].decode('ascii', errors='replace')
                # Windows syscall stubs are leaves with no unwind entry. Only
                # recognize their short stub, not an arbitrary following address.
                limit = unwind.get(addr, addr + (32 if name.startswith(('Nt', 'Zw')) else 1))
                result.append((addr, limit, name))
            return sorted(result)
    except (OSError, ValueError, struct.error):
        return []


class ExternalModule:
    def __init__(self, base, size, path):
        self.base, self.size, self.path = base, size, path
        self.label = os.path.basename(path)
        self.exports = None

    def name(self, addr):
        if self.exports is None:
            self.exports = pe_exports(self.path)
            self.starts = [v[0] for v in self.exports]
        rva = addr - self.base
        i = bisect.bisect_right(self.starts, rva) - 1
        if i >= 0 and rva < self.exports[i][1]:
            return self.label + '!' + self.exports[i][2]
        return self.label + '+0x%x' % rva


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('samples', nargs='?', default=os.path.join(ROOT, 'build', 'samples.txt'))
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--thread', type=lambda v: int(v, 0))
    ap.add_argument('--busy', action='store_true')
    ap.add_argument('--frames', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                    help='Inclusive presented-frame range from sample round markers')
    ap.add_argument('--ns', type=int, nargs=2, metavar=('BEGIN', 'END'),
                    help='Sample round timestamp range [BEGIN, END), in plat_now_ns clock')
    ap.add_argument('--exe', default=os.path.join(ROOT, 'build', 'savagewind.exe'), help='Exact runtime binary from the sampled run')
    ap.add_argument('--game-library', default=os.path.join(ROOT, 'build', 'swgame.dll'), help='Exact game library from the sampled run')
    ap.add_argument('--exclude', action='append', default=[], metavar='FUNCTION',
                    help='leave out the samples with this function on the stack (e.g. the cost of a debug log)')
    a = ap.parse_args()
    bases, threads, samples, modules = {}, {}, [], []
    sample_ns = sample_frame = None
    coverage = None
    if a.frames and a.frames[1] < a.frames[0]: ap.error('LAST frame must be >= FIRST')
    if a.ns and a.ns[1] <= a.ns[0]: ap.error('END timestamp must be > BEGIN')
    for line in open(a.samples):
        p = line.split()
        if p[0] == 'P':
            coverage = [int(v) for v in p[1:]]
        elif p[0] == 'M':
            bases[p[1]] = int(p[2], 16)
        elif p[0] == 'D':
            _, base, size, path = line.rstrip().split(' ', 3)
            modules.append(ExternalModule(int(base, 16), int(size, 16), path))
        elif p[0] == 'T':
            threads[int(p[1])] = p[2]
        elif p[0] == 'R':
            sample_ns, sample_frame = int(p[1]), int(p[2])
        elif p[0] == 'S':
            if a.frames and (sample_frame is None or not a.frames[0] <= sample_frame <= a.frames[1]): continue
            if a.ns and (sample_ns is None or not a.ns[0] <= sample_ns < a.ns[1]): continue
            samples.append((int(p[1]), [int(x, 16) for x in p[2:]]))
    exe = Map(symbols(a.exe), 0x140000000, bases.get('exe', 0x140000000))
    dll = Map(symbols(a.game_library), 0x180000000, bases.get('swgame', 0x180000000))
    dll_lo = bases.get('swgame', 0)
    dll_size = image_size(a.game_library)
    exe_size = image_size(a.exe)

    modules.sort(key=lambda m: m.base)
    module_starts = [m.base for m in modules]

    @functools.lru_cache(maxsize=None)
    def name(addr):
        if dll_lo and dll_lo <= addr < dll_lo + dll_size:
            return dll.name(addr) or '?swgame'
        if bases.get('exe', 0) <= addr < bases.get('exe', 0) + exe_size:
            return exe.name(addr) or '?exe'
        i = bisect.bisect_right(module_starts, addr) - 1
        if i >= 0 and addr < modules[i].base + modules[i].size:
            return modules[i].name(addr)
        return '?%x' % (addr >> 20 << 20)  # system libraries: by megabyte

    selfc, incl, per_thread, kinds = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    total = 0
    for th, stack in samples:
        if a.thread is not None and th != a.thread:
            continue
        names = [name(x) for x in stack]
        plain = [n.rsplit('!', 1)[-1] for n in names]
        kind = 'core' if CORE_WAIT in plain else 'wait' if any(n in WAITS or n in SYSTEM_WAITS for n in plain) else 'busy'
        kinds[kind] += 1
        if a.busy and kind != 'busy':
            continue
        if a.exclude and any(n in a.exclude for n in names):
            continue
        total += 1
        per_thread[th] += 1
        if names:
            selfc[names[0]] += 1
        for n in set(names):
            incl[n] += 1
    if coverage and len(coverage) == 6:
        begin, end, first, last, rounds, full = coverage
        print('Sampler coverage: %.3f of %.3f requested seconds, %d rounds%s' % (
            (last-first)/1e9 if first else 0, (end-begin)/1e9, rounds,
            '; BUFFER EXHAUSTED; later events have no stacks' if full else ''))
    if not total:
        print('no samples')
        return
    n_all = sum(kinds.values())
    missing = [hex(th) for th in threads if th >= 0xf000 and not any(s[0] == th for s in samples)]
    if missing:
        print('WARNING: registered host threads have no samples: ' + ', '.join(missing))
    print('%d samples: busy %.1f%%, waiting %.1f%%, waiting for a core %.1f%%' % (
        n_all, 100.0 * kinds['busy'] / n_all, 100.0 * kinds['wait'] / n_all, 100.0 * kinds['core'] / n_all))
    print('%d counted' % total)
    print('\nby thread:')
    for th, c in per_thread.most_common(12):
        print('  %5.1f%%  handle %#x entry %s' % (100.0 * c / total, th, threads.get(th, '?')))
    print('\nself:')
    for n, c in selfc.most_common(a.top):
        print('  %5.1f%%  %s' % (100.0 * c / total, n))
    print('\ninclusive:')
    for n, c in incl.most_common(a.top):
        print('  %5.1f%%  %s' % (100.0 * c / total, n))


if __name__ == '__main__':
    main()
