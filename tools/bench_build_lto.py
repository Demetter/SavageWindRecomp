"""Build an isolated ThinLTO game library for a same-runtime A/B experiment.

Reads generated C from --source; never edits it or replaces the normal DLL.
The unchanged ABI/source ID is retained only for this local experiment. Compiler
flags and input hashes are recorded separately; this is not a release build.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    source = args.source.resolve()
    out = args.output.resolve()
    if not out.is_relative_to(repo / 'build' / 'codex-perf'):
        raise SystemExit('Experimental output must be under build/codex-perf.')
    out.mkdir(parents=True, exist_ok=True)
    clang = Path('C:/msys64/ucrt64/bin/clang.exe')
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ['PATH'])
    flags = ['-O2', '-fstrict-aliasing', '-Wno-return-type', '-Wno-unused-label',
             '-fvisibility=hidden', '-Iruntime', '-fdenormal-fp-math=dynamic',
             '-D_FILE_OFFSET_BITS=64', '-march=x86-64-v3', '-flto=thin']
    base_id = (repo / 'build/swgame.dll.id').read_text().strip()
    sources = sorted(source.rglob('*.c')) + [repo / 'tools/gamelib_entry.c']
    header_hashes = {p: digest(repo / p) for p in
                     ['runtime/cpu.h', 'runtime/simd.h', 'runtime/simd_fp.h', 'runtime/simd_softfma.h', 'runtime/gamelib.h']}
    manifest = {'purpose': 'local same-runtime ThinLTO A/B, not a release library',
                'compiler': subprocess.check_output([str(clang), '--version'], env=env, text=True),
                'flags': flags, 'base_id': base_id, 'headers': header_hashes,
                'sources': {str(s): digest(s) for s in sources}}
    manifest_path = out / 'inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit('Input/flag change: choose a new output directory.')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()

    def compile_one(src):
        relative = src.relative_to(source) if src.is_relative_to(source) else Path('tools/gamelib_entry.c')
        obj = (out / 'obj' / relative).with_suffix('.o')
        obj.parent.mkdir(parents=True, exist_ok=True)
        if obj.exists():
            return obj
        extra = [f'-DGAMELIB_ID="{base_id}"'] if src.name == 'gamelib_entry.c' else []
        cmd = [str(clang), *flags, *extra, '-c', str(src), '-o', str(obj)]
        result = subprocess.run(cmd, cwd=repo, env=env, text=True, capture_output=True)
        if result.returncode:
            obj.with_suffix('.error.txt').write_text(result.stdout + result.stderr)
            if obj.exists():
                obj.unlink()
            raise RuntimeError(f'Compile failed: {src}\n{result.stderr[-3000:]}')
        return obj

    objects = []
    print(f'Compiling {len(sources)} files, {args.jobs} jobs', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(compile_one, s) for s in sources]
        for f in concurrent.futures.as_completed(pending):
            objects.append(f.result())
            if len(objects) % 50 == 0:
                print(f'{len(objects)}/{len(sources)} compiled, {time.monotonic()-start:.0f}s', flush=True)
    # Compiler response file, not shell text. Forward slashes keep its escaping simple.
    rsp = out / 'link.rsp'
    rsp.write_text('\n'.join('"' + p.as_posix() + '"' for p in sorted(objects)))
    dll = out / 'swgame.dll'
    cmd = [str(clang), '-shared', '-fuse-ld=lld', '-flto=thin',
           f'-Wl,--thinlto-jobs={args.jobs}', '-Wl,--lto-O2',
           '-Wl,--thinlto-cache-dir=' + (out / 'cache').as_posix(),
           '@' + str(rsp), str(repo / 'build/savagewind.dll.a'), '-o', str(dll)]
    print('Linking ThinLTO library', flush=True)
    with (out / 'link.log').open('w') as log:
        subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    Path(str(dll) + '.id').write_text(base_id + '\n')
    (out / 'result.json').write_text(json.dumps({'seconds': time.monotonic()-start,
        'dll_sha256': digest(dll), 'bytes': dll.stat().st_size, 'link': cmd}, indent=2))
    print(f'Done: {dll} ({time.monotonic()-start:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
