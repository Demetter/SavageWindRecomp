"""Isolated A1 compile/link. Never overwrites normal objects, EXE, or import library."""
from pathlib import Path
import hashlib, os, shlex, subprocess
root=Path(__file__).resolve().parents[1]
os.chdir(root)
out=Path('build/codex-perf/a1b-buffer-cache');out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ);env['PATH']='C:/msys64/ucrt64/bin;'+env['PATH']
cc='C:/msys64/ucrt64/bin/clang.exe'
flags=shlex.split('-O2 -g -fno-strict-aliasing -Wall -Wno-unused-function -Iruntime -Iruntime/nvn -isystem third_party -isystem third_party/stb -D_FILE_OFFSET_BITS=64 -D_POSIX_THREAD_SAFE_FUNCTIONS -D_CRT_RAND_S -DWIN32_LEAN_AND_MEAN -DNOMINMAX')
def run(args):
    print(' '.join(args),flush=True)
    subprocess.run(args,env=env,check=True)
for name in ('exec','spirv','vkq','a1d'):
    extra=[]
    # Preserve the existing compiler policy for unchanged graphics shaders.
    # A1 compute source carries its own version and hashes independently, in
    # an isolated cache directory selected by the benchmark script.
    if name=='spirv':extra=['-DSPIRV_ID="90fabe4b68807b8e"']
    run([cc,*flags,*extra,'-c',f'runtime/nvn/vulkan/{name}.c','-o',str(out/f'{name}.o')])
run([cc,*flags,'-c','runtime/nvn/objects.c','-o',str(out/'objects.o')])
run([cc,*flags,'-c','runtime/gpu_consumer_probe.c','-o',str(out/'gpu_consumer_probe.o')])
old=Path('build/codex-perf/result-delivery-coverage-build.log').read_text(encoding='utf-8')
line=next(line for line in old.splitlines() if line.startswith('clang++ '))
args=shlex.split(line);args[0]='C:/msys64/ucrt64/bin/clang++.exe'
args[args.index('-o')+1]=str(out/'savagewind.exe')
for i,arg in enumerate(args):
    for name in ('exec','spirv','vkq'):
        if arg==f'build/rt/nvn/vulkan/{name}.o':args[i]=str(out/f'{name}.o')
    if arg.startswith('-Wl,--out-implib,'):args[i]='-Wl,--out-implib,'+str(out/'savagewind.dll.a')
    if arg=='build/rt/nvn/objects.o':args[i]=str(out/'objects.o')
    if arg=='build/rt/gpu_consumer_probe.o':args[i]=str(out/'gpu_consumer_probe.o')
args.append(str(out/'a1d.o'))
run(args)
run([cc,'-O2','tools/test_buffer_cache_shader.c','-lshaderc_shared','-o',str(out/'test_buffer_cache_shader.exe')])
ids=('d3a45e67ef9bacaf','b6ffbe601061b198','ec5709dd8702c7dd','4e2d60a3f420b3ba','05615853c139aef1','92f24d7ae50e5945','353d56c825f0617f')
run([str(out/'test_buffer_cache_shader.exe'),*[f'build/codex-perf/consumer-shaders/{s}.glsl' for s in ids]])
run([str(out/'test_buffer_cache_shader.exe'),'build/codex-perf/a1b-buffer-cache/shaders/c3e522014faa8ebd.glsl'])
for f in ('build/savagewind.exe','build/swgame.dll',str(out/'savagewind.exe')):
    print(hashlib.sha256(Path(f).read_bytes()).hexdigest(),f,flush=True)
