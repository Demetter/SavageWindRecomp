"""Prepare exact-symbol debugger capture. This does not launch the game."""
import ctypes as c, ctypes.wintypes as w, json, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'build/codex-perf/native-abi-sizing'
pid=int(sys.argv[1]);variant=sys.argv[2] if len(sys.argv)>2 else 'prototype';k=c.WinDLL('kernel32',use_last_error=True)
class MODULE(c.Structure):
    _fields_=[('dwSize',w.DWORD),('th32ModuleID',w.DWORD),('th32ProcessID',w.DWORD),('GlblcntUsage',w.DWORD),('ProccntUsage',w.DWORD),('modBaseAddr',c.c_void_p),('modBaseSize',w.DWORD),('hModule',w.HMODULE),('szModule',w.WCHAR*256),('szExePath',w.WCHAR*260)]
k.CreateToolhelp32Snapshot.argtypes=[w.DWORD,w.DWORD];k.CreateToolhelp32Snapshot.restype=w.HANDLE
k.Module32FirstW.argtypes=[w.HANDLE,c.POINTER(MODULE)];k.Module32NextW.argtypes=[w.HANDLE,c.POINTER(MODULE)];k.CloseHandle.argtypes=[w.HANDLE]
h=k.CreateToolhelp32Snapshot(8|16,pid);m=MODULE();m.dwSize=c.sizeof(m);ok=k.Module32FirstW(h,c.byref(m));base=None
while ok:
    if m.szModule.lower()=='swgame.dll':base=m.modBaseAddr;dll=m.szExePath;break
    ok=k.Module32NextW(h,c.byref(m))
k.CloseHandle(h)
if base is None:raise RuntimeError('game DLL not loaded')
symbols={p[2]:int(p[0],16) for line in subprocess.check_output(['C:/msys64/ucrt64/bin/nm.exe','--defined-only',dll],text=True).splitlines() if len(p:=line.split())==3 and p[1] in 'Tt'}
manifest=json.loads((OUT/variant/'manifest.json').read_text())
config={'pid':pid,'dll':dll,'module_base':hex(base),'harness':str(OUT/variant/('savagewind.exe' if variant!='prototype' else 'abi-replay.exe')),'out':str(OUT/('captures' if variant=='prototype' else 'captures-boundary')),
        'targets':{n[2:]:base+symbols[n]-0x180000000 for n in manifest['roots']}}
(OUT/'capture-config.json').write_text(json.dumps(config,indent=2))
path=(OUT/'capture-config.json').as_posix()
(OUT/'capture.gdb').write_text('set auto-solib-add off\nset pagination off\nset confirm off\nset $abi_config = "'+path+'"\nattach '+str(pid)+'\nsource '+(ROOT/'tools/abi_capture_gdb.py').as_posix()+'\nquit\n')
print(json.dumps({'pid':pid,'module_base':hex(base),'targets':len(config['targets'])}))
