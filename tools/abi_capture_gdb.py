"""Run by GDB only, using a config written by abi_capture_prepare.py.

The debugger stops all CPU threads at function entry. The separate replay
process reads original guest pages on demand and writes only its own copies.
No game executable or DLL on disk is changed.
"""
import gdb, json, pathlib, subprocess, time, os

cfg=json.loads(pathlib.Path(gdb.convenience_variable('abi_config').string()).read_text())
out=pathlib.Path(cfg['out']);out.mkdir(exist_ok=True)
results=[]
for command in ('set pagination off','set confirm off','handle SIGSEGV nostop noprint pass'):
    gdb.execute(command)
class Capture(gdb.Breakpoint):
    def __init__(self,name,pc):
        super().__init__('*0x%x'%pc,internal=True);self.name=name;self.count=0
    def stop(self):
        return True
bps=[Capture(n,a) for n,a in cfg['targets'].items()]
deadline=time.monotonic()+90
try:
    while any(b.enabled for b in bps) and time.monotonic()<deadline:
        gdb.execute('continue')
        if not gdb.selected_inferior().threads():break
        pc=int(gdb.parse_and_eval('$pc'))
        bp=next((b for b in bps if b.enabled and cfg['targets'][b.name]==pc),None)
        if bp is None:
            print('non-target stop',hex(pc));break
        inferior=gdb.selected_inferior();cpu=int(gdb.parse_and_eval('$rcx'))
        raw=bytes(inferior.read_memory(cpu,1024))
        stem=out/(bp.name+'-'+str(bp.count));cpufile=stem.with_suffix('.cpu');cpufile.write_bytes(raw)
        env=dict(os.environ);env['ABI_LIBRARY']=cfg['dll']
        result=subprocess.run([cfg['harness'],'capture',str(inferior.pid),bp.name,str(cpufile),str(stem.with_suffix('.bin'))],capture_output=True,text=True,timeout=12,env=env)
        record={'root':bp.name,'case':bp.count,'pid':inferior.pid,'thread':list(gdb.selected_thread().ptid),'cpu_address':hex(cpu),
                'lr':hex(int.from_bytes(raw[240:248],'little')),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        print(json.dumps(record),flush=True);results.append(record)
        (out/'capture-results.json').write_text(json.dumps(results,indent=2))
        bp.count+=1
        if bp.count>=3 or result.returncode!=0:bp.enabled=False
finally:
    try:
        for bp in bps:bp.delete()
        gdb.execute('detach')
    except gdb.error:pass
