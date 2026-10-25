"""Read-only capture analysis; writes derived evidence beside the run logs."""
from pathlib import Path
import json,re,statistics,collections,shutil
root=Path(__file__).resolve().parents[1]
run=root/'build/codex-perf/runs/a1_cache_validation_1'
lines=(run/'stderr.log').read_text(encoding='utf-8',errors='replace').splitlines()
frames=[];ft={};sync=[];current=None
for line in lines:
    if line.startswith('[ft] '):
        p=line.split();current=int(p[2]);ft[current]=float(p[1])
    if line.startswith('[a1-frame] '):
        p=line.split()[1:];frames.append({p[i]:int(p[i+1]) for i in range(0,len(p),2)})
    if '[sync-probe] lr 7000e94a98 ' in line:
        pairs=re.findall(r'(queue|release|gpu|core) ([0-9.]+)',line)
        sync.append({'frame':current,**{k:float(v) for k,v in pairs}})
def stats(v):
    v=sorted(v)
    if not v:return None
    return {'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'min':v[0],'max':v[-1],
        'p95':v[round((len(v)-1)*.95)],'p99':v[round((len(v)-1)*.99)]}
on=[r for r in frames if r['on'] and r['draws']>1000]
lo=min(r['frame'] for r in on);hi=max(r['frame'] for r in on)
reasonbits={'uncovered_gmem_read':1,'uncovered_gmem_write':2,'unsupported_shader':4,'capacity':8,'live_vertex_input':16,
    'pending_heap_copy':32,'cpu_update_invalidation':64,'bad_or_unbounded_binding':128,'unrouted_graphics_gmem':256}
out={'run':run.name,'normal_fences':True,'validation_timing_not_normal_performance':True,
    'on_frame_range':[lo,hi],'on_frames':len(on),'eligible_frames':sum(r['eligible'] for r in on),
    'reason_masks':dict(collections.Counter(str(r['reasons']) for r in on)),
    'reasons':{name:sum(bool(r['reasons']&bit) for r in on) for name,bit in reasonbits.items()},
    'per_on_frame':{name:stats([r[name] for r in on]) for name in ['copy_bytes','writeback_bytes','read_misses','write_misses','aliases','dispatches','draws','readback_bytes','gpu_only_bytes']},
    'vulkan_validation_errors':sum(('VUID-' in line or 'SYNC-HAZARD' in line or 'Validation Error' in line) for line in lines),
    'fatal_lines':[line for line in lines if 'FATAL' in line or '[fatal]' in line],
    'timing':{}}
# Within this validation launch only; trim one second at mode boundaries.
start=ft[lo];end=ft.get(hi+1,ft[hi]);off_windows=[(start-9000,start-1000),(end+1000,min(end+9000,max(ft.values())))]
for mode in ['off_before','on','off_after']:
    a,b=(start+1000,end-1000) if mode=='on' else off_windows[mode=='off_after']
    chosen=[f for f,t in ft.items() if a<=t<=b and f-1 in ft]
    out['timing'][mode]={'time_ms':[a,b],'frame_ms':stats([ft[f]-ft[f-1] for f in chosen]),
        'main_sync_ms':{kind:stats([r[kind] for r in sync if r['frame'] in chosen]) for kind in ['queue','gpu','core']}}
dest=run/'readbacks';dest.mkdir(exist_ok=True)
for line in lines:
    m=re.search(r'saved as (build/linrt_\S+\.raw)',line)
    if m:
        src=root/m[1]
        if src.exists():shutil.copyfile(src,dest/src.name)
(run/'a1-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
