"""Derive A1b evidence from an existing capture; never launches the game."""
from pathlib import Path
import collections,json,re,statistics,sys
root=Path(__file__).resolve().parents[1];run=root/'build/codex-perf/runs'/sys.argv[1]
lines=(run/'stderr.log').read_text(encoding='utf-8',errors='replace').splitlines()
def stats(v):
    if not v:return None
    v=sorted(v);return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),min=v[0],max=v[-1],p95=v[round((len(v)-1)*.95)],p99=v[round((len(v)-1)*.99)])
def pairs(line):
    p=line.split()[1:];return dict(zip(p[::2],p[1::2]))
frames=[];shaders=[];updates=[];fts=[];clocks=[];shots=[];checks=[];toggles=[];on=0
for line in lines:
    if line.startswith('[a1-toggle] '):
        d={k:int(v) for k,v in pairs(line).items()};toggles.append(d);on=d['on']
    elif line.startswith('[a1-frame] '):frames.append({k:int(v) for k,v in pairs(line).items()})
    elif line.startswith('[a1b-shader] '):shaders.append(pairs(line))
    elif line.startswith('[a1b-updates] '):updates.append({k:int(v) for k,v in pairs(line).items()})
    elif line.startswith('[ft] '):
        p=line.split();fts.append(dict(ms=float(p[1]),frame=int(p[2]),on=on))
    elif line.startswith('[gcp-readback-verify] '):checks.append({**pairs(line),'mode':on})
    elif line.startswith('[clock] '):
        m=re.search(r't ([\d.]+) frames (\d+) game ([\d.]+) tod ([\d.-]+) pos ([\d.-]+) ([\d.-]+) ([\d.-]+).* at ([\d.]+)',line)
        if m:clocks.append(dict(t=float(m[1]),frame=int(m[2]),game=float(m[3]),tod=float(m[4]),pos=[float(m[i]) for i in (5,6,7)],ns=round(float(m[8])*1e9),mode=on))
    elif '[vk] saved build/frames/' in line:shots.append(dict(file=line.split('build/frames/')[1].split()[0],mode=on,frame=fts[-1]['frame'] if fts else 0))
ret=[f for f in frames if f['on'] and f['draws']>1000]
names={1:'uncovered_read',2:'uncovered_write',4:'unsupported',8:'capacity',16:'live_vertex',32:'input_copy',64:'cpu_update',128:'bad_range',256:'graphics_gmem'}
out=dict(run=run.name,on_steady_frames=len(ret),eligible=sum(f['eligible'] for f in ret),
    rejection_masks=dict(collections.Counter(str(f['reasons']) for f in ret)),
    rejection_counts={n:sum(bool(f['reasons']&bit) for f in ret) for bit,n in names.items()},
    traffic={k:stats([f[k] for f in ret]) for k in ['copy_bytes','writeback_bytes','aliases','read_misses','write_misses','gpu_only_bytes']},
    updates={k:stats([u[k] for u in updates if u['on']]) for k in ['changed_overlaps','drains','rejected']},
    shaders=[],shots=shots,toggles=toggles,clocks=clocks,readback_checks=checks,
    errors=[l for l in lines if any(x in l for x in ('VUID-','SYNC-HAZARD','Validation Error','FATAL','[fatal]','mismatch at first'))])
for id in sorted({s['id'] for s in shaders}):
    all_s=[s for s in shaders if s['id']==id];ss=[s for s in all_s if s['on']=='1']
    out['shaders'].append(dict(id=id,stage=all_s[0]['stage'],proved=all_s[0]['proved'],on_frames=len(ss),
        unbounded_frames=sum(int(s['unbounded'])>0 for s in ss),unbounded_draws=sum(int(s['unbounded']) for s in ss),
        gpu_reads=sum(int(s['reads']) for s in ss),gpu_writes=sum(int(s['writes']) for s in ss),outside_map=sum(int(s['outside_map']) for s in ss),
        minimum_address=min((s['min'] for s in ss if int(s['reads'])+int(s['writes'])),default=None),maximum_address=max((s['max'] for s in ss if int(s['reads'])+int(s['writes'])),default=None)))
# Use frame markers and trim 1.5 seconds from each switch, warm up >=45s.
ftmap={f['frame']:f['ms'] for f in fts};edges=[(ftmap[t['frame']],t['on']) for t in toggles if t['frame'] in ftmap]
start=edges[0][0]-10000 if edges else 45000;edges=[(start,0)]+edges+[(fts[-1]['ms'],None)] if fts else []
blocks=[]
for (a,mode),(b,_) in zip(edges,edges[1:]):
    vals=[fts[i]['ms']-fts[i-1]['ms'] for i in range(1,len(fts)) if a+1500<=fts[i]['ms']<=b-1500]
    if vals:blocks.append(dict(on=mode,begin_ms=a,end_ms=b,frame_ms=stats(vals),over_8ms=sum(v>8 for v in vals),over_16ms=sum(v>16.667 for v in vals)))
out['blocks']=blocks
# Equal-duration, balanced ten-second ABBA blocks; use only complete blocks and
# retain 7s from each. This avoids weighting OFF more just because it is faster.
ten=[];values={0:[],1:[]}
if edges:
    origin=edges[0][0]
    for k in range(int((fts[-1]['ms']-origin)//10000)):
        a=origin+k*10000;b=a+10000;mode=int(k%4 in (1,2))
        chosen=[i for i in range(1,len(fts)) if a+1500<=fts[i]['ms']<=b-1500]
        assert all(fts[i]['on']==mode for i in chosen), 'Timing window crosses a mode transition'
        vals=[fts[i]['ms']-fts[i-1]['ms'] for i in chosen]
        if vals:
            ten.append(dict(block=k,on=mode,begin_ms=a,end_ms=b,frame_ms=stats(vals)))
            values[mode].extend(vals)
out['ten_second_blocks']=ten
out['equal_block_summary']={}
for mode in (0,1):
    ss=[b['frame_ms']['mean'] for b in ten if b['on']==mode]
    if ss:
        avg=statistics.mean(ss);vv=values[mode]
        out['equal_block_summary'][str(mode)]=dict(blocks=len(ss),mean_ms=avg,fps=1000/avg,pooled_frames=stats(vv),over_8ms=sum(v>8 for v in vv),over_16ms=sum(v>16.667 for v in vv))
(run/'a1b-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('clocks','readback_checks')},indent=2))
