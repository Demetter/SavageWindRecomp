"""Read a finished B2 capture; compare the four balanced ABBA blocks only."""
from pathlib import Path
import json, re, statistics, sys

root = Path(__file__).resolve().parents[1]
run = root / 'build/codex-perf/runs' / sys.argv[1]
lines = (run / 'stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()
frames, toggles, profiles = [], [], []
for line in lines:
    if line.startswith('[ft] '):
        p = line.split()
        frames.append((int(p[2]), float(p[1])))
    elif line.startswith('[b2-toggle] '):
        p = line.split()[1:]
        toggles.append({k: int(v) for k, v in zip(p[::2], p[1::2])})
    elif line.startswith('[svcprof] '):
        m = re.match(r'\[svcprof\] t(\d+)', line)
        if m:
            empty = re.search(r'waitsync wakes with nothing signalled (\d+)', line)
            sync = re.search(r' 18:(\d+)/(\d+)ms', line)
            profiles.append(dict(thread=int(m[1]), frame=frames[-1][0] if frames else 0,
                                 ms=frames[-1][1] if frames else 0,
                                 empty_wakes=int(empty[1]) if empty else 0,
                                 sync_calls=int(sync[1]) if sync else 0,
                                 sync_wait_ms=int(sync[2]) if sync else 0))

def stats(values):
    if not values:
        return None
    v = sorted(values)
    return dict(n=len(v), mean=statistics.mean(v), p95=v[round((len(v)-1)*.95)],
                p99=v[round((len(v)-1)*.99)], maximum=v[-1],
                over_8ms=sum(x>8 for x in v), over_16ms=sum(x>16.667 for x in v))

ftmap = dict(frames)
edges = {t['block']: ftmap[t['frame']] for t in toggles if t['frame'] in ftmap}
assert 0 in edges and frames[-1][1] >= edges[0]+40000, 'Incomplete four-block capture'
blocks, pooled = [], {0: [], 1: []}
for b in range(4):
    a = edges[b]
    end = edges.get(b+1, a+10000)
    on = int(b in (1,2))
    v = [frames[i][1]-frames[i-1][1] for i in range(1,len(frames))
         if a+1500 <= frames[i][1] <= end-1500]
    blocks.append(dict(block=b, on=on, begin_ms=a, end_ms=end, frame_ms=stats(v)))
    pooled[on].extend(v)
summary = {}
for on in (0,1):
    mean = statistics.mean(b['frame_ms']['mean'] for b in blocks if b['on']==on)
    summary[on] = dict(mean_ms=mean, fps=1000/mean, pooled=stats(pooled[on]))

# svcprof aggregates ~2-second intervals. Accept a thread's interval only when
# BOTH its previous and current reports lie in the same trimmed ABBA block.
# Thread IDs are id&31 buckets, not unique handles; keep that distinction here.
previous, bins = {}, {0: {}, 1: {}}
for p in profiles:
    prev = previous.get(p['thread'])
    previous[p['thread']] = p
    if not prev:
        continue
    for b in blocks:
        if b['begin_ms']+1500 <= prev['ms'] < p['ms'] <= b['end_ms']-1500:
            t = bins[b['on']].setdefault(p['thread'], dict(intervals=0, frames=0, duration_ms=0,
                                                         empty_wakes=0, sync_calls=0, sync_wait_ms=0))
            t['intervals'] += 1
            t['frames'] += p['frame']-prev['frame']
            t['duration_ms'] += p['ms']-prev['ms']
            for key in ('empty_wakes','sync_calls','sync_wait_ms'):
                t[key] += p[key]
for mode in bins.values():
    for t in mode.values():
        t['empty_wakes_per_frame'] = t['empty_wakes']/t['frames'] if t['frames'] else None
        t['sync_wait_ms_per_frame'] = t['sync_wait_ms']/t['frames'] if t['frames'] else None

out = dict(run=run.name, toggles=toggles, blocks=blocks, summary=summary,
           saving_ms=summary[0]['mean_ms']-summary[1]['mean_ms'], svc_profile=bins,
           errors=[s for s in lines if any(x in s for x in ('FATAL','[fatal]','VUID-','SYNC-HAZARD'))])
out['gate_pass'] = out['saving_ms'] >= .2
(run/'b2-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
