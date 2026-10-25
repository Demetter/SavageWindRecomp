"""Offline, exact-binary attribution for the opt-in native ABI sizing exercise."""
import bisect, collections, functools, json, os, pathlib, sys
import profile_samples as ps

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/codex-perf/native-abi-sizing'
os.environ['PATH'] = 'C:/msys64/ucrt64/bin;' + os.environ['PATH']

def analyze(run):
    d = ROOT / 'build/codex-perf/runs' / run
    m = json.loads((d/'manifest.json').read_text())
    exe = d/'savagewind.exe'
    if not exe.exists(): exe = pathlib.Path(m.get('executable_snapshot', m['executable']))
    dll = pathlib.Path(m.get('game_library_snapshot', m['game_library']))
    bases, modules, threads = {}, [], {}
    maps = None
    counts = {t: collections.Counter() for t in (0x101, 0x120, 0x121)}
    selfs = {t: collections.Counter() for t in counts}
    gameparents = {t: collections.Counter() for t in counts}
    allparents = {t: collections.Counter() for t in counts}
    first = last = None
    rounds = 0
    @functools.lru_cache(None)
    def name(a):
        for low, high, mp in maps:
            if low <= a < high: return mp.name(a) or '?'
        i = bisect.bisect_right(modstarts, a) - 1
        if i >= 0 and a < modules[i].base + modules[i].size: return modules[i].name(a)
        return '?external'
    for line in (d/'samples.txt').open():
        p=line.split()
        if not p: continue
        if p[0]=='M': bases[p[1]]=int(p[2],16)
        elif p[0]=='D':
            _,b,s,path=line.rstrip().split(' ',3)
            modules.append(ps.ExternalModule(int(b,16),int(s,16),path))
        elif p[0]=='T': threads[p[1]]=p[2]
        elif p[0]=='R':
            point=(int(p[1]),int(p[2])); first=first or point; last=point; rounds+=1
        elif p[0]=='S' and int(p[1]) in counts:
            if maps is None:
                maps=[(bases['exe'],bases['exe']+ps.image_size(exe),ps.Map(ps.symbols(exe),0x140000000,bases['exe'])),
                      (bases['swgame'],bases['swgame']+ps.image_size(dll),ps.Map(ps.symbols(dll),0x180000000,bases['swgame']))]
                modules.sort(key=lambda x:x.base); modstarts=[x.base for x in modules]
            t=int(p[1]); names=[name(int(x,16)) for x in p[2:]]
            plain=[n.rsplit('!',1)[-1] for n in names]
            waits = any(n in ps.WAITS or n in ps.SYSTEM_WAITS or n.startswith(('ZwWait','NtWait')) for n in plain)
            if ps.CORE_WAIT in plain: kind='core_wait'
            elif waits: kind='wait'
            elif names[0].startswith('f_'): kind='game'
            elif names[0] in ('rt_dispatch','dispatch_slow'): kind='dispatch'
            elif 'svc_impl' in names or 'rt_svc' in names: kind='kernel_active'
            elif any(n.startswith('hle_') for n in names): kind='hle_active'
            else: kind='runtime_or_external_active'
            counts[t][kind]+=1; selfs[t][names[0]]+=1
            allparents[t].update(set(names))
            if kind=='game': gameparents[t].update(set(n for n in names if n.startswith('f_')))
    ms=(last[0]-first[0])/1e6/(last[1]-first[1])
    out={'run':run,'exe':str(exe),'dll':str(dll),'manifest':m,'first':first,'last':last,'rounds':rounds,'sampled_frame_ms':ms,
         'limitations':'Wall-time stack samples, not scheduled CPU. Suspends target threads. Exclusive buckets; game parents only count stacks whose leaf is game code. No call counts can be inferred.',
         'threads':{hex(t):{'samples':sum(counts[t].values()),'split':dict(counts[t]),'ms_per_frame':{k:ms*v/sum(counts[t].values()) for k,v in counts[t].items()},
                          'self':selfs[t].most_common(150),'game_only_inclusive':gameparents[t].most_common(100),'all_inclusive':allparents[t].most_common(60)} for t in counts if counts[t]}}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/(run+'.json')).write_text(json.dumps(out,indent=2))
    print(run, 'sampled ms/frame',round(ms,4))
    for t,x in out['threads'].items():
        print(t,x['samples'], {k:round(v,4) for k,v in x['ms_per_frame'].items()})
        print('hot game',[(n,c) for n,c in x['self'] if n.startswith('f_')][:20])

if __name__=='__main__':
    for run in sys.argv[1:]: analyze(run)
