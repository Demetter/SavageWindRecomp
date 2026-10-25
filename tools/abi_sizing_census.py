"""Read-only census of existing generated code; prototype evidence, not an ABI proof."""
import collections, json, pathlib, re
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'build/codex-perf/native-abi-sizing'
FUNCTION=re.compile(r'^void (f_[0-9a-f]+)\(Cpu \*restrict c\) \{\n(.*?)^\}',re.M|re.S)
X=re.compile(r'c->x\[(\d+)\]')
ASSIGN=re.compile(r'c->x\[(\d+)\]\s*=(?!=)')
CALL=re.compile(r'\b(f_[0-9a-f]+)\(c\)')

def flag_census(body):
    # Conservative: labels, branches, calls and exits consume flags. Only count
    # certain straight-line overwritten flags; therefore a lower bound.
    last=None; dead=total=0
    for line in body.splitlines():
        use=bool(re.search(r'COND_|adcs\d|sbcs\d|c->nzcv(?!\s*=)',line))
        define=bool(re.search(r'\b(?:adds|subs|logic|adcs|sbcs)\d+\(c,|c->nzcv\s*=',line))
        boundary=bool(re.search(r'^L_|\bf_[0-9a-f]+\(c\)|rt_|return|goto',line))
        if use or boundary: last=None
        if define:
            total+=1
            if last is not None: dead+=1
            last=True
    return total,dead

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    profiles=[json.loads((OUT/(p+'.json')).read_text()) for p in ('complete_cpu_profile','resource_work_stutters','corrected_fence_sensitivities')]
    hot=set(n for j in profiles for t in j['threads'].values() for n,c in t['self'] if n.startswith('f_'))
    records={}; total=collections.Counter(); extract=OUT/'hot-source';extract.mkdir(exist_ok=True)
    for path in sorted((ROOT/'generated').glob('*/*.c')):
        if path.name.endswith('_table.c'): continue
        source=path.read_text()
        for m in FUNCTION.finditer(source):
            n,body=m.groups(); calls=CALL.findall(body); stores=len(ASSIGN.findall(body)); refs=len(X.findall(body))
            flags,dead=flag_census(body)
            # Definite reads before any preceding assignment, ignoring pure
            # prologue stores of callee-saved registers. Branch-sensitive ABI
            # certification is deliberately a separate analysis.
            defined=set(); extra=set()
            for line in body.splitlines():
                rhs=ASSIGN.sub('',line)
                extra.update(int(x) for x in X.findall(rhs) if 9<=int(x)<=18 and int(x) not in defined)
                defined.update(map(int,ASSIGN.findall(line)))
                if CALL.search(line) or 'rt_dispatch(c)' in line: defined.update(range(19))
            v={'module':path.parent.name,'path':str(path.relative_to(ROOT)),'lines':body.count('\n'),
               'calls':calls,'dispatch_sites':body.count('rt_dispatch(c)'),
               'tail_sites':body.count('__attribute__((musttail))'),'setjmp': 'setjmp(' in body,
               'gpr_stores':stores,'gpr_reads':refs-stores,'flags_def':flags,'flags_proven_dead_straightline':dead,
               'implicit_x9_x18_linear_candidates':sorted(extra),
               'simd':bool(re.search(r'c->v\[|VS\w*\(|VREG|fp_\w+\(c|SV_|sl_\w+\(c',body)),
               'runtime_boundary':bool(re.search(r'\b(?:hle_|rt_|kernel_)',body)),
               'cold_fallback':bool(re.search(r'f_[0-9a-f]+_r\(c\)',body)),
               'guest_constants':len(re.findall(r'0x7[0123456789][0-9a-f]{8}ull',body))}
            records[n]=v;total['functions']+=1;total['direct_call_sites']+=len(calls);total['dispatch_sites']+=v['dispatch_sites']
            for k in ('gpr_stores','gpr_reads','flags_def','flags_proven_dead_straightline'): total[k]+=v[k]
            if n in hot: (extract/(n+'.c')).write_text(m[0]+'\n')
    callers=collections.defaultdict(list)
    for n,v in records.items():
        for c in set(v['calls']):callers[c].append(n)
    report={'totals':dict(total),'functions':records,'limitations':['Source counts are NOT retired x86 loads/stores; restrict and ThinLTO already remove traffic.', 'Implicit-input linear scan gives candidates only, not CFG-certified counts. Flags count only proven straight-line overwrite without use; calls/exits conservative.']}
    (OUT/'census.json').write_text(json.dumps(report))
    rows=[]
    for n in hot:
        v=records.get(n)
        if not v:continue
        row={'name':n,**v,'self_counts':{p['run']:{t:x.get('self') and dict(x['self']).get(n,0) for t,x in p['threads'].items()} for p in profiles},'callers':callers[n][:20]}
        rows.append(row)
    rows.sort(key=lambda x:-x['self_counts']['resource_work_stutters'].get('0x101',0))
    (OUT/'hot-census.json').write_text(json.dumps(rows,indent=2))
    print(dict(total))
    for x in rows[:60]:print(x['name'],x['self_counts']['resource_work_stutters'].get('0x101',0),x['lines'],'calls',len(x['calls']),'dispatch',x['dispatch_sites'],'simd',x['simd'],'global',x['guest_constants'])

if __name__=='__main__':main()
