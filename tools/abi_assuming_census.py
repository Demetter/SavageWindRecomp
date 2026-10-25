"""Conservative CFG certificate for the FINAL isolated ABI sizing variant.

No generated/game files are modified. Unknown control flow or ABI contracts
fail closed. Counts are certified lower bounds, not a claim that all rejected
functions actually violate AAPCS. x18 is platform-specific: keep it unmasked
and leave functions writing it conservative in this experiment.
"""
import bisect, collections, json, re, sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
from capstone.arm64_const import ARM64_OP_IMM, ARM64_OP_FP, ARM64_OP_REG, ARM64_VAS_8B, ARM64_VAS_4H, ARM64_VAS_2S, ARM64_VAS_1D
from abi_sizing_census import ROOT, OUT
from nso import Nso
from analyze import eh_frame_extents
from layout import read

md=Cs(CS_ARCH_ARM64,CS_MODE_ARM); md.detail=True
def canonical(n):
    if n=='fp': return 'x29'
    if n=='lr': return 'x30'
    if re.fullmatch('[wx][0-9]+',n): return 'x'+n[1:]
    if re.fullmatch('[bhsdqv][0-9]+',n): return 'v'+n[1:]
    return n

def inspect(nso,base,a,b):
    ins={i.address:i for i in md.disasm(nso.image[a-base:b-base],a)}
    pending=[a]; reachable={}; reasons=set(); calls=set(); vreferred=set()
    while pending:
        p=pending.pop()
        if p in reachable: continue
        i=ins.get(p)
        if i is None: reasons.add('undecoded_or_outside_extent');continue
        raw_reads,raw_writes=i.regs_access()
        reads,writes=map(lambda rs:{canonical(i.reg_name(r)) for r in rs},(raw_reads,raw_writes))
        # Only the low 64 bits of v8..v15 survive a call. Keep upper-half
        # liveness separate so a full-width use cannot borrow that guarantee.
        for reg in raw_reads:
            rn=i.reg_name(reg)
            if re.fullmatch('[qv](?:8|9|1[0-5])',rn):
                operands=[o for o in i.operands if o.type==ARM64_OP_REG and o.reg==reg and o.access&1]
                low_only=operands and all(o.vas in (ARM64_VAS_8B,ARM64_VAS_4H,ARM64_VAS_2S,ARM64_VAS_1D) and o.vector_index<0 for o in operands)
                if not low_only:reads.add('vh'+rn[1:])
        for reg in raw_writes:
            rn=canonical(i.reg_name(reg))
            if rn in {f'v{k}' for k in range(8,16)}:writes.add('vh'+rn[1:])
        # Capstone 5's alias metadata incorrectly reads NZCV for CBZ/CBNZ and
        # the destination for non-inserting bitfield aliases (LSR/UBFX/etc.).
        # Correct from the encoding; BFM really DOES read its destination.
        if i.mnemonic in ('cbz','cbnz'): reads.discard('nzcv')
        if i.mnemonic=='fmov' and len(i.operands)==2 and i.operands[1].type==ARM64_OP_FP:
            reads.discard(canonical(i.reg_name(i.operands[0].reg)))
        word=int.from_bytes(i.bytes,'little')
        if word&0x1f800000==0x13000000 and (word>>29)&3 in (0,2):
            rd=word&31;rn=(word>>5)&31
            if rd!=rn:reads.discard('x'+str(rd))
        vreferred|={int(r[1:]) for r in reads|writes if r.startswith('v') and r[1:].isdigit()}
        m=i.mnemonic; successors=[]; call=None
        if m=='ret':
            if i.op_str not in ('','x30'): reasons.add('nonstandard_return')
        elif m=='bl':
            call=int(i.operands[0].imm);calls.add('f_'+format(call,'x'));successors=[p+4]
        elif m=='b' or m.startswith('b.') or m in ('cbz','cbnz','tbz','tbnz'):
            target=i.operands[-1].imm if i.operands[-1].type==ARM64_OP_IMM else -1
            successors=[target]+([] if m=='b' else [p+4])
            if not a<=target<b: reasons.add('tail_or_external_branch')
        elif m in ('br','blr','eret','drps') or m.startswith(('ret','bra','blr')):
            reasons.add('indirect_or_nonstandard_control')
        elif m in ('brk','udf','svc','hvc','smc'):
            reasons.add('trap_or_runtime_boundary')
        else: successors=[p+4]
        reachable[p]=(reads,writes,successors,call)
        pending.extend(q for q in successors if a<=q<b)
        if any(not a<=q<b for q in successors):reasons.add('tail_or_external_branch')
    if not any(ins[p].mnemonic=='ret' for p in reachable): reasons.add('no_standard_return')
    if any('x18' in w for r,w,s,c in reachable.values()):reasons.add('x18_platform_contract_unproved')
    # Forward may-be-undefined analysis: unions at joins, fixed point for loops.
    # At entry x0..x7 and v0..v7 are arguments; callee-saved values are valid.
    # A call kills scratch GPRs/flags, but can return x0/x1 and v0..v7.
    initial={f'x{i}' for i in range(8,19)}|{f'v{i}' for i in range(16,32)}|{'nzcv'}
    states={a:initial}; queue=collections.deque([a]);bad=set();hits=[]
    while queue:
        p=queue.popleft()
        if p not in reachable:continue
        reads,writes,succ,call=reachable[p];u=states[p]
        missing=reads&u
        if missing:bad|=missing;hits.append((hex(p),sorted(missing)))
        after=u-writes
        if call:
            after|={f'x{i}' for i in range(2,19)}|{f'v{i}' for i in range(16,32)}|{f'vh{i}' for i in range(8,16)}|{'nzcv'}
            after-={'x0','x1','x30'}|{f'v{i}' for i in range(8)}
        for q in succ:
            if q not in reachable:continue
            merged=states.get(q,set())|after
            if q not in states or merged!=states[q]:states[q]=merged;queue.append(q)
    if bad:reasons.add('implicit_or_call_clobbered_input')
    return dict(reasons=sorted(reasons),missing_registers=sorted(bad),missing_sites=hits[:12],calls=sorted(calls),vectors=sorted(vreferred),reachable_instructions=len(reachable))

def main():
    records=json.loads((OUT/'census.json').read_text())['functions'];result={}
    # Reuse the completed census only for this narrow precision correction:
    # rescan every upper-vector rejection, then recompute caller closure.
    prior=json.loads((OUT/'abi-assuming-census.json').read_text())['functions'] if '--refine-vector-lanes' in sys.argv else {}
    for module,base,size,bid in read(ROOT/'generated/layout.txt'):
        nso=Nso(ROOT/'extracted/exefs'/module)
        extents={base+a:base+b for a,b in eh_frame_extents(nso) if b>a}
        addresses=sorted(int(n[2:],16) for n,r in records.items() if r['module']==module)
        for k,a in enumerate(addresses):
            name='f_'+format(a,'x');meta=records[name];reasons=[];b=extents.get(a)
            if name in prior and not any(x.startswith('vh') for x in prior[name].get('missing_registers',[])):
                r=prior[name];r['reasons']=[x for x in r['reasons'] if x!='callee_contract_unproved'];result[name]=r;continue
            if b is None:reasons.append('not_unwind_entry')
            elif k+1<len(addresses) and addresses[k+1]<b:reasons.append('interior_generated_entry')
            if meta['cold_fallback']:reasons.append('unresolved_interior_flow')
            if meta['setjmp']:reasons.append('setjmp_boundary')
            if reasons:r=dict(reasons=reasons,calls=sorted(set(meta['calls'])))
            else:r=inspect(nso,base,a,b)
            result[name]=r
        print(module,'scanned',len(addresses),flush=True)
    # A direct call is only trusted when the callee is certified too. This
    # intentionally excludes opaque HLE/indirect/private calling conventions.
    clean={n for n,r in result.items() if not r['reasons']}
    local=len(clean)
    while True:
        excluded={n for n in clean if any(c not in clean for c in result[n]['calls'])}
        if not excluded:break
        for n in excluded:result[n]['reasons'].append('callee_contract_unproved')
        clean-=excluded
    cluster=json.loads((OUT/'closed-prototype-snapshot/manifest.json').read_text())
    summary={'total_functions':len(result),'locally_certified':local,'certified_closed_abi':len(clean),
             'cluster_clean':[n for n in cluster['functions'] if n in clean],
             'qualified_roots':[n for n in cluster['roots'] if n in clean],
             'reasons':dict(collections.Counter(x for r in result.values() for x in r['reasons']))}
    for n,r in result.items():r['clean']=n in clean
    (OUT/'abi-assuming-census.json').write_text(json.dumps({'summary':summary,'functions':result}))
    print(json.dumps(summary,indent=2))
    for n in cluster['functions']:print(n,result[n]['reasons'],result[n].get('missing_registers',[]))
if __name__=='__main__':main()
