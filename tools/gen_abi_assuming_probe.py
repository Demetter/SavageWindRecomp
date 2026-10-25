"""One final, isolated ABI-assuming variant of the saved 31-function cluster.

Qualified functions use x0..x7 scalar args and a scalar 128-bit x0/x1 result.
Scratch GPRs and flags are not returned. Saved registers are C locals. A small
call environment supplies guest stack/LR and initial callee-saved values ONLY
to retain the original guest stack spill bytes for exact memory verification.
SIMD values used by the cluster have scalar input/output parameters, not an
aggregate return. Uncertified roots retain the saved conservative generator.
"""
import json,re,shutil
from abi_sizing_census import ROOT,OUT,FUNCTION

def main():
    saved=OUT/'closed-prototype-snapshot';m=json.loads((saved/'manifest.json').read_text())
    cert=json.loads((OUT/'abi-assuming-census.json').read_text())
    clean=set(cert['summary']['cluster_clean']);out=OUT/'prototype-abi-assuming';out.mkdir(exist_ok=True)
    rec=json.loads((OUT/'census.json').read_text())['functions'];bodies={};cache={}
    for n in m['functions']:
        p=rec[n]['path']
        if p not in cache:cache[p]={x[1]:x[2] for x in FUNCTION.finditer((ROOT/p).read_text())}
        bodies[n]=cache[p][n]
    shutil.copy2(saved/'old.c',out/'old.c');shutil.copy2(saved/'abi_types.h',out/'abi_types.h')
    shutil.copytree(OUT/'prototype/native-headers',out/'native-headers',dirs_exist_ok=True)
    conservative=re.sub(r'\bb_([0-9a-f]+)\b',r'conservative_\1',(saved/'native.c').read_text())
    code=[conservative,
          'typedef struct { uint64_t sp, lr, saved[11], fpcr, excl_addr, excl_val; __uint128_t saved_v[8]; } NativeCall;']
    vs={n:[i for i in m['vector_parameters_results'][n] if i<8] for n in m['functions']}
    def signature(n):
        return ', '.join(['const NativeCall *env']+[f'uint64_t x{i}' for i in range(8)]+
                        [f'__uint128_t v{i}' for i in vs[n]]+[f'__uint128_t *out_v{i}' for i in vs[n]])
    def result(n):
        return '{ '+' '.join(f'*out_v{i}=fp.v[{i}];' for i in vs[n])+' return ((__uint128_t)x1<<64)|x0; }'
    for n in m['functions']:
        if n in clean:code.append(f'static __uint128_t a_{n[2:]}({signature(n)});')
    for n in reversed(m['functions']):
        if n not in clean:continue
        b=bodies[n]
        b=re.sub(r'c->x\[(\d+)\]',r'x\1',b).replace('c->sp','sp')
        def call(match):
            callee=match[1];assert callee in clean, (n,callee)
            env='{.sp=sp,.lr=x30,.saved={'+','.join(f'x{i}' for i in range(19,30))+'},.fpcr=fp.fpcr,.excl_addr=fp.excl_addr,.excl_val=fp.excl_val,.saved_v={'+','.join(f'fp.v[{i}]' for i in range(8,16))+'}}'
            args=['&next']+[f'x{i}' for i in range(8)]+[f'fp.v[{i}]' for i in vs[callee]]+[f'&fp.v[{i}]' for i in vs[callee]]
            return '{ NativeCall next='+env+'; __uint128_t pair=a_'+callee[2:]+'('+','.join(args)+'); x0=(uint64_t)pair; x1=(uint64_t)(pair>>64); }'
        b=re.sub(r'\b(f_[0-9a-f]+)\(c\);',call,b)
        assert 'musttail' not in b and not re.search(r'\bf_[0-9a-f]+\(c\)',b)
        b=re.sub(r'\breturn;',result(n),b)
        locals_='uint64_t sp=env->sp, x30=env->lr; '+''.join(f'uint64_t x{i}='+('0' if i<19 else f'env->saved[{i-19}]')+';' for i in range(8,30))
        init='NativeFp fp={.fpcr=env->fpcr,.excl_addr=env->excl_addr,.excl_val=env->excl_val,.v={'+','.join([f'[{i}]=v{i}' for i in vs[n]]+[f'[{i}]=env->saved_v[{i-8}]' for i in range(8,16)])+'}}; NativeFp *c=&fp;'
        attr='__attribute__((always_inline)) ' if n in m['roots'] else ''
        code.append(f'static {attr}__uint128_t a_{n[2:]}({signature(n)}) {{ {locals_}\n{init}\n{b}\n}}')
    for n in m['roots']:
        ident=n[2:]
        if n not in clean:
            code.append(f'void b_{ident}(struct Cpu *p) {{ conservative_{ident}(p); }}');continue
        env='{.sp=raw[32],.lr=raw[30],.saved={'+','.join(f'raw[{i}]' for i in range(19,30))+'},.fpcr=raw[40],.excl_addr=raw[37],.excl_val=raw[38],.saved_v={'+','.join(f'*(__uint128_t*)((char*)opaque+{512+16*i})' for i in range(8,16))+'}}'
        args=['&env']+[f'raw[{i}]' for i in range(8)]+[f'*(__uint128_t*)((char*)opaque+{512+16*i})' for i in vs[n]]+[f'(__uint128_t*)((char*)opaque+{512+16*i})' for i in vs[n]]
        code.append(f'void b_{ident}(struct Cpu *opaque) {{ uint64_t *raw=(uint64_t*)(void*)opaque; NativeCall env={env}; __uint128_t pair=a_{ident}('+','.join(args)+'); raw[0]=(uint64_t)pair; raw[1]=(uint64_t)(pair>>64); }')
    (out/'native.c').write_text('\n'.join(code))
    api=['#pragma once','#include "cpu.h"','#define ABI_ASSUMING_VARIANT 1']
    for n in m['roots']:api.extend([f'void o_{n[2:]}(Cpu*);',f'void b_{n[2:]}(Cpu*);'])
    api+=['typedef struct { const char *name; GuestFn old, native; int abi_clean; } ProbeEntry;','static const ProbeEntry entries[] = {']
    for n in m['roots']:api.append(f'{{"{n[2:]}",o_{n[2:]},b_{n[2:]},{int(n in clean)}}},')
    api+=['};'];(out/'entries.h').write_text('\n'.join(api))
    m.update(variant='final-abi-assuming',qualified_functions=sorted(clean),qualified_roots=[n for n in m['roots'] if n in clean],
             mask={'gpr':list(range(2,18)),'nzcv':True,'vectors_full':list(range(16,32)),'vectors_upper64':list(range(8,16))},
             gamewide_count=cert['summary']['certified_closed_abi'],
             limits=['x18 stays checked; functions writing x18 are uncertified.',
                     'Callee-saved/guest-SP/LR environment retained solely for exact guest stack byte fidelity.',
                     'All v0-v7 used by the body or callees are scalar inputs/outputs; no aggregate return on qualified path.',
                     'No new captured inputs and no production outputs.'])
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    # Separate harness file: the earlier full-state harness and its binaries
    # remain reproducible and unchanged.
    harness=(ROOT/'tools/abi_replay_harness.c').read_text()
    harness=harness.replace('static int compare(Cpu *a,Cpu *b) {',
        'static int compare(Cpu *a,Cpu *b,int abi_clean) {\n'
        '    if (abi_clean) {\n'
        '        for(int i=2;i<=17;i++) a->x[i]=b->x[i]=0;\n'
        '        a->nzcv=b->nzcv=0;\n'
        '        for(int i=16;i<32;i++) a->v[i]=b->v[i]=0;\n'
        '        for(int i=8;i<16;i++){a->v[i]&=(__uint128_t)UINT64_MAX;b->v[i]&=(__uint128_t)UINT64_MAX;}\n'
        '    }')
    harness=harness.replace('return compare(&a,&b);','return compare(&a,&b,p->abi_clean);')
    (out/'harness.c').write_text(harness)
    print('final variant',len(clean),'/',len(m['functions']),'qualified;',len(m['qualified_roots']),'roots')
if __name__=='__main__':main()
