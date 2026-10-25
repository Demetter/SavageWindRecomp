"""Isolated sizing generator. Never writes the normal generated tree or game DLL.

--native-abi emits explicit scalar GPR arguments and aggregate C results; direct
callees receive those values, not Cpu*. FP helpers keep a *local* NativeFp value.
Full boundary state is preserved, including scratch registers and final flags.
This deliberately does NOT assume AAPCS makes scratch state unobservable.
"""
import argparse, json, pathlib, re
from abi_sizing_census import FUNCTION, ROOT, OUT

def generate(roots, native, boundary_depth=None, output='prototype'):
    rec=json.loads((OUT/'census.json').read_text())['functions']
    funcs={}; source_cache={}
    def collect(n,depth=0):
        if n in funcs:return
        if boundary_depth is not None and depth>boundary_depth:return
        meta=rec[n];path=meta['path']
        if path not in source_cache:
            source_cache[path]={m[1]:m[2] for m in FUNCTION.finditer((ROOT/path).read_text())}
        body=source_cache[path][n]
        bad = bool(re.search(r'rt_(?!udf'+('|dispatch' if boundary_depth is not None else '')+r')|hle_|__asm__|tpidr|setjmp',body) or rec[n]['cold_fallback'])
        if boundary_depth is None:bad |= 'c->pc' in body
        if bad:
            if n in roots:raise ValueError(f'{n}: unsupported native root; must remain a Cpu boundary')
            return
        funcs[n]=body
        for callee in set(meta['calls']):collect(callee,depth+1)
    for n in roots:collect(n)
    path=OUT/output;path.mkdir(exist_ok=True)
    body=['#include "cpu.h"']
    for n in funcs:body.append(f'void o_{n[2:]}(Cpu *c);')
    for n,b in funcs.items():
        b=re.sub(r'\bf_([0-9a-f]+)\(c\)',r'o_\1(c)',b)
        body.append(f'void o_{n[2:]}(Cpu *restrict c) {{\n{b}}}')
    (path/'old.c').write_text('\n'.join(body))
    if not native:return
    regs={n:set(map(int,re.findall(r'c->x\[(\d+)\]',b)))|{0,30} for n,b in funcs.items()}
    writes={n:set(map(int,re.findall(r'c->x\[(\d+)\]\s*=(?!=)',b))) for n,b in funcs.items()}
    native_calls={n:set(rec[n]['calls'])&funcs.keys() for n in funcs}
    has_boundary={n:('rt_dispatch(c)' in b or bool(set(rec[n]['calls'])-funcs.keys())) for n,b in funcs.items()}
    if boundary_depth is None and any(has_boundary.values()):raise ValueError('unclosed cluster')
    vectors={}
    for n,b in funcs.items():
        vs=set()
        vs.update(map(int,re.findall(r'c->v\[(\d+)\]',b)))
        vs.update(map(int,re.findall(r'VS\w*\((\d+)\)',b)))
        for pat in (r'VREG\(\w+,\s*(\d+)',r'VQ\(\w+,\s*\d+,\s*(\d+)',r'VW\(\w+,\s*\d+,\s*(\d+)',r'FP_SET\w*\((\d+)',r'SV_FOP_\w+\([^,]+,\s*\d+,\s*(\d+)'):
            vs.update(map(int,re.findall(pat,b)))
        for m in re.finditer(r'fp_(?:op|fma)_[sd]\(c,([^;]+?)\);',b):
            vs.update(int(a.strip()) for a in m[1].split(',') if a.strip().isdigit())
        for m in re.finditer(r'sl_(ldr|lds|sts)\(c,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),',b):
            count=int(m[4] if m[1]=='ldr' else m[3]);first=int(m[5]);vs.update((first+i)&31 for i in range(count))
        # General structure loads/stores may span several SIMD registers. Keep
        # all vector state until their exact range has been proved.
        if re.search(r'sl_(?!ldr|lds|sts)',b):vs.update(range(32))
        vectors[n]=vs
    effects={n:({ 'nzcv' } if rec[n]['flags_def'] else set())|({'sp'} if re.search(r'c->sp\s*=',b) else set())|({'excl_addr','excl_val'} if re.search(r'\b(?:ldx|stx)\d+\(c',b) else set()) for n,b in funcs.items()}
    for n in funcs:
        if has_boundary[n]:
            regs[n]=set(range(32));writes[n]=set(range(32));vectors[n]=set(range(32));effects[n].update(('sp','nzcv','fpcr','fpsr','pc','excl_addr','excl_val'))
        if 'c->pc' in funcs[n]:effects[n].add('pc')
    for _ in range(len(funcs)+1):
        changed=False
        for n in funcs:
            for table in (regs,writes,vectors,effects):
                new=table[n]|set().union(*(table[c] for c in native_calls[n]))
                if new!=table[n]:table[n]=new;changed=True
        if not changed:break
    regs={n:sorted(rs) for n,rs in regs.items()}
    writes={n:sorted(rs) for n,rs in writes.items()}
    vectors={n:sorted(rs) for n,rs in vectors.items()}
    types=['#pragma once','#include <stdint.h>',
           'typedef struct { uint64_t nzcv, fpcr, fpsr, excl_addr, excl_val, pc; __uint128_t v[32]; } NativeFp;']
    fp_inputs=('nzcv','fpcr','fpsr','pc','excl_addr','excl_val')
    def args(n,decl):
        return ', '.join((['void *boundary' if decl else 'boundary'] if boundary_depth is not None else [])+[('uint64_t ' if decl else '')+f'x{i}' for i in regs[n]]+['uint64_t sp' if decl else 'sp']+
                         [('__uint128_t v'+str(i)) if decl else f'fp.v[{i}]' for i in vectors[n]]+
                         [('uint64_t in_'+k) if decl else 'fp.'+k for k in fp_inputs])
    def field_expr(k):return k if k=='sp' else 'fp.'+k
    def pack(n):return '('+'R_'+n[2:]+'){'+', '.join([f'.x{i}=x{i}' for i in writes[n]]+[f'.v{i}=fp.v[{i}]' for i in vectors[n]]+[f'.{k}={field_expr(k)}' for k in sorted(effects[n])]+['.dummy=0'])+'}'
    for n in funcs:
        fields=' '.join([f'uint64_t x{i};' for i in writes[n]]+[f'__uint128_t v{i};' for i in vectors[n]]+[f'uint64_t {k};' for k in sorted(effects[n])])
        types.append(f'typedef struct {{ {fields} uint64_t dummy; }} R_{n[2:]};')
        types.append(f'static R_{n[2:]} n_{n[2:]}({args(n,True)});')
    (path/'abi_types.h').write_text('\n'.join(types)+'\n')
    headers=path/'native-headers';headers.mkdir(exist_ok=True)
    h=(ROOT/'runtime/cpu.h').read_text()
    h=re.sub(r'typedef struct __attribute__\(\(aligned\(16\)\)\) Cpu \{.*?\} Cpu;', '#include "../abi_types.h"',h,flags=re.S)
    h=re.sub(r'_Static_assert\(.*?;','',h)
    h=re.sub(r'\bCpu\b','NativeFp',h)
    (headers/'cpu.h').write_text(h)
    for file in ('simd.h','simd_fp.h','simd_softfma.h'):
        (headers/file).write_text(re.sub(r'\bCpu\b','NativeFp',(ROOT/'runtime'/file).read_text()))
    native_code=['#include "native-headers/cpu.h"','#undef FN_ENTER',
                 '#define FN_ENTER(a) do { if (__builtin_expect(g_fnprof,0)) fnprof_hit(a,x30,x0); } while(0)']
    if boundary_depth is not None:native_code.append('void abi_boundary(void*, uint64_t*, NativeFp*, uint64_t);')
    for n,b in reversed(list(funcs.items())):
        b=re.sub(r'c->x\[(\d+)\]',r'x\1',b).replace('c->sp','sp')
        # c names only the local flags/SIMD helper state, with no GPR file.
        def boundary_call(target,tail=False):
            text='{ uint64_t regs_[33]={'+','.join(f'x{i}' for i in range(32))+',sp}; abi_boundary(boundary,regs_,&fp,'+target+'); '
            text+=' '.join(f'x{i}=regs_[{i}];' for i in range(32))+' sp=regs_[32]; }'
            return text+(' return '+pack(n)+';' if tail else '')
        def call(m):
            callee=m[1];id=callee[2:]
            tail=bool(m[0].startswith('__attribute__'))
            if callee not in funcs:return boundary_call('0x'+id+'ull',tail)
            assigns=' '.join([f'x{i}=r_.x{i};' for i in writes[callee]]+[f'fp.v[{i}]=r_.v{i};' for i in vectors[callee]]+[f'{field_expr(k)}=r_.{k};' for k in sorted(effects[callee])])
            return '{ R_'+id+' r_ = n_'+id+'('+args(callee,False)+'); '+assigns+' }'+(' return '+pack(n)+';' if tail else '')
        b=re.sub(r'(?:__attribute__\(\(musttail\)\) return )?\b(f_[0-9a-f]+)\(c\);',call,b)
        b=re.sub(r'(?:__attribute__\(\(musttail\)\) return )?rt_dispatch\(c\);',lambda m:boundary_call('fp.pc',m[0].startswith('__attribute__')),b)
        b=re.sub(r'\breturn;', 'return '+pack(n)+';',b)
        attr='__attribute__((always_inline)) ' if n in roots else ''
        init='NativeFp fp={'+','.join('.'+k+'=in_'+k for k in fp_inputs)+',.v={'+','.join(f'[{i}]=v{i}' for i in vectors[n])+'}}; NativeFp *c=&fp;'
        native_code.append(f'static {attr}R_{n[2:]} n_{n[2:]}({args(n,True)}) {{ {init}\n{b}}}')
    # Boundary wrapper is in this TU so the root's C-value interface is fully
    # scalar-replaced. Otherwise the sizing accidentally measures a redundant
    # root aggregate round trip that production would never need.
    bridge=['struct Cpu;']
    for n in roots:
        ident=n[2:]
        bridge.append(f'void b_{ident}(struct Cpu *opaque) {{ uint64_t *raw=(uint64_t*)(void*)opaque;')
        values=', '.join((['opaque'] if boundary_depth is not None else [])+[f'raw[{i}]' for i in regs[n]]+['raw[32]']+[f'*(__uint128_t*)((char*)opaque+{512+16*i})' for i in vectors[n]]+['raw[34]','raw[40]','raw[41]','raw[33]','raw[37]','raw[38]'])
        bridge.append(f'R_{ident} r=n_{ident}({values});')
        bridge.append(' '.join(f'raw[{i}]=r.x{i};' for i in writes[n]))
        bridge.append(' '.join(f'*(__uint128_t*)((char*)opaque+{512+16*i})=r.v{i};' for i in vectors[n]))
        offsets={'sp':32,'nzcv':34,'fpcr':40,'fpsr':41,'pc':33,'excl_addr':37,'excl_val':38}
        bridge.append(' '.join(f'raw[{offsets[k]}]=r.{k};' for k in sorted(effects[n]))+' }')
    (path/'native.c').write_text('\n'.join(native_code+bridge))
    (path/'bridge.c').write_text('// Boundary wrappers are emitted into native.c for scalar replacement.\n')
    api=['#pragma once','#include "cpu.h"']
    for n in roots:
        api.extend([f'void o_{n[2:]}(Cpu*);',f'void b_{n[2:]}(Cpu*);'])
    api.append('typedef struct { const char *name; GuestFn old, native; } ProbeEntry;')
    api.append('static const ProbeEntry entries[] = {')
    for n in roots:api.append(f'{{"{n[2:]}",o_{n[2:]},b_{n[2:]}}},')
    api.append('};')
    (path/'entries.h').write_text('\n'.join(api))
    legacy=['#include "cpu.h"','void abi_guest_call(uint64_t,Cpu*);']+[f'void o_{n[2:]}(Cpu *c) {{ abi_guest_call(0x{n[2:]}ull,c); }}' for n in roots]
    (path/'legacy_roots.c').write_text('\n'.join(legacy))
    manifest={'roots':roots,'functions':list(funcs),'boundary_depth':boundary_depth,'gpr_parameters':regs,'gpr_results':writes,'vector_parameters_results':vectors,'effects':{n:sorted(v) for n,v in effects.items()},'native_abi':True,
              'limits':['Prototype retains original volatile guest memory including stack spills.', 'FP/SIMD remains local helper state; general registers are explicit C arguments/locals/results.', 'Aggregate results preserve ALL boundary state; no unsound caller-saved-register assumption.', 'No production generator or DLL changes.']}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('generated',len(funcs),'functions',path)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--functions',required=True,nargs='+');p.add_argument('--native-abi',action='store_true');p.add_argument('--boundary-depth',type=int);p.add_argument('--output',default='prototype');a=p.parse_args()
    generate([n if n.startswith('f_') else 'f_'+n for n in a.functions],a.native_abi,a.boundary_depth,a.output)
