"""Native functions (BOTW-NATIVE-CODEGEN.md, change 3): which generated functions keep the general registers
x0-x29, sp and the flags in C locals, and the text rewrite that moves their code there.

AAPCS64, which the game's compiler followed: arguments in x0-x7 (x8: where a big result goes), results in x0-x1;
a function preserves x19-x29 and sp for its caller and may change x0-x18; no flags pass in or out of a call. A
function written that way needs from its caller only x0-x8 and sp, and gives back only x0-x1 (and x19-x29, sp as
they were). So in a native function every general register is a local the C compiler gives a host register, and
only those go through the Cpu at a call: the arguments, sp and x29 (the frame chain) to the callee, x0-x1 back,
x19-x29 kept in host registers across it. A function that doesn't keep to it - discovery's pieces of a split
function (they continue another piece with every register live), hand-written code, a function whose callers read
more of its registers - stays as before, with all registers in the Cpu: a native function stores all its locals
there before calling or jumping to one, and loads them back after a call.

analyze() decides from the lifted code (the C text each instruction became): its reads and writes of c->x[0..29]
and of the flags, the stores to [sp, ...] that save a register, and the control flow."""
import re

NREG = 30  # x0-x29 (x30, the link register, stays in the Cpu: every call writes it there)
# Which registers are locals: the callee-saved ones only. With x0-x18 too (ALL_REGS) a hot routine compiled to 9% more
# instructions, not fewer: they are dead after every call anyway, and in call-free code the compiler kept c->x[] in
# host registers already (the Cpu is restrict); 30 locals only made it spill more (f_7000267228: 3,068 -> 3,357).
ALL_REGS = False
# (bisecting) SW_NATIVE_VLOCAL=0: native functions keep V and FPCR in the Cpu
import os
V_LOCAL = os.environ.get('SW_NATIVE_VLOCAL', '1') != '0'
# (debugging) SW_NATIVE_CHECK=1: each native function checks at its returns and tail calls that sp and x19-x29 are
# what they were at entry (cpu.h rt_native_fail logs the first ones that aren't)
CHECK = os.environ.get('SW_NATIVE_CHECK') == '1'
CS = tuple(range(19, 30))  # x19-x28 callee-saved, x29 the frame pointer
ARGS = tuple(range(0, 9))  # x0-x7 arguments, x8 the indirect result address
RES = (0, 1)


def mask(rs):
    m = 0
    for r in rs:
        m |= 1 << r
    return m


M_CS, M_ARGS, M_RES = mask(CS), mask(ARGS), mask(RES)
M_TEMP = mask(range(9, 19))  # x9-x18: never passed in
M_CALLER = mask(range(0, 19))  # what a call may change
M_SVC = mask(range(0, 8))  # what the kernel returns in
RE_REG = re.compile(r'c->x\[(\d+)\](\s*=(?!=))?')
RE_FLAG_W = re.compile(r'\b(adds|subs|adcs|sbcs|logic)(32|64)\(|c->nzcv\s*=(?!=)')
RE_FLAG_R = re.compile(r'\bCOND_[A-Z]{2}\(c\)|\bFLAG_[NZCV]\(c\)|\b(adc|sbc|adcs|sbcs)(32|64)\(|(?<![>\w])c->nzcv(?!\s*=(?!=))')


def bit(r):
    return 1 << r if 0 <= r < NREG else 0


def names(m):
    return ','.join(str(r) for r in range(NREG) if m & bit(r))


def text_of(ins):
    """The C text an instruction reads and writes registers in (for branches: the condition or target)."""
    if ins.kind in ('op', 'trap'):
        return ins.code
    if ins.kind == 'bcond':
        return ins.cond
    if ins.kind in ('br', 'blr'):
        return str(ins.target)
    return ''


def regs(text):
    """(read mask, write mask) of x0-x29 in one instruction's C text: uses come before the writes."""
    use = dfn = 0
    for m in RE_REG.finditer(text):
        b = bit(int(m.group(1)))
        if m.group(2):
            dfn |= b
        else:
            use |= b
    return use, dfn


def flags(text):
    """(reads, writes) the flags; ccmp reads them first (its condition), then writes."""
    return RE_FLAG_R.search(text) is not None, RE_FLAG_W.search(text) is not None


def save_regs(w):
    """The callee-saved registers an instruction stores to [sp, ...] (STP, STR / STUR), as a mask: a prologue's
    saves."""
    if (w >> 26) & 1:  # (FP/SIMD)
        return 0
    if ((w >> 27) & 7) == 0b101 and not (w >> 22) & 1 and ((w >> 5) & 31) == 31:  # STP / STNP, any index mode
        return (bit(w & 31) | bit((w >> 10) & 31)) & M_CS
    if ((w >> 27) & 7) == 0b111 and ((w >> 22) & 3) == 0 and ((w >> 5) & 31) == 31:  # STR / STUR (imm, reg)
        return bit(w & 31) & M_CS
    return 0


class Result:
    __slots__ = ('ok', 'why', 'extra_results', 'calls')

    def __init__(self, ok, why='', extra_results=False, calls=()):
        self.ok, self.why, self.extra_results, self.calls = ok, why, extra_results, calls


def analyze(m, entry, end, tables, is_function):
    """Result for the C function entry..end of module m (gen.Module). is_function: entry is where a function of the
    game's begins (unwind info, a symbol, a relocation, a call, a taken address), not a piece discovery split off:
    only a function has a prologue, so only there is a store of a caller's register to [sp, ...] its save (in a
    piece it may spill a value the piece before computed). extra_results: the function reads a register other than
    x0-x1 that a call it made left (its callees, .calls, give back more than the convention: they stay in the Cpu)."""
    n = (end - entry) // 4
    calls = sorted({m.lift(a).target for a in range(entry, end, 4) if m.lift(a).kind == 'bl'})
    if any(t == 'all' for t in tables.values()):
        return Result(False, 'unresolved table', False, calls)
    info = []
    for i in range(n):
        a = entry + 4 * i
        ins = m.lift(a)
        t = text_of(ins)
        use, dfn = regs(t)
        fr, fw = flags(t)
        info.append((ins, use, dfn, save_regs(m.word(a)) if is_function else 0, fr, fw, 'rt_svc(' in t))
    # forward "set on every path" dataflow over the instructions: registers set (mask), flags set (bool), registers a
    # call left undefined (mask)
    state = [None] * n
    state[0] = (M_ARGS, False, 0)
    work = [0]
    live_in = 0
    flags_in = extra = False
    written = saved = 0

    def succ(i):
        ins = info[i][0]
        a = entry + 4 * i
        k = ins.kind
        out = []
        if k in ('op', 'asm', 'bl', 'blr', 'bcond') and i + 1 < n:
            out.append(i + 1)
        if k in ('b', 'bcond') and entry <= ins.target < end:
            out.append((ins.target - entry) // 4)
        if k == 'br':
            jt = tables.get(a)
            if jt:
                out += [(t - entry) // 4 for t in jt if entry <= t < end]
        return out

    while work:
        i = work.pop()
        d, f, clob = state[i]
        ins, use, dfn, sv, fr, fw, svc = info[i]
        sv_now = sv & ~d  # (storing a callee-saved register's entry value to [sp, ...] is its save, not a use)
        saved |= sv_now
        real = use & ~sv_now & ~d
        extra |= bool(real & clob)
        live_in |= real & ~clob
        if fr and not f:
            flags_in = True
        written |= dfn
        d2, f2, clob2 = d | dfn, f or fw, clob & ~dfn
        if svc:
            d2 |= M_SVC
        if ins.kind in ('bl', 'blr'):
            # (after a call: x0-x1 its results, x2-x18 undefined, x19-x29 kept; the flags are loaded back from the Cpu)
            d2, clob2, f2 = (d2 & ~M_CALLER) | M_RES, (clob2 | M_CALLER) & ~M_RES, True
        for j in succ(i):
            if state[j] is None:
                state[j] = (d2, f2, clob2)
                work.append(j)
            else:
                od, of, oc = state[j]
                nd, nf, nc = od & d2, of and f2, oc | clob2
                if (nd, nf, nc) != (od, of, oc):
                    state[j] = (nd, nf, nc)
                    work.append(j)
    extra = extra and ALL_REGS  # (with x0-x18 in the Cpu a caller reads its callees' registers there anyway)
    bad_in = live_in & ((M_CS | M_TEMP) if ALL_REGS else M_CS)
    if bad_in:
        return Result(False, 'reads x%s from its caller' % names(bad_in), extra, calls)
    if flags_in:
        return Result(False, 'reads flags from its caller', extra, calls)
    unsaved = written & ~saved & M_CS
    if unsaved:
        return Result(False, 'writes x%s without saving it' % names(unsaved), extra, calls)
    return Result(True, '', extra, calls)  # (extra results: its callees stay in the Cpu, gen.generate)


# ---- the rewrite of a native function's lifted code
RE_X = re.compile(r'c->x\[([0-9]|1[0-9]|2[0-9])\]' if ALL_REGS else r'c->x\[(19|2[0-9])\]')
RE_FLAGOP = re.compile(r'\b(adds|subs|adcs|sbcs|adc|sbc|logic)(32|64)\(c,')
RE_COND = re.compile(r'\b(COND_[A-Z]{2}|FLAG_[NZCV])\(c\)')


# A native function is called with the Cpu as cg. Its c is a Cpu of its own on its stack, of which only the V registers
# and FPCR are used: the SIMD lowering and simd.h's helpers keep working on c->v[] and c->fpcr, and since the address
# never leaves the function (the out-of-line slow paths get the FPCR's value, simd.h SIMD_SLOW_CPU), the compiler keeps
# those fields in host registers, across calls too. Everything else of the Cpu (x0-x18, x30, pc, the exclusive monitor,
# TLS, FPSR) stays cg's; the kernel, exclusive loads and stores, dispatch and calls get cg.
RE_CG = re.compile(r'(?<![\w>])c->(x\[\d+\]|(?:pc|tpidrro|tpidr|excl_addr|excl_val|thread|fpsr)\b)')
RE_CG_CALL = re.compile(r'\b(ldx\d+|stx\d+|rt_svc|rt_udf|rt_brk)\(c,')


def rewrite(code):
    code = RE_X.sub(r'x\1', code)
    code = RE_FLAGOP.sub(r'n\1\2(&fl,', code)
    code = RE_COND.sub(r'\1(&fl)', code)
    code = code.replace('c->nzcv', 'fl.nzcv')
    code = re.sub(r'c->sp\b', 'sp', code)
    code = RE_CG.sub(r'cg->\1', code)
    code = RE_CG_CALL.sub(r'\1(cg,', code)
    # MSR FPCR: the thread's (MXCSR, cg) and this function's copy
    return re.sub(r'rt_fpcr_write\(c, ([^;]*)\);', r'rt_fpcr_write(cg, \1); c->fpcr = cg->fpcr;', code)


# Placeholders the body is written with; expanded once the locals it uses are known (finish):
#   PRE_CALL / POST_CALL     around a call to a native function (or any blr): arguments, sp and x29 to the callee;
#                            x0-x1 and the flags back
#   PRE_SVC / POST_SVC       around a supervisor call: the same, x0-x7 back
#   PRE_TAIL                 before a tail call to a native function
#   PRE_LEGACY / POST_LEGACY around a call or jump to code that keeps its registers in the Cpu: all of them there and back
#   PRE_RET                  before returning: x0-x1, sp, the flags
# A function that ever stores its locals in the Cpu (PRE_LEGACY) has put values of its own in c->x[19..29], where its
# caller may keep its registers (a caller that isn't native): before it returns or tail-calls, it stores them again,
# by then restored to what they were at entry (its epilogue loaded them back), so the caller finds its own.
PRE_CALL, POST_CALL, PRE_SVC, POST_SVC, PRE_TAIL, PRE_LEGACY, POST_LEGACY, PRE_RET = (
    '/*@pre*/', '/*@post*/', '/*@presvc*/', '/*@postsvc*/', '/*@tail*/', '/*@spill*/', '/*@fill*/', '/*@ret*/')
# PRE_RT / PRE_RT_TAIL: before a call or tail call that may reach the runtime (an HLE function, or a forwarder - a PLT
# stub, a thunk - that jumps to one): the runtime reads a caller's x19 and x29 in places (nvnWindowSetPresentInterval:
# x19 is the frame loop's framework; hle/input.c), so x19-x29 go to the Cpu too
PRE_RT, PRE_RT_TAIL = '/*@prert*/', '/*@prerttail*/'
RE_LOCAL = re.compile(r'\b(x(?:[0-9]|1[0-9]|2[0-9])|sp|fl)\b' if ALL_REGS else r'\b(x(?:19|2[0-9])|sp|fl)\b')


def finish(lines):
    """Expands the placeholders and puts the locals' declarations after FN_ENTER_N (lines[1])."""
    body = '\n'.join(lines[2:])
    used = set(RE_LOCAL.findall(re.sub(r'/\*@\w+\*/|c->sp\b|c->x\[\d+\]|c->nzcv', '', body)))
    xs = [r for r in range(NREG) if f'x{r}' in used]
    has_sp, has_fl = 'sp' in used, 'fl' in used
    # (its own Cpu, for V and FPCR: c named, or one of simd.h's macros that use c)
    has_v = re.search(r'(?<![\w>])c\b|\b(VU|VREG|VQ|VW|VS8|VS16|VS32|VS64|VSF32|VSF64|VHI64|VGET|VSETL|FP_SETS|FP_SETD|'
                      r'SV_FOP_S|SV_FOP_D)\(', re.sub(r'/\*@\w+\*/', '', body)) is not None
    # (it stores its own x19-x29 in the Cpu somewhere - for code keeping its registers there, or before a call that
    # may reach the runtime: its returns and tail calls store them again, restored by then to their entry values)
    spills = PRE_LEGACY in body or PRE_RT in body or PRE_RT_TAIL in body
    if has_v and not V_LOCAL:  # (SW_NATIVE_VLOCAL=0: V and FPCR stay in the Cpu, c is cg)
        has_v, alias_c = False, True
    else:
        alias_c = False

    def st(rs):
        return ''.join(f' cg->x[{r}] = x{r};' for r in xs if r in rs)

    def ld(rs):
        return ''.join(f' x{r} = cg->x[{r}];' for r in xs if r in rs)

    def st_v(rs):  # (a store of a value unchanged since it was loaded from cg is dropped by the compiler)
        return ''.join(f' cg->v[{r}] = c->v[{r}];' for r in rs) if has_v else ''

    def ld_v(rs):
        return (''.join(f' c->v[{r}] = cg->v[{r}];' for r in rs) + ' c->fpcr = cg->fpcr;') if has_v else ''

    st_sp = ' cg->sp = sp;' if has_sp else ''
    st_fl = ' cg->nzcv = fl.nzcv;' if has_fl else ''
    ld_fl = ' fl.nzcv = cg->nzcv;' if has_fl else ''
    keep = CS if spills else (29,)  # (the frame chain always; x19-x28 again once spilled)
    v_args, v_res, v_cs, v_all = range(8), range(4), range(8, 16), range(32)
    # (after a call v0-v7 are loaded back, not only the results: the next call's stores of the ones not changed since
    # then are dropped)
    rep = {PRE_CALL: st(ARGS) + st((29,)) + st_sp + st_v(v_args), POST_CALL: ld(RES) + ld_fl + ld_v(v_args),
           PRE_RT: st(ARGS) + st(CS) + st_sp + st_v(v_args),
           PRE_RT_TAIL: st(ARGS) + st(CS) + st_sp + st_v(v_args) + (st_v(v_cs) if spills else ''),
           PRE_SVC: st(ARGS) + st_sp, POST_SVC: ld(range(8)) + ld_fl,
           PRE_TAIL: st(ARGS) + st(keep) + st_sp + st_v(v_args) + (st_v(v_cs) if spills else ''),
           PRE_LEGACY: st(range(NREG)) + st_sp + st_fl + st_v(v_all),
           POST_LEGACY: ld(range(NREG)) + ld_fl + ld_v(v_all),
           # (x29 always: every call stored this function's frame pointer in the Cpu, where a caller that keeps its
           # registers there has its own; by now it is the entry value again)
           PRE_RET: st(RES) + (st(CS) if spills else st((29,))) + st_sp + st_fl + st_v(v_res) +
           (st_v(v_cs) if spills else '')}
    if CHECK:
        fn = re.search(r'FN_ENTER_N\((0x[0-9a-f]+)ull\)', lines[1]).group(1)
        chk = (''.join(f' if (x{r} != x{r}_in) rt_native_fail({fn}, {r}, x{r}, x{r}_in);' for r in xs if r in CS) +
               (f' if (sp != sp_in) rt_native_fail({fn}, 31, sp, sp_in);' if has_sp else ''))
        for k in (PRE_RET, PRE_TAIL, PRE_RT_TAIL):
            rep[k] = chk + rep[k]
    out = lines[:2]
    decl = [f'uint64_t x{r} = cg->x[{r}];' for r in xs]
    if CHECK:
        decl += [f'const uint64_t x{r}_in = x{r};' for r in xs if r in CS] + (['const uint64_t sp_in = cg->sp;'] if has_sp else [])
    if has_sp:
        decl.append('uint64_t sp = cg->sp;')
    if has_fl:
        decl.append('NFlags fl = {cg->nzcv};')
    if alias_c:
        decl.append('Cpu *const c = cg;')
    if has_v:
        decl.append('Cpu c_; Cpu *const restrict c = &c_; c_.fpcr = cg->fpcr;' +
                    ''.join(f' c_.v[{r}] = cg->v[{r}];' for r in v_all))
    if decl:
        out.append('    ' + ' '.join(decl))
    for l in lines[2:]:
        for k, v in rep.items():
            if k in l:
                l = l.replace(k, v)
        out.append(l)
    return out
