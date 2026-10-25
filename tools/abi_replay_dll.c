// Isolated replay adapter. Original DLL is loaded read-only; unsupported HLE,
// scheduling, clock, or nonlocal-jump operations FAIL the replay, never fake a
// result. This is not a replacement runtime and cannot launch the game.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "gamelib.h"
#include "abi_types.h"
static const GameLibTables *tables;
static uint64_t dispatches, boundaries;
static void unsupported(const char *s) {fprintf(stderr,"unreplayable runtime boundary: %s\n",s);exit(101);}
GuestFn abi_lookup(uint64_t pc) {
    if(!tables) {
        const char *path=getenv("ABI_LIBRARY");
        HMODULE h=path?LoadLibraryA(path):NULL;
        if(!h) {fprintf(stderr,"DLL load failed %lu\n",GetLastError());exit(102);}
        GameLibEntry e=(GameLibEntry)(void*)GetProcAddress(h,GAMELIB_ENTRY);if(!e)exit(103);tables=e();
    }
    for(uint32_t i=0;i<*tables->nmodules;i++) {
        const GameLibModule *m=&tables->modules[i];uint64_t lo=0,hi=*m->count;
        if(pc<m->base||pc>=m->base+m->size)continue;
        while(lo<hi){uint64_t mid=(lo+hi)/2;if(m->addrs[mid]<pc)lo=mid+1;else hi=mid;}
        if(lo<*m->count&&m->addrs[lo]==pc)return m->fns[lo];
    }
    fprintf(stderr,"unreplayable dispatch target %llx\n",(unsigned long long)pc);exit(104);
}
void abi_guest_call(uint64_t pc,Cpu *c) {abi_lookup(pc)(c);}
void rt_dispatch(Cpu *c) {dispatches++;abi_lookup(c->pc)(c);}
void abi_boundary(void *opaque,uint64_t *regs,NativeFp *fp,uint64_t target) {
    boundaries++;Cpu *c=opaque;
    memcpy(c->x,regs,sizeof c->x);c->sp=regs[32];c->nzcv=fp->nzcv;c->pc=fp->pc;
    c->fpcr=fp->fpcr;c->fpsr=fp->fpsr;c->excl_addr=fp->excl_addr;c->excl_val=fp->excl_val;memcpy(c->v,fp->v,sizeof c->v);
    abi_guest_call(target,c);
    memcpy(regs,c->x,sizeof c->x);regs[32]=c->sp;fp->nzcv=c->nzcv;fp->pc=c->pc;
    fp->fpcr=c->fpcr;fp->fpsr=c->fpsr;fp->excl_addr=c->excl_addr;fp->excl_val=c->excl_val;memcpy(fp->v,c->v,sizeof c->v);
}
void abi_counts_reset(void) {dispatches=boundaries=0;}
void abi_counts_print(void) {fprintf(stderr,"replay_dispatches=%llu replay_boundaries=%llu\n",(unsigned long long)dispatches,(unsigned long long)boundaries);}
void rt_svc(Cpu *c,uint32_t i,uint64_t p) {(void)c;(void)i;(void)p;unsupported("rt_svc");}
uint64_t rt_ticks(void) {unsupported("rt_ticks");return 0;}
#define FAIL(name) void name(void) {unsupported(#name);}
FAIL(hle_longjmp) FAIL(rt_frame_wait) FAIL(rt_frustum) FAIL(rt_gui_speed)
FAIL(rt_opt_back) FAIL(rt_opt_decide) FAIL(rt_opt_flag_type) FAIL(rt_opt_msg)
FAIL(rt_opt_pause_tick) FAIL(rt_opt_read) FAIL(rt_opt_reset) FAIL(rt_opt_rows)
FAIL(rt_opt_screen) FAIL(rt_opt_store) FAIL(rt_opt_tick) FAIL(rt_opt_title_tick)
FAIL(rt_persp) FAIL(rt_persp_offset) FAIL(rt_quit_dialog_back)
FAIL(rt_quit_dialog_begin) FAIL(rt_quit_dialog_texts) FAIL(rt_setjmp_slot)
FAIL(rt_title_event) FAIL(rt_title_msg) FAIL(rt_vfr_update)
