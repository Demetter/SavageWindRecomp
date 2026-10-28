// (diagnostics) SAVAGEWIND_VK_PTCL_PROBE=1: could the game's compute dispatches inside a pass all run before the pass's
// draws, as one batch? BOTW's GPU particles interleave them with drawing, per emitter: "compute binds, DISPATCH 1x1x1,
// BARRIER 1, graphics state, its draws" - each dispatch ends the render pass and waits for all before it. Per
// dispatch, against the draws and dispatches since the pass's targets were set (the group): does it write what those
// draws read (vertex, index, uniform, storage ranges; sampled textures by texel address), sample one of the group's
// targets, follow a draw that stores to memory, or read or write what an earlier dispatch of the group wrote (they
// would then need a barrier between them). Memory-ordering ops (copies, clears of textures or buffers, fences, inline
// uniform updates, counters, barriers other than the particle one) end the group. "[ptcl-probe]" every 600 frames;
// the first hazards of each kind logged once with their ranges. Render thread only (draw_front, dispatch_compute).
static int ptp_on = -1;
typedef struct { uint64_t lo, hi; } PpR;
#define PP_MAX 16384
static PpR pp_rd[PP_MAX];  // the group's draws' reads
static uint8_t pp_rk[PP_MAX];  // ... what each is: 0 vertex or index, 1 uniform, 2 storage
static int pp_nrd;
static uint64_t pp_rd_lo = ~0ull, pp_rd_hi;
static uint64_t pp_tex[1024];  // texel addresses the group's draws sample
static int pp_ntex;
static PpR pp_dw[512], pp_dr[1024];  // the group's dispatches' writes, reads
static int pp_ndw, pp_ndr;
static uint64_t pp_rt[9];  // the group's targets (texel addresses)
static int pp_nrt = -1;    // (-1: no group)
static int pp_draws, pp_sites, pp_disp;
static bool pp_gwrite;     // a draw of the group stores to memory
static bool pp_dwrite;     // a draw of the group writes depth
static int pp_draws_since; // draws since the group's last dispatch (the trace)
static uint64_t pp_trace_frame;  // SAVAGEWIND_VK_PTCL_PROBE=<frame> (> 1): each dispatch of that frame logged
static uint64_t pp_frame_logged;
enum { PP_DISPATCH, PP_SITE, PP_WAR, PP_RT, PP_GW, PP_DD, PP_IMG, PP_GROUPS, PP_GROUP_SITES, PP_MAX_SITES,
       PP_UUB, PP_FIRST, PP_WAR_VI, PP_WAR_UB, PP_WAR_SB, PP_WAR_TEX, PP_CLEAN, PP_SAME_PROG, PP_RT_DEPTH,
       PP_RT_DEPTH_WRITTEN, PP_RT_COLOR, PP_DD_SBUF, PP_DD_UBUF, PP_DD_WAR, PP_N };
static uint64_t pp_stat[PP_N];
static const char *const pp_names[PP_N] = {"dispatches", "after draws (sites)", "write what a draw read",
    "sample a target", "after a draw storing", "need a barrier between dispatches", "bind images",
    "groups with sites", "sites in them", "most sites in a group", "groups ended by a uniform update", "before any draw",
    "(write: vertex/index)", "(write: uniform)", "(write: storage)", "(write: sampled texture)", "sites with no hazard",
    "same compute program as the last", "(target: the depth)", "(... written by a draw before)", "(target: a colour one)",
    "(between: storage over a write)", "(between: uniform over a write)", "(between: a write over a read)"};
static uint32_t pp_logged;  // hazard kinds logged once

static bool pp_overlap(const PpR *a, int n, uint64_t lo, uint64_t hi) {
    for (int i = 0; i < n; i++) if (lo < a[i].hi && a[i].lo < hi) return true;
    return false;
}

static void pp_group_end(void) {
    if (pp_sites) {
        pp_stat[PP_GROUPS]++;
        pp_stat[PP_GROUP_SITES] += (uint64_t)pp_sites;
        if ((uint64_t)pp_sites > pp_stat[PP_MAX_SITES]) pp_stat[PP_MAX_SITES] = (uint64_t)pp_sites;
    }
    pp_nrd = pp_ntex = pp_ndw = pp_ndr = 0;
    pp_rd_lo = ~0ull, pp_rd_hi = 0;
    pp_nrt = -1;
    pp_draws = pp_sites = pp_disp = 0;
    pp_gwrite = pp_dwrite = false;
    pp_draws_since = 0;
}

static int pp_rt_now(uint64_t *rt) {
    int n = 0;
    for (int i = 0; i < st.rt.ncolor && i < 8; i++) rt[n++] = st.rt.color[i] ? st.rt.color[i]->addr : 0;
    rt[n++] = st.rt.depth ? st.rt.depth->addr : 0;
    return n;
}

static bool pp_same_rt(void) {
    uint64_t rt[9];
    int n = pp_rt_now(rt);
    return n == pp_nrt && !memcmp(rt, pp_rt, sizeof rt[0] * (size_t)n);
}

static void pp_join(void) {  // (a draw or dispatch) the group of the current targets
    if (pp_nrt >= 0 && pp_same_rt()) return;
    pp_group_end();
    pp_nrt = pp_rt_now(pp_rt);
}

static void pp_read(uint64_t lo, uint64_t size, int kind) {
    if (!lo || !size || pp_nrd >= PP_MAX) return;
    uint64_t hi = lo + size;
    if (pp_nrd && pp_rd[pp_nrd - 1].lo == lo && pp_rd[pp_nrd - 1].hi == hi) return;
    pp_rk[pp_nrd] = (uint8_t)kind;
    pp_rd[pp_nrd++] = (PpR){lo, hi};
    if (lo < pp_rd_lo) pp_rd_lo = lo;
    if (hi > pp_rd_hi) pp_rd_hi = hi;
}

static void pp_report(void) {
    uint64_t fr = renderer_present_count();
    if (fr < pp_frame_logged + 600) return;
    pp_frame_logged = fr;
    LOG("[ptcl-probe] frame %llu:", (unsigned long long)fr);
    for (int i = 0; i < PP_N; i++) LOG(" %s %llu;", pp_names[i], (unsigned long long)pp_stat[i]);
    LOG("\n");
}

static void pp_op(CmdRec *r) {  // (exec_list, each record) ops that order memory end the group
    if (ptp_on < 0) {
        const char *e = getenv("SAVAGEWIND_VK_PTCL_PROBE");
        ptp_on = e ? 1 : 0;
        pp_trace_frame = e && strtoull(e, NULL, 10) > 1 ? strtoull(e, NULL, 10) : 0;
    }
    if (!ptp_on) return;
    switch (r->op) {
    case CMD_CLEAR_TEXTURE: case CMD_CLEAR_BUFFER: case CMD_COPY_BUFFER_TO_BUFFER: case CMD_COPY_BUFFER_TO_TEXTURE:
    case CMD_COPY_TEXTURE_TO_BUFFER: case CMD_COPY_TEXTURE_TO_TEXTURE: case CMD_DOWNSAMPLE: case CMD_FENCE_SYNC:
    case CMD_WAIT_SYNC: case CMD_SIGNAL_EVENT: case CMD_WAIT_EVENT: case CMD_REPORT_COUNTER: case CMD_RESOLVE_DEPTH:
    case CMD_DISPATCH_INDIRECT: case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
        pp_group_end();
        break;
    case CMD_UPDATE_UNIFORM_BUFFER:
        if (pp_disp) pp_stat[PP_UUB]++;
        pp_group_end();
        break;
    case CMD_BARRIER:
        if (r->x[0] != 1) pp_group_end();
        break;
    case CMD_END:
        pp_report();
        break;
    default: break;
    }
}

static void pp_draw(CmdRec *r, bool indexed, Prog *pr) {  // (draw_, after draw_front) its reads into the group
    pp_join();
    pp_draws++;
    pp_draws_since++;
    if (st.ds[0] && st.ds[1] && st.rt.depth) pp_dwrite = true;
    for (int i = 0; i < st.nstream && i < 16; i++) pp_read(st.vbuf[i].addr, st.vbuf[i].size, 0);
    if (indexed) pp_read(r->x[3], (uint64_t)r->x[2] * (r->x[1] == 0 ? 1 : r->x[1] == 1 ? 2 : 4), 0);
    ShaderInfo *si[2] = {pr->vi, pr->fi};
    for (int s = 0; s < 2; s++) {
        if (!si[s]) continue;
        for (int k = 0; k < 18; k++)
            if (si[s]->cbuf_mask & (1u << k)) pp_read(st.ubuf[s][k].addr, st.ubuf[s][k].size, 1);
        if (si[s]->uses_gmem)
            for (int k = 0; k < 16; k++) pp_read(st.sbuf[s][k].addr, st.sbuf[s][k].size, 2);
        if (si[s]->writes_gmem) pp_gwrite = true;
        for (int k = 0; k < si[s]->ntex && pp_ntex < 1024; k++) {
            NvnTexture *t = shader_texture_object(s, &si[s]->tex[k]);
            if (t && t->addr) pp_tex[pp_ntex++] = t->addr;
        }
    }
}

static void pp_hazard(int kind, ShaderInfo *si, uint64_t lo, uint64_t hi) {
    pp_stat[kind]++;
    if (pp_logged & (1u << kind)) return;
    pp_logged |= 1u << kind;
    LOG("[ptcl-probe] first '%s': frame %llu cs %llx range %llx..%llx (group: %d draws, %d dispatches before, %d reads "
        "%llx..%llx)\n", pp_names[kind], (unsigned long long)renderer_present_count(),
        (unsigned long long)st.program->stages[5].code_addr, (unsigned long long)lo, (unsigned long long)hi, pp_draws,
        pp_disp, pp_nrd, (unsigned long long)pp_rd_lo, (unsigned long long)pp_rd_hi);
    (void)si;
}

static void pp_dispatch(ShaderInfo *si) {  // (dispatch_compute) against the group so far, then into it
    static uint64_t last_cs;
    pp_join();
    pp_stat[PP_DISPATCH]++;
    uint64_t cs = st.program->stages[5].code_addr;
    if (cs == last_cs) pp_stat[PP_SAME_PROG]++;
    last_cs = cs;
    if (!pp_draws) pp_stat[PP_FIRST]++;
    else {
        pp_stat[PP_SITE]++;
        pp_sites++;
        uint64_t before = pp_stat[PP_WAR] + pp_stat[PP_RT] + pp_stat[PP_GW] + pp_stat[PP_DD];
        bool war = false;
        if (si->writes_gmem)
            for (int k = 0; k < 16 && !war; k++) {
                Range w = st.sbuf[5][k];
                if (!w.addr || !w.size || w.addr >= pp_rd_hi || w.addr + w.size <= pp_rd_lo) continue;
                for (int i = 0; i < pp_nrd; i++)
                    if (w.addr < pp_rd[i].hi && pp_rd[i].lo < w.addr + w.size) {
                        pp_hazard(PP_WAR, si, w.addr, w.addr + w.size);
                        pp_stat[PP_WAR_VI + pp_rk[i]]++;
                        war = true;
                        break;
                    }
            }
        for (int k = 0; k < si->ntex; k++) {
            NvnTexture *t = shader_texture_object(5, &si->tex[k]);
            if (!t || !t->addr) continue;
            for (int i = 0; i < pp_nrt; i++)
                if (pp_rt[i] == t->addr) {
                    bool depth = i == pp_nrt - 1;
                    pp_stat[depth ? PP_RT_DEPTH : PP_RT_COLOR]++;
                    if (depth && pp_dwrite) pp_stat[PP_RT_DEPTH_WRITTEN]++;
                    pp_hazard(PP_RT, si, t->addr, t->addr);
                    break;
                }
            if (si->tex[k].is_image)
                for (int i = 0; i < pp_ntex; i++)
                    if (pp_tex[i] == t->addr) { pp_hazard(PP_WAR_TEX, si, t->addr, t->addr); break; }
        }
        if (pp_gwrite) pp_hazard(PP_GW, si, 0, 0);
        if (pp_stat[PP_WAR] + pp_stat[PP_RT] + pp_stat[PP_GW] + pp_stat[PP_DD] == before) pp_stat[PP_CLEAN]++;
    }
    // between the group's dispatches (sites or not): reads or writes over an earlier one's writes, writes over its reads
    bool dd = false;
    for (int k = 0; k < 16 && !dd; k++) {
        Range b = st.sbuf[5][k];
        if (!b.addr || !b.size) continue;
        bool raw = pp_overlap(pp_dw, pp_ndw, b.addr, b.addr + b.size);
        bool war = si->writes_gmem && pp_overlap(pp_dr, pp_ndr, b.addr, b.addr + b.size);
        if (raw || war) pp_stat[raw ? PP_DD_SBUF : PP_DD_WAR]++, pp_hazard(PP_DD, si, b.addr, b.addr + b.size), dd = true;
    }
    for (int k = 0; k < 18 && !dd; k++) {
        Range u = st.ubuf[5][k];
        if (u.addr && u.size && ((si->cbuf_mask & (1u << k)) || si->uses_gmem) && pp_overlap(pp_dw, pp_ndw, u.addr, u.addr + u.size))
            pp_stat[PP_DD_UBUF]++, pp_hazard(PP_DD, si, u.addr, u.addr + u.size), dd = true;
    }
    if (pp_trace_frame && renderer_present_count() == pp_trace_frame) {
        LOG("[ptcl-trace] group rt %d (%llx..) draws %d since %d dwrite %d | cs %llx w %d gmem %d dd %d sbuf:", pp_nrt,
            (unsigned long long)pp_rt[0], pp_draws, pp_draws_since, pp_dwrite, (unsigned long long)st.program->stages[5].code_addr,
            si->writes_gmem, si->uses_gmem, dd);
        for (int k = 0; k < 16; k++)
            if (st.sbuf[5][k].addr) LOG(" %d=%llx+%llx", k, (unsigned long long)st.sbuf[5][k].addr, (unsigned long long)st.sbuf[5][k].size);
        LOG(" ubuf:");
        for (int k = 0; k < 18; k++)
            if (st.ubuf[5][k].addr) LOG(" %d=%llx+%llx", k, (unsigned long long)st.ubuf[5][k].addr, (unsigned long long)st.ubuf[5][k].size);
        LOG(" tex:");
        for (int k = 0; k < si->ntex; k++) {
            NvnTexture *t = shader_texture_object(5, &si->tex[k]);
            int rt = -1;
            for (int i = 0; t && i < pp_nrt; i++) if (pp_rt[i] == t->addr) rt = i;
            LOG(" %llx%s", t ? (unsigned long long)t->addr : 0ull, rt < 0 ? "" : rt == pp_nrt - 1 ? "(depth)" : "(colour)");
        }
        LOG("\n");
    }
    pp_draws_since = 0;
    bool img = false;
    for (int k = 0; k < si->ntex; k++) img |= si->tex[k].is_image;
    if (img) pp_stat[PP_IMG]++;
    for (int k = 0; k < 16; k++) {
        Range b = st.sbuf[5][k];
        if (!b.addr || !b.size) continue;
        if (si->writes_gmem && pp_ndw < 512) pp_dw[pp_ndw++] = (PpR){b.addr, b.addr + b.size};
        if (pp_ndr < 1024) pp_dr[pp_ndr++] = (PpR){b.addr, b.addr + b.size};
    }
    for (int k = 0; k < 18; k++) {
        Range u = st.ubuf[5][k];
        if (u.addr && u.size && ((si->cbuf_mask & (1u << k)) || si->uses_gmem) && pp_ndr < 1024)
            pp_dr[pp_ndr++] = (PpR){u.addr, u.addr + u.size};
    }
    pp_disp++;
}
