// ---- (research, off) Dispatches inside a pass, run as one batch before its draws (SAVAGEWIND_VK_DISPATCH_BATCH=1).
// BOTW's GPU particles update each emitter just before drawing it, inside the pass that draws them: "dispatch, barrier
// 1, the emitter's draws", 19 and 15 times in Hyrule Castle's two translucent passes (28 Oct probes, particle_probe.inc,
// SAVAGEWIND_LOG_BARRIERS). A dispatch can't
// run inside a Vulkan render pass, so each one ended the pass, drained the GPU and began the pass again. A native
// renderer runs its particle simulation as one compute batch ahead of the pass it feeds (a render graph's order): at a
// dispatch inside a pass, the list ahead is read with the state it sets, up to the end of the pass, and each later
// dispatch of the pass runs now, right after this one, while it depends on nothing drawn in between - it writes
// nothing those draws read, samples no target they wrote (the scene depth only while none of them wrote depth), and
// uses nothing they may store to; the barriers between the batch's dispatches follow what they use
// (gb_dispatch_barrier). The walk of the list then skips the dispatches already run and the game's barrier after each,
// and the draws stay in one pass after one barrier. SAVAGEWIND_VK_DISPATCH_BATCH=0: off; =2: on in even frames only
// (an A/B within one run); SAVAGEWIND_PERFX=33554432 with _AB: off / on in blocks of frames. "[dispatch-batch]" every
// 600 frames: dispatches run ahead, batches, the look-ahead's time, records and draws a frame, why it ended.
// 28 Oct, 5120x2160 Hyrule Castle: 257 -> 284 fps between launches (passes a frame 168 -> 138), +6.75% in in-run
// blocks; Kakariko 720p in-run blocks: +0.017 ms a frame (within noise). OFF by default (research): the look-ahead
// re-reads each particle pass's records on the render thread - ~0.3 ms a frame at Kakariko, ~0.8 ms at the castle in
// the frames it runs - hidden where the GPU or the game thread is the limit, not where the render thread is. The shape
// without a look-ahead: the draws after a pass's first update recorded into a pass segment that later updates are
// recorded ahead of (deferred pass recording: BOTW-WORKLOG).
static int db_mode = -1;
static CmdRec *db_ahead[64];  // the dispatches run ahead, in list order
static int db_n, db_next;
static bool db_skip_barrier;  // the game's barrier after a dispatch run ahead: skipped
typedef struct { uint64_t lo, hi; } DbR;
static DbR db_rd[4096], db_wr[1024];  // the look-ahead's draws: ranges read, ranges they may store to
static int db_nrd, db_nwr;
enum { DB_END, DB_CALL, DB_TARGETS, DB_DEPTH, DB_COLOR, DB_READ, DB_STORE, DB_IMAGE, DB_UPDATE, DB_BARRIER, DB_CLEAR,
       DB_OTHER, DB_LONG, DB_FULL, DB_NOT_READY, DB_STOPS };
static const char *const db_why[DB_STOPS] = {"list end", "call", "other targets", "samples depth drawn since",
    "samples a target drawn since", "writes what a draw read", "uses what a draw may store to", "binds an image",
    "uniform update", "other barrier", "clear", "other command", "64 draws without one", "64 run ahead", "program not ready"};
static uint64_t db_stops[DB_STOPS], db_ran, db_batches, db_frame_logged;
static uint64_t db_ns, db_records, db_draws, db_frame0;  // (statistics) the look-ahead's own time, records and draws read

static bool db_on(void) {
    if (db_mode < 0) { const char *e = getenv("SAVAGEWIND_VK_DISPATCH_BATCH"); db_mode = e ? atoi(e) : 0; }
    if (perfx_trial(PERFX_DBATCH)) return perfx(PERFX_DBATCH);  // (SAVAGEWIND_PERFX=33554432 + _AB: blocks of frames)
    return db_mode == 2 ? !(renderer_present_count() & 1) : db_mode != 0;
}

// (exec_list, a dispatch) run already with the first of its pass: skip it, and the game's barrier after it
static bool db_take(CmdRec *r) {
    if (db_next < db_n && db_ahead[db_next] == r) { db_next++; db_skip_barrier = true; return true; }
    db_n = db_next = 0;  // (anything else: that batch is over)
    return false;
}

static bool db_overlap(const DbR *a, int n, uint64_t lo, uint64_t hi) {
    for (int i = 0; i < n; i++) if (lo < a[i].hi && a[i].lo < hi) return true;
    return false;
}
static bool db_add(DbR *a, int *n, int cap, uint64_t lo, uint64_t size) {
    if (!lo || !size) return true;
    if (*n && a[*n - 1].lo == lo && a[*n - 1].hi == lo + size) return true;
    if (*n >= cap) return false;
    a[(*n)++] = (DbR){lo, lo + size};
    return true;
}

// A draw ahead: what it reads (vertex, index, uniform and storage ranges) and may store to; false: no room
static bool db_draw(CmdRec *r, bool indexed, bool *depth_written, bool *drawn) {
    bool ok = true;
    for (int i = 0; i < st.nstream && i < 16; i++) ok &= db_add(db_rd, &db_nrd, 4096, st.vbuf[i].addr, st.vbuf[i].size);
    if (indexed) ok &= db_add(db_rd, &db_nrd, 4096, r->x[3], (uint64_t)r->x[2] * (r->x[1] == 0 ? 1 : r->x[1] == 1 ? 2 : 4));
    Prog *pr = st.program && (st.program->stage_mask & 3) == 3 ? state_program(nvn_window_origin(), nvn_depth_mode()) : NULL;
    bool known = pr && atomic_load(&pr->state) == 1 && pr->vi && pr->fi;
    ShaderInfo *si[2] = {known ? pr->vi : NULL, known ? pr->fi : NULL};
    for (int s = 0; s < 2; s++) {  // (a program not translated yet: everything bound, and its storage written)
        for (int k = 0; k < 18; k++)
            if (!si[s] || (si[s]->cbuf_mask & (1u << k))) ok &= db_add(db_rd, &db_nrd, 4096, st.ubuf[s][k].addr, st.ubuf[s][k].size);
        if (si[s] && !si[s]->uses_gmem) continue;
        for (int k = 0; k < 16; k++) {
            ok &= db_add(db_rd, &db_nrd, 4096, st.sbuf[s][k].addr, st.sbuf[s][k].size);
            if (!si[s] || si[s]->writes_gmem) ok &= db_add(db_wr, &db_nwr, 1024, st.sbuf[s][k].addr, st.sbuf[s][k].size);
        }
    }
    if (st.ds[0] && st.ds[1] && st.rt.depth) *depth_written = true;
    *drawn = true;
    return ok;
}

// A dispatch ahead: whether it can run now (before the draws read so far); DB_STOPS if so, else why not
static int db_check(ShaderInfo *si, bool depth_written, bool drawn) {
    for (int k = 0; k < si->ntex; k++) {
        if (si->tex[k].is_image) return DB_IMAGE;
        NvnTexture *t = shader_texture_object(5, &si->tex[k]);
        TexBackend *b = t ? t->backend : NULL;
        if (!b) continue;
        if (st.rt.depth && st.rt.depth->backend == b && depth_written) return DB_DEPTH;
        for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i] && st.rt.color[i]->backend == b && drawn) return DB_COLOR;
    }
    for (int k = 0; k < 16; k++) {
        Range b = st.sbuf[5][k];
        if (!b.addr || !b.size) continue;
        if (si->writes_gmem && db_overlap(db_rd, db_nrd, b.addr, b.addr + b.size)) return DB_READ;
        if (db_overlap(db_wr, db_nwr, b.addr, b.addr + b.size)) return DB_STORE;
    }
    for (int k = 0; k < 18; k++) {
        Range u = st.ubuf[5][k];
        if (u.addr && u.size && ((si->cbuf_mask & (1u << k)) || si->uses_gmem) && db_overlap(db_wr, db_nwr, u.addr, u.addr + u.size))
            return DB_STORE;
    }
    return DB_STOPS;
}

static void db_report(void) {
    uint64_t fr = renderer_present_count();
    if (fr < db_frame_logged + 600) return;
    db_frame_logged = fr;
    double frames = fr > db_frame0 ? (double)(fr - db_frame0) : 1;
    LOG("[dispatch-batch] frame %llu: %llu dispatches run ahead in %llu batches; a frame: look-ahead %.1f us, %.0f records, "
        "%.0f draws; look-ahead ended:", (unsigned long long)fr, (unsigned long long)db_ran, (unsigned long long)db_batches,
        db_ns / 1e3 / frames, db_records / frames, db_draws / frames);
    for (int i = 0; i < DB_STOPS; i++) if (db_stops[i]) LOG(" %s %llu;", db_why[i], (unsigned long long)db_stops[i]);
    LOG("\n");
    db_ns = db_records = db_draws = 0, db_frame0 = fr;
}

// (exec_list) right after a dispatch that ended a pass: the pass's later dispatches that can run now, run now
static void db_batch(CmdRec *d1) {
    uint64_t t0 = plat_now_ns(), nrec = 0, ndraw = 0;
    static XState saved;
    memcpy(&saved, &st, sizeof saved);
    RtState rt = st.rt;
    bool depth_written = false, drawn = false, barrier = false;  // (barrier: the game's, since the last dispatch)
    int draws = 0;  // since the last dispatch
    db_nrd = db_nwr = 0;
    db_n = db_next = 0;
    int why = DB_LONG;
    uint64_t p = (uint64_t)(uintptr_t)d1 + d1->size;
    for (int guard = 0; guard < 20000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        void *x = nvn_cmd_extra(r);
        nrec++;
        if (r->op == CMD_DRAW_ARRAYS || r->op == CMD_DRAW_ELEMENTS) ndraw++;
        switch (r->op) {
        case CMD_END: why = DB_END; goto done;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_CALL: why = DB_CALL; goto done;
        case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR:
        case CMD_BIND_DEPTH_STENCIL: case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB:
        case CMD_BIND_VERTEX_STREAM: case CMD_BIND_VERTEX_BUFFERS: case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS:
        case CMD_BIND_TEXTURES: case CMD_BIND_TEXTURES_DEFERRED: case CMD_BIND_IMAGES: case CMD_SET_TEXTURE_POOL:
        case CMD_SET_SAMPLER_POOL: case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS: case CMD_SET_DEPTH_RANGES:
        case CMD_SET_BLEND_COLOR: case CMD_SET_STENCIL: case CMD_SET_LINE_WIDTH: case CMD_SET_DEPTH_CLAMP:
        case CMD_SET_RENDER_ENABLE: case CMD_SET_RASTERIZER_DISCARD: case CMD_SET_POLYGON_OFFSET:
        case CMD_SET_PRIMITIVE_RESTART: case CMD_SET_COPY_STRIDES:
            if (r->op == CMD_BIND_PROGRAM) key_gen++, prog_gen++;
            state_apply(r, x);
            break;
        case CMD_SET_RENDER_TARGETS:
            key_gen++, prog_gen++;
            state_apply(r, x);
            if (memcmp(&rt, &st.rt, sizeof rt)) { why = DB_TARGETS; goto done; }
            break;
        case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS:  // (64 draws without a dispatch: the rest of the pass is drawing)
            if (++draws > 64 || (!st.render_off && !db_draw(r, r->op == CMD_DRAW_ELEMENTS, &depth_written, &drawn))) {
                why = DB_LONG;
                goto done;
            }
            break;
        case CMD_DISPATCH: {
            if (!st.program || !r->x[0] || !r->x[1] || !r->x[2]) { why = DB_OTHER; goto done; }
            if (db_n == 64) { why = DB_FULL; goto done; }
            CsProg *cp = compute_program(st.program, g_vk.gmem_bda == 2 && (renderer_present_count() & 1));
            if (!cp || atomic_load(&cp->p.state) != 1) { why = DB_NOT_READY; goto done; }
            int stop = db_check(cp->si, depth_written, drawn);
            if (stop != DB_STOPS) { why = stop; goto done; }
            if (barrier) gb_pending = true, gb_src = 1, gb_bits |= 1;  // (the game's barrier before it: gb_dispatch_barrier)
            barrier = false;
            dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]);
            db_ahead[db_n++] = r;
            db_ran++;
            draws = 0;
            break;
        }
        case CMD_BARRIER:
            if (r->x[0] != 1) { why = DB_BARRIER; goto done; }
            barrier = true;
            break;
        case CMD_UPDATE_UNIFORM_BUFFER: why = DB_UPDATE; goto done;
        case CMD_CLEAR_COLOR: case CMD_CLEAR_DEPTH_STENCIL: why = DB_CLEAR; goto done;
        case CMD_CLEAR_TEXTURE: case CMD_COPY_BUFFER_TO_BUFFER: case CMD_COPY_BUFFER_TO_TEXTURE:
        case CMD_COPY_TEXTURE_TO_BUFFER: case CMD_COPY_TEXTURE_TO_TEXTURE: case CMD_DOWNSAMPLE: case CMD_FENCE_SYNC:
        case CMD_REPORT_COUNTER: case CMD_RESET_COUNTER: {  // (what the walk acts on that orders work or memory)
            static uint64_t seen;  // (each such command logged once)
            if (r->op < 64 && !(seen & (1ull << r->op))) seen |= 1ull << r->op, LOG("[dispatch-batch] look-ahead ends at command %u\n", r->op);
            why = DB_OTHER;
            goto done;
        }
        default: break;  // (commands the walk ignores: depth bounds, alpha reference, point size, sample mask, discards)
        }
        p += r->size;
    }
done:
    db_stops[why]++;
    if (db_n) db_batches++;
    memcpy(&st, &saved, sizeof saved);
    key_gen++, prog_gen++, sbuf_gen++, material_tex_gen++;  // (the memos keyed on them saw the state ahead)
    db_ns += plat_now_ns() - t0, db_records += nrec, db_draws += ndraw;
    db_report();
}
