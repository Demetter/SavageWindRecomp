// Diagnostic only: consecutive-pair opportunities in the expensive scene pass.
// These are upper bounds for a redesign, NOT permission to merge draws. Uniform
// addresses do not prove byte identity; intervening barriers, queries, writes and
// draw-local shader semantics still need preserving by an actual implementation.
#include "native_draw_geometry.h"
static int draw_batch_probe_on = -1;
static struct {
    uint64_t draws, pairs, pipeline, dynamic, buffers, offsets, textures, uniforms;
    uint64_t state_buffers, state_offsets, state_textures, state_uniforms, unknown_textures;
    uint64_t rebase, state_textures_rebase, rebase_vertex_id;
    uint64_t eligible, groups, grouped_draws, saved_draw_calls, max_group, direct_ubo, ordered_pairs;
} draw_batch_counts;
static void draw_batch_probe_frame(uint64_t frame) {
    if (draw_batch_probe_on < 0) {
        const char *e = getenv("SAVAGEWIND_VK_DRAW_BATCH_PROBE");
        draw_batch_probe_on = e && atoi(e);
    }
    if (!draw_batch_probe_on || !draw_batch_counts.draws) return;
    LOG("[draw-batch-probe] frame %llu draws %llu pairs %llu pipeline %llu dynamic %llu buffers %llu offsets %llu textures %llu uniform_sources %llu state_buffers %llu state_offsets %llu state_textures %llu state_uniform_sources %llu unknown_textures %llu rebase %llu state_textures_rebase %llu rebase_vertex_id %llu eligible %llu groups %llu grouped_draws %llu saved_draw_calls %llu max_group %llu direct_ubo %llu ordered_pairs %llu\n",
        (unsigned long long)frame, (unsigned long long)draw_batch_counts.draws,
        (unsigned long long)draw_batch_counts.pairs, (unsigned long long)draw_batch_counts.pipeline,
        (unsigned long long)draw_batch_counts.dynamic, (unsigned long long)draw_batch_counts.buffers,
        (unsigned long long)draw_batch_counts.offsets, (unsigned long long)draw_batch_counts.textures,
        (unsigned long long)draw_batch_counts.uniforms, (unsigned long long)draw_batch_counts.state_buffers,
        (unsigned long long)draw_batch_counts.state_offsets, (unsigned long long)draw_batch_counts.state_textures,
        (unsigned long long)draw_batch_counts.state_uniforms, (unsigned long long)draw_batch_counts.unknown_textures,
        (unsigned long long)draw_batch_counts.rebase, (unsigned long long)draw_batch_counts.state_textures_rebase,
        (unsigned long long)draw_batch_counts.rebase_vertex_id,
        (unsigned long long)draw_batch_counts.eligible, (unsigned long long)draw_batch_counts.groups,
        (unsigned long long)draw_batch_counts.grouped_draws, (unsigned long long)draw_batch_counts.saved_draw_calls,
        (unsigned long long)draw_batch_counts.max_group, (unsigned long long)draw_batch_counts.direct_ubo,
        (unsigned long long)draw_batch_counts.ordered_pairs);
    memset(&draw_batch_counts, 0, sizeof draw_batch_counts);
}

static void draw_batch_probe(Prog *pr, Pipe *pp, VkCommandBuffer cb, bool indexed,
                             int32_t base_vertex, DrawJob *job, const VkDescriptorImageInfo *fii) {
    if (draw_batch_probe_on <= 0) return;
    typedef struct {
        uint64_t frame, pass, seq, generation, boundary, draw;
        VkCommandBuffer cb;
        VkPipeline pipeline;
        uint32_t streams, uniform_pos, ntex[2];
        bool indexed, texture_known, eligible;
        __typeof__(dc) state;
        Range ubuf[2][18], sbuf[2][16];
        VkDescriptorImageInfo images[2][DJ_TEX];
        float rs0[4], tint[8], scales[2][DJ_TEX][2];
    } Snapshot;
    static Snapshot previous;
    static bool valid;
    static Snapshot group;
    static unsigned group_size;
    static uint64_t group_min_index, group_max_index;
    // The six-target native-resolution scene pass, including its submitted pieces.
    if (pass_ncolor != 6 || pass_w != 5120 || pass_h != 2160 || !in_pass) { valid = false; group_size = 0; return; }
    Snapshot current = {0};
    current.frame = renderer_present_count(); current.pass = stat_passes;
    current.seq = vk_pending_seq(); current.cb = cb; current.pipeline = pp->pipe;
    current.boundary = vkq_batch_probe_token(dj_bind_replay); current.draw = stat_draws;
    current.streams = pp->k.s.used_streams; current.indexed = indexed;
    current.state = dc;
    current.generation = gcopy_gen; current.uniform_pos = atomic_load(&uub_nlog);
    current.texture_known = job && pr->vi->ntex <= DJ_TEX && pr->fi->ntex <= DJ_TEX;
    if (current.texture_known) {
        current.ntex[0] = pr->vi->ntex; current.ntex[1] = pr->fi->ntex;
        // The producer owns ii; the worker only reads it. This producer cannot
        // reuse the job slot until a subsequent draw, even if replay has finished.
        memcpy(current.images[0], job->ii[0], pr->vi->ntex * sizeof current.images[0][0]);
        memcpy(current.images[1], fii, pr->fi->ntex * sizeof current.images[1][0]);
        memcpy(current.rs0, job->rs0, sizeof current.rs0); memcpy(current.tint, job->tint, sizeof current.tint);
        for (unsigned stage = 0; stage < 2; stage++)
            memcpy(current.scales[stage], job->sc[stage], current.ntex[stage] * sizeof current.scales[stage][0]);
    }
    bool direct = job && (job->direct_ubo_mask[0] || job->direct_ubo_mask[1]);
    draw_batch_counts.direct_ubo += direct;
    // oq_counting remains set after the bounded query run fills; only draws
    // which actually create a query must be excluded here.
    bool query = oq_counting && oq_nrun < OQ_RUN && oq_on();
    current.eligible = current.texture_known && vkq_on && indexed && !query && !direct && base_vertex >= 0 &&
                       !pr->vi->writes_gmem && !pr->fi->writes_gmem;
    for (int stage = 0; stage < 2; stage++) {
        const ShaderInfo *si = stage ? pr->fi : pr->vi;
        for (unsigned k = 3; k < 17; k++)
            if (si->cbuf_mask & (1u << k)) current.ubuf[stage][k - 3] = st.ubuf[stage][k - 3];
        memcpy(current.sbuf[stage], st.sbuf[stage], sizeof current.sbuf[stage]);
    }
    draw_batch_counts.draws++;
    if (!current.texture_known) draw_batch_counts.unknown_textures++;
    if (valid && previous.frame == current.frame && previous.pass == current.pass &&
        previous.seq == current.seq && previous.cb == current.cb) {
        draw_batch_counts.pairs++;
        draw_batch_counts.ordered_pairs += previous.boundary == current.boundary && previous.draw + 1 == current.draw;
        bool pipeline = previous.pipeline == current.pipeline;
        bool dynamic = !memcmp(&previous.state.dt, &current.state.dt,
                                offsetof(__typeof__(dc), ibuf) - offsetof(__typeof__(dc), dt));
        bool buffers = previous.streams == current.streams && previous.indexed == current.indexed;
        bool offsets = true;
        if (indexed) {
            buffers &= previous.state.ibuf == dc.ibuf && previous.state.itype == dc.itype;
            offsets &= previous.state.ioff == dc.ioff;
        }
        for (int s = 0; s < 17; s++) if (current.streams & (1u << s)) {
            buffers &= previous.state.vbuf[s] == dc.vbuf[s];
            offsets &= previous.state.voff[s] == dc.voff[s];
        }
        offsets &= buffers;
        bool textures = current.texture_known && previous.texture_known &&
            !memcmp(current.ntex, previous.ntex, sizeof current.ntex) &&
            !memcmp(current.images, previous.images, sizeof current.images);
        bool uniforms = current.generation == previous.generation && current.uniform_pos == previous.uniform_pos &&
            !memcmp(current.ubuf, previous.ubuf, sizeof current.ubuf) &&
            !memcmp(current.sbuf, previous.sbuf, sizeof current.sbuf);
        uint64_t index_binding; uint32_t first_a, first_b; int32_t new_base; int64_t vertex_id_delta = 0;
        bool rebase = pipeline && buffers && indexed &&
            native_draw_index_rebase(previous.state.ioff, dc.ioff,
                dc.itype == VK_INDEX_TYPE_UINT16 ? 2 : dc.itype == VK_INDEX_TYPE_UINT32 ? 4 : 0,
                &index_binding, &first_a, &first_b) &&
            native_draw_vertex_rebase(current.streams, previous.state.voff, dc.voff,
                pp->k.s.stride, pp->k.s.divisor, base_vertex, &new_base, &vertex_id_delta);
        draw_batch_counts.pipeline += pipeline; draw_batch_counts.dynamic += dynamic;
        draw_batch_counts.buffers += buffers; draw_batch_counts.offsets += offsets;
        draw_batch_counts.textures += textures; draw_batch_counts.uniforms += uniforms;
        draw_batch_counts.state_buffers += pipeline && dynamic && buffers;
        draw_batch_counts.state_offsets += pipeline && dynamic && offsets;
        draw_batch_counts.state_textures += pipeline && dynamic && buffers && textures;
        draw_batch_counts.state_uniforms += pipeline && dynamic && buffers && textures && uniforms;
        draw_batch_counts.rebase += rebase;
        draw_batch_counts.state_textures_rebase += rebase && dynamic && textures;
        draw_batch_counts.rebase_vertex_id += rebase && vertex_id_delta && pr->vs_uses_vid;
    }
    // A conservative pilot estimate: actual contiguous runs, rebased against
    // their FIRST draw, with native-upload-only constants, identical scales/tint,
    // and no intervening queued command except the bindings a batch replaces.
    // This still does not establish shader math/depth equivalence or an FPS gain.
    if (current.eligible) {
        draw_batch_counts.eligible++;
        bool append = group_size && group_size < 128 && valid && previous.draw + 1 == current.draw &&
            group.frame == current.frame && group.pass == current.pass && group.seq == current.seq &&
            group.cb == cb && group.boundary == current.boundary && group.pipeline == current.pipeline &&
            group.streams == current.streams && group.state.ibuf == dc.ibuf && group.state.itype == dc.itype &&
            !memcmp(&group.state.dt, &dc.dt, offsetof(__typeof__(dc), ibuf) - offsetof(__typeof__(dc), dt)) &&
            !memcmp(group.ntex, current.ntex, sizeof group.ntex) && !memcmp(group.images, current.images, sizeof group.images) &&
            !memcmp(group.rs0, current.rs0, sizeof group.rs0) && !memcmp(group.tint, current.tint, sizeof group.tint) &&
            !memcmp(group.scales, current.scales, sizeof group.scales);
        for (unsigned s = 0; s < 17 && append; s++) if (current.streams & (1u << s))
            append = group.state.vbuf[s] == dc.vbuf[s];
        uint64_t min_index = group_min_index < dc.ioff ? group_min_index : dc.ioff;
        uint64_t max_index = group_max_index > dc.ioff ? group_max_index : dc.ioff;
        uint64_t binding; uint32_t first_a, first_b; int32_t rebased; int64_t delta;
        append = append && native_draw_index_rebase(min_index, max_index,
            dc.itype == VK_INDEX_TYPE_UINT16 ? 2 : dc.itype == VK_INDEX_TYPE_UINT32 ? 4 : 0,
            &binding, &first_a, &first_b) && native_draw_vertex_rebase(current.streams,
            group.state.voff, dc.voff, pp->k.s.stride, pp->k.s.divisor, base_vertex, &rebased, &delta) &&
            (!delta || !pr->vs_uses_vid);
        if (append) {
            group_size++; group_min_index = min_index; group_max_index = max_index;
            draw_batch_counts.saved_draw_calls++;
            draw_batch_counts.grouped_draws += group_size == 2 ? 2 : 1;
        } else {
            group = current; group_size = 1; group_min_index = group_max_index = dc.ioff;
            draw_batch_counts.groups++;
        }
        if (group_size > draw_batch_counts.max_group) draw_batch_counts.max_group = group_size;
    } else group_size = 0;
    previous = current; valid = true;
}
