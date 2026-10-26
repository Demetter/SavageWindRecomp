// Producer-side resource groups. This bypasses repeated feedback discovery,
// depth preflight, native texture resolution and job-image snapshots for ordinary
// consecutive draws. Mutable uniforms, geometry and dependencies remain per draw.
// It deliberately does not change the worker/recorder queue protocol or GPU work.
typedef struct MaterialChunk {
    struct MaterialChunk *next;
    unsigned used;
    ProducerMaterial items[32];
} MaterialChunk;
static struct { uint64_t seq; MaterialChunk *head, *current; } material_arenas[SUBS];
static const ProducerMaterial *last_material;
static uint64_t material_hits, material_built, material_lookups, material_verified, material_report_frame;

static void producer_material_reset(void) {
    last_material = NULL;
    for (unsigned i = 0; i < SUBS; i++) {
        MaterialChunk *c = material_arenas[i].head;
        while (c) { MaterialChunk *next = c->next; free(c); c = next; }
        memset(&material_arenas[i], 0, sizeof material_arenas[i]);
    }
    material_tex_gen++; material_pass_gen++;
}

static bool producer_material_allowed(Prog *pr) {
    if (material_ab_frames) {
        // Quiet ABBA blocks in one launch avoid attributing launch drift to reuse.
        // mode stays immutable for descriptor workers; only producer eligibility changes.
        uint64_t frame = renderer_present_count();
        if (frame < material_ab_start) return false;
        uint64_t block = (frame - material_ab_start) / material_ab_frames;
        bool enabled = (block & 3) == 1 || (block & 3) == 2;
        static uint64_t reported = UINT64_MAX;
        if (block != reported) {
            reported = block;
            LOG("[producer-material-ab] frame %llu block %llu enabled %u\n",
                (unsigned long long)frame, (unsigned long long)block, enabled);
        }
        if (!enabled) return false;
    }
    return material_mode && mt_list && vkq_on && !g_debug_frame && !uub_log_full &&
        !stage_reuse_mode && !g_vk.native_multi_draw && draw_batch_probe_on <= 0 &&
        st.program->stage_mask == 3 && pr->vi->ntex <= DJ_TEX && pr->fi->ntex <= DJ_TEX &&
        in_pass && !st.rt_dirty && !pass_out && !pass_cube_layers &&
        // (PERFX_MATUP) with an upscaler, DLAA or sharpening too: what they change per draw (the composite's scene,
        // scaled textures) is never captured (producer_material_capture)
        (perfx(PERFX_MATUP) || (g_settings.upscaler == UPSCALE_OFF && !g_settings.sharpening && !dlaa_wanted()));
}

static bool producer_material_current(const ProducerMaterial *m, Prog *pr) {
    extern uint32_t vk_view_gen;
    return m && m->pr == pr && m->seq == submit_seq + 1 && m->frame == vk_frame_counter() &&
        m->pass_gen == material_pass_gen && m->tex_gen == material_tex_gen &&
        m->tp == st.tpool && m->sp == st.spool && m->view_gen == vk_view_gen &&
        m->trev == (st.tpool ? st.tpool->revision : 0) && m->srev == (st.spool ? st.spool->revision : 0);
}

static bool producer_material_object_ready(const NvnTexture *t) {
    const TexBackend *b = t->backend;
    return !__atomic_load_n(&t->cpu_levels, __ATOMIC_ACQUIRE) &&
        b && b->image && !b->in_feedback_layout;
}

static void producer_material_audit(const ProducerMaterial *m) {
    // Exercise the original paths on every reuse. They must discover no feedback
    // and produce byte-identical bindings/scales/extents on these eligible draws.
    resolve_feedback(m->pr);
    if (nfeedback || draw_feedback) FATAL("producer material missed attachment feedback\n");
    for (int s = 0; s < 2; s++) {
        ShaderInfo *si = s ? m->pr->fi : m->pr->vi;
        prepare_depth_comparisons(s, si);
        for (int i = 0; i < si->ntex; i++) {
            VkDescriptorImageInfo ref; float scale[2]; uint32_t extent[2];
            texture_info_uncached(s, si, i, &ref, scale, extent);
            if (ref.sampler != m->ii[s][i].sampler || ref.imageView != m->ii[s][i].imageView ||
                ref.imageLayout != m->ii[s][i].imageLayout || memcmp(scale, m->sc[s][i], sizeof scale) ||
                memcmp(extent, m->wh[s][i], sizeof extent))
                AUDIT_FAIL("producer material mismatch frame %llu stage %d texture %d\n",
                      (unsigned long long)renderer_present_count(), s, i);
        }
    }
    material_verified++;
}

static const ProducerMaterial *producer_material_find(Prog *pr) {
    if (!producer_material_allowed(pr)) return NULL;
    if (material_mode == 2) {
        material_lookups++;
        uint64_t frame = renderer_present_count();
        if (frame >= material_report_frame + 300) {
            LOG("[producer-material-audit] frame %llu lookups %llu shared %llu built %llu verified %llu; zero mismatches\n",
                (unsigned long long)frame, (unsigned long long)material_lookups,
                (unsigned long long)material_hits, (unsigned long long)material_built,
                (unsigned long long)material_verified);
            material_report_frame = frame;
        }
    }
    const ProducerMaterial *m = last_material;
    if (!producer_material_current(m, pr)) return NULL;
    // WriteTexels may arrive between draws without a pool registration. Check
    // each distinct object before reusing the entire native resource group.
    for (int i = 0; i < m->nobjects; i++)
        if (!producer_material_object_ready(m->objects[i])) return NULL;
    if (material_mode == 2) { producer_material_audit(m); material_hits++; }
    return m;
}

static ProducerMaterial *producer_material_alloc(void) {
    Sub *s = sub(); // waits before this slot's previous submission can be recycled
    __typeof__(material_arenas[0]) *a = &material_arenas[cur_sub];
    if (a->seq != s->seq) {
        for (MaterialChunk *c = a->head; c; c = c->next) c->used = 0;
        a->seq = s->seq; a->current = a->head;
    }
    if (!a->head) a->head = a->current = calloc(1, sizeof *a->head);
    if (!a->current) FATAL("producer material allocation failed\n");
    if (a->current->used == 32) {
        if (!a->current->next) a->current->next = calloc(1, sizeof *a->head);
        a->current = a->current->next;
        if (!a->current) FATAL("producer material allocation failed\n");
    }
    return &a->current->items[a->current->used++];
}

static const ProducerMaterial *producer_material_capture(Prog *pr, const DrawJob *job) {
    if (!producer_material_allowed(pr) || nfeedback || draw_feedback || sdw_fix.s || aa_fix.sx || scene_up.t) return NULL;
    NvnTexture *objects[2 * DJ_TEX]; int nobjects = 0;
    for (int s = 0; s < 2; s++) {
        ShaderInfo *si = s ? pr->fi : pr->vi;
        for (int i = 0; i < si->ntex; i++) {
            if (si->tex[i].shadow) return NULL; // native comparison mirrors remain dynamic
            NvnTexture *t = shader_texture_object(s, &si->tex[i]);
            if (is_attachment(t)) return NULL;
            if (!t) continue;
            if (!producer_material_object_ready(t)) return NULL;
            // Shadow scaling can patch constants even without a comparison opcode.
            if (job->sc[s][i][0] > 1 || job->sc[s][i][1] > 1) return NULL;
            int k = 0; while (k < nobjects && objects[k] != t) k++;
            if (k == nobjects) objects[nobjects++] = t;
        }
    }
    extern uint32_t vk_view_gen;
    ProducerMaterial *m = producer_material_alloc();
    m->pr = pr; m->seq = submit_seq + 1; m->frame = vk_frame_counter();
    m->tex_gen = material_tex_gen; m->pass_gen = material_pass_gen; m->view_gen = vk_view_gen;
    m->tp = st.tpool; m->sp = st.spool;
    m->trev = st.tpool ? st.tpool->revision : 0; m->srev = st.spool ? st.spool->revision : 0;
    m->nobjects = nobjects; memcpy(m->objects, objects, (size_t)nobjects * sizeof objects[0]);
    for (int s = 0; s < 2; s++) {
        m->ntex[s] = job->ntex[s];
        memcpy(m->ii[s], job->ii[s], (size_t)m->ntex[s] * sizeof m->ii[s][0]);
        memcpy(m->sc[s], job->sc[s], (size_t)m->ntex[s] * sizeof m->sc[s][0]);
        memcpy(m->wh[s], job->wh[s], (size_t)m->ntex[s] * sizeof m->wh[s][0]);
    }
    last_material = m;
    if (material_mode == 2) material_built++;
    return m;
}
