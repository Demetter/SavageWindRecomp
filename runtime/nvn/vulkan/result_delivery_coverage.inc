// Eligibility stage of result delivery. No snapshot, page guard or fence
// elision runs until the approved coverage rules permit the frame to release.
// Default OFF; producer thread only. This is not an implemented early-release path.
static bool rdc_on(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *e = getenv("SAVAGEWIND_VK_RESULT_DELIVERY_COVERAGE");
        enabled = e && atoi(e) != 0;
        if (enabled) LOG("[result-delivery] coverage stage only; normal fences; no guard faults or early releases\n");
    }
    return enabled != 0;
}
#define RDC_MAX 2048
static struct {
    uint64_t frame, draws, alias_seq, unknown_seq;
    unsigned nw, nt, overflow;
    struct { uint64_t addr, size, seq, shader; unsigned slot; } w[RDC_MAX];
    struct { VkImage image; uint64_t draw, seq, shader, addr; } t[RDC_MAX];
} rdc;
static void rdc_begin(void) {
    if (!rdc_on()) return;
    uint64_t f = renderer_present_count();
    if (rdc.frame != f) { memset(&rdc, 0, sizeof rdc); rdc.frame = f; }
}
static void rdc_writer(int stage, const ShaderInfo *si) {
    if (!rdc_on()) return;
    uint64_t shader = vk_translate_id(si), seq = vk_pending_seq();
    // Static review of these cached shaders places their store bases in the
    // first three driver storage bindings. Other stages remain conservative.
    bool known = stage == 5 && (shader == 0xd3a45e67ef9bacafull || shader == 0xb6ffbe601061b198ull ||
        shader == 0xec5709dd8702c7ddull || shader == 0x4e2d60a3f420b3baull ||
        shader == 0x05615853c139aef1ull || shader == 0x92f24d7ae50e5945ull || shader == 0x353d56c825f0617full);
    for (int i = 0; i < (known ? 3 : 16); i++) {
        Range r = st.sbuf[stage][i]; if (!r.addr || !r.size) continue;
        unsigned k = 0;
        while (k < rdc.nw && (rdc.w[k].addr != r.addr || rdc.w[k].size != r.size || rdc.w[k].shader != shader)) k++;
        if (k == RDC_MAX) { rdc.overflow = 1; continue; }
        if (k == rdc.nw) rdc.nw++;
        rdc.w[k] = (typeof(rdc.w[0])){r.addr, r.size, seq, shader, (unsigned)i};
    }
}
static void rdc_direct_read(uint64_t addr, uint64_t size) {
    if (!rdc_on()) return;
    for (unsigned i = 0; i < rdc.nw; i++) {
        typeof(rdc.w[0]) *w = &rdc.w[i];
        if (addr >= w->addr + w->size || w->addr >= addr + size) continue;
        if (!rdc.alias_seq && rdc.frame % 60 == 0)
            TRACE_LOG("[result-alias] frame %llu addr %llx size %llu writer %llx slot %u write_seq %llu read_seq %llu\n",
                (unsigned long long)rdc.frame, (unsigned long long)addr, (unsigned long long)size,
                (unsigned long long)w->shader, w->slot, (unsigned long long)w->seq, (unsigned long long)vk_pending_seq());
        rdc.alias_seq = vk_pending_seq();
        return;
    }
}
static void rdc_unknown_gmem(void) {
    // A storage-buffer declaration alone does not bound loads through pointers
    // fetched from guest memory. No dynamic-address proof exists for this path.
    if (rdc_on()) rdc.unknown_seq = vk_pending_seq();
}
static void rdc_draw(ShaderInfo *fi, bool uses_gmem) {
    if (!rdc_on()) return;
    rdc.draws++;
    if (uses_gmem) rdc_unknown_gmem();
    if (g_cur_fs_hash == 0xa929e75a271d6792ull && rdc.frame % 60 == 0) {
        for (int i = 0; i < fi->ntex; i++) {
            NvnTexture *t = shader_texture_object(1, &fi->tex[i]); TexBackend *b = t ? t->backend : NULL;
            unsigned k = 0; while (k < rdc.nt && (!b || rdc.t[k].image != b->image)) k++;
            TRACE_LOG("[result-probe-input] frame %llu probe_draw %llu probe_seq %llu input %d addr %llx image %llx dimensions %ux%u last_draw %llu last_seq %llu last_fs %llx\n",
                (unsigned long long)rdc.frame, (unsigned long long)rdc.draws, (unsigned long long)vk_pending_seq(), i,
                (unsigned long long)(t ? t->addr : 0), (unsigned long long)(b ? (uintptr_t)b->image : 0), b ? b->width : 0, b ? b->height : 0,
                (unsigned long long)(k < rdc.nt ? rdc.t[k].draw : 0), (unsigned long long)(k < rdc.nt ? rdc.t[k].seq : 0),
                (unsigned long long)(k < rdc.nt ? rdc.t[k].shader : 0));
        }
    }
    // Draw attachment history is sufficient to rule out moving a probe ahead
    // of an observed producer. It does not certify a complete relocation plan:
    // clears, uploads, aliases and subresources need proof before a real move.
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i]; TexBackend *b = t ? t->backend : NULL; if (!b) continue;
        unsigned k = 0; while (k < rdc.nt && rdc.t[k].image != b->image) k++;
        if (k == RDC_MAX) { rdc.overflow = 1; continue; }
        if (k == rdc.nt) rdc.nt++;
        rdc.t[k] = (typeof(rdc.t[0])){b->image, rdc.draws, vk_pending_seq(), g_cur_fs_hash, t->addr};
    }
}
static void rdc_complete(void) {
    if (!rdc_on()) return;
    uint64_t done = vk_completed_seq();
    unsigned reason = (rdc.alias_seq > done ? 1u : 0u) | (rdc.unknown_seq > done ? 2u : 0u) | (rdc.overflow ? 16u : 0u);
    unsigned pending = 0, unmapped = 0, streaming = 0;
    pthread_mutex_lock(&readback_lock);
    for (int i = 0; i < nreadbacks; i++) {
        Readback *r = &readbacks[i]; if (r->seq <= done) continue;
        pending++;
        bool mapped = r->fs == 0xa929e75a271d6792ull || r->fs == 0x003bf0cbe6afef14ull || r->fs == 0xfdfa1d1c9088bfa7ull;
        if (!mapped) unmapped++;
        if (r->fmt == 6 || r->fmt == 0xa || r->row * (uint64_t)r->rows > 1048576) streaming++;
    }
    pthread_mutex_unlock(&readback_lock);
    if (unmapped) reason |= 4u;
    if (streaming) reason |= 8u;
    TRACE_LOG("[result-coverage] frame %llu draws %llu submitted %llu done %llu reasons %u alias_seq %llu unknown_seq %llu pending_outputs %u unmapped_outputs %u streaming %u\n",
        (unsigned long long)rdc.frame, (unsigned long long)rdc.draws, (unsigned long long)submit_seq, (unsigned long long)done,
        reason, (unsigned long long)rdc.alias_seq, (unsigned long long)rdc.unknown_seq, pending, unmapped, streaming);
}
