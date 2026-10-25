// DLSS (NVIDIA): the bridge swdlss.dll (dlss/swdlss.cpp, tools/build_dlss_bridge.bat) loaded from next to the
// executable, with NVIDIA's nvngx_dlss.dll. Neither is part of this repository; without them DLSS is unavailable and
// nothing else changes. The anti-aliasing itself (jitter, motion vectors, the evaluation) is in exec.c (dlss_*).
// SAVAGEWIND_NO_DLSS=1: not loaded.
#include <stdlib.h>
#include <string.h>

#include "vk.h"
#include "dlss/swdlss.h"

#ifdef _WIN32
#include <windows.h>
#endif

static struct {
    PFN_swdlss_required_extensions required_extensions;
    PFN_swdlss_init init;
    PFN_swdlss_create create;
    PFN_swdlss_evaluate evaluate;
    PFN_swdlss_release release;
    PFN_swdlss_shutdown shutdown;
    PFN_swdlss_last_error last_error;
} g;
static int g_loaded = -1;  // -1 not tried, 0 absent, 1 loaded
static bool g_ready;       // NGX up on the device and DLSS available

static bool load(void) {
    if (g_loaded >= 0) return g_loaded;
    g_loaded = 0;
#ifdef _WIN32
    if (getenv("SAVAGEWIND_NO_DLSS")) return false;
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, path, MAX_PATH);
    while (n && path[n - 1] != L'\\' && path[n - 1] != L'/') n--;
    if (!n || n + 12 >= MAX_PATH) return false;
    wcscpy(path + n, L"swdlss.dll");
    HMODULE m = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) return false;
#define GET(f) (g.f = (PFN_swdlss_##f)(void *)GetProcAddress(m, "swdlss_" #f))
    if (!GET(required_extensions) || !GET(init) || !GET(create) || !GET(evaluate) || !GET(release) || !GET(shutdown) || !GET(last_error)) {
        LOG("[dlss] swdlss.dll is incomplete: DLSS unavailable\n");
        FreeLibrary(m);
        return false;
    }
#undef GET
    g_loaded = 1;
#endif
    return g_loaded;
}

static void add_unique(const char **ext, uint32_t *n, uint32_t cap, const char *name) {
    for (uint32_t i = 0; i < *n; i++) if (!strcmp(ext[i], name)) return;
    if (*n < cap) ext[(*n)++] = name;
}

// (create_instance) NGX's instance extensions, added to ext.
void dlss_instance_extensions(const char **ext, uint32_t *n, uint32_t cap) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!load() || !g.required_extensions(&ni, &ie, &nd, &de)) return;
    for (unsigned i = 0; i < ni; i++) add_unique(ext, n, cap, ie[i]);
}

// (create_device) NGX's device extensions, added to ext when the device has them all; false: DLSS stays off.
bool dlss_device_extensions(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t cap) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (!load() || !g.required_extensions(&ni, &ie, &nd, &de)) return false;
    for (unsigned i = 0; i < nd; i++) {
        bool ok = false;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) { LOG("[dlss] the device lacks %s: DLSS unavailable\n", de[i]); return false; }
    }
    for (unsigned i = 0; i < nd; i++) add_unique(ext, n, cap, de[i]);
    return true;
}

// (after the device) NGX on it.
void dlss_device_init(void) {
    if (!load()) return;
    g_ready = g.init(g_vk.inst, g_vk.phys, g_vk.dev, vkGetInstanceProcAddr, vkGetDeviceProcAddr, L".");
    if (g_ready) LOG("[dlss] DLSS available\n");
    else LOG("[dlss] unavailable: %s\n", g.last_error());
}

bool dlss_available(void) { return g_ready; }
bool dlss_create(VkCommandBuffer cb, uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h, int quality, unsigned flags) {
    if (!g_ready) return false;
    if (g.create(cb, in_w, in_h, out_w, out_h, quality, flags)) return true;
    LOG("[dlss] %s\n", g.last_error());
    return false;
}
bool dlss_evaluate(VkCommandBuffer cb, const SwDlssImage *color, const SwDlssImage *depth, const SwDlssImage *mv, const SwDlssImage *out,
                   float jx, float jy, float mvx, float mvy, int reset) {
    if (!g_ready) return false;
    if (g.evaluate(cb, color, depth, mv, out, jx, jy, mvx, mvy, reset)) return true;
    static int nlog;
    if (nlog++ < 5) LOG("[dlss] %s\n", g.last_error());
    return false;
}
void dlss_release(void) { if (g_ready) g.release(); }
