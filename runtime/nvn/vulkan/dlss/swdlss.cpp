// The DLSS bridge: NVIDIA NGX (DLSS SDK, Vulkan) behind the plain C entry points of swdlss.h, built with MSVC (the
// SDK's libraries need its C++ runtime; the runtime is built with MinGW) by tools/build_dlss_bridge.bat. One DLSS
// feature at a time, made again when its sizes or quality change.
#define SWDLSS_BUILD
#include "swdlss.h"

#include <stdio.h>
#include <string.h>

#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_vk.h"

static NVSDK_NGX_Parameter *g_params;
static NVSDK_NGX_Handle *g_feature;
static VkDevice g_dev;
static char g_err[256];
static struct { uint32_t in_w, in_h, out_w, out_h; int quality; unsigned flags; } g_made;

static int fail(const char *what, NVSDK_NGX_Result r) {
    snprintf(g_err, sizeof g_err, "%s: 0x%08x", what, (unsigned)r);
    return 0;
}

extern "C" SWDLSS_API const char *swdlss_last_error(void) { return g_err; }

extern "C" SWDLSS_API int swdlss_required_extensions(unsigned *ninst, const char ***inst, unsigned *ndev, const char ***dev) {
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(ninst, inst, ndev, dev);
    return NVSDK_NGX_SUCCEED(r) ? 1 : fail("RequiredExtensions", r);
}

extern "C" SWDLSS_API int swdlss_init(VkInstance inst, VkPhysicalDevice pd, VkDevice dev, PFN_vkGetInstanceProcAddr gipa,
                                      PFN_vkGetDeviceProcAddr gdpa, const wchar_t *data_dir) {
    // (a project id of its own: NGX identifies custom engines by one)
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID("5a7e6f3c-8b1d-4e2a-9c47-1f0d3b6a2e95", NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                                              "1.0", data_dir, inst, pd, dev, gipa, gdpa);
    if (NVSDK_NGX_FAILED(r)) return fail("Init", r);
    g_dev = dev;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_params);
    if (NVSDK_NGX_FAILED(r)) return fail("GetCapabilityParameters", r);
    int available = 0;
    r = g_params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (NVSDK_NGX_FAILED(r) || !available) {
        int why = 0;
        g_params->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &why);
        return fail("DLSS not available on this GPU / driver (feature init result)", (NVSDK_NGX_Result)why);
    }
    return 1;
}

extern "C" SWDLSS_API void swdlss_release(void) {
    if (g_feature) NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
    g_feature = NULL;
    memset(&g_made, 0, sizeof g_made);
}

extern "C" SWDLSS_API int swdlss_create(VkCommandBuffer cb, uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h, int quality,
                                        unsigned flags) {
    if (!g_params) return fail("create before init", NVSDK_NGX_Result_Fail);
    if (g_feature && g_made.in_w == in_w && g_made.in_h == in_h && g_made.out_w == out_w && g_made.out_h == out_h &&
        g_made.quality == quality && g_made.flags == flags)
        return 1;
    swdlss_release();
    static const NVSDK_NGX_PerfQuality_Value q[] = {NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
                                                    NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
                                                    NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = in_w, cp.Feature.InHeight = in_h, cp.Feature.InTargetWidth = out_w, cp.Feature.InTargetHeight = out_h;
    cp.Feature.InPerfQualityValue = q[quality >= 0 && quality <= 4 ? quality : 0];
    int f = 0;
    if (flags & SWDLSS_HDR) f |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (flags & SWDLSS_DEPTH_INVERTED) f |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (flags & SWDLSS_MV_LOW_RES) f |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (flags & SWDLSS_AUTO_EXPOSURE) f |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    cp.InFeatureCreateFlags = f;
    // the model: flags' bits 8-15 (SWDLSS_PRESET), 0 = NVIDIA's default for the mode (transformer K for DLAA)
    unsigned preset = (flags >> 8) & 0xff;
    const char *const hint[] = {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance};
    for (int i = 0; i < 5; i++) g_params->Set(hint[i], preset);
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(g_dev, cb, 1, 1, &g_feature, g_params, &cp);
    if (NVSDK_NGX_FAILED(r)) { g_feature = NULL; return fail("CreateFeature (DLSS)", r); }
    g_made.in_w = in_w, g_made.in_h = in_h, g_made.out_w = out_w, g_made.out_h = out_h, g_made.quality = quality, g_made.flags = flags;
    return 1;
}

static NVSDK_NGX_Resource_VK res(const SwDlssImage *i, bool rw, bool depth) {
    VkImageSubresourceRange sr = {depth ? (VkImageAspectFlags)VK_IMAGE_ASPECT_DEPTH_BIT : (VkImageAspectFlags)VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(i->view, i->image, sr, i->format, i->width, i->height, rw);
}

extern "C" SWDLSS_API int swdlss_evaluate(VkCommandBuffer cb, const SwDlssImage *color, const SwDlssImage *depth, const SwDlssImage *mv,
                                          const SwDlssImage *out, float jitter_x, float jitter_y, float mv_scale_x, float mv_scale_y,
                                          int reset) {
    if (!g_feature) return fail("evaluate before create", NVSDK_NGX_Result_Fail);
    NVSDK_NGX_Resource_VK c = res(color, false, false), d = res(depth, false, true), m = res(mv, false, false), o = res(out, true, false);
    NVSDK_NGX_VK_DLSS_Eval_Params ep = {};
    ep.Feature.pInColor = &c;
    ep.Feature.pInOutput = &o;
    ep.pInDepth = &d;
    ep.pInMotionVectors = &m;
    ep.InJitterOffsetX = jitter_x, ep.InJitterOffsetY = jitter_y;
    ep.InRenderSubrectDimensions.Width = color->width, ep.InRenderSubrectDimensions.Height = color->height;
    ep.InReset = reset;
    ep.InMVScaleX = mv_scale_x, ep.InMVScaleY = mv_scale_y;
    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cb, g_feature, g_params, &ep);
    return NVSDK_NGX_SUCCEED(r) ? 1 : fail("EvaluateFeature (DLSS)", r);
}

extern "C" SWDLSS_API void swdlss_shutdown(VkDevice dev) {
    swdlss_release();
    if (g_params) NVSDK_NGX_VULKAN_DestroyParameters(g_params);
    g_params = NULL;
    NVSDK_NGX_VULKAN_Shutdown1(dev);
}
