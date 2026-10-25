// The DLSS bridge (swdlss.dll, built with MSVC from swdlss.cpp against NVIDIA's DLSS SDK, which is not part of this
// repository: tools/build_dlss_bridge.bat). The runtime loads it at start when it sits next to the executable with
// nvngx_dlss.dll, and calls it through these plain C entry points; without it the DLSS option stays unavailable.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef SWDLSS_BUILD
#define SWDLSS_API __declspec(dllexport)
#else
#define SWDLSS_API
#endif

typedef struct {
    VkImage image;
    VkImageView view;  // (a 2D view of mip 0, layer 0; the depth aspect for depth)
    VkFormat format;
    uint32_t width, height;
} SwDlssImage;

enum { SWDLSS_DLAA = 0, SWDLSS_QUALITY = 1, SWDLSS_BALANCED = 2, SWDLSS_PERFORMANCE = 3, SWDLSS_ULTRA_PERFORMANCE = 4 };
enum { SWDLSS_HDR = 1, SWDLSS_DEPTH_INVERTED = 2, SWDLSS_MV_LOW_RES = 4, SWDLSS_AUTO_EXPOSURE = 8 };
#define SWDLSS_PRESET(p) ((unsigned)(p) << 8)  // the model (NVSDK_NGX_DLSS_Hint_Render_Preset: 5 E (CNN), 11 K...)

#ifdef __cplusplus
extern "C" {
#endif
// The Vulkan extensions NGX needs (before the instance and the device are made).
typedef int (*PFN_swdlss_required_extensions)(unsigned *ninst, const char ***inst, unsigned *ndev, const char ***dev);
// NGX on the device; 1: DLSS available (driver and GPU), 0: not, with the reason in swdlss_last_error.
typedef int (*PFN_swdlss_init)(VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
                               const wchar_t *data_dir);
// (Re)makes the DLSS feature for these sizes, recorded into cb; 1: made.
typedef int (*PFN_swdlss_create)(VkCommandBuffer cb, uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h, int quality,
                                 unsigned flags);
// One frame into out, recorded into cb (images in GENERAL layout; out storage). jitter in input pixels, motion vectors in
// pixels (scale: to input pixels), reset at camera cuts. 1: done.
typedef int (*PFN_swdlss_evaluate)(VkCommandBuffer cb, const SwDlssImage *color, const SwDlssImage *depth, const SwDlssImage *mv,
                                   const SwDlssImage *out, float jitter_x, float jitter_y, float mv_scale_x, float mv_scale_y,
                                   int reset);
typedef void (*PFN_swdlss_release)(void);
typedef void (*PFN_swdlss_shutdown)(VkDevice);
typedef const char *(*PFN_swdlss_last_error)(void);

SWDLSS_API int swdlss_required_extensions(unsigned *ninst, const char ***inst, unsigned *ndev, const char ***dev);
SWDLSS_API int swdlss_init(VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
                           const wchar_t *data_dir);
SWDLSS_API int swdlss_create(VkCommandBuffer cb, uint32_t in_w, uint32_t in_h, uint32_t out_w, uint32_t out_h, int quality,
                             unsigned flags);
SWDLSS_API int swdlss_evaluate(VkCommandBuffer cb, const SwDlssImage *color, const SwDlssImage *depth, const SwDlssImage *mv,
                               const SwDlssImage *out, float jitter_x, float jitter_y, float mv_scale_x, float mv_scale_y,
                               int reset);
SWDLSS_API void swdlss_release(void);
SWDLSS_API void swdlss_shutdown(VkDevice);
SWDLSS_API const char *swdlss_last_error(void);
#ifdef __cplusplus
}
#endif
