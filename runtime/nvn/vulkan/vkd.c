// Device-level entry points of the submission thread's Vulkan calls (vkd.h).
#define VKD_NO_REDIRECT
#include "vkd.h"

#define VKD_DEFINE(n) PFN_vk##n vkd_##n = vk##n;  // (the loader's until vkd_load)
VKD_FUNCS(VKD_DEFINE)

void vkd_load(VkDevice dev) {
#define VKD_LOAD(n) { PFN_vk##n p = (PFN_vk##n)vkGetDeviceProcAddr(dev, "vk" #n); vkd_##n = p ? p : vk##n; }
    VKD_FUNCS(VKD_LOAD)
}
