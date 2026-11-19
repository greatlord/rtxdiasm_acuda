#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"
#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_core.h>
#endif



static VKAPI_ATTR VkBool32 VKAPI_CALL validation_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *user) {
    (void)type;
    acuda_context_t ctx = user;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ctx->validation_errors++;
        fprintf(stderr, "ACuda validation ERROR: %s\n",
                data && data->pMessage ? data->pMessage : "(no message)");
    }
    return VK_FALSE;
}

int has_validation_layer(void) {
    // Validation (and its spirv-opt pass) is off by default; opt in with
    // ACUDA_VALIDATION=1. ACUDA_NO_VALIDATION still forces it off.
    if (getenv("ACUDA_NO_VALIDATION"))
        return 0;
    if (!getenv("ACUDA_VALIDATION"))
        return 0;
    uint32_t n = 0;
    if (vkEnumerateInstanceLayerProperties(&n, NULL) != VK_SUCCESS)
        return 0;
    VkLayerProperties *layers = malloc(sizeof(*layers) * (n ? n : 1));
    if (!layers)
        return 0;
    int found = 0;
    if (vkEnumerateInstanceLayerProperties(&n, layers) == VK_SUCCESS) {
        for (uint32_t i = 0; i < n; i++) {
            if (strcmp(layers[i].layerName, ACUDA_VALIDATION_LAYER) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(layers);
    return found;
}

VkResult create_messenger(acuda_context_t ctx) {
    PFN_vkCreateDebugUtilsMessengerEXT fn =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            ctx->instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn)
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = validation_cb,
        .pUserData = ctx,
    };
    return fn(ctx->instance, &mci, NULL, &ctx->messenger);
}

void destroy_messenger(acuda_context_t ctx) {
    if (ctx->messenger == VK_NULL_HANDLE)
        return;
    PFN_vkDestroyDebugUtilsMessengerEXT fn =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            ctx->instance, "vkDestroyDebugUtilsMessengerEXT");
    if (fn)
        fn(ctx->instance, ctx->messenger, NULL);
    ctx->messenger = VK_NULL_HANDLE;
}

// Timestamp queries are optional; 0 valid bits means unsupported.
uint32_t timestamp_valid_bits(VkPhysicalDevice dev, uint32_t family) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, NULL);
    if (family >= n)
        return 0;
    VkQueueFamilyProperties *p = malloc(sizeof(*p) * (n ? n : 1));
    if (!p)
        return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, p);
    uint32_t bits = family < n ? p[family].timestampValidBits : 0;
    free(p);
    return bits;
}

uint32_t find_compute_queue(VkPhysicalDevice dev) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, NULL);
    if (count == 0)
        return UINT32_MAX;
    VkQueueFamilyProperties *props = malloc(sizeof(*props) * count);
    if (!props)
        return UINT32_MAX;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props);
    const char *wanted = getenv("ACUDA_QUEUE_FAMILY");
    if (wanted && *wanted) {
        int v = atoi(wanted);
        if (v >= 0 && (uint32_t)v < count &&
            (props[v].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            free(props);
            return (uint32_t)v;
        }
    }
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            idx = i;
            break;
        }
    }
    if (idx != UINT32_MAX) {
        // Prefer a family that also supports graphics: depth-aspect
        // image copies (depth-compare textures) are only legal on
        // graphics queues without maintenance10.
        for (uint32_t i = 0; i < count; i++) {
            if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                idx = i;
                break;
            }
        }
    }
    free(props);
    return idx;
}

uint32_t find_host_memory(VkPhysicalDevice dev) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev, &mp);
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

uint32_t find_device_memory(VkPhysicalDevice dev, uint32_t host_type) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev, &mp);
    uint32_t fallback = host_type;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            continue;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            return i; // VRAM
        fallback = i;
    }
    return fallback;
}

int has_timeline(VkPhysicalDevice dev) {
    VkPhysicalDeviceTimelineSemaphoreFeatures tf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &tf,
    };
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    return tf.timelineSemaphore;
}

int has_int64(VkPhysicalDevice dev) {
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(dev, &feats);
    return feats.shaderInt64;
}

int has_float64(VkPhysicalDevice dev) {
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(dev, &feats);
    return feats.shaderFloat64;
}

int has_device_ext(VkPhysicalDevice dev, const char *want) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, NULL, &n, NULL) !=
        VK_SUCCESS)
        return 0;
    VkExtensionProperties *props = malloc(sizeof(*props) * (n ? n : 1));
    if (!props)
        return 0;
    int found = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, NULL, &n, props) ==
        VK_SUCCESS) {
        for (uint32_t i = 0; i < n; i++) {
            if (strcmp(props[i].extensionName, want) == 0) {
                found = 1;
                break;
            }
        }
    }
    free(props);
    return found;
}

// Vulkan 1.4 core features this backend relies on. maintenance4
// (LocalSizeId) and synchronization2 (the 1.3 barrier API) are
// required; the integer dot product is optional and only enables the
// native dp4a path.
void query_vk13(VkPhysicalDevice dev,
                       VkPhysicalDeviceVulkan13Features *out) {
    memset(out, 0, sizeof(*out));
    out->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = out,
    };
    vkGetPhysicalDeviceFeatures2(dev, &f2);
}

int device_is_dozen(VkPhysicalDevice dev) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(dev, &p);
    return strstr(p.deviceName, "Direct3D12") != NULL ||
           strstr(p.deviceName, "Dozen") != NULL;
}

int has_vk13_core(VkPhysicalDevice dev) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(dev, &p);
    if (p.apiVersion < VK_API_VERSION_1_4)
        return 0;
    VkPhysicalDeviceVulkan13Features f;
    query_vk13(dev, &f);
    return f.maintenance4 != VK_FALSE && f.synchronization2 != VK_FALSE;
}

// Guaranteed-fused OpFmaKHR for directed-rounding residuals. Optional:
// devices without VK_KHR_shader_fma use the compiler's integer fallback.
#ifndef VK_KHR_SHADER_FMA_EXTENSION_NAME
#define VK_KHR_SHADER_FMA_EXTENSION_NAME "VK_KHR_shader_fma"
#endif
#if !defined(VK_KHR_shader_fma)
typedef struct VkPhysicalDeviceShaderFmaFeaturesKHR {
    VkStructureType sType;
    void*           pNext;
    VkBool32        shaderFmaFloat32;
    VkBool32        shaderFmaFloat64;
    VkBool32        shaderFmaInt64;
} VkPhysicalDeviceShaderFmaFeaturesKHR;
#endif
#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR ((VkStructureType)1000579000)
#endif

int has_shader_fma(VkPhysicalDevice dev, int *f64_ok) {
    *f64_ok = 0;
#if defined(VK_KHR_SHADER_FMA_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR)
    if (!has_device_ext(dev, VK_KHR_SHADER_FMA_EXTENSION_NAME))
        return 0;
    VkPhysicalDeviceShaderFmaFeaturesKHR fma = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &fma,
    };
    vkGetPhysicalDeviceFeatures2(dev, &f2);
    *f64_ok = fma.shaderFmaFloat64;
    return fma.shaderFmaFloat32;
#else
    (void)dev;
    return 0;
#endif
}
