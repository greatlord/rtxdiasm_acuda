#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"
#include "gpu_clock.h"
#include "acuda/runtime/time_compat.h"
#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_core.h>
#define acuda_setenv(name, value) _putenv_s((name), (value))
#define acuda_unsetenv(name) _putenv_s((name), "")
#define acuda_strdup _strdup
#else
#include <pthread.h>
#include <dlfcn.h>
#define acuda_setenv(name, value) setenv((name), (value), 1)
#define acuda_unsetenv(name) unsetenv((name))
#define acuda_strdup strdup
#endif

#if !defined(VK_API_VERSION_1_4)
#error "ACuda requires Vulkan headers with Vulkan 1.4 support"
#endif

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

#ifndef VK_KHR_SHADER_CLOCK_EXTENSION_NAME
#define VK_KHR_SHADER_CLOCK_EXTENSION_NAME "VK_KHR_shader_clock"
#endif
#if !defined(VK_KHR_shader_clock)
typedef struct VkPhysicalDeviceShaderClockFeaturesKHR {
    VkStructureType sType;
    void*           pNext;
    VkBool32        shaderSubgroupClock;
    VkBool32        shaderDeviceClock;
} VkPhysicalDeviceShaderClockFeaturesKHR;
#endif
#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR ((VkStructureType)1000181000)
#endif


// Some drivers expose VK_KHR_shader_fma but still algebraically
// simplify fma(1, -a, a+b) to b (observed on RADV), which would make
// directed rounding wrong. Probe the actual pipeline once at init and
// fall back to the compiler's integer path when the probe disagrees.



acuda_status_t acuda_init(acuda_context_t *out_ctx) {
    if (getenv("ACUDA_TRACE")) {
        ACUDA_DEBUG_PRINT("ACuda-debug: acuda_init(out=%p)\n", (void *)out_ctx);
        fflush(stderr);
    }
    if (!out_ctx)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_ctx = NULL;

    acuda_context_t ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    acuda_compiler_caps_init(&ctx->compiler_caps);

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "acuda",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    const char *layer = ACUDA_VALIDATION_LAYER;
    const char *ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkResult vr;
    ctx->validation_on = has_validation_layer();
    if (ctx->validation_on) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &ext;
        vr = vkCreateInstance(&ici, NULL, &ctx->instance);
        if (vr != VK_SUCCESS) {
            // Fall back to a plain instance if validation breaks creation.
            ctx->validation_on = 0;
            memset(&ici, 0, sizeof(ici));
            ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            ici.pApplicationInfo = &app;
        }
    }
    if (!ctx->validation_on)
        vr = vkCreateInstance(&ici, NULL, &ctx->instance);
    if (vr != VK_SUCCESS) {
        if (vr == VK_ERROR_INCOMPATIBLE_DRIVER)
            fprintf(stderr,
                    "ACuda: Vulkan 1.4 is required, but the loader/driver "
                    "refused the instance\n");
        free(ctx);
        return acuda_from_vk(vr);
    }
    if (ctx->validation_on && create_messenger(ctx) != VK_SUCCESS)
        ctx->validation_on = 0; // keep going without the callback

    uint32_t total = 0;
    vr = vkEnumeratePhysicalDevices(ctx->instance, &total, NULL);
    if (vr != VK_SUCCESS || total == 0) {
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return vr == VK_SUCCESS ? ACUDA_ERROR_NO_DEVICE : acuda_from_vk(vr);
    }
    ctx->devices = malloc(sizeof(*ctx->devices) * total);
    if (!ctx->devices) {
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    ctx->device_total = total;
    vr = vkEnumeratePhysicalDevices(ctx->instance, &total, ctx->devices);
    if (vr != VK_SUCCESS && vr != VK_INCOMPLETE) {
        free(ctx->devices);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return acuda_from_vk(vr);
    }
    ctx->device_total = total;
    ctx->device_original = malloc(sizeof(*ctx->device_original) * total);
    if (!ctx->device_original) {
        free(ctx->devices);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < total; i++)
        ctx->device_original[i] = i;

    // Allow forcing a specific GPU: ACUDA_DEVICE / ACUDA_GPU can be an
    // index ("0", "1", ...) or a substring matched against the device name
    // or vendor ("nvidia", "amd", "intel").
    const char *devsel = getenv("ACUDA_DEVICE");
    if (!devsel)
        devsel = getenv("ACUDA_GPU");
    int force_index = -1;
    char devsel_lower[256] = { 0 };
    if (devsel) {
        int all_digits = 1;
        for (const char *c = devsel; *c; c++) {
            if (*c < '0' || *c > '9') { all_digits = 0; break; }
        }
        if (all_digits && *devsel)
            force_index = atoi(devsel);
        else {
            size_t n = strlen(devsel);
            if (n >= sizeof(devsel_lower)) n = sizeof(devsel_lower) - 1;
            for (size_t k = 0; k < n; k++)
                devsel_lower[k] = (char)tolower((unsigned char)devsel[k]);
            devsel_lower[n] = '\0';
        }
    }

    if (getenv("ACUDA_LIST_DEVICES")) {
        fprintf(stderr, "ACuda: %u Vulkan device(s) found:\n", total);
        for (uint32_t i = 0; i < total; i++) {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(ctx->devices[i], &pp);
            fprintf(stderr, "  [%u] %s (vendor 0x%04x, type %d)\n",
                    i, pp.deviceName, pp.vendorID, (int)pp.deviceType);
        }
    }

    // Prefer a discrete GPU with compute + timeline support.
    int found = 0;
    for (uint32_t pass = 0; pass < 2 && !found; pass++) {
        for (uint32_t i = 0; i < total; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(ctx->devices[i], &p);
            if (force_index >= 0) {
                if ((int)i != force_index)
                    continue;
            } else if (devsel_lower[0]) {
                // match vendor or name substring
                int match = 0;
                char name_lower[256];
                size_t nn = strlen(p.deviceName);
                if (nn >= sizeof(name_lower)) nn = sizeof(name_lower) - 1;
                for (size_t k = 0; k < nn; k++)
                    name_lower[k] = (char)tolower((unsigned char)p.deviceName[k]);
                name_lower[nn] = '\0';
                if (strstr(name_lower, devsel_lower))
                    match = 1;
                if (!match && strstr(devsel_lower, "nvidia") &&
                    p.vendorID == 0x10DE)
                    match = 1;
                if (!match && strstr(devsel_lower, "amd") &&
                    p.vendorID == 0x1002)
                    match = 1;
                if (!match && strstr(devsel_lower, "intel") &&
                    p.vendorID == 0x8086)
                    match = 1;
                if (!match)
                    continue;
            } else {
                if (pass == 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                    continue;
            }
            if (find_compute_queue(ctx->devices[i]) == UINT32_MAX)
                continue;
            if (!has_timeline(ctx->devices[i]))
                continue;
            if (!has_int64(ctx->devices[i]))
                continue;
            if (!has_vk13_core(ctx->devices[i]))
                continue;
            // NOTE: shaderFloat64 is NOT required for selection: f32-only
            // devices must still init (f64 kernels fail cleanly at launch
            // when the feature is missing; see ctx->f64_ok).
            fprintf(stderr, "ACuda: selected device [%u] %s\n",
                    i, p.deviceName);
            // Make the selected (discrete) device ordinal 0 so that
            // cuDeviceGet(0) / cudaSetDevice(0) target it by default.
            if (i != 0) {
                VkPhysicalDevice tmp = ctx->devices[0];
                ctx->devices[0] = ctx->devices[i];
                ctx->devices[i] = tmp;
                uint32_t ti = ctx->device_original[0];
                ctx->device_original[0] = ctx->device_original[i];
                ctx->device_original[i] = ti;
            }
            ctx->selected = 0;
            ctx->phys = ctx->devices[0];
            ctx->props = p;
            query_pci_info(ctx->phys, &ctx->pci_domain, &ctx->pci_bus,
                           &ctx->pci_device, &ctx->pci_function);
            /* GPU clock reporting: prefer ADL2 (AMD, Windows) when available,
             * then fall back to hwinfo for any still-missing values. */
#if defined(_WIN32)
            if (ctx->props.vendorID == 0x1002) {
                int adl_core = 0, adl_mem = 0;
                if (acuda_query_adl_gpu_clock_khz(ctx->pci_domain, ctx->pci_bus,
                                                  ctx->pci_device, ctx->pci_function,
                                                  &adl_core, &adl_mem)) {
                    if (adl_core > 0)
                        ctx->core_clock_khz = adl_core;
                    if (adl_mem > 0)
                        ctx->mem_clock_khz = adl_mem;
                }
            }
#endif
#ifdef ACUDA_USE_HWINFO
            if (ctx->core_clock_khz <= 0 || ctx->mem_clock_khz <= 0) {
                int hw_core = 0, hw_mem = 0;
                (void)acuda_query_real_gpu_clock_khz(ctx->pci_domain, ctx->pci_bus,
                                                     ctx->pci_device, ctx->pci_function,
                                                     &hw_core, &hw_mem);
                if (ctx->core_clock_khz <= 0 && hw_core > 0)
                    ctx->core_clock_khz = hw_core;
                if (ctx->mem_clock_khz <= 0 && hw_mem > 0)
                    ctx->mem_clock_khz = hw_mem;
            }
#endif
            /* GPU memory reporting: prefer ADL2 (AMD, Windows) when available,
             * then fall back to hwinfo for any still-missing value. */
#if defined(_WIN32)
            if (ctx->device_mem_bytes == 0 && ctx->props.vendorID == 0x1002) {
                uint64_t adl_mem_bytes = 0;
                if (acuda_query_adl_gpu_memory_bytes(ctx->pci_domain, ctx->pci_bus,
                                                     ctx->pci_device, ctx->pci_function,
                                                     &adl_mem_bytes) &&
                    adl_mem_bytes > 0)
                    ctx->device_mem_bytes = adl_mem_bytes;
            }
#endif
#ifdef ACUDA_USE_HWINFO
            if (ctx->device_mem_bytes == 0) {
                uint64_t hw_mem_bytes = 0;
                if (acuda_query_real_gpu_memory_bytes(ctx->pci_domain, ctx->pci_bus,
                                                      ctx->pci_device, ctx->pci_function,
                                                      &hw_mem_bytes) &&
                    hw_mem_bytes > 0)
                    ctx->device_mem_bytes = hw_mem_bytes;
            }
#endif
            if (ctx->device_mem_bytes == 0) {
                VkPhysicalDeviceMemoryProperties mp;
                vkGetPhysicalDeviceMemoryProperties(ctx->phys, &mp);
                for (uint32_t hi = 0; hi < mp.memoryHeapCount; hi++) {
                    if (mp.memoryHeaps[hi].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                        uint64_t sz = (uint64_t)mp.memoryHeaps[hi].size;
                        if (sz > ctx->device_mem_bytes)
                            ctx->device_mem_bytes = sz;
                    }
                }
            }
            found = 1;
            break;
        }
    }
    if (!found) {
        fprintf(stderr,
                "ACuda: no Vulkan 1.3 compute device with maintenance4 + "
                "synchronization2 support found\n");
        free(ctx->devices);
        free(ctx->device_original);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return ACUDA_ERROR_NO_DEVICE;
    }
    ctx->dozen = device_is_dozen(ctx->phys);
    ctx->max_block_x = ctx->props.limits.maxComputeWorkGroupSize[0];
    if (ctx->dozen) {
#ifndef _WIN32
        setenv("ACUDA_DZN", "1", 1);
#endif
        fprintf(stderr,
                "ACuda: maintenance4 unavailable; using per-launch "
                "LocalSize fallback\n");
    }
    ctx->queue_family = find_compute_queue(ctx->phys);
    ctx->timestamp_bits = timestamp_valid_bits(ctx->phys, ctx->queue_family);
    ctx->mem_type = find_host_memory(ctx->phys);
    ctx->mem_type_dev = ctx->mem_type == UINT32_MAX
                            ? UINT32_MAX
                            : find_device_memory(ctx->phys, ctx->mem_type);
    if (ctx->mem_type == UINT32_MAX) {
        free(ctx->devices);
        free(ctx->device_original);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }

    VkPhysicalDeviceFeatures2 f2en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };
    vkGetPhysicalDeviceFeatures(ctx->phys, &f2en.features);
    f2en.features.shaderInt64 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features v12q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &v11q,
    };
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pexecq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    };
    v11q.pNext = &pexecq;
    VkPhysicalDeviceVulkan13Features v13q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &v12q,
    };
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT i64imgq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
        .pNext = &v13q,
    };
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
        .pNext = &i64imgq,
    };
    VkPhysicalDeviceShaderClockFeaturesKHR clockq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
        .pNext = &af2q,
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &clockq,
    };
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        .pNext = &afq,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = &bf16q,
    };
#else
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmq = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = &afq,
    };
#endif
    acuda_mfdp_features_t mfq = { .sType = ACUDA_ST_MFDP };
    mfq.pNext = &cmq;
    VkPhysicalDeviceFeatures2 f2q = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &mfq,
    };
    vkGetPhysicalDeviceFeatures2(ctx->phys, &f2q);
    ctx->maintenance4_ok = v13q.maintenance4 != VK_FALSE;
    if (getenv("ACUDA_NO_MAINTENANCE4"))
        ctx->maintenance4_ok = 0;
    // VK_EXT_shader_atomic_float gates native OpAtomicFAddEXT on shared
    // f32 accumulators. Only the shared feature is usable here: the
    // compiler's storage buffers are u32-word views and cannot expose a
    // typed float pointer for the buffer form.
    int atomic_float_ext =
        has_device_ext(ctx->phys, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    int atomic_float2_ext =
        has_device_ext(ctx->phys, VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    unsigned atomic_mask = 0;
    if (atomic_float_ext) {
        if (afq.shaderSharedFloat32AtomicAdd)
            atomic_mask |= ACUDA_ATOMIC_F32_SHARED;
        if (afq.shaderBufferFloat32AtomicAdd)
            atomic_mask |= ACUDA_ATOMIC_F32_BUFFER;
        if (afq.shaderSharedFloat64AtomicAdd)
            atomic_mask |= ACUDA_ATOMIC_F64_SHARED;
        if (afq.shaderBufferFloat64AtomicAdd)
            atomic_mask |= ACUDA_ATOMIC_F64_BUFFER;
        if (afq.shaderImageFloat32AtomicAdd)
            atomic_mask |= ACUDA_ATOMIC_IMAGE_F32_ADD;
        if (atomic_float2_ext && af2q.shaderImageFloat32AtomicMinMax)
            atomic_mask |= ACUDA_ATOMIC_IMAGE_F32_MINMAX;
    }
    if (getenv("ACUDA_PTX_NO_ATOMIC_FLOAT") ||
        getenv("ACUDA_PTX_NO_ATOMIC_F32_ADD"))
        atomic_mask = 0;
    if (getenv("ACUDA_DEBUG_ATOMIC"))
        fprintf(stderr, "ACuda: atomic float features=0x%x\n", atomic_mask);
    int mfdot_ok =
        mfq.shaderMixedFloatDotProductFloat16AccFloat32 != VK_FALSE &&
        has_device_ext(ctx->phys, ACUDA_MFDP_EXT);
    if (getenv("ACUDA_NO_FDOT2"))
        mfdot_ok = 0;
    // The installed validation layer predates the extension's feature
    // structure; only enable it when validation is off (or forced).
    if (ctx->validation_on && !getenv("ACUDA_FDOT2"))
        mfdot_ok = 0;
    int cm_ext = has_device_ext(ctx->phys,
                                VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)
    int bf16_ext = has_device_ext(ctx->phys,
                                  VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
#else
    int bf16_ext = 0;
#endif
    int cm_f16 = 0, cm_bf16 = 0;
    if (cm_ext && !getenv("ACUDA_NO_COOPMAT")) {
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR cmget =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
                vkGetInstanceProcAddr(ctx->instance,
                    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (cmget) {
            uint32_t pc = 0;
            if (cmget(ctx->phys, &pc, NULL) == VK_SUCCESS && pc > 0) {
                VkCooperativeMatrixPropertiesKHR *cp =
                    calloc(pc, sizeof(*cp));
                if (cp) {
                    for (uint32_t i = 0; i < pc; i++)
                        cp[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                    if (cmget(ctx->phys, &pc, cp) == VK_SUCCESS) {
                        for (uint32_t i = 0; i < pc; i++) {
                            if (cp[i].MSize != 16 || cp[i].NSize != 16 ||
                                cp[i].KSize != 16 ||
                                cp[i].CType != VK_COMPONENT_TYPE_FLOAT32_KHR ||
                                cp[i].ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR)
                                continue;
                            if (cp[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                cp[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR)
                                cm_f16 = 1;
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR) && defined(VK_COMPONENT_TYPE_BFLOAT16_KHR)
                            if (bf16_ext && bf16q.shaderBFloat16CooperativeMatrix &&
                                cp[i].AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                                cp[i].BType == VK_COMPONENT_TYPE_BFLOAT16_KHR)
                                cm_bf16 = 1;
#endif
                        }
                    }
                    free(cp);
                }
            }
        }
    }
    if (getenv("ACUDA_NO_COOPMAT")) {
        cm_f16 = 0;
        cm_bf16 = 0;
    }
    // The installed validation layer predates cooperative matrix and
    // rejects its memory operands. Keep the validated path on the scalar
    // fallback unless the user explicitly forces the extension.
    if (ctx->validation_on && !getenv("ACUDA_COOPMAT")) {
        cm_f16 = 0;
        cm_bf16 = 0;
    }
    acuda_compiler_set_cooperative_matrix(&ctx->compiler_caps, cm_f16, cm_bf16);
    if (getenv("ACUDA_DEBUG_COOPMAT"))
        fprintf(stderr,
                "ACuda: cooperative matrix ext=%d bf16ext=%d f16=%d bf16=%d\n",
                cm_ext, bf16_ext, cm_f16, cm_bf16);
    VkPhysicalDeviceSubgroupProperties subprops = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties scprops = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
        .pNext = &subprops,
    };
    VkPhysicalDeviceFloatControlsProperties fcprops = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
        .pNext = &scprops,
    };
    VkPhysicalDeviceShaderIntegerDotProductProperties dprops = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
        .pNext = &fcprops,
    };
    VkPhysicalDeviceVulkan13Properties v13props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
        .pNext = &dprops,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &v13props,
    };
    vkGetPhysicalDeviceProperties2(ctx->phys, &p2);
    ctx->subgroup_size = (int)subprops.subgroupSize;
    ctx->subgroup_ops = (uint32_t)subprops.supportedOperations;
    ctx->subgroup_stages = (uint32_t)subprops.supportedStages;
    ctx->max_workgroup_subgroups = v13props.maxComputeWorkgroupSubgroups;
    ctx->max_buffer_size = v13props.maxBufferSize;
    ctx->max_image_dimension1d = ctx->props.limits.maxImageDimension1D;
    ctx->max_image_dimension2d = ctx->props.limits.maxImageDimension2D;
    // The only graphics path (the MSAA fill) prefers dynamic rendering.
    // It is optional: when absent the fill keeps using its render pass.
    ctx->dynamic_rendering_ok = v13q.dynamicRendering != VK_FALSE;
    if (getenv("ACUDA_PTX_NO_DYNAMIC_RENDERING"))
        ctx->dynamic_rendering_ok = 0;
    // computeFullSubgroups lets a pipeline promise subgroup-multiple
    // workgroups. Launches stay arbitrary-sized, so the flag is only set
    // per pipeline when blockDim.x is already a multiple of the subgroup.
    ctx->full_subgroups_ok = v13q.computeFullSubgroups != VK_FALSE;
    if (getenv("ACUDA_PTX_NO_FULL_SUBGROUPS"))
        ctx->full_subgroups_ok = 0;
    // Optional ISA introspection hook (ACUDA_DUMP_ISA).
    ctx->pipe_exec_ok =
        has_device_ext(ctx->phys,
                       VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) &&
        pexecq.pipelineExecutableInfo != VK_FALSE;
    ctx->subgroup_min = scprops.minSubgroupSize;
    ctx->subgroup_max = scprops.maxSubgroupSize;
    ctx->subgroup32_ok =
        v13q.subgroupSizeControl &&
        (scprops.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        scprops.minSubgroupSize <= 32 && scprops.maxSubgroupSize >= 32;
    // Optional occupancy experiment: force a subgroup size for kernels
    // that do not need PTX's 32-lane warp semantics.
    if (getenv("ACUDA_SUBGROUP_SIZE")) {
        unsigned v = (unsigned)strtoul(getenv("ACUDA_SUBGROUP_SIZE"),
                                       NULL, 10);
        if (v13q.subgroupSizeControl &&
            (scprops.requiredSubgroupSizeStages &
             VK_SHADER_STAGE_COMPUTE_BIT) &&
            v >= scprops.minSubgroupSize && v <= scprops.maxSubgroupSize)
            ctx->pref_subgroup_size = v;
        else
            fprintf(stderr,
                    "ACuda: ignoring ACUDA_SUBGROUP_SIZE=%u (outside "
                    "device range %u..%u)\n",
                    v, scprops.minSubgroupSize, scprops.maxSubgroupSize);
    }
    if (getenv("ACUDA_PTX_NO_SUBGROUP32"))
        ctx->subgroup32_ok = 0;
    if (getenv("ACUDA_DEBUG_SUBGROUP"))
        fprintf(stderr,
                "ACuda: subgroup default=%d control=%d range=[%u,%u] "
                "require32=%d pref=%u\n",
                ctx->subgroup_size, (int)v13q.subgroupSizeControl,
                scprops.minSubgroupSize, scprops.maxSubgroupSize,
                ctx->subgroup32_ok, ctx->pref_subgroup_size);
    ctx->i64_atomics_ok = v12q.shaderBufferInt64Atomics &&
                          v12q.shaderSharedInt64Atomics;
    int int64_image_ok =
        ctx->i64_atomics_ok &&
        has_device_ext(ctx->phys,
                       VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME) &&
        i64imgq.shaderImageInt64Atomics;
    if (getenv("ACUDA_PTX_NO_INT64_IMAGE"))
        int64_image_ok = 0;
    if (getenv("ACUDA_DEBUG_ATOMIC"))
        fprintf(stderr, "ACuda: image int64 atomics=%d\n", int64_image_ok);
    int shader_clock_ext =
        has_device_ext(ctx->phys, VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    int shader_clock_device_ok =
        shader_clock_ext && clockq.shaderDeviceClock != VK_FALSE;
    int shader_clock_subgroup_ok =
        shader_clock_ext && clockq.shaderSubgroupClock != VK_FALSE;
    if (getenv("ACUDA_NO_SHADER_CLOCK")) {
        shader_clock_device_ok = 0;
        shader_clock_subgroup_ok = 0;
    }
    if (getenv("ACUDA_DEBUG_CLOCK"))
        fprintf(stderr, "ACuda: shader clock ext=%d device=%d subgroup=%d\n",
                shader_clock_ext, shader_clock_device_ok,
                shader_clock_subgroup_ok);
    ctx->desc_idx_ok =
        v12q.runtimeDescriptorArray &&
        v12q.shaderStorageImageArrayNonUniformIndexing &&
        v12q.descriptorBindingVariableDescriptorCount &&
        f2q.features.shaderStorageImageArrayDynamicIndexing;
    VkPhysicalDeviceVulkan11Features v11en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f2en,
    };
    VkPhysicalDeviceVulkan12Features v12en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .timelineSemaphore = VK_TRUE,
        .pNext = &v11en,
    };
    VkPhysicalDeviceVulkan13Features v13en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .maintenance4 = ctx->maintenance4_ok ? VK_TRUE : VK_FALSE,
        .synchronization2 = VK_TRUE,
        .pNext = &v12en,
    };
    if (v13q.subgroupSizeControl)
        v13en.subgroupSizeControl = VK_TRUE;
    if (ctx->dynamic_rendering_ok)
        v13en.dynamicRendering = VK_TRUE;
    if (ctx->full_subgroups_ok)
        v13en.computeFullSubgroups = VK_TRUE;
    if (v13q.shaderZeroInitializeWorkgroupMemory)
        v13en.shaderZeroInitializeWorkgroupMemory = VK_TRUE;
    if (ctx->i64_atomics_ok) {
        v12en.shaderBufferInt64Atomics = VK_TRUE;
        v12en.shaderSharedInt64Atomics = VK_TRUE;
    }
    if (ctx->desc_idx_ok) {
        v12en.runtimeDescriptorArray = VK_TRUE;
        v12en.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
        v12en.descriptorBindingVariableDescriptorCount = VK_TRUE;
        f2en.features.shaderStorageImageArrayDynamicIndexing = VK_TRUE;
    }
    // Optional 1.2 features. 8-bit storage enables the native byte SSBO
    // path; bufferDeviceAddress is enabled as groundwork for the
    // raw-pointer work (the compiler still uses binding+offset
    // addressing). The Vulkan memory model is deliberately not enabled
    // until the shader side adopts the capability: turning it on
    // without vulkanMemoryModelDeviceScope makes the validator reject
    // the existing Device-scope atomics/barriers.
    int ssbo8_ok = v12q.storageBuffer8BitAccess != VK_FALSE;
    if (getenv("ACUDA_PTX_NO_SSBO8"))
        ssbo8_ok = 0;
    if (ssbo8_ok)
        v12en.storageBuffer8BitAccess = VK_TRUE;
    // 16-bit storage types also need the Int16 capability, which the
    // Vulkan environment gates on the shaderInt16 feature.
    int int16_ok = f2q.features.shaderInt16 != VK_FALSE;
    if (int16_ok)
        f2en.features.shaderInt16 = VK_TRUE;
    int f16_ok = v12q.shaderFloat16 != VK_FALSE;
    if (f16_ok)
        v12en.shaderFloat16 = VK_TRUE;
    int ssbo16_ok = v11q.storageBuffer16BitAccess != VK_FALSE && int16_ok;
    if (getenv("ACUDA_PTX_NO_SSBO16"))
        ssbo16_ok = 0;
    if (ssbo16_ok)
        v11en.storageBuffer16BitAccess = VK_TRUE;
    ctx->bda_ok = v12q.bufferDeviceAddress != VK_FALSE;
    if (ctx->bda_ok)
        v12en.bufferDeviceAddress = VK_TRUE;
    // Vulkan memory model: the compiler declares the capability and
    // uses explicit scopes/semantics on atomics and barriers. Device
    // scope additionally requires vulkanMemoryModelDeviceScope.
    int vmm_ok = v12q.vulkanMemoryModel && v12q.vulkanMemoryModelDeviceScope;
    if (getenv("ACUDA_PTX_NO_VMM"))
        vmm_ok = 0;
    if (vmm_ok) {
        v12en.vulkanMemoryModel = VK_TRUE;
        v12en.vulkanMemoryModelDeviceScope = VK_TRUE;
    }
    // Enable Float64 only where the device has it; f64 kernels check
    // ctx->f64_ok at launch and fail cleanly otherwise.
    ctx->f64_ok = has_float64(ctx->phys);
    if (ctx->f64_ok)
        f2en.features.shaderFloat64 = VK_TRUE;
    // Float controls (Vulkan 1.2 core): preserve denormals and the
    // signed-zero/Inf/NaN rules instead of accepting the driver's
    // default (RADV historically flushes f32 denormal *results* on
    // add). Declaring the modes requires both feature bits.
    int fc32_ok = fcprops.shaderDenormPreserveFloat32 &&
                  fcprops.shaderSignedZeroInfNanPreserveFloat32;
    int fc64_ok = ctx->f64_ok && fcprops.shaderDenormPreserveFloat64 &&
                  fcprops.shaderSignedZeroInfNanPreserveFloat64;
    if (getenv("ACUDA_PTX_NO_FLOAT_CONTROLS")) {
        fc32_ok = 0;
        fc64_ok = 0;
    }
    // Float controls are properties, not enableable feature bits: the
    // shader declares the execution modes and the implementation must
    // report support here.
    if (getenv("ACUDA_DEBUG_FLOAT"))
        fprintf(stderr,
                "ACuda: float controls 32=%d 64=%d (denorm preserve "
                "%d/%d, signed-zero/NaN %d/%d)\n",
                fc32_ok, fc64_ok,
                (int)fcprops.shaderDenormPreserveFloat32,
                (int)fcprops.shaderDenormPreserveFloat64,
                (int)fcprops.shaderSignedZeroInfNanPreserveFloat32,
                (int)fcprops.shaderSignedZeroInfNanPreserveFloat64);
    int fma64_ok = 0;
    int fma_ok = has_shader_fma(ctx->phys, &fma64_ok);
    int fma_cap_f32 = fma_ok;
    int fma_cap_f64 = fma64_ok;
    int fma_mode = 0; /* 0=auto (probe), 1=prefer hardware, 2=always software */
    {
        const char *fm = getenv("ACUDA_FMA_MODE");
        if (fm && *fm) {
            if (strcmp(fm, "1") == 0) fma_mode = 1;
            else if (strcmp(fm, "2") == 0) fma_mode = 2;
            else fma_mode = 0;
        }
    }
    if (getenv("ACUDA_DEBUG_FMA"))
        fprintf(stderr, "ACuda: shader FMA detect f32=%d f64=%d mode=%d\n",
                fma_ok, fma64_ok, fma_mode);
    if (fma_mode == 2) {
        fma_ok = 0;
        fma64_ok = 0;
    }
#if defined(VK_KHR_SHADER_FMA_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR)
    VkPhysicalDeviceShaderFmaFeaturesKHR fmaf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR,
    };
#endif
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afen = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT i64imgen = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
    };
    VkPhysicalDeviceShaderClockFeaturesKHR clocken = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
    };
    // shaderIntegerDotProduct is optional in Vulkan 1.4. It only
    // selects the native dp4a lowering; the compiler keeps the exact
    // fallback when it is absent (or disabled by the env knob).
    int dot_ok = v13q.shaderIntegerDotProduct != VK_FALSE;
    int dot_signed = dot_ok &&
        dprops.integerDotProduct4x8BitPackedSignedAccelerated;
    int dot_unsigned = dot_ok &&
        dprops.integerDotProduct4x8BitPackedUnsignedAccelerated;
    int dot_mixed = dot_ok &&
        dprops.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    if (getenv("ACUDA_PTX_NO_DOT_ACCEL"))
        dot_signed = dot_unsigned = dot_mixed = 0;
    if (getenv("ACUDA_PTX_NO_DOT_PRODUCT"))
        dot_ok = 0;
    if (!dot_ok)
        dot_signed = dot_unsigned = dot_mixed = 0;
    if (dot_ok)
        v13en.shaderIntegerDotProduct = VK_TRUE;
    if (getenv("ACUDA_DEBUG_DOT"))
        fprintf(stderr,
                "ACuda: integer dot feature=%d packed S=%d U=%d SU=%d\n",
                dot_ok, dot_signed, dot_unsigned, dot_mixed);
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pexecen = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
        .pipelineExecutableInfo = VK_TRUE,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmen = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
    };
#endif
    void *chain = &v13en;
    acuda_mfdp_features_t mfen = { .sType = ACUDA_ST_MFDP };
    if (mfdot_ok) {
        mfen.shaderMixedFloatDotProductFloat16AccFloat32 = VK_TRUE;
        mfen.pNext = chain;
        chain = &mfen;
    }
    if (cm_f16 || cm_bf16) {
        cmen.cooperativeMatrix = VK_TRUE;
        cmen.pNext = chain;
        chain = &cmen;
    }
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)
    if (cm_bf16) {
        bf16en.shaderBFloat16Type = VK_TRUE;
        bf16en.shaderBFloat16CooperativeMatrix = VK_TRUE;
        bf16en.pNext = chain;
        chain = &bf16en;
    }
#endif
    if (ctx->pipe_exec_ok) {
        pexecen.pNext = chain;
        chain = &pexecen;
    }
#if defined(VK_KHR_SHADER_FMA_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR)
    if (fma_ok || fma64_ok) {
        fmaf.shaderFmaFloat32 = fma_ok ? VK_TRUE : VK_FALSE;
        fmaf.shaderFmaFloat64 = fma64_ok ? VK_TRUE : VK_FALSE;
        fmaf.pNext = chain;
        chain = &fmaf;
    }
#endif
    if (atomic_mask) {
        afen.shaderSharedFloat32AtomicAdd =
            (atomic_mask & ACUDA_ATOMIC_F32_SHARED) ? VK_TRUE : VK_FALSE;
        afen.shaderBufferFloat32AtomicAdd =
            (atomic_mask & ACUDA_ATOMIC_F32_BUFFER) ? VK_TRUE : VK_FALSE;
        afen.shaderSharedFloat64AtomicAdd =
            (atomic_mask & ACUDA_ATOMIC_F64_SHARED) ? VK_TRUE : VK_FALSE;
        afen.shaderBufferFloat64AtomicAdd =
            (atomic_mask & ACUDA_ATOMIC_F64_BUFFER) ? VK_TRUE : VK_FALSE;
        afen.shaderImageFloat32AtomicAdd =
            (atomic_mask & ACUDA_ATOMIC_IMAGE_F32_ADD) ? VK_TRUE : VK_FALSE;
        afen.pNext = chain;
        chain = &afen;
    }
    if (atomic_mask & ACUDA_ATOMIC_IMAGE_F32_MINMAX) {
        af2en.shaderImageFloat32AtomicMinMax = VK_TRUE;
        af2en.pNext = chain;
        chain = &af2en;
    }
    if (int64_image_ok) {
        i64imgen.shaderImageInt64Atomics = VK_TRUE;
        i64imgen.pNext = chain;
        chain = &i64imgen;
    }
    if (shader_clock_device_ok || shader_clock_subgroup_ok) {
        clocken.shaderDeviceClock = shader_clock_device_ok ? VK_TRUE : VK_FALSE;
        clocken.shaderSubgroupClock = shader_clock_subgroup_ok ? VK_TRUE : VK_FALSE;
        clocken.pNext = chain;
        chain = &clocken;
    }
    // Request a second queue from the same family when ACUDA_HIGH_PERF is set,
    // so the keep-busy thread can submit without racing user submissions.
    // Queue priorities are hints only; the real clock ramp comes from sustained
    // load. Always ask for at least one queue.
    int high_perf = 1;
    int high_perf_forced = 0;
    {
        const char *hp = getenv("ACUDA_HIGH_PERF");
        if (hp) {
            if (strcmp(hp, "0") == 0)
                high_perf = 0;
            else
                high_perf_forced = 1;
        }
    }
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, NULL);
    VkQueueFamilyProperties *qfps = nqf ? calloc(nqf, sizeof(*qfps)) : NULL;
    if (qfps) {
        vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, qfps);
        if (getenv("ACUDA_DEBUG_STREAMS")) {
            for (uint32_t i = 0; i < nqf; i++)
                fprintf(stderr,
                        "ACuda queue family %u: flags=0x%x queues=%u "
                        "timestamp=%u\n",
                        i, qfps[i].queueFlags, qfps[i].queueCount,
                        qfps[i].timestampValidBits);
        }
    }
    uint32_t avail_q = 1;
    if (qfps && ctx->queue_family < nqf)
        avail_q = qfps[ctx->queue_family].queueCount;
    if (avail_q < 1)
        avail_q = 1;
    // Async multi-queue is the default.  Set ACUDA_ASYNC_STREAMS=0 to opt
    // back into one serialized Vulkan queue (the legacy behavior).
    int async_streams = 1;
    {
        const char *as = getenv("ACUDA_ASYNC_STREAMS");
        if (as && *as && atoi(as) == 0)
            async_streams = 0;
    }
    // Reserve queue 0 for the default stream and use the remaining queues
    // from the same family for user streams.  Same-family queues avoid
    // Vulkan ownership transfers while allowing independent submissions.
    uint32_t want_users = 0;
    const char *stream_q = getenv("ACUDA_STREAM_QUEUES");
    if (!async_streams) {
        want_users = 0;
    } else if (stream_q && *stream_q) {
        int v = atoi(stream_q);
        if (v < 0)
            v = 0;
        if (v > ACUDA_MAX_USER_QUEUES)
            v = ACUDA_MAX_USER_QUEUES;
        want_users = (uint32_t)v;
    } else if (avail_q > 1) {
        want_users = avail_q - 1;
        if (want_users > 4)
            want_users = 4;
    }
    uint32_t nqueues = 1 + want_users;
    if (nqueues > avail_q)
        nqueues = avail_q;
    // Keep-busy only gets a queue when the user did not ask for all of
    // them; stream concurrency has priority over the clock-ramp hack.
    uint32_t bg_index = UINT32_MAX;
    if (async_streams && high_perf && avail_q > nqueues) {
        bg_index = nqueues;
        nqueues++;
    }
    (void)high_perf_forced;
    free(qfps);
    float prios[ACUDA_MAX_USER_QUEUES + 2];
    for (uint32_t i = 0; i < nqueues; i++)
        prios[i] = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = nqueues,
        .pQueuePriorities = prios,
    };
    uint32_t nqci = 1;
    const char *dev_exts[9] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    uint32_t ndev_exts = 0;
#if defined(VK_KHR_SHADER_FMA_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR)
    if (fma_ok || fma64_ok)
        dev_exts[ndev_exts++] = VK_KHR_SHADER_FMA_EXTENSION_NAME;
#endif
    if (ctx->pipe_exec_ok)
        dev_exts[ndev_exts++] =
            VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
    if (mfdot_ok)
        dev_exts[ndev_exts++] = ACUDA_MFDP_EXT;
    if (atomic_mask)
        dev_exts[ndev_exts++] = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;
    if (atomic_mask & ACUDA_ATOMIC_IMAGE_F32_MINMAX)
        dev_exts[ndev_exts++] =
            VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME;
    if (int64_image_ok)
        dev_exts[ndev_exts++] =
            VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME;
    if (cm_f16 || cm_bf16)
        dev_exts[ndev_exts++] =
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
#if defined(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR)
    if (cm_bf16)
        dev_exts[ndev_exts++] =
            VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
#endif
    if (shader_clock_device_ok || shader_clock_subgroup_ok)
        dev_exts[ndev_exts++] = VK_KHR_SHADER_CLOCK_EXTENSION_NAME;
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = chain,
        .queueCreateInfoCount = nqci,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = ndev_exts,
        .ppEnabledExtensionNames = dev_exts,
    };
    vr = vkCreateDevice(ctx->phys, &dci, NULL, &ctx->device);
    if (vr != VK_SUCCESS) {
        free(ctx->devices);
        free(ctx->device_original);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return acuda_from_vk(vr);
    }
    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);
    if (getenv("ACUDA_TRACE"))
        ACUDA_DEBUG_PRINT("acuda-trace .. device exts: map_memory_placed=%d "
                          "map_memory2=%d\n",
                          has_device_ext(ctx->phys,
                                         "VK_EXT_map_memory_placed"),
                          has_device_ext(ctx->phys, "VK_KHR_map_memory2"));
    ctx->bg_queue = VK_NULL_HANDLE;
    ctx->bg_queue_family = ctx->queue_family;
    ctx->nuser_queues = 0;
    ctx->next_user_queue = 0;
    for (uint32_t i = 1; i < nqueues; i++) {
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(ctx->device, ctx->queue_family, i, &q);
        if (i == bg_index) {
            ctx->bg_queue = q;
        } else if (ctx->nuser_queues < ACUDA_MAX_USER_QUEUES) {
            ctx->user_queues[ctx->nuser_queues++] = q;
        }
    }
    if (getenv("ACUDA_DEBUG_STREAMS"))
        fprintf(stderr,
                "ACuda streams: primary family %u, %u user queues, "
                "keep-busy queue %s\n",
                ctx->queue_family, ctx->nuser_queues,
                ctx->bg_queue ? "enabled" : "disabled");
    if (ctx->pipe_exec_ok) {
        ctx->get_pipe_exec_props =
            (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(
                ctx->device, "vkGetPipelineExecutablePropertiesKHR");
        ctx->get_pipe_exec_reprs =
            (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)
                vkGetDeviceProcAddr(
                    ctx->device,
                    "vkGetPipelineExecutableInternalRepresentationsKHR");
        if (!ctx->get_pipe_exec_props || !ctx->get_pipe_exec_reprs)
            ctx->pipe_exec_ok = 0;
    }

    acuda_status_t status =
        acuda_stream_create_primary(ctx, &ctx->default_stream);
    if (status != ACUDA_SUCCESS) {
        vkDestroyDevice(ctx->device, NULL);
        free(ctx->devices);
        free(ctx->device_original);
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return status;
    }
    // Nudge the GPU into its boosted clock state (cheap, ~50ms by default).
    acuda_warmup_gpu(ctx);
    // Optional sustained keep-busy (opt-in via ACUDA_HIGH_PERF=1).
    acuda_start_high_perf(ctx);
    /* ACUDA_FMA_MODE: 0=auto (probe when capability exists), 1=prefer hardware
     * when capability exists (skip probes), 2=always software. */
    int fma_probe_off = (fma_mode != 0);
    int had_fma = fma_ok;
    if (fma_ok && !fma_probe_off) {
        if (!probe_shader_fma(ctx) || !probe_shader_fma_rn(ctx)) {
            fma_ok = 0;
            acuda_compiler_set_shader_fma(&ctx->compiler_caps, 0);
        }
    }
    if (had_fma && !fma_ok && fma_mode == 0)
        fprintf(stderr,
                "ACuda: VK_KHR_shader_fma miscompiles fused f32 "
                "rounding here; using the portable fallback\n");
    int had_fma64 = fma64_ok;
    if (fma64_ok) {
        if (!fma_ok || !ctx->f64_ok)
            fma64_ok = 0;
        else if (!fma_probe_off && !probe_shader_fma64(ctx))
            fma64_ok = 0;
        if (!fma64_ok)
            acuda_compiler_set_shader_fma_f64(&ctx->compiler_caps, 0);
    }
    if (had_fma64 && !fma64_ok && fma_ok && fma_mode == 0)
        fprintf(stderr,
                "ACuda: VK_KHR_shader_fma f64 miscompiles directed f64 "
                "rounding here; f64 uses the portable fallback\n");
    // Tell the compiler which directed-rounding path this device needs.
    acuda_compiler_set_shader_fma(&ctx->compiler_caps, fma_ok);
    acuda_compiler_set_shader_fma_f64(&ctx->compiler_caps, fma64_ok);
    acuda_compiler_set_soft_fma(&ctx->compiler_caps, fma_mode == 2 ? 1 : 0);
    if (getenv("ACUDA_DEBUG_FMA"))
        fprintf(stderr,
                "ACuda: shader FMA final f32=%d f64=%d mode=%d soft=%d\n",
                fma_ok, fma64_ok, fma_mode, fma_mode == 2 ? 1 : 0);
    {
        int f16_native = f16_ok;
        if (getenv("ACUDA_F16_EMULATED"))
            f16_native = 0;
        else if (f16_ok && !getenv("ACUDA_F16_PROBE_OFF"))
            f16_native = probe_f16_native(ctx);
        acuda_compiler_set_f16_native(&ctx->compiler_caps, f16_native);
        acuda_compiler_set_mixed_fdot(&ctx->compiler_caps, mfdot_ok);
        if (getenv("ACUDA_DEBUG_FDOT2"))
            fprintf(stderr, "ACuda: mixed f16 dot (VALVE) = %d\n",
                    mfdot_ok);
    }
    acuda_compiler_set_dot_product(&ctx->compiler_caps, dot_ok);
    acuda_compiler_set_dot_product_forms(&ctx->compiler_caps, dot_signed, dot_unsigned,
                                         dot_mixed);
    acuda_compiler_set_zero_init_workgroup(&ctx->compiler_caps,
        v13q.shaderZeroInitializeWorkgroupMemory != VK_FALSE);
    if (getenv("ACUDA_PTX_NO_ZERO_INIT_WG"))
        acuda_compiler_set_zero_init_workgroup(&ctx->compiler_caps, 0);
    acuda_compiler_set_ssbo8(&ctx->compiler_caps, ssbo8_ok);
    acuda_compiler_set_ssbo16(&ctx->compiler_caps, ssbo16_ok);
    acuda_compiler_set_float_controls(&ctx->compiler_caps, fc32_ok, fc64_ok);
    acuda_compiler_set_vulkan_memory_model(&ctx->compiler_caps, vmm_ok);
    int bda_use = ctx->bda_ok;
    if (getenv("ACUDA_PTX_NO_BDA"))
        bda_use = 0;
    acuda_compiler_set_bda(&ctx->compiler_caps, bda_use);
    acuda_compiler_set_atomic_float(&ctx->compiler_caps, atomic_mask);
    int alias_ok = acuda_surface_alias_supported(ctx);
    if (getenv("ACUDA_NO_SURF_ALIAS"))
        alias_ok = 0;
    acuda_compiler_set_surface_alias(&ctx->compiler_caps, alias_ok);
    if (getenv("ACUDA_DEBUG_ATOMIC"))
        fprintf(stderr, "ACuda: surface uint alias view=%d\n", alias_ok);
    acuda_compiler_set_int64_image(&ctx->compiler_caps, int64_image_ok);
    acuda_compiler_set_shader_clock(&ctx->compiler_caps,
                                    shader_clock_device_ok,
                                    shader_clock_subgroup_ok);
    if (getenv("ACUDA_DEBUG_CLOCK"))
        fprintf(stderr, "ACuda: shader clock caps device=%d subgroup=%d\n",
                shader_clock_device_ok, shader_clock_subgroup_ok);
    if (getenv("ACUDA_DEBUG_BDA"))
        fprintf(stderr, "ACuda: bufferDeviceAddress=%d shader_bda=%d\n",
                ctx->bda_ok, bda_use);

    *out_ctx = ctx;
    return ACUDA_SUCCESS;
}

// Live GPU clock/utilisation refresh.
//
// The context clocks are sampled once at init; monitoring UIs re-read them
// every second and showed stale (or zero) values.  Refresh on demand, with a
// short throttle so attribute-heavy callers do not hammer the backends.
// Source priority: ADL2 (AMD, Windows) -> hwinfo (x64/Linux) -> NVML
// (NVIDIA, and the only live source in the 32-bit build where hwinfo cannot
// be compiled).
void acuda_context_refresh_clocks(acuda_context_t ctx) {
    int core = 0, mem = 0, load = -1;
    uint32_t now = 0;
    if (!ctx)
        return;
#if defined(_WIN32)
    now = (uint32_t)GetTickCount();
#else
    {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
            now = (uint32_t)((uint32_t)ts.tv_sec * 1000u +
                             (uint32_t)(ts.tv_nsec / 1000000));
    }
#endif
    if (ctx->clock_tick && now - ctx->clock_tick < 500u)
        return;
    ctx->clock_tick = now ? now : 1u;
#if defined(_WIN32)
    if (ctx->props.vendorID == 0x1002)
        (void)acuda_query_adl_gpu_clock_khz(
            ctx->pci_domain, ctx->pci_bus, ctx->pci_device, ctx->pci_function,
            &core, &mem);
#endif
#ifdef ACUDA_USE_HWINFO
    if (core <= 0 || mem <= 0) {
        int hc = 0, hm = 0;
        (void)acuda_query_real_gpu_clock_khz(
            ctx->pci_domain, ctx->pci_bus, ctx->pci_device, ctx->pci_function,
            &hc, &hm);
        if (core <= 0)
            core = hc;
        if (mem <= 0)
            mem = hm;
    }
#endif
    if (core <= 0 || mem <= 0)
        (void)acuda_query_nvml_gpu_clock_khz(
            ctx->pci_domain, ctx->pci_bus, ctx->pci_device, ctx->pci_function,
            &core, &mem);
    if (acuda_query_nvml_gpu_load_percent(ctx->pci_domain, ctx->pci_bus,
                                          ctx->pci_device, ctx->pci_function,
                                          &load))
        ctx->gpu_load_percent = load;
    if (core > 0)
        ctx->core_clock_khz = core;
    if (mem > 0)
        ctx->mem_clock_khz = mem;
}
