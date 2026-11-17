#ifndef ACUDA_CONTEXT_INTERNAL_H
#define ACUDA_CONTEXT_INTERNAL_H

#include "acuda/runtime/internal.h"

#define ACUDA_VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

// VK_VALVE_shader_mixed_float_dot_product (headers may predate it).
#define ACUDA_MFDP_EXT "VK_VALVE_shader_mixed_float_dot_product"
#define ACUDA_ST_MFDP ((VkStructureType)1000673000)
typedef struct {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderMixedFloatDotProductFloat16AccFloat32;
    VkBool32 shaderMixedFloatDotProductFloat16AccFloat16;
    VkBool32 shaderMixedFloatDotProductBFloat16Acc;
    VkBool32 shaderMixedFloatDotProductFloat8AccFloat32;
} acuda_mfdp_features_t;

void query_pci_info(VkPhysicalDevice dev, uint32_t *domain,
                    uint32_t *bus, uint32_t *device,
                    uint32_t *function);

int has_validation_layer(void);
VkResult create_messenger(acuda_context_t ctx);
void destroy_messenger(acuda_context_t ctx);
uint32_t timestamp_valid_bits(VkPhysicalDevice dev, uint32_t family);
uint32_t find_compute_queue(VkPhysicalDevice dev);
uint32_t find_host_memory(VkPhysicalDevice dev);
uint32_t find_device_memory(VkPhysicalDevice dev, uint32_t host_type);
int has_timeline(VkPhysicalDevice dev);
int has_int64(VkPhysicalDevice dev);
int has_float64(VkPhysicalDevice dev);
int has_device_ext(VkPhysicalDevice dev, const char *want);
void query_vk13(VkPhysicalDevice dev, VkPhysicalDeviceVulkan13Features *out);
int device_is_dozen(VkPhysicalDevice dev);
int has_vk13_core(VkPhysicalDevice dev);
int has_shader_fma(VkPhysicalDevice dev, int *f64_ok);

void acuda_warmup_gpu(acuda_context_t ctx);
void acuda_start_high_perf(acuda_context_t ctx);
void acuda_stop_high_perf(acuda_context_t ctx);

int probe_shader_fma(acuda_context_t ctx);
int probe_shader_fma64(acuda_context_t ctx);
int probe_f16_native(acuda_context_t ctx);
int probe_shader_fma_rn(acuda_context_t ctx);

#endif
