#include <string.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"

acuda_status_t acuda_device_count(acuda_context_t ctx, uint32_t *out_count) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!out_count)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_count = ctx->device_total;
    return ACUDA_SUCCESS;
}

static uint64_t device_local_memory_bytes(VkPhysicalDevice dev) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(dev, &mp);
    uint64_t best = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (mp.memoryHeaps[i].size > best)
                best = mp.memoryHeaps[i].size;
        }
    }
    if (best == 0) {
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
            if (mp.memoryHeaps[i].size > best)
                best = mp.memoryHeaps[i].size;
        }
    }
    return best;
}

void query_pci_info(VkPhysicalDevice dev, uint32_t *domain,
                           uint32_t *bus, uint32_t *device,
                           uint32_t *function) {
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &pci,
    };
    vkGetPhysicalDeviceProperties2(dev, &p2);
    *domain = pci.pciDomain;
    *bus = pci.pciBus;
    *device = pci.pciDevice;
    *function = pci.pciFunction;
}

static void fill_info(VkPhysicalDevice dev, acuda_device_info_t *out) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(dev, &p);
    memset(out, 0, sizeof(*out));
    memcpy(out->name, p.deviceName, ACUDA_DEVICE_NAME_MAX - 1);
    out->name[ACUDA_DEVICE_NAME_MAX - 1] = '\0';
    out->vendor_id = p.vendorID;
    out->device_id = p.deviceID;
    out->is_discrete_gpu = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    out->api_major = VK_VERSION_MAJOR(p.apiVersion);
    out->api_minor = VK_VERSION_MINOR(p.apiVersion);
    out->api_patch = VK_VERSION_PATCH(p.apiVersion);
    query_pci_info(dev, &out->pci_domain, &out->pci_bus, &out->pci_device,
                   &out->pci_function);
    out->memory_bytes = device_local_memory_bytes(dev);
}

acuda_status_t acuda_device_get(acuda_context_t ctx, uint32_t index,
                                acuda_device_info_t *out_info) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!out_info || index >= ctx->device_total)
        return ACUDA_ERROR_INVALID_VALUE;
    fill_info(ctx->devices[index], out_info);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_device_vulkan_index(acuda_context_t ctx, uint32_t index,
                                         uint32_t *out_index) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!out_index || index >= ctx->device_total ||
        !ctx->device_original)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_index = ctx->device_original[index];
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_device_selected(acuda_context_t ctx, uint32_t *out_index) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!out_index)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_index = ctx->selected;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_validation_errors(acuda_context_t ctx, uint32_t *out_count) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!out_count)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_count = ctx->validation_errors;
    return ACUDA_SUCCESS;
}


const acuda_compiler_caps_t *acuda_context_get_compiler_caps(acuda_context_t ctx) {
    return ctx ? &ctx->compiler_caps : NULL;
}
