// ACuda context + device discovery. C ABI.
#ifndef ACUDA_CONTEXT_H
#define ACUDA_CONTEXT_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct acuda_context_impl *acuda_context_t;
typedef struct acuda_stream_impl *acuda_stream_t;
typedef struct acuda_event_impl *acuda_event_t;
typedef struct acuda_buffer_impl *acuda_buffer_t;

#define ACUDA_DEVICE_NAME_MAX 256

typedef struct acuda_device_info {
    char name[ACUDA_DEVICE_NAME_MAX];
    uint32_t vendor_id;
    uint32_t device_id;
    int is_discrete_gpu;
    uint32_t api_major;
    uint32_t api_minor;
    uint32_t api_patch;
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
    uint64_t memory_bytes;
} acuda_device_info_t;

// Creates instance, picks discrete GPU w/ compute queue + timeline
// semaphores, creates logical device and default stream.
acuda_status_t acuda_init(acuda_context_t *out_ctx);
acuda_status_t acuda_shutdown(acuda_context_t ctx);

acuda_status_t acuda_device_count(acuda_context_t ctx, uint32_t *out_count);
acuda_status_t acuda_device_get(acuda_context_t ctx, uint32_t index,
                                acuda_device_info_t *out_info);
// Refreshes the live GPU clock/utilisation cache (ADL2 for AMD, hwinfo where
// available, NVML for NVIDIA).  Throttled internally; called by the driver
// before answering clock/utilisation queries so the values are not stale.
void acuda_context_refresh_clocks(acuda_context_t ctx);
// Original Vulkan enumeration index for an exposed device slot.
acuda_status_t acuda_device_vulkan_index(acuda_context_t ctx, uint32_t index,
                                         uint32_t *out_index);
// Index of the device selected by acuda_init.
acuda_status_t acuda_device_selected(acuda_context_t ctx, uint32_t *out_index);
// Number of validation-layer errors reported since acuda_init
// (0 unless the KHRONOS validation layer is active; see docs).
acuda_status_t acuda_validation_errors(acuda_context_t ctx, uint32_t *out_count);

#ifdef __cplusplus
}
#endif

#endif
