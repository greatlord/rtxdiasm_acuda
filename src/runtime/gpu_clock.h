#ifndef ACUDA_GPU_CLOCK_H
#define ACUDA_GPU_CLOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Query real GPU clocks (graphics/core and memory) for the PCI device at the
 * given BDF. Returns 1 if any clock was obtained, 0 otherwise.
 * Output values are in kHz. Pointers may be NULL to skip a value.
 */
int acuda_query_real_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                   uint32_t pci_device, uint32_t pci_function,
                                   int *out_graphics_khz, int *out_memory_khz);

/*
 * Query GPU clocks via AMD ADL2/PMLog (Windows, runtime-loaded atiadlxx.dll).
 * Intended for AMD GPUs; falls back to 0/unsupported when unavailable.
 * Returns 1 if any clock was obtained, 0 otherwise. Values in kHz.
 */
int acuda_query_adl_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                  uint32_t pci_device, uint32_t pci_function,
                                  int *out_graphics_khz, int *out_memory_khz);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_GPU_CLOCK_H */
