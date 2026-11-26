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

/*
 * Query real GPU dedicated memory for the PCI device at the given BDF.
 * Returns 1 if a memory value was obtained, 0 otherwise.
 * Output value is in bytes.
 */
int acuda_query_real_gpu_memory_bytes(uint32_t pci_domain, uint32_t pci_bus,
                                      uint32_t pci_device, uint32_t pci_function,
                                      uint64_t *out_bytes);

/*
 * Query GPU dedicated memory via AMD ADL2 (Windows, runtime-loaded atiadlxx.dll).
 * Returns 1 if a memory value was obtained, 0 otherwise. Bytes.
 */
int acuda_query_adl_gpu_memory_bytes(uint32_t pci_domain, uint32_t pci_bus,
                                     uint32_t pci_device, uint32_t pci_function,
                                     uint64_t *out_bytes);

/*
 * Query live NVIDIA clocks via NVML (runtime-loaded nvml.dll on Windows).
 * Returns 1 if any clock was obtained, 0 otherwise. Values in kHz.
 * This is the Win32 source because hwinfo cannot be built for Win32.
 */
int acuda_query_nvml_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                   uint32_t pci_device, uint32_t pci_function,
                                   int *out_graphics_khz,
                                   int *out_memory_khz);

/*
 * Query live NVIDIA utilisation via NVML.  Returns 1 and 0-100 percent.
 */
int acuda_query_nvml_gpu_load_percent(uint32_t pci_domain, uint32_t pci_bus,
                                      uint32_t pci_device,
                                      uint32_t pci_function, int *out_percent);

#ifdef __cplusplus
}
#endif

#endif /* ACUDA_GPU_CLOCK_H */
