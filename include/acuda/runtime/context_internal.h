#ifndef ACUDA_CONTEXT_INTERNAL_H
#define ACUDA_CONTEXT_INTERNAL_H

#include "acuda/runtime/internal.h"

void query_pci_info(VkPhysicalDevice dev, uint32_t *domain,
                    uint32_t *bus, uint32_t *device,
                    uint32_t *function);

void acuda_warmup_gpu(acuda_context_t ctx);
void acuda_start_high_perf(acuda_context_t ctx);
void acuda_stop_high_perf(acuda_context_t ctx);

int probe_shader_fma(acuda_context_t ctx);
int probe_shader_fma64(acuda_context_t ctx);
int probe_f16_native(acuda_context_t ctx);
int probe_shader_fma_rn(acuda_context_t ctx);

#endif
