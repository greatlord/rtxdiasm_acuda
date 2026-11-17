#ifndef ACUDA_CONTEXT_INTERNAL_H
#define ACUDA_CONTEXT_INTERNAL_H

#include "acuda/runtime/internal.h"

void query_pci_info(VkPhysicalDevice dev, uint32_t *domain,
                    uint32_t *bus, uint32_t *device,
                    uint32_t *function);

#endif
