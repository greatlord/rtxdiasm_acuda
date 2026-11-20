#include "gpu_clock.h"

#include <hwinfo/gpu.h>
#include <hwinfo/monitoring.h>
#include <hwinfo/units.h>

#include <cstdio>
#include <string>

extern "C" int acuda_query_real_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                              uint32_t pci_device, uint32_t pci_function,
                                              int *out_graphics_khz, int *out_memory_khz) {
    char bdf[32];
    std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    hwinfo::GpuQuery query{};
    query.cuda = false;
    query.opencl = false;
    auto gpus = hwinfo::gpus(query);
    if (!gpus) {
        return 0;
    }

    const hwinfo::Gpu *match = nullptr;
    for (const auto &g : *gpus) {
        if (g.pci && g.pci->address && *g.pci->address == bdf) {
            match = &g;
            break;
        }
    }
    if (!match) {
        return 0;
    }

    int gkhz = 0;
    int mkhz = 0;

    // Prefer current/live clocks when available.
    {
        hwinfo::GpuQuery sample_query{};
        sample_query.cuda = false;
        sample_query.opencl = false;
        hwinfo::GpuSampler sampler(*match, sample_query);
        auto st = sampler.sample();
        if (st) {
            if (st->frequency) {
                gkhz = static_cast<int>(st->frequency->to(hwinfo::FrequencyUnit::kHz));
            }
            if (st->memory_frequency) {
                mkhz = static_cast<int>(st->memory_frequency->to(hwinfo::FrequencyUnit::kHz));
            }
        }
    }

    // Fall back to maximum reported clocks.
    if (gkhz <= 0 && match->max_frequency) {
        gkhz = static_cast<int>(match->max_frequency->to(hwinfo::FrequencyUnit::kHz));
    }
    if (mkhz <= 0 && match->max_memory_frequency) {
        mkhz = static_cast<int>(match->max_memory_frequency->to(hwinfo::FrequencyUnit::kHz));
    }

    int ok = 0;
    if (out_graphics_khz) {
        *out_graphics_khz = 0;
        if (gkhz > 0) {
            *out_graphics_khz = gkhz;
            ok = 1;
        }
    } else if (gkhz > 0) {
        ok = 1;
    }
    if (out_memory_khz) {
        *out_memory_khz = 0;
        if (mkhz > 0) {
            *out_memory_khz = mkhz;
            ok = 1;
        }
    } else if (mkhz > 0) {
        ok = 1;
    }

    return ok;
}
