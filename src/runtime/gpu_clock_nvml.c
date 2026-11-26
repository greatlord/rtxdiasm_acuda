// Live NVIDIA clocks and utilisation through NVML.
//
// NVML is loaded dynamically and is entirely optional: it is not present on
// every machine (notably there is no 32-bit nvml.dll on this one), and hwinfo
// -- the primary cross-platform source -- cannot be built for Win32 because
// its Level Zero headers static-assert on 64-bit pointer sizes.  The source
// order is therefore: ADL2 for AMD, hwinfo where it is compiled in, then
// NVML, then whatever the context sampled at init.
//
// Windows: LoadLibraryA("nvml.dll")
// Linux:   dlopen("libnvidia-ml.so.1"), then "libnvidia-ml.so"
#include "gpu_clock.h"

#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#define ACUDA_NVML_OPEN() LoadLibraryA("nvml.dll")
#define ACUDA_NVML_SYM(lib, name) GetProcAddress((HMODULE)(lib), name)
typedef HMODULE acuda_nvml_lib_t;
#else
#include <dlfcn.h>
static void *acuda_nvml_open_linux(void) {
    void *h = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h)
        h = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL);
    return h;
}
#define ACUDA_NVML_OPEN() acuda_nvml_open_linux()
#define ACUDA_NVML_SYM(lib, name) dlsym((lib), (name))
typedef void *acuda_nvml_lib_t;
#endif

typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} acuda_nvml_util_t;
typedef nvmlReturn_t (*nvml_init_t)(void);
typedef nvmlReturn_t (*nvml_by_bus_t)(const char *, nvmlDevice_t *);
typedef nvmlReturn_t (*nvml_clock_t)(nvmlDevice_t, int, unsigned int *);
typedef nvmlReturn_t (*nvml_util_t)(nvmlDevice_t, acuda_nvml_util_t *);

static struct {
    int state; // 0 untried, 1 ready, -1 unavailable
    nvmlDevice_t dev;
    nvml_clock_t get_clock;
    nvml_util_t get_util;
} g_nvml;

static void nvml_probe(uint32_t domain, uint32_t bus, uint32_t device,
                       uint32_t function) {
    acuda_nvml_lib_t lib;
    nvml_init_t init;
    nvml_by_bus_t by_bus;
    char bdf[32];
    if (g_nvml.state)
        return;
    g_nvml.state = -1;
    lib = ACUDA_NVML_OPEN();
    if (!lib)
        return;
    init = (nvml_init_t)ACUDA_NVML_SYM(lib, "nvmlInit_v2");
    by_bus = (nvml_by_bus_t)ACUDA_NVML_SYM(lib,
                                           "nvmlDeviceGetHandleByPciBusId_v2");
    g_nvml.get_clock = (nvml_clock_t)ACUDA_NVML_SYM(lib,
                                                    "nvmlDeviceGetClockInfo");
    g_nvml.get_util = (nvml_util_t)ACUDA_NVML_SYM(
        lib, "nvmlDeviceGetUtilizationRates");
    if (!init || !by_bus || !g_nvml.get_clock)
        return;
    if (init() != 0)
        return;
    snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", (unsigned)domain,
             (unsigned)bus, (unsigned)device, (unsigned)function);
    if (by_bus(bdf, &g_nvml.dev) != 0)
        return;
    g_nvml.state = 1;
}

int acuda_query_nvml_gpu_clock_khz(uint32_t pci_domain, uint32_t pci_bus,
                                   uint32_t pci_device, uint32_t pci_function,
                                   int *out_graphics_khz,
                                   int *out_memory_khz) {
    unsigned int v = 0;
    int got = 0;
    nvml_probe(pci_domain, pci_bus, pci_device, pci_function);
    if (g_nvml.state != 1)
        return 0;
    if (out_graphics_khz &&
        g_nvml.get_clock(g_nvml.dev, 0 /* NVML_CLOCK_GRAPHICS */, &v) == 0 &&
        v > 0) {
        *out_graphics_khz = (int)v;
        got = 1;
    }
    v = 0;
    if (out_memory_khz &&
        g_nvml.get_clock(g_nvml.dev, 2 /* NVML_CLOCK_MEM */, &v) == 0 &&
        v > 0) {
        *out_memory_khz = (int)v;
        got = 1;
    }
    return got;
}

int acuda_query_nvml_gpu_load_percent(uint32_t pci_domain, uint32_t pci_bus,
                                      uint32_t pci_device,
                                      uint32_t pci_function, int *out_percent) {
    acuda_nvml_util_t u;
    nvml_probe(pci_domain, pci_bus, pci_device, pci_function);
    if (g_nvml.state != 1 || !g_nvml.get_util || !out_percent)
        return 0;
    if (g_nvml.get_util(g_nvml.dev, &u) != 0)
        return 0;
    *out_percent = (int)u.gpu;
    return 1;
}
