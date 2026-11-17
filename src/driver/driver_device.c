#include "driver_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuInit(unsigned int flags) {
    /* Allow trace capture from GUI apps whose stderr is not wired
     * anywhere (FluidMark): ACUDA_TRACE_FILE=<path> redirects stderr. */
    static int trace_file_done = 0;
    if (!trace_file_done) {
        trace_file_done = 1;
        const char *tp = getenv("ACUDA_TRACE_FILE");
        if (tp && *tp) {
            FILE *tf = freopen(tp, "a", stderr);
            (void)tf;
        }
        /* Unbuffered: a crash must not lose the trailing trace lines
         * that identify the faulting call. */
        setvbuf(stderr, NULL, _IONBF, 0);
#if defined(_M_IX86)
        acuda_install_veh();
#endif
    }
#if defined(_WIN32)
    /* Reserve the pinned-host heap before the device arena fragments the
     * 32-bit address space.  PhysX requests a 128 MB staging buffer after
     * the arena has grown past 1 GB; a late VirtualAlloc then fails. */
    acuda_host_heap_init(128u << 20);
#endif
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuInit(flags=%u) g@%p\n", flags,
                (void *)&g_init);
    if (flags != 0)
        return CUDA_ERROR_INVALID_VALUE;
    g_init = 1;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuInit done g=%d\n", g_init);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion)
        return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = ACUDA_DRIVER_VERSION;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount\n");
    if (!count)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    uint32_t n = 0;
    acuda_status_t st = acuda_device_count(g_actx, &n);
    if (st != ACUDA_SUCCESS)
        return map_status(st);
    // Expose only GPUs with compute queues to driver clients; software
    // rasterizers (llvmpipe) confuse client heuristics. Matches ZLUDA.
    // TODO: filter by queue support instead of clamping to 1.
    if (n > 1)
        n = 1;
    *count = (int)n;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount -> %d\n", (int)n);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGet(%d)\n", ordinal);
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (ordinal < 0 || ordinal >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (!name || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    acuda_device_info_t info;
    acuda_status_t st = acuda_device_get(g_actx, (uint32_t)dev, &info);
    if (st != ACUDA_SUCCESS)
        return CUDA_ERROR_INVALID_DEVICE;
    strncpy(name, info.name, (size_t)len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

int cc_major(void) {
    const char *v = getenv("ACUDA_CC_MAJOR");
    return v && *v ? atoi(v) : ACUDA_CC_MAJOR;
}

int cc_minor(void) {
    const char *v = getenv("ACUDA_CC_MINOR");
    return v && *v ? atoi(v) : ACUDA_CC_MINOR;
}

CUresult ACUDA_CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (!major || !minor)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *major = cc_major();
    *minor = cc_minor();
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetProperties(dev=%d)\n", dev);
    if (!prop)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    memset(prop, 0, sizeof(*prop));
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock = 49152;
    prop->totalConstantMemory = 65536;
    prop->SIMDWidth = 32;
    prop->memPitch = 2147483647;
    prop->regsPerBlock = 65536;
    prop->clockRate = 1290000;
    prop->textureAlign = 512;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!bytes)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *bytes = (size_t)acuda_device_mem_total();
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem(bytes, dev);
}

CUresult ACUDA_CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
}

 int device_pci_info(CUdevice dev, acuda_device_info_t *info) {
    if (!g_actx || acuda_device_get(g_actx, (uint32_t)dev, info) != ACUDA_SUCCESS)
        return 0;
    return 1;
}


CUresult ACUDA_CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!uuid)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    // Deterministic pseudo-UUID: "ACUD" + ordinal + FNV of nothing (stable).
    memset(uuid->bytes, 0, 16);
    uuid->bytes[0] = 'A';
    uuid->bytes[1] = 'C';
    uuid->bytes[2] = 'U';
    uuid->bytes[3] = 'D';
    uuid->bytes[4] = (char)dev;
    uint32_t h = 2166136261u + (uint32_t)dev * 16777619u;
    memcpy(uuid->bytes + 8, &h, 4);
    memcpy(uuid->bytes + 12, &h, 4);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    acuda_device_info_t info;
    const char *id = getenv("ACUDA_PCI_BUS_ID");
    char generated[32];
    if (!pciBusId || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (!id || !*id) {
        if (device_pci_info(dev, &info)) {
            snprintf(generated, sizeof(generated), "%04x:%02x:%02x.%x",
                     info.pci_domain, info.pci_bus, info.pci_device,
                     info.pci_function);
            id = generated;
        } else {
            id = (dev == 0) ? "0000:00:00.0" : "0000:00:00.0";
        }
    }
    strncpy(pciBusId, id, (size_t)len - 1);
    pciBusId[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId) {
    if (!device || !pciBusId)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_init();
    if (r != CUDA_SUCCESS)
        return r;
    {
        const char *id = getenv("ACUDA_PCI_BUS_ID");
        if (id && *id && strcmp(pciBusId, id) == 0) {
            *device = 0;
            return CUDA_SUCCESS;
        }
    }
    if (g_actx) {
        uint32_t n = 0;
        if (acuda_device_count(g_actx, &n) == ACUDA_SUCCESS) {
            for (uint32_t i = 0; i < n; i++) {
                char bdf[32];
                if (cuDeviceGetPCIBusId(bdf, sizeof(bdf), (CUdevice)i) == CUDA_SUCCESS &&
                    strcmp(pciBusId, bdf) == 0) {
                    *device = (CUdevice)i;
                    return CUDA_SUCCESS;
                }
            }
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult ACUDA_CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                            CUarray_format format,
                                            unsigned numChannels,
                                            CUdevice dev) {
    (void)format;
    (void)numChannels;
    if (!maxWidthInElements)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    *maxWidthInElements = (size_t)1 << 27;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetP2PAttribute(int *value, int attrib,
                                 CUdevice srcDevice, CUdevice dstDevice) {
    (void)attrib;
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(srcDevice);
    if (r != CUDA_SUCCESS)
        return r;
    r = check_dev(dstDevice);
    if (r != CUDA_SUCCESS)
        return r;
    *value = 0; // no peer mappings in this model
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev) {
    if (!canAccessPeer)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    r = check_dev(peerDev);
    if (r != CUDA_SUCCESS)
        return r;
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuFlushGPUDirectRDMAWrites(int target, int scope) {
    (void)target;
    (void)scope;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS; // no GPUDirect transports here
}
