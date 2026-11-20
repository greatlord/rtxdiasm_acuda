#include "acuda/driver/driver_internal.h"

int acuda_dd9e98a4_check_ready_canary(const char *where);

/* Diagnostics: PhysX-era capability gates may key off total device memory
 * (the 32-bit default is 512 MiB; real cards of the era had 1+ GiB).
 * ACUDA_DEVICE_MEM_MB=<n> overrides the reported total without touching
 * the arenas.  Cached once per process. */
unsigned long long acuda_device_mem_total(void) {
    static unsigned long long cached;
    static int cached_ok;
    if (!cached_ok) {
        const char *s = getenv("ACUDA_DEVICE_MEM_MB");
        unsigned long mb = s && *s ? strtoul(s, NULL, 0) : 0;
        cached = mb ? (unsigned long long)mb << 20 : ACUDA_DEVICE_MEM;
        cached_ok = 1;
    }
    return cached;
}

CUresult ACUDA_CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    acuda_dd9e98a4_check_ready_canary("cuMemAlloc_v2_in");
    CUresult r = mem_alloc(dptr, bytesize);
    acuda_dd9e98a4_check_ready_canary("cuMemAlloc_v2_out");
    return r;
}

CUresult ACUDA_CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    acuda_dd9e98a4_check_ready_canary("cuMemAlloc_in");
    CUresult r = mem_alloc(dptr, bytesize);
    acuda_dd9e98a4_check_ready_canary("cuMemAlloc_out");
    return r;
}

CUresult ACUDA_CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
    acuda_dd9e98a4_check_ready_canary("cuMemFree_v2_in");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (!dptr)
        return CUDA_ERROR_INVALID_VALUE;
    // Order after queued default-stream work before recycling the range.
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    if (acuda_arena_free((uint64_t)dptr) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_dd9e98a4_check_ready_canary("cuMemFree_v2_out");
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

CUresult ACUDA_CUDAAPI cuMemGetInfo(size_t *freeBytes, size_t *totalBytes) {
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (totalBytes)
        *totalBytes = (size_t)acuda_device_mem_total();
    if (freeBytes) {
        size_t live = acuda_arena_live();
        unsigned long long total = acuda_device_mem_total();
        *freeBytes = total > live ? (size_t)(total - live) : 0;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemGetInfo_v2(size_t *freeBytes, size_t *totalBytes) {
    return cuMemGetInfo(freeBytes, totalBytes);
}

CUresult ACUDA_CUDAAPI cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                                         size_t widthInBytes, size_t height,
                                         unsigned int elemSize) {
    if (!dptr || !pPitch)
        return CUDA_ERROR_INVALID_VALUE;
    if (elemSize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (height == 0)
        height = 1;
    size_t pitch = (widthInBytes + 511u) & ~(size_t)511u;
    if (pitch == 0)
        pitch = 512;
    if (pitch > (size_t)-1 / height)
        return CUDA_ERROR_OUT_OF_MEMORY;
    CUresult r = cuMemAlloc_v2(dptr, pitch * height);
    if (r == CUDA_SUCCESS)
        *pPitch = pitch;
    return r;
}

CUresult ACUDA_CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                              CUdeviceptr dptr) {
    // PhysX 2.8.4 calls this ~138k times during a benchmark to map a
    // sub-pointer back to its allocation (fluid solver heap).  Report the
    // real arena allocation and fall back to {dptr, 0} for untracked
    // pointers, which is what the runtime tolerated before.
    uint64_t base = 0;
    size_t size = 0;
    if (acuda_arena_lookup((uint64_t)dptr, &base, &size) == 0) {
        if (pbase)
            *pbase = (CUdeviceptr)base;
        if (psize)
            *psize = size;
        return CUDA_SUCCESS;
    }
    if (pbase)
        *pbase = dptr;
    if (psize)
        *psize = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize,
                                 CUdeviceptr dptr) {
    return cuMemGetAddressRange(pbase, psize, dptr);
}

CUresult ACUDA_CUDAAPI cuDeviceGetDefaultMemPool(CUmemPool *pool_out, CUdevice dev) {
    if (!pool_out)
        return CUDA_ERROR_INVALID_VALUE;
    *pool_out = NULL;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;
    if (!default_pool[dev]) {
        CUmemPool p = calloc(1, sizeof(*p));
        if (!p)
            return CUDA_ERROR_OUT_OF_MEMORY;
        p->device = dev;
        default_pool[dev] = p;
    }
    *pool_out = default_pool[dev];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetMemPool(CUmemPool *pool_out, CUdevice dev) {
    return cuDeviceGetDefaultMemPool(pool_out, dev);
}

CUresult ACUDA_CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemPool pool) {
    (void)pool;
    return check_dev(dev);
}

CUresult ACUDA_CUDAAPI cuMemPoolTrimTo(CUmemPool pool, size_t minBytesToKeep) {
    (void)minBytesToKeep;
    if (!pool)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult host_for(CUdeviceptr dptr, size_t bytes, void **host) {
    if (!dptr || !host)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_host((uint64_t)dptr, bytes, host) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

CUresult bind_for(CUdeviceptr dptr, size_t bytes, VkBuffer *buf,
                         size_t *offset) {
    if (!dptr || !buf || !offset)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_bind((uint64_t)dptr, bytes, buf, offset) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}


 int device_pci_info(CUdevice dev, acuda_device_info_t *info);




CUresult mem_alloc(CUdeviceptr *dptr, size_t bytesize) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAlloc(%zu)\n", bytesize);
    if (!dptr)
        return CUDA_ERROR_INVALID_VALUE;
    *dptr = 0;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint64_t virt = 0;
    VkBuffer buf = VK_NULL_HANDLE;
    (void)buf;
    int ar = acuda_arena_alloc(bytesize, &virt, &buf);
    if (ar != 0) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAlloc(%zu) -> OOM (arena %d)\n", bytesize, ar);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *dptr = (CUdeviceptr)virt;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAlloc(%zu) -> 0x%llx\n", bytesize, (unsigned long long)virt);
#if defined(_WIN32) && !defined(_WIN64)
    acuda_trace_va("cuMemAlloc");
#endif
    return CUDA_SUCCESS;
}




uint32_t rd32le(const unsigned char *p);

