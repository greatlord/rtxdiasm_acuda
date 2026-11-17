#include "acuda/driver/driver_internal.h"
#include <string.h>
#include <stddef.h>

CUresult ACUDA_CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                             size_t alignment, CUdeviceptr addr,
                             unsigned long long flags) {
    (void)alignment; (void)addr; (void)flags;
    if (!ptr)
        return CUDA_ERROR_INVALID_VALUE;
    return cuMemAlloc_v2(ptr, size);
}
CUresult ACUDA_CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)size;
    return cuMemFree_v2(ptr);
}
/* Minimal virtual-memory model: reserve/create allocate real arena
 * memory, map/unmap/access are no-ops, so probes during device
 * initialisation succeed without exposing a full VMM. */
CUresult ACUDA_CUDAAPI cuMemCreate(void **handle, size_t size, const void *prop,
                     unsigned long long flags) {
    (void)prop; (void)flags;
    if (!handle)
        return CUDA_ERROR_INVALID_VALUE;
    {
        CUdeviceptr dptr = 0;
        CUresult r = cuMemAlloc_v2(&dptr, size);
        if (r != CUDA_SUCCESS)
            return r;
        *handle = (void *)(uintptr_t)dptr;
        return CUDA_SUCCESS;
    }
}
CUresult ACUDA_CUDAAPI cuMemRelease(void *handle) {
    return cuMemFree_v2((CUdeviceptr)(uintptr_t)handle);
}
CUresult ACUDA_CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  void *handle, unsigned long long flags) {
    (void)ptr; (void)size; (void)offset; (void)handle; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const void *desc, size_t count) {
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const void *prop,
                                       int option) {
    (void)prop; (void)option;
    if (!granularity)
        return CUDA_ERROR_INVALID_VALUE;
    *granularity = 4096;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemGetAllocationPropertiesFromHandle(void *prop, void *handle) {
    (void)handle;
    if (prop)
        memset(prop, 0, 64);
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemExportToShareableHandle(void *shareableHandle, void *handle,
                                      int handleType,
                                      unsigned long long flags) {
    (void)handle; (void)handleType; (void)flags;
    if (shareableHandle)
        *(unsigned long long *)shareableHandle = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemImportFromShareableHandle(void **handle, void *osHandle,
                                        int shHandleType) {
    (void)osHandle; (void)shHandleType;
    if (handle)
        *handle = NULL;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemRetainAllocationHandle(void **handle, void *addr) {
    (void)addr;
    if (handle)
        *handle = NULL;
    return CUDA_SUCCESS;
}
