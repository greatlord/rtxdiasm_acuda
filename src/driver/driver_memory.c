#include "driver_internal.h"

CUresult ACUDA_CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    return mem_alloc(dptr, bytesize);
}

CUresult ACUDA_CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return mem_alloc(dptr, bytesize);
}

CUresult ACUDA_CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
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
    // The runtime only needs a successful answer here; the arena does not
    // expose per-allocation metadata yet.
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
