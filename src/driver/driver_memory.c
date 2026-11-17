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
