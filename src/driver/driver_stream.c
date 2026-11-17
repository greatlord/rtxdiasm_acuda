#include "driver_internal.h"
#include <stdlib.h>

CUresult ACUDA_CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int flags) {
    if (!phStream)
        return CUDA_ERROR_INVALID_VALUE;
    *phStream = NULL;
    if (flags != 0 && flags != 1)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    CUstream s = calloc(1, sizeof(*s));
    if (!s)
        return CUDA_ERROR_OUT_OF_MEMORY;
    s->ctx = c;
    acuda_status_t st = acuda_stream_create_flags(g_actx, flags, &s->s);
    if (st != ACUDA_SUCCESS) {
        free(s);
        return map_status(st);
    }
    *phStream = s;
    return CUDA_SUCCESS;
}

static CUresult stream_free(CUstream hStream) {
    if (!hStream)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    acuda_status_t st = acuda_stream_destroy(g_actx, hStream->s);
    free(hStream);
    return map_status(st);
}

CUresult ACUDA_CUDAAPI cuStreamDestroy_v2(CUstream hStream) {
    return stream_free(hStream);
}

CUresult ACUDA_CUDAAPI cuStreamDestroy(CUstream hStream) {
    return stream_free(hStream);
}

CUresult ACUDA_CUDAAPI cuStreamSynchronize(CUstream hStream) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    acuda_stream_t s = NULL;
    if (hStream) {
        if (hStream->ctx != c)
            return CUDA_ERROR_INVALID_HANDLE;
        s = hStream->s;
    }
    return map_status(acuda_stream_synchronize(g_actx, s));
}
