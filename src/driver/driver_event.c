#include "acuda/driver/driver_internal.h"
#include <stdlib.h>
#include <string.h>

CUresult ACUDA_CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int flags) {
    if (!phEvent)
        return CUDA_ERROR_INVALID_VALUE;
    *phEvent = NULL;
    if (flags > 3)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    CUevent e = calloc(1, sizeof(*e));
    if (!e)
        return CUDA_ERROR_OUT_OF_MEMORY;
    e->ctx = c;
    acuda_status_t st = acuda_event_create(g_actx, &e->e);
    if (st != ACUDA_SUCCESS) {
        free(e);
        return map_status(st);
    }
    *phEvent = e;
    return CUDA_SUCCESS;
}

static CUresult event_free(CUevent hEvent) {
    if (!hEvent)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    acuda_status_t st = acuda_event_destroy(g_actx, hEvent->e);
    free(hEvent);
    return map_status(st);
}

CUresult ACUDA_CUDAAPI cuEventDestroy_v2(CUevent hEvent) {
    return event_free(hEvent);
}

CUresult ACUDA_CUDAAPI cuEventDestroy(CUevent hEvent) {
    return event_free(hEvent);
}

CUresult ACUDA_CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    if (!hEvent)
        return CUDA_ERROR_INVALID_HANDLE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (hEvent->ctx != c)
        return CUDA_ERROR_INVALID_HANDLE;
    acuda_stream_t s = NULL;
    if (hStream) {
        if (hStream->ctx != c)
            return CUDA_ERROR_INVALID_HANDLE;
        s = hStream->s;
    }
    return map_status(acuda_event_record(g_actx, hEvent->e, s));
}

CUresult ACUDA_CUDAAPI cuEventSynchronize(CUevent hEvent) {
    if (!hEvent)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    acuda_status_t st = acuda_event_synchronize(g_actx, hEvent->e);
    if (st == ACUDA_ERROR_UNINITIALIZED)
        return CUDA_ERROR_INVALID_VALUE;
    return map_status(st);
}

CUresult ACUDA_CUDAAPI cuEventQuery(CUevent hEvent) {
    if (!hEvent)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    acuda_status_t st = acuda_event_query(g_actx, hEvent->e);
    if (st == ACUDA_ERROR_UNINITIALIZED)
        return CUDA_ERROR_INVALID_VALUE;
    return map_status(st);
}

CUresult ACUDA_CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (!pMilliseconds)
        return CUDA_ERROR_INVALID_VALUE;
    if (!hStart || !hEnd)
        return CUDA_ERROR_INVALID_HANDLE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (hStart->ctx != c || hEnd->ctx != c)
        return CUDA_ERROR_INVALID_HANDLE;
    uint64_t ns = 0;
    acuda_status_t st = acuda_event_elapsed(g_actx, hStart->e, hEnd->e, &ns);
    if (st == ACUDA_ERROR_UNINITIALIZED)
        return CUDA_ERROR_INVALID_VALUE;
    if (st != ACUDA_SUCCESS)
        return map_status(st);
    *pMilliseconds = (float)ns / 1e6f;
    return CUDA_SUCCESS;
}
