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

/* Stream memory operations are accepted as no-ops; the software paths
 * that use them are not scheduled on this backend. */
CUresult ACUDA_CUDAAPI cuStreamWriteValue32(void *stream, CUdeviceptr addr,
                              unsigned int value, unsigned int flags) {
    (void)stream; (void)addr; (void)value; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamWaitValue32(void *stream, CUdeviceptr addr,
                             unsigned int value, unsigned int flags) {
    (void)stream; (void)addr; (void)value; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamWriteValue64(void *stream, CUdeviceptr addr,
                              unsigned long long value, unsigned int flags) {
    (void)stream; (void)addr; (void)value; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamWaitValue64(void *stream, CUdeviceptr addr,
                             unsigned long long value, unsigned int flags) {
    (void)stream; (void)addr; (void)value; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamBatchMemOp(void *stream, unsigned int count, void *params,
                            unsigned int flags) {
    (void)stream; (void)count; (void)params; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                     int *greatestPriority) {
    if (leastPriority)
        *leastPriority = 0;
    if (greatestPriority)
        *greatestPriority = -1;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamIsCapturing(void *stream, int *captureStatus) {
    (void)stream;
    if (captureStatus)
        *captureStatus = 0; /* CU_STREAM_CAPTURE_STATUS_NONE */
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamBeginCapture(void *stream, int mode) {
    (void)stream; (void)mode;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamEndCapture(void *stream, void **graph) {
    (void)stream;
    if (graph)
        *graph = NULL;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuThreadExchangeStreamCaptureMode(int *mode) {
    (void)mode;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo(void *stream, int *status,
                                unsigned long long *id) {
    (void)stream;
    if (status)
        *status = 0;
    if (id)
        *id = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo_v2(void *stream, int *status,
                                   unsigned long long *id, void **graph,
                                   const void **deps, size_t *numDeps) {
    (void)stream;
    if (status)
        *status = 0;
    if (id)
        *id = 0;
    if (graph)
        *graph = NULL;
    if (deps)
        *deps = NULL;
    if (numDeps)
        *numDeps = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo_v3(void *stream, int *status,
                                   unsigned long long *id, void **graph,
                                   const void **deps, size_t *numDeps,
                                   void **edgeData) {
    if (edgeData)
        *edgeData = NULL;
    return cuStreamGetCaptureInfo_v2(stream, status, id, graph, deps,
                                     numDeps);
}

CUresult ACUDA_CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (hStream) {
        *pctx = hStream->ctx;
        return CUDA_SUCCESS;
    }
    return need_ctx(pctx) == CUDA_SUCCESS ? CUDA_SUCCESS
                                          : CUDA_ERROR_NOT_INITIALIZED;
}

CUresult ACUDA_CUDAAPI cuStreamGetCtx_v2(CUstream hStream, CUcontext *pctx,
                           void *pGreenCtx) {
    if (pGreenCtx)
        *(void **)pGreenCtx = NULL;
    return cuStreamGetCtx(hStream, pctx);
}

CUresult ACUDA_CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device) {
    (void)hStream;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
    if (!streamId)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *streamId = (unsigned long long)(uintptr_t)hStream;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamQuery(CUstream hStream) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!hStream)
        return CUDA_SUCCESS; // default stream: host-synchronous view
    if (hStream->ctx != c)
        return CUDA_ERROR_INVALID_HANDLE;
    uint64_t done = 0, last = hStream->s->last_submitted;
    if (last == 0)
        return CUDA_SUCCESS;
    if (vkGetSemaphoreCounterValue(g_actx->device, hStream->s->timeline,
                                   &done) != VK_SUCCESS)
        return CUDA_ERROR_UNKNOWN;
    return done >= last ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

CUresult ACUDA_CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                           unsigned int flags) {
    (void)flags;
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
    return map_status(acuda_stream_wait_event(g_actx, s, hEvent->e));
}

CUresult ACUDA_CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
    (void)hStream;
    if (!priority)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *priority = 0;
    return CUDA_SUCCESS;
}


/* Stream extensions. */
CUresult ACUDA_CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                                  unsigned int flags,
                                                  int priority) {
    (void)priority;
    return cuStreamCreate(phStream, flags);
}

CUresult ACUDA_CUDAAPI cuStreamAddCallback(CUstream hStream,
                                           acuda_stream_cb_t cb,
                                           void *userData,
                                           unsigned int flags) {
    (void)flags;
    if (!cb)
        return CUDA_ERROR_INVALID_VALUE;
    /* Synchronous backend: drain the stream, then run inline. */
    CUresult r = cuStreamSynchronize(hStream);
    if (r != CUDA_SUCCESS)
        return r;
    cb(hStream, CUDA_SUCCESS, userData);
    return CUDA_SUCCESS;
}
