#include "driver_internal.h"
#include <stdlib.h>

CUresult ctx_push(CUcontext ctx) {
    if (g_depth >= ACUDA_CTX_STACK)
        return CUDA_ERROR_OUT_OF_MEMORY;
    g_stack[g_depth++] = ctx;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxCreate_v2(dev=%d flags=0x%x pctx=%p)\n",
                dev, flags, (void *)pctx);
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = NULL;
    if (flags & ~ACUDA_CTX_FLAGS_MASK) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuCtxCreate_v2 bad flags 0x%x\n", flags);
        return CUDA_ERROR_INVALID_VALUE; // sched/map-host/lmem bits only
    }
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    CUcontext c = calloc(1, sizeof(*c));
    if (!c)
        return CUDA_ERROR_OUT_OF_MEMORY;
    c->device = dev;
    r = ctx_push(c);
    if (r != CUDA_SUCCESS) {
        free(c);
        return r;
    }
    *pctx = c;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxCreate_v2 -> %p\n", (void *)c);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

CUresult ACUDA_CUDAAPI cuCtxDestroy_v2(CUcontext ctx) {
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    for (size_t i = 0; i < g_depth; i++) {
        if (g_stack[i] == ctx) {
            memmove(&g_stack[i], &g_stack[i + 1],
                    sizeof(*g_stack) * (g_depth - i - 1));
            g_depth--;
            break;
        }
    }
    free(ctx);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxDestroy(CUcontext ctx) {
    return cuCtxDestroy_v2(ctx);
}

CUresult ACUDA_CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx) {
    if (!ctx)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return ctx_push(ctx);
}

CUresult ACUDA_CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
    return cuCtxPushCurrent_v2(ctx);
}

CUresult ACUDA_CUDAAPI cuCtxPopCurrent_v2(CUcontext *pctx) {
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = NULL;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (g_depth == 0)
        return CUDA_ERROR_INVALID_CONTEXT;
    *pctx = g_stack[--g_depth];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
    return cuCtxPopCurrent_v2(pctx);
}

CUresult ACUDA_CUDAAPI cuCtxSynchronize(void) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxSynchronize\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    return map_status(acuda_stream_synchronize(g_actx, NULL));
}

