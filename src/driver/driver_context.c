#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_dd9e98a4.h"
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


CUcontext current_ctx(void) {
    return g_depth > 0 ? g_stack[g_depth - 1] : NULL;
}

CUresult ACUDA_CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxGetCurrent\n");
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *pctx = current_ctx();
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxSetCurrent(%p)\n", (void *)ctx);
    acuda_dd9e98a4_check_ready_canary("cuCtxSetCurrent");
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!ctx) {
        g_depth = 0;
        return CUDA_SUCCESS;
    }
    return ctx_push(ctx);
}

CUresult ACUDA_CUDAAPI cuCtxGetDevice(CUdevice *device) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuCtxGetDevice\n");
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    *device = c->device;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx;
    if (!version)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *version = ACUDA_DRIVER_VERSION;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetLimit(size_t *pvalue, int limit) {
    (void)limit;
    if (!pvalue)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *pvalue = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxSetLimit(int limit, size_t value) {
    (void)limit;
    (void)value;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxSetCacheConfig(int config) {
    (void)config;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetCacheConfig(int *config) {
    if (!config)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *config = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxDetach(CUcontext ctx) {
    (void)ctx;
    return CUDA_SUCCESS; // legacy no-op
}

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (!pctx)
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = NULL;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;
    if (!primary_ctx[dev]) {
        CUcontext c = calloc(1, sizeof(*c));
        if (!c)
            return CUDA_ERROR_OUT_OF_MEMORY;
        c->device = dev;
        primary_ctx[dev] = c;
    }
    primary_refs[dev]++;
    r = ctx_push(primary_ctx[dev]);
    if (r != CUDA_SUCCESS) {
        if (--primary_refs[dev] == 0) {
            free(primary_ctx[dev]);
            primary_ctx[dev] = NULL;
        }
        return r;
    }
    *pctx = primary_ctx[dev];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES || !primary_ctx[dev])
        return CUDA_ERROR_INVALID_DEVICE;
    if (primary_refs[dev] > 0)
        primary_refs[dev]--;
    // Wrapper is freed on last release; Vulkan device stays (shared).
    if (primary_refs[dev] == 0) {
        for (size_t i = 0; i < g_depth; i++) {
            if (g_stack[i] == primary_ctx[dev]) {
                memmove(&g_stack[i], &g_stack[i + 1],
                        sizeof(*g_stack) * (g_depth - i - 1));
                g_depth--;
                break;
            }
        }
        free(primary_ctx[dev]);
        primary_ctx[dev] = NULL;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;
    return CUDA_SUCCESS; // wrappers are stateless; nothing to reset
}

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    if (flags & ~ACUDA_CTX_FLAGS_MASK)
        return CUDA_ERROR_INVALID_VALUE;
    return check_dev(dev);
}

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active) {
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;
    if (flags)
        *flags = 0;
    if (active)
        *active = primary_ctx[dev] ? 1 : 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetDevice_v2(CUdevice *device) { return cuCtxGetDevice(device); }
CUresult ACUDA_CUDAAPI cuCtxSynchronize_v2(void) { return cuCtxSynchronize(); }

CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    return cuDevicePrimaryCtxRelease(dev);
}
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    return cuDevicePrimaryCtxReset(dev);
}
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

CUcontext primary_ctx[ACUDA_MAX_DEVICES];
int primary_refs[ACUDA_MAX_DEVICES];
CUmemPool default_pool[ACUDA_MAX_DEVICES];

CUresult check_dev(int dev) {
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    return (dev >= 0 && dev < n) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

int g_init;
acuda_context_t g_actx;
int g_actx_exit_registered;
CUcontext g_stack[ACUDA_CTX_STACK];
size_t g_depth;

// The driver keeps one process-wide runtime context that outlives individual
// CUDA contexts. At exit, stop its ACUDA_HIGH_PERF keep-busy thread so it is
// not still submitting while other atexit handlers (e.g. the Vulkan
// validation layer) tear the device down. The context itself is left intact:
// destroying it here would report the driver's intentionally persistent
// resources as leaks.
static void acuda_driver_atexit(void) {
    if (g_actx)
        acuda_high_perf_shutdown(g_actx);
}

CUresult map_status(acuda_status_t st) {
    switch (st) {
    case ACUDA_SUCCESS: return CUDA_SUCCESS;
    case ACUDA_ERROR_INVALID_VALUE: return CUDA_ERROR_INVALID_VALUE;
    case ACUDA_ERROR_OUT_OF_MEMORY: return CUDA_ERROR_OUT_OF_MEMORY;
    case ACUDA_ERROR_NO_DEVICE: return CUDA_ERROR_NO_DEVICE;
    case ACUDA_ERROR_NOT_SUPPORTED: return CUDA_ERROR_NOT_SUPPORTED;
    case ACUDA_ERROR_INVALID_HANDLE: return CUDA_ERROR_INVALID_HANDLE;
    case ACUDA_ERROR_NOT_READY: return CUDA_ERROR_NOT_READY;
    default: return CUDA_ERROR_UNKNOWN;
    }
}

CUresult need_init(void) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? need_init g=%d @%p g_actx=%p\n", g_init,
                (void *)&g_init, (void *)g_actx);
    if (g_init) {
        if (!acuda_dd9e98a4_check_ready_canary("need_init")) {
            /* canary check already logged */
        }
        if (acuda_dd9e98a4_ready_state() == 0) {
            ACUDA_DEBUG_PRINT("acuda-trace ?? need_init: dd9e98a4_state=0 was=%d\n",
                              acuda_dd9e98a4_was_ready_flag());
        }
    }
    return g_init ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

// Lazily creates the one shared Vulkan context for the process.

CUresult need_actx(void) {
    CUresult r = need_init();
    if (r != CUDA_SUCCESS)
        return r;
    if (!g_actx) {
        ACUDA_DEBUG_PRINT("acuda-trace !! need_actx: g_actx NULL while g_init=%d, reinitializing\n", g_init);
        acuda_context_t ctx = NULL;
        acuda_status_t st = acuda_init(&ctx);
        if (st != ACUDA_SUCCESS)
            return map_status(st);
        if (acuda_arena_boot(ctx) != 0) {
            acuda_shutdown(ctx);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        g_actx = ctx;
        if (!g_actx_exit_registered) {
            g_actx_exit_registered = 1;
            atexit(acuda_driver_atexit);
        }
    }
    return CUDA_SUCCESS;
}

CUresult need_ctx(CUcontext *out) {
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    CUcontext c = current_ctx();
    if (!c)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (out)
        *out = c;
    return CUDA_SUCCESS;
}
