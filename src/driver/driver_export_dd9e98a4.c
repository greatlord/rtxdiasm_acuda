#include "acuda/driver/driver_export_dd9e98a4.h"
#include "acuda/driver/driver_export_dd9e98a4_types.h"
#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_stub.h"
#include "acuda/driver.h"
#include "acuda/runtime_api.h"

extern CUresult ACUDA_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

/* --- dd9e98a4 interface functions mapped to CUDA runtime API entry points ---
 * The CUDA 13.x hybrid runtime resolves these IDs through the dispatcher and
 * tail-calls them from its cuda* wrapper functions (see docs/unknown notes).
 * id 0x3 is cudaGetDeviceCount(). */
extern CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count);
extern CUresult ACUDA_CUDAAPI cuInit(unsigned int flags);

/* The CUDA 13.x hybrid runtime never calls cuInit() or cuCtxCreate(): it
 * expects the driver interface to be usable right away, so bring up the
 * primary context before serving interface requests. */
extern int g_init;
extern acuda_context_t g_actx;
static unsigned int g_dd9e98a4_init_magic; /* 0xDDE998A4 once init completed */
static unsigned g_dd9e98a4_nfatbins;
static unsigned g_dd9e98a4_nkernels;
static int g_dd9e98a4_was_ready; /* latched once canonical ready observed */
static int g_dd9e98a4_implicit_init_failed;
static void acuda_dd9e98a4_htrace(const char *name, const void *a, const void *b, long long v, int phase);
extern int (*acuda_runtime_canary_hook)(const char *where);

int acuda_dd9e98a4_check_ready_canary(const char *where) {
    /* Canonical driver ready state is g_init. Detect resets/corruption where
     * the interface was ready before but core init markers are gone. */
    if (g_init) {
        g_dd9e98a4_was_ready = 1;
        g_dd9e98a4_init_magic = 0xDDE998A4u;
        return 1;
    }
    if (g_dd9e98a4_was_ready || g_dd9e98a4_init_magic == 0xDDE998A4u) {
        ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 READY_RESET %s was=%d magic=0x%x g_init=%d g_actx=%p fb=%u kern=%u\n",
                          where ? where : "?",
                          g_dd9e98a4_was_ready, g_dd9e98a4_init_magic,
                          g_init, g_actx, g_dd9e98a4_nfatbins, g_dd9e98a4_nkernels);
        if (getenv("ACUDA_TRACE")) fflush(stderr);
        return 0;
    }
    return 1; /* not yet initialized */
}

int acuda_dd9e98a4_ready_state(void) {
    if (g_init)
        return 1;
    if (g_dd9e98a4_implicit_init_failed)
        return -1;
    return 0;
}

int acuda_dd9e98a4_was_ready_flag(void) { return g_dd9e98a4_was_ready; }

int acuda_dd9e98a4_driver_ready(void) {
#if defined(_WIN32)
    {
        static int _errmode_set = 0;
        if (!_errmode_set) {
            _errmode_set = 1;
            SetErrorMode(SetErrorMode(0) | 0x0001u /*SEM_FAILCRITICALERRORS*/ |
                         0x0002u /*SEM_NOGPFAULTERRORBOX*/ |
                         0x8000u /*SEM_NOOPENFILEERRORBOX*/);
        }
    }
#endif
    acuda_runtime_canary_hook = acuda_dd9e98a4_check_ready_canary;

    if (g_init) {
        g_dd9e98a4_was_ready = 1;
        g_dd9e98a4_init_magic = 0xDDE998A4u;
        g_dd9e98a4_implicit_init_failed = 0;
        return 1;
    }

    /* The CUDA 13.x hybrid runtime never calls cuInit()/cuCtxCreate(): bring up
     * the driver and a usable primary context on first interface request.
     * Canonical ready state is g_init (set by cuInit). */
    {
#if defined(_MSC_VER)
        void *_ra = _ReturnAddress();
#else
        void *_ra = __builtin_return_address(0);
#endif
        const char *_vt = getenv("ACUDA_DD9E98A4_TRACE_VERBOSE");
        if (_vt && *_vt) {
            fprintf(stderr, "ACuda-debug: dd9e98a4 implicit init start g_init=%d g_actx=%p fb=%u kern=%u caller=%p\n",
                    g_init, g_actx, g_dd9e98a4_nfatbins, g_dd9e98a4_nkernels, _ra);
            fflush(stderr);
        }
        CUresult r = cuInit(0);
        if (r == CUDA_SUCCESS) {
            CUcontext ctx = NULL;
            CUresult rctx = cuDevicePrimaryCtxRetain(&ctx, 0);
            if (rctx == CUDA_SUCCESS)
                rctx = cuCtxSetCurrent(ctx);
            if (rctx != CUDA_SUCCESS) {
                fprintf(stderr, "ACuda-debug: dd9e98a4 implicit init ctx setup failed r=%d (continuing as ready)\n", (int)rctx);
                fflush(stderr);
            }
            g_dd9e98a4_was_ready = 1;
            g_dd9e98a4_init_magic = 0xDDE998A4u;
            g_dd9e98a4_implicit_init_failed = 0;
            if (_vt && *_vt) {
                fprintf(stderr, "ACuda-debug: dd9e98a4 implicit init -> %d g_init=%d g_actx=%p caller=%p\n",
                        (int)r, g_init, g_actx, _ra);
                fflush(stderr);
            }
            return 1;
        }
        g_dd9e98a4_implicit_init_failed = 1;
        fprintf(stderr, "ACuda-debug: dd9e98a4 implicit init failed r=%d caller=%p\n", (int)r, _ra);
        fflush(stderr);
        return 0;
    }
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_count(
        void *count, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_count");
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    r = cuDeviceGetCount((int *)count);
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 id=0x3 cudaGetDeviceCount -> %d (count=%d)\n",
                      (int)r, count ? *(int *)count : -1);
    return r;
}

/* id 0xc8: cudaDeviceGetAttribute(int *value, int attr, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_attribute(
        void *value, void *attr, void *device, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_attribute");
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuDeviceGetAttribute((int *)value, (int)(intptr_t)attr,
                                (CUdevice)(intptr_t)device);
}

/* id 0x14: cudaMalloc(void **devPtr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc(
        void *devPtr, void *size, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_malloc");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    return cuMemAlloc((CUdeviceptr *)devPtr, (size_t)size);
}

/* id 0x19: cudaMallocHost(void **ptr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc_host(
        void *ptr, void *size, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_malloc_host");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!ptr)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)ptr = NULL;
    *(void **)ptr = malloc((size_t)size ? (size_t)size : 1);
    return *(void **)ptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

/* id 0x1b: cudaMallocHost/cudaHostAlloc-style host allocation used by
 * statically linked cudart (CUDA 13.5) internal dispatch. Mirrors 0x19 and
 * remains compatible with cudaFreeHost (0x1a). */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_host_alloc(
        void *ptr, void *size, void *flags, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_host_alloc");
    (void)flags; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!ptr)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)ptr = NULL;
    *(void **)ptr = malloc((size_t)size ? (size_t)size : 1);
    return *(void **)ptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

/* id 0x10: cudaSetDevice(int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device(
        void *device, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_set_device");
    CUcontext ctx = NULL;
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL set_device dev=%d\n", (int)(intptr_t)device);
    r = cuDevicePrimaryCtxRetain(&ctx, (CUdevice)(intptr_t)device);
    if (r != CUDA_SUCCESS)
        return r;
    return cuCtxSetCurrent(ctx);
}

/* id 0x11: cudaGetDevice(int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device(
        void *device, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_device");
    CUdevice d = 0;
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    r = cuCtxGetDevice(&d);
    if (r != CUDA_SUCCESS)
        return r;
    *(int *)device = (int)d;
    return CUDA_SUCCESS;
}

/* id 0x1b8: cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device_properties(
        void *prop, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_device_properties");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    return acuda_query_device_properties((struct cudaDeviceProp *)prop,
                                         (int)(intptr_t)device);
}
/* id 0xd1: const char *cudaGetErrorString(cudaError_t error) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_error_string(
        void *error, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_error_string");
    const char *str = "unknown error";
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    cuGetErrorString((CUresult)(intptr_t)error, &str);
    return (CUresult)(uintptr_t)str;
}

/* --- memory, events, synchronisation, profiler ------------------------- */

/* id 0x31: cudaMemset(void *devPtr, int value, size_t count) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memset(
        void *devPtr, void *value, void *count, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_memset");
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 memset(%p,%d,%zu)\n",
                      devPtr, (int)(intptr_t)value, (size_t)count);
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!devPtr)
        return CUDA_ERROR_INVALID_VALUE;
    if (!(size_t)count)
        return CUDA_SUCCESS;
    return cuMemsetD8((CUdeviceptr)(uintptr_t)devPtr,
                      (unsigned char)(uintptr_t)value, (size_t)count);
}

/* id 0x1f: cudaMemcpy(void *dst, const void *src, size_t count,
 *                       cudaMemcpyKind kind) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy(
        void *dst, void *src, void *count, void *kind,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_memcpy");
    int copy_kind = (int)(intptr_t)kind;
    size_t bytes = (size_t)count;
    acuda_dd9e98a4_htrace("memcpy", dst, src, (long long)bytes, 0);
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if ((!dst || !src) && bytes)
        return CUDA_ERROR_INVALID_VALUE;
    if (!bytes)
        return CUDA_SUCCESS;
    switch (copy_kind) {
    case 0: /* cudaMemcpyHostToHost */
        memcpy(dst, src, bytes);
        return CUDA_SUCCESS;
    case 1: /* cudaMemcpyHostToDevice */
        return cuMemcpyHtoD_v2((CUdeviceptr)(uintptr_t)dst, src, bytes);
    case 2: /* cudaMemcpyDeviceToHost */
        return cuMemcpyDtoH_v2(dst, (CUdeviceptr)(uintptr_t)src, bytes);
    case 3: /* cudaMemcpyDeviceToDevice */
        return cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst,
                               (CUdeviceptr)(uintptr_t)src, bytes);
    case 4: /* cudaMemcpyDefault */
        memcpy(dst, src, bytes);
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

/* id 0x29: cudaMemcpyAsync(void *dst, const void *src, size_t count,
 *                          cudaMemcpyKind kind, cudaStream_t stream)
 * ACuda's device memory is directly addressable host memory, so the async
 * copies are executed synchronously, exactly like ACuda's cudart does. */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy_async(
        void *dst, void *src, void *count, void *kind, void *stream,
        void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_memcpy_async");
    int copy_kind = (int)(intptr_t)kind;
    size_t bytes = (size_t)count;
    CUresult _r;
    acuda_dd9e98a4_htrace("memcpy_async", dst, src, (long long)bytes, 0);
    (void)unused_f; (void)unused_g; (void)unused_h;
    if ((!dst || !src) && bytes) { _r = CUDA_ERROR_INVALID_VALUE; goto out; }
    if (!bytes) { _r = CUDA_SUCCESS; goto out; }
    /* Async copies are implemented synchronously but must be ordered with
     * prior work on the specified stream (host-visible memory). */
    if (stream) {
        _r = cuStreamSynchronize((CUstream)stream);
        if (_r != CUDA_SUCCESS)
            goto out;
    }
    switch (copy_kind) {
    case 0: /* cudaMemcpyHostToHost */
        memmove(dst, src, bytes);
        _r = CUDA_SUCCESS; break;
    case 1: /* cudaMemcpyHostToDevice */
        _r = cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, bytes); break;
    case 2: /* cudaMemcpyDeviceToHost */
        _r = cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, bytes); break;
    case 3: /* cudaMemcpyDeviceToDevice */
        _r = cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst,
                          (CUdeviceptr)(uintptr_t)src, bytes); break;
    default:
        _r = CUDA_ERROR_INVALID_VALUE; break;
    }
out:
    acuda_dd9e98a4_htrace("memcpy_async", dst, src, (long long)_r, 1);
    return _r;
}

/* id 0x16: cudaFree(void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free(
        void *devPtr, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_free");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL free ptr=%p\n", devPtr);
    if (!devPtr)
        return CUDA_SUCCESS;
    return cuMemFree((CUdeviceptr)(uintptr_t)devPtr);
}

/* id 0x1a: cudaFreeHost(void *ptr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free_host(
        void *ptr, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_free_host");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    free(ptr);
    return CUDA_SUCCESS;
}

/* id 0xa4: cudaDeviceReset(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_reset(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_reset");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuCtxSynchronize();
}

/* id 0xa5: cudaDeviceSynchronize(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_synchronize(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_synchronize");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuCtxSynchronize();
}

/* id 0x85: cudaEventCreate(cudaEvent_t *event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_create");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    return cuEventCreate((CUevent *)event, 0);
}

/* id 0x86: cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create_with_flags(
        void *event, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_create_with_flags");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    return cuEventCreate((CUevent *)event, (unsigned int)(uintptr_t)flags);
}

/* id 0x87: cudaEventRecord(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record(
        void *event, void *stream, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_record");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    acuda_dd9e98a4_htrace("evt_record", event, stream, 0, 0);
    CUresult _r = cuEventRecord((CUevent)event, (CUstream)stream);
    acuda_dd9e98a4_htrace("evt_record", event, stream, (long long)_r, 1);
    return _r;
}

/* id 0x89: cudaEventSynchronize(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_synchronize(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_synchronize");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    acuda_dd9e98a4_htrace("evt_sync", event, NULL, 0, 0);
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL event_synchronize event=%p\n", event);
    if (!event) {
        acuda_dd9e98a4_htrace("evt_sync", event, NULL, -1, 1);
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUresult _r = cuEventSynchronize((CUevent)event);
    acuda_dd9e98a4_htrace("evt_sync", event, NULL, (long long)_r, 1);
    return _r;
}

/* id 0x8a: cudaEventQuery(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_query(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_query");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    return cuEventQuery((CUevent)event);
}

/* id 0x1e6: cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_elapsed_time(
        void *ms, void *start, void *end, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_elapsed_time");
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuEventElapsedTime((float *)ms, (CUevent)start, (CUevent)end);
}

/* id 0x88: cudaEventDestroy(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_destroy(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_destroy");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL event_destroy event=%p\n", event);
    return cuEventDestroy((CUevent)event);
}

/* id 0xab / 0xac: cudaProfilerStart() / cudaProfilerStop() */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_profiler_noop(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_profiler_noop");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* --- kernel launch -----------------------------------------------------
 * The hybrid runtime reports the executable's fatbin once at startup and
 * then every __global__ host stub that gets launched.  Launches arrive with
 * the host stub address as the function key, which is exactly what ACuda's
 * cudart does for statically linked executables. */



static acuda_dd9e98a4_fatbin g_dd9e98a4_fatbins[ACUDA_DD9E98A4_MAX_FATBINS];
static acuda_dd9e98a4_kernel g_dd9e98a4_kernels[ACUDA_DD9E98A4_MAX_FUNCS];
static acuda_dd9e98a4_var g_dd9e98a4_vars[ACUDA_DD9E98A4_MAX_VARS];
static unsigned g_dd9e98a4_nvars;

/* id 0x1ee: __cudaRegisterFatBinary(void *fatCubin) -> handle */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary(
        void *fatCubin, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_register_fat_binary");
    unsigned i;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!fatCubin)
        return CUDA_ERROR_INVALID_VALUE;
    /* The hybrid runtime keeps the returned handle in a 32-bit field, so a
     * pointer cannot be handed back; use a 1-based index instead. */
    for (i = 0; i < g_dd9e98a4_nfatbins; i++) {
        if (g_dd9e98a4_fatbins[i].blob == fatCubin)
            return (CUresult)(i + 1);
    }
    if (g_dd9e98a4_nfatbins >= ACUDA_DD9E98A4_MAX_FATBINS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    i = g_dd9e98a4_nfatbins++;
    g_dd9e98a4_fatbins[i].blob = fatCubin;
    g_dd9e98a4_fatbins[i].module = NULL;
    g_dd9e98a4_fatbins[i].state = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register fatbin %p -> %u\n",
                      fatCubin, i + 1);
    return (CUresult)(i + 1);
}

/* id 0x1ef: __cudaRegisterFatBinaryEnd(void **fatCubinHandle) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary_end(
        void *fatCubinHandle, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_register_fat_binary_end");
    (void)fatCubinHandle; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

static acuda_dd9e98a4_fatbin *acuda_dd9e98a4_fatbin_from_handle(void *fatCubinHandle) {
    unsigned idx = (unsigned)(uintptr_t)fatCubinHandle;
    if (idx < 1 || idx > g_dd9e98a4_nfatbins)
        return NULL;
    return &g_dd9e98a4_fatbins[idx - 1];
}

static CUresult acuda_dd9e98a4_ensure_fatbin_module(acuda_dd9e98a4_fatbin *fb) {
    if (!fb)
        return CUDA_ERROR_INVALID_VALUE;
    if (fb->state == 0) {
        CUresult r = cuModuleLoadFatBinary(&fb->module, fb->blob);
        fb->state = (r == CUDA_SUCCESS) ? 1 : -1;
        ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 module load -> %d\n", (int)r);
        return r;
    }
    return (fb->state == 1) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_IMAGE;
}

static acuda_dd9e98a4_var *acuda_dd9e98a4_find_var(const void *host_var) {
    unsigned i;
    if (!host_var)
        return NULL;
    for (i = 0; i < g_dd9e98a4_nvars; i++) {
        if (g_dd9e98a4_vars[i].host_var == host_var)
            return &g_dd9e98a4_vars[i];
    }
    return NULL;
}

static CUresult acuda_dd9e98a4_resolve_symbol_ptr(const void *symbol,
                                                  CUdeviceptr *out_dptr,
                                                  size_t *out_bytes) {
    acuda_dd9e98a4_var *v = acuda_dd9e98a4_find_var(symbol);
    acuda_dd9e98a4_fatbin *fb;
    CUresult r;
    size_t i;
    if (out_dptr) *out_dptr = 0;
    if (out_bytes) *out_bytes = 0;
    if (!v)
        return CUDA_ERROR_NOT_FOUND;
    fb = acuda_dd9e98a4_fatbin_from_handle(v->fatbin_handle);
    r = acuda_dd9e98a4_ensure_fatbin_module(fb);
    if (r != CUDA_SUCCESS)
        return r;
    if (!fb->module || !v->device_name)
        return CUDA_ERROR_NOT_FOUND;
    for (i = 0; i < fb->module->nglobals; i++) {
        if (strcmp(fb->module->globals[i].name, v->device_name) == 0) {
            if (out_dptr) *out_dptr = fb->module->globals[i].dptr;
            if (out_bytes) *out_bytes = fb->module->globals[i].bytes;
            return CUDA_SUCCESS;
        }
    }
    /* Fallback to driver global lookup (may lazily allocate known names). */
    return cuModuleGetGlobal_v2(out_dptr, out_bytes, fb->module, v->device_name);
}

/* id 0x1f1: __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_var(
        void *fatCubinHandle, void *hostVar, void *deviceAddress, void *deviceName,
        void *ext, void *size, void *constant, void *global) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_register_var");
    acuda_dd9e98a4_var *v;
    unsigned i;
    if (!fatCubinHandle || !hostVar || !deviceName)
        return CUDA_ERROR_INVALID_VALUE;
    for (i = 0; i < g_dd9e98a4_nvars; i++) {
        if (g_dd9e98a4_vars[i].host_var == hostVar) {
            v = &g_dd9e98a4_vars[i];
            goto update;
        }
    }
    if (g_dd9e98a4_nvars >= ACUDA_DD9E98A4_MAX_VARS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    v = &g_dd9e98a4_vars[g_dd9e98a4_nvars++];
    memset(v, 0, sizeof(*v));
update:
    v->fatbin_handle = fatCubinHandle;
    v->host_var = hostVar;
    v->device_address = deviceAddress;
    v->device_name = (const char *)deviceName;
    v->ext = (int)(intptr_t)ext;
    v->size = (size_t)(uintptr_t)size;
    v->constant = (int)(intptr_t)constant;
    v->global = (int)(intptr_t)global;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register var %p name=%s size=%zu const=%d global=%d\n",
                      hostVar, v->device_name ? v->device_name : "(null)", v->size, v->constant, v->global);
    return CUDA_SUCCESS;
}

/* id 0x27: cudaMemcpyToSymbol(symbol, src, count, offset, kind) */
static cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    CUresult r = acuda_dd9e98a4_resolve_symbol_ptr((void *)(uintptr_t)symbol, &dptr, &sym_size);
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 cudaMemcpyToSymbol sym=%p dptr=0x%llx sym=%zu count=%zu off=%zu kind=%d r=%d\n",
                      symbol, (unsigned long long)dptr, sym_size, count, offset, (int)kind, (int)r);
    if (r != CUDA_SUCCESS)
        return (cudaError_t)r;
    if (count == 0)
        count = sym_size > offset ? sym_size - offset : 0;
    if (offset > sym_size || count > sym_size - offset)
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    if (count == 0)
        return (cudaError_t)CUDA_SUCCESS;
    switch (kind) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDefault:
        if (!src) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
        return (cudaError_t)cuMemcpyHtoD_v2(dptr + (CUdeviceptr)offset, src, count);
    case cudaMemcpyDeviceToDevice:
        if (!src) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
        return (cudaError_t)cuMemcpyDtoD_v2(dptr + (CUdeviceptr)offset, (CUdeviceptr)(uintptr_t)src, count);
    default:
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    }
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy_to_symbol(
        void *symbol, void *src, void *count, void *offset,
        void *kind, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_memcpy_to_symbol");
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!symbol)
        return CUDA_ERROR_INVALID_VALUE;
    return (CUresult)cudaMemcpyToSymbol(symbol, src, (size_t)(uintptr_t)count, (size_t)(uintptr_t)offset,
                                        (enum cudaMemcpyKind)(int)(intptr_t)kind);
}

static cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    CUdeviceptr dptr = 0;
    size_t sym_size = 0;
    CUresult r = acuda_dd9e98a4_resolve_symbol_ptr((void *)(uintptr_t)symbol, &dptr, &sym_size);
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 cudaMemcpyFromSymbol sym=%p dptr=0x%llx sym=%zu count=%zu off=%zu kind=%d r=%d\n",
                      symbol, (unsigned long long)dptr, sym_size, count, offset, (int)kind, (int)r);
    if (r != CUDA_SUCCESS)
        return (cudaError_t)r;
    if (count == 0)
        count = sym_size > offset ? sym_size - offset : 0;
    if (offset > sym_size || count > sym_size - offset)
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    if (count == 0)
        return (cudaError_t)CUDA_SUCCESS;
    switch (kind) {
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDefault:
        if (!dst) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
        return (cudaError_t)cuMemcpyDtoH_v2(dst, dptr + (CUdeviceptr)offset, count);
    case cudaMemcpyDeviceToDevice:
        if (!dst) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
        return (cudaError_t)cuMemcpyDtoD_v2((CUdeviceptr)(uintptr_t)dst, dptr + (CUdeviceptr)offset, count);
    default:
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    }
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy_from_symbol(
        void *dst, void *symbol, void *count, void *offset,
        void *kind, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_memcpy_from_symbol");
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!symbol)
        return CUDA_ERROR_INVALID_VALUE;
    return (CUresult)cudaMemcpyFromSymbol(dst, symbol, (size_t)(uintptr_t)count, (size_t)(uintptr_t)offset,
                                          (enum cudaMemcpyKind)(int)(intptr_t)kind);
}

/* id 0x1f3: __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
 *          deviceName, thread_limit, tid, bid, bDim, gDim, wSize) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_function(
        void *fatCubinHandle, void *hostFun, void *deviceFun, void *deviceName,
        void *thread_limit, void *tid, void *bid, void *bDim) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_register_function");
    acuda_dd9e98a4_kernel *k;
    (void)deviceFun; (void)thread_limit; (void)tid; (void)bid; (void)bDim;
    if (!fatCubinHandle || !hostFun || !deviceName)
        return CUDA_ERROR_INVALID_VALUE;
    if (g_dd9e98a4_nkernels >= ACUDA_DD9E98A4_MAX_FUNCS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    k = &g_dd9e98a4_kernels[g_dd9e98a4_nkernels++];
    k->host_fn = hostFun;
    k->device_name = (const char *)deviceName;
    k->fatbin = NULL;
    {
        unsigned idx = (unsigned)(uintptr_t)fatCubinHandle;
        if (idx >= 1 && idx <= g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[idx - 1];
        else if (g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[g_dd9e98a4_nfatbins - 1];
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register kernel %p -> %s\n",
                      hostFun, k->device_name);
    return CUDA_SUCCESS;
}

static void acuda_dd9e98a4_htrace(const char *name, const void *a, const void *b, long long v, int phase)
{
    static FILE *fp = NULL;
    static int init = 0;
    static long long seq = 0;
#if defined(_WIN32)
    static SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
#endif
    if (!init) {
        const char *p = getenv("ACUDA_DD9E98A4_HANDLER_TRACE_FILE");
        init = 1;
        if (p && p[0])
            fp = fopen(p, "w");
    }
    if (!fp) {
#if defined(_WIN32)
        ReleaseSRWLockExclusive(&lock);
#endif
        return;
    }
    {
        long long cur = seq++;
        fprintf(fp, "%lld %s %d %p %p %lld st=%d g=%d a=%p\n",
                cur, name ? name : "?", phase, a, b, v,
                acuda_dd9e98a4_ready_state(), g_init, g_actx);
        if ((cur & 0x3f) == 0x3f)
            fflush(fp);
    }
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&lock);
#endif
}

/* id 0x1f8: __cudaGetKernel(void **kernel, const void *func) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_kernel(
        void *kernel, void *func, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_kernel");
    acuda_dd9e98a4_htrace("get_kern", kernel, func, 0, 0);
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 get kernel slot=%p func=%p\n",
                      kernel, func);
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!kernel || !func) {
        acuda_dd9e98a4_htrace("get_kern", kernel, func, -1, 1);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *(const void **)kernel = func;
    acuda_dd9e98a4_htrace("get_kern", kernel, func, 0, 1);
    return CUDA_SUCCESS;
}

/* Pushed <<<grid, block, shared, stream>>> configuration. */

static ACUDA_DD9E98A4_TLS acuda_dd9e98a4_launch_config
    g_dd9e98a4_configs[16];
static ACUDA_DD9E98A4_TLS unsigned g_dd9e98a4_config_depth;

/* id 0x1f6: __cudaPushCallConfiguration(dim3 *gridDim, dim3 *blockDim,
 *                                       size_t sharedMem, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_push_config(
        void *gridDim, void *blockDim, void *sharedMem, void *stream,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_push_config");
    acuda_dd9e98a4_launch_config *cfg;
    acuda_dd9e98a4_htrace("push_cfg", gridDim, stream, (long long)(size_t)sharedMem, 0);
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 push config\n");
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!gridDim || !blockDim || g_dd9e98a4_config_depth >= 16) {
        acuda_dd9e98a4_htrace("push_cfg", gridDim, stream, -1, 1);
        return CUDA_ERROR_INVALID_VALUE;
    }
    cfg = &g_dd9e98a4_configs[g_dd9e98a4_config_depth++];
    cfg->grid = *(const acuda_dd9e98a4_dim3 *)gridDim;
    cfg->block = *(const acuda_dd9e98a4_dim3 *)blockDim;
    cfg->shared = (size_t)sharedMem;
    cfg->stream = stream;
    acuda_dd9e98a4_htrace("push_cfg", gridDim, stream, 0, 1);
    return CUDA_SUCCESS;
}

/* id 0x1f7: __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
 *                                      size_t *sharedMem, cudaStream_t *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_pop_config(
        void *gridDim, void *blockDim, void *sharedMem, void *stream,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_pop_config");
    acuda_dd9e98a4_launch_config *cfg;
    acuda_dd9e98a4_htrace("pop_cfg", gridDim, stream, (long long)g_dd9e98a4_config_depth, 0);
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 pop config\n");
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (g_dd9e98a4_config_depth == 0) {
        acuda_dd9e98a4_htrace("pop_cfg", gridDim, stream, -1, 1);
        return CUDA_ERROR_INVALID_VALUE;
    }
    cfg = &g_dd9e98a4_configs[--g_dd9e98a4_config_depth];
    if (gridDim)
        *(acuda_dd9e98a4_dim3 *)gridDim = cfg->grid;
    if (blockDim)
        *(acuda_dd9e98a4_dim3 *)blockDim = cfg->block;
    if (sharedMem)
        *(size_t *)sharedMem = cfg->shared;
    if (stream)
        *(void **)stream = cfg->stream;
    acuda_dd9e98a4_htrace("pop_cfg", gridDim, stream, 0, 1);
    return CUDA_SUCCESS;
}

/* id 0x1f9: cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
 *                            void **args, size_t sharedMem, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_launch_kernel(
        void *func, void *gridDim, void *blockDim, void *args, void *sharedMem,
        void *stream, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_launch_kernel");
    const acuda_dd9e98a4_dim3 *grid = (const acuda_dd9e98a4_dim3 *)gridDim;
    const acuda_dd9e98a4_dim3 *block = (const acuda_dd9e98a4_dim3 *)blockDim;
    acuda_dd9e98a4_kernel *k = NULL;
    acuda_dd9e98a4_fatbin *fb;
    CUfunction fn = NULL;
    CUresult r;
    unsigned i;
    (void)unused_g; (void)unused_h;
    acuda_dd9e98a4_htrace("launch", func, stream, (long long)(size_t)sharedMem, 0);
    acuda_dd9e98a4_check_ready_canary("api_launch_kernel");
    if (!func || !grid || !block) {
        acuda_dd9e98a4_htrace("launch", func, stream, -1, 1);
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (i = 0; i < g_dd9e98a4_nkernels; i++) {
        if (g_dd9e98a4_kernels[i].host_fn == func) {
            k = &g_dd9e98a4_kernels[i];
            break;
        }
    }
    if (!k || !k->fatbin) {
        acuda_dd9e98a4_htrace("launch", func, stream, -2, 1);
        return CUDA_ERROR_NOT_FOUND;
    }
    fb = k->fatbin;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 launch %s grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                      k->device_name, grid->x, grid->y, grid->z, block->x,
                      block->y, block->z);
    if (fb->state == 0) {
        r = cuModuleLoadFatBinary(&fb->module, fb->blob);
        fb->state = (r == CUDA_SUCCESS) ? 1 : -1;
        ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 module load -> %d\n", (int)r);
    }
    if (fb->state != 1) {
        acuda_dd9e98a4_htrace("launch", func, stream, -3, 1);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    r = cuModuleGetFunction(&fn, fb->module, k->device_name);
    if (r != CUDA_SUCCESS) {
        acuda_dd9e98a4_htrace("launch", func, stream, (long long)r, 1);
        return r;
    }
    acuda_dd9e98a4_check_ready_canary("api_launch_kernel_pre_cuLaunch");
    r = cuLaunchKernel(fn, grid->x, grid->y, grid->z, block->x, block->y,
                       block->z, (unsigned int)(size_t)sharedMem,
                       (CUstream)stream, args, NULL);
    acuda_dd9e98a4_check_ready_canary("api_launch_kernel_post_cuLaunch");
    acuda_dd9e98a4_htrace("launch", func, stream, (long long)r, 1);
    return r;
}


/* id 0x5: cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_choose_device(
        void *device, void *prop, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_choose_device");
    (void)prop; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (device)
        *(int *)device = 0;
    return CUDA_SUCCESS;
}

/* id 0xa: cudaGetLastError(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_last_error(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_last_error");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xb: cudaPeekAtLastError(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_peek_at_last_error(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_peek_at_last_error");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xc: cudaGetErrorString(cudaError_t error) -> const char * */
static const char *ACUDA_CUDAAPI acuda_dd9e98a4_api_get_error_string_runtime(
        void *error, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_error_string_runtime");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    switch ((int)(intptr_t)error) {
    case 0: return "no error";
    case 1: return "invalid argument";
    case 2: return "out of memory";
    case 3: return "initialization error";
    case 201: return "invalid device context";
    case 700: return "an illegal memory access was encountered";
    default: return "unknown error";
    }
}

/* id 0x12: cudaSetValidDevices(int *device_arr, int len) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_valid_devices(
        void *device_arr, void *len, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_set_valid_devices");
    (void)device_arr; (void)len; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x13: cudaSetDeviceFlags(unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device_flags(
        void *flags, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_set_device_flags");
    (void)flags; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xd4: cudaGetDeviceFlags(unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device_flags(
        void *flags, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_get_device_flags");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (flags)
        *(unsigned int *)flags = 0;
    return CUDA_SUCCESS;
}

/* id 0x81: cudaStreamCreate(cudaStream_t *pStream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create(
        void *pStream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_create");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, 0);
}

/* id 0xc6: cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create_with_flags(
        void *pStream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_create_with_flags");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, (unsigned int)(uintptr_t)flags);
}

/* id 0xca: cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create_with_priority(
        void *pStream, void *flags, void *priority, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_create_with_priority");
    (void)priority; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, (unsigned int)(uintptr_t)flags);
}

/* id 0xa6: cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_limit(
        void *pValue, void *limit, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_limit");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!pValue)
        return CUDA_ERROR_INVALID_VALUE;
    *(size_t *)pValue = 0;
    return cuCtxGetLimit((size_t *)pValue, (int)(intptr_t)limit);
}

/* id 0xa7: cudaDeviceSetLimit(enum cudaLimit limit, size_t value) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_limit(
        void *limit, void *value, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_set_limit");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    return cuCtxSetLimit((int)(intptr_t)limit, (size_t)(uintptr_t)value);
}

/* id 0xa8: cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_cache_config(
        void *pCacheConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_cache_config");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (pCacheConfig)
        *(int *)pCacheConfig = 0;
    return CUDA_SUCCESS;
}

/* id 0xa9: cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_cache_config(
        void *cacheConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_set_cache_config");
    (void)cacheConfig; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xad: cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_by_pci_bus_id(
        void *device, void *pciBusId, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_by_pci_bus_id");
    (void)pciBusId; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (device)
        *(int *)device = 0;
    return CUDA_SUCCESS;
}

/* id 0xae: cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) */
static cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (!pciBusId || len <= 0)
        return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    return (cudaError_t)cuDeviceGetPCIBusId(pciBusId, len, (CUdevice)device);
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_pci_bus_id(
        void *pciBusId, void *len, void *device, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_pci_bus_id");
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return (CUresult)cudaDeviceGetPCIBusId((char *)pciBusId, (int)(intptr_t)len, (int)(intptr_t)device);
}

/* id 0xcd: cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_stream_priority_range(
        void *leastPriority, void *greatestPriority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_stream_priority_range");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (leastPriority)
        *(int *)leastPriority = 0;
    if (greatestPriority)
        *(int *)greatestPriority = 0;
    return CUDA_SUCCESS;
}

/* id 0xb0: cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ipc_get_event_handle(
        void *handle, void *event, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ipc_get_event_handle");
    (void)event; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (handle)
        memset(handle, 0, 64);
    return CUDA_SUCCESS;
}

/* id 0xb2: cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ipc_get_mem_handle(
        void *handle, void *devPtr, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ipc_get_mem_handle");
    (void)devPtr; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (handle)
        memset(handle, 0, 64);
    return CUDA_SUCCESS;
}

/* id 0xb4: cudaIpcCloseMemHandle(void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ipc_close_mem_handle(
        void *devPtr, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ipc_close_mem_handle");
    (void)devPtr; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xb7: cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_shared_mem_config(
        void *pConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_shared_mem_config");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (pConfig)
        *(int *)pConfig = 0;
    return CUDA_SUCCESS;
}

/* id 0xb8: cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_shared_mem_config(
        void *config, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_set_shared_mem_config");
    (void)config; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x7: cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f)
 * Returns pointer to cudaChannelFormatDesc { int x,y,z,w; enum cudaChannelFormatKind f; } (20 bytes). */
static cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
    cudaChannelFormatDesc d;
    d.x = x; d.y = y; d.z = z; d.w = w; d.f = f;
    return d;
}

static void *ACUDA_CUDAAPI acuda_dd9e98a4_api_create_channel_desc(
        void *a, void *b, void *c, void *d,
        void *e, void *f, void *g, void *h) {
    static cudaChannelFormatDesc s_desc;
    cudaChannelFormatDesc *out = (cudaChannelFormatDesc *)a;
    if (!out) out = &s_desc;
    *out = cudaCreateChannelDesc((int)(size_t)b, (int)(size_t)c, (int)(size_t)d, (int)(size_t)e, (enum cudaChannelFormatKind)(int)(size_t)f);
    (void)g; (void)h;
    return (void *)out;
}

/* id 0x1: cudaDriverGetVersion(int *driverVersion) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_driver_get_version(
        void *driverVersion, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    return cuDriverGetVersion((int *)driverVersion);
}

/* id 0x2: cudaRuntimeGetVersion(int *runtimeVersion) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_runtime_get_version(
        void *runtimeVersion, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    /* Keep runtime and driver version aligned to ACuda's single CUDA version. */
    return cuDriverGetVersion((int *)runtimeVersion);
}

/* id 0x6: cudaGetChannelDesc(cudaChannelFormatDesc *desc, cudaArray_t array) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_channel_desc(
        void *desc, void *array, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    return (CUresult)cudaGetChannelDesc((cudaChannelFormatDesc *)desc, (cudaArray_t)(uintptr_t)array);
}

typedef struct acuda_graphics_resource_st *acuda_cudaGraphicsResource_t;

/* id 0x4c: cudaGraphicsMapResources */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_graphics_map_resources(
        void *count, void *resources, void *stream, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuGraphicsMapResources((unsigned int)(uintptr_t)count, (CUgraphicsResource *)resources, (CUstream)stream);
}
static cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, acuda_cudaGraphicsResource_t resource) {
    CUdeviceptr dptr = 0;
    size_t sz = 0;
    CUresult r;
    if (!devPtr || !size) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    *devPtr = NULL; *size = 0;
    r = cuGraphicsResourceGetMappedPointer_v2(&dptr, &sz, (CUgraphicsResource)(uintptr_t)resource);
    if (r != CUDA_SUCCESS) return (cudaError_t)r;
    *devPtr = (void *)(uintptr_t)dptr;
    *size = sz;
    return (cudaError_t)CUDA_SUCCESS;
}

static cudaError_t cudaGraphicsSubResourceGetMappedArray(void **array, acuda_cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
    CUresult r;
    if (!array) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    *array = NULL;
    r = cuGraphicsSubResourceGetMappedArray((CUarray *)array, (CUgraphicsResource)(uintptr_t)resource, arrayIndex, mipLevel);
    return (cudaError_t)r;
}

static cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(void **mipmappedArray, acuda_cudaGraphicsResource_t resource) {
    CUresult r;
    if (!mipmappedArray) return (cudaError_t)CUDA_ERROR_INVALID_VALUE;
    *mipmappedArray = NULL;
    r = cuGraphicsResourceGetMappedMipmappedArray((CUmipmappedArray *)mipmappedArray, (CUgraphicsResource)(uintptr_t)resource);
    return (cudaError_t)r;
}

/* id 0x4d: cudaGraphicsUnmapResources */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_graphics_unmap_resources(
        void *count, void *resources, void *stream, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return cuGraphicsUnmapResources((unsigned int)(uintptr_t)count, (CUgraphicsResource *)resources, (CUstream)stream);
}
/* id 0x4e: cudaGraphicsResourceGetMappedPointer */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_graphics_resource_get_mapped_pointer(
        void *devPtr, void *size, void *resource, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return (CUresult)cudaGraphicsResourceGetMappedPointer((void **)devPtr, (size_t *)size,
                                                          (acuda_cudaGraphicsResource_t)(uintptr_t)resource);
}
/* id 0x4f: cudaGraphicsSubResourceGetMappedArray */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_graphics_sub_resource_get_mapped_array(
        void *array, void *resource, void *arrayIndex, void *mipLevel,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return (CUresult)cudaGraphicsSubResourceGetMappedArray((void **)array,
                                                           (acuda_cudaGraphicsResource_t)(uintptr_t)resource,
                                                           (unsigned int)(uintptr_t)arrayIndex,
                                                           (unsigned int)(uintptr_t)mipLevel);
}
/* id 0xc4: cudaGraphicsResourceGetMappedMipmappedArray */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_graphics_resource_get_mapped_mipmapped_array(
        void *mipmappedArray, void *resource, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return (CUresult)cudaGraphicsResourceGetMappedMipmappedArray((void **)mipmappedArray,
                                                                 (acuda_cudaGraphicsResource_t)(uintptr_t)resource);
}

typedef struct { unsigned long long tex; unsigned char view[64]; int has; } acuda_tex_view_entry_t;
static acuda_tex_view_entry_t g_tex_views[256];
#if defined(_WIN32)
static SRWLOCK g_tex_views_lock = SRWLOCK_INIT;
#endif
static void acuda_dd9e98a4_store_tex_view(unsigned long long tex, const void *view) {
#if defined(_WIN32)
    AcquireSRWLockExclusive(&g_tex_views_lock);
#endif
    for (int i = 0; i < 256; i++) {
        if (g_tex_views[i].has && g_tex_views[i].tex == tex) {
            if (view) memcpy(g_tex_views[i].view, view, 64);
            else memset(g_tex_views[i].view, 0, 64);
            g_tex_views[i].has = view ? 1 : 0;
#if defined(_WIN32)
            ReleaseSRWLockExclusive(&g_tex_views_lock);
#endif
            return;
        }
    }
    if (view) {
        for (int i = 0; i < 256; i++) {
            if (!g_tex_views[i].has) {
                g_tex_views[i].tex = tex;
                memcpy(g_tex_views[i].view, view, 64);
                g_tex_views[i].has = 1;
                break;
            }
        }
    }
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&g_tex_views_lock);
#endif
}
static int acuda_dd9e98a4_load_tex_view(unsigned long long tex, void *view) {
    int ok = 0;
#if defined(_WIN32)
    AcquireSRWLockShared(&g_tex_views_lock);
#endif
    for (int i = 0; i < 256; i++) {
        if (g_tex_views[i].has && g_tex_views[i].tex == tex) {
            if (view) memcpy(view, g_tex_views[i].view, 64);
            ok = 1; break;
        }
    }
#if defined(_WIN32)
    ReleaseSRWLockShared(&g_tex_views_lock);
#endif
    return ok;
}

static CUresult acuda_dd9e98a4_to_cu_resdesc(const void *pResDesc, CUDA_RESOURCE_DESC *rd) {
    /* cudart cudaResourceDesc layout: int resType; +pad; union { array, mipmap, linear, pitch2D } */
    typedef struct { int x,y,z,w,f; } acuda_chfmt;
    typedef struct { void *array; } acuda_res_array;
    typedef struct { void *mipmap; } acuda_res_mip;
    typedef struct { void *devPtr; acuda_chfmt desc; size_t sizeInBytes; } acuda_res_linear;
    typedef struct { void *devPtr; acuda_chfmt desc; size_t pitchInBytes; size_t width; size_t height; } acuda_res_pitch2d;
    typedef struct { int resType; int pad; union { acuda_res_array array; acuda_res_mip mipmap; acuda_res_linear linear; acuda_res_pitch2d pitch2D; char raw[64]; } res; } acuda_cuda_resdesc;
    const acuda_cuda_resdesc *src = (const acuda_cuda_resdesc *)pResDesc;
    memset(rd, 0, sizeof(*rd));
    if (!src) return CUDA_ERROR_INVALID_VALUE;
    rd->resType = (CUresourcetype)src->resType;
    if (src->resType == 0) { /* cudaResourceTypeArray */
        rd->res.array.hArray = (CUarray)(uintptr_t)src->res.array.array;
        return CUDA_SUCCESS;
    } else if (src->resType == 1) { /* cudaResourceTypeMipmappedArray */
        rd->res.mipmap.hMipmappedArray = (CUmipmappedArray)(uintptr_t)src->res.mipmap.mipmap;
        return CUDA_SUCCESS;
    } else if (src->resType == 2) { /* cudaResourceTypeLinear */
        unsigned n = (src->res.linear.desc.x?1:0)+(src->res.linear.desc.y?1:0)+(src->res.linear.desc.z?1:0)+(src->res.linear.desc.w?1:0);
        CUarray_format fmt = CU_AD_FORMAT_FLOAT;
        if (src->res.linear.desc.f == 1) fmt = CU_AD_FORMAT_UNSIGNED_INT32;
        else if (src->res.linear.desc.f == 2) fmt = CU_AD_FORMAT_SIGNED_INT32;
        rd->res.linear.devPtr = (CUdeviceptr)(uintptr_t)src->res.linear.devPtr;
        rd->res.linear.format = fmt;
        rd->res.linear.numChannels = n;
        rd->res.linear.sizeInBytes = src->res.linear.sizeInBytes;
        return CUDA_SUCCESS;
    } else if (src->resType == 3) { /* cudaResourceTypePitch2D */
        unsigned n = (src->res.pitch2D.desc.x?1:0)+(src->res.pitch2D.desc.y?1:0)+(src->res.pitch2D.desc.z?1:0)+(src->res.pitch2D.desc.w?1:0);
        CUarray_format fmt = CU_AD_FORMAT_FLOAT;
        if (src->res.pitch2D.desc.f == 1) fmt = CU_AD_FORMAT_UNSIGNED_INT32;
        else if (src->res.pitch2D.desc.f == 2) fmt = CU_AD_FORMAT_SIGNED_INT32;
        rd->res.pitch2D.devPtr = (CUdeviceptr)(uintptr_t)src->res.pitch2D.devPtr;
        rd->res.pitch2D.format = fmt;
        rd->res.pitch2D.numChannels = n;
        rd->res.pitch2D.width = src->res.pitch2D.width;
        rd->res.pitch2D.height = src->res.pitch2D.height;
        rd->res.pitch2D.pitchInBytes = src->res.pitch2D.pitchInBytes;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0xb9: cudaCreateTextureObject */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_create_texture_object(
        void *pTexObject, void *pResDesc, void *pTexDesc, void *pResViewDesc,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_RESOURCE_DESC rd;
    CUDA_TEXTURE_DESC td;
    CUtexObject tex = 0;
    CUresult r;
    (void)pResViewDesc; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pTexObject || !pResDesc || !pTexDesc) return CUDA_ERROR_INVALID_VALUE;
    r = acuda_dd9e98a4_to_cu_resdesc(pResDesc, &rd);
    if (r != CUDA_SUCCESS) return r;
    memset(&td, 0, sizeof(td));
    {
        /* cudart cudaTextureDesc: addressMode[3], filterMode, readMode, sRGB, borderColor[4], normalizedCoords, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp */
        const int *src = (const int *)pTexDesc;
        for (int i = 0; i < 3; i++) td.addressMode[i] = (CUaddress_mode)src[i];
        td.filterMode = (CUfilter_mode)src[3];
        /* src[4]=readMode, src[5]=sRGB */
        td.borderColor[0] = ((const float *)pTexDesc)[6];
        td.borderColor[1] = ((const float *)pTexDesc)[7];
        td.borderColor[2] = ((const float *)pTexDesc)[8];
        td.borderColor[3] = ((const float *)pTexDesc)[9];
        td.flags = src[10] ? 0x2u : 0u; /* normalizedCoords -> CU_TRSF_NORMALIZED_COORDINATES */
        td.maxAnisotropy = (unsigned)src[11];
        td.mipmapFilterMode = (CUfilter_mode)src[12];
        td.mipmapLevelBias = ((const float *)pTexDesc)[13];
        td.minMipmapLevelClamp = ((const float *)pTexDesc)[14];
        td.maxMipmapLevelClamp = ((const float *)pTexDesc)[15];
    }
    {
        CUDA_RESOURCE_VIEW_DESC vd;
        CUDA_RESOURCE_VIEW_DESC *pvd = NULL;
        if (pResViewDesc) {
            const unsigned int *v = (const unsigned int *)pResViewDesc;
            memset(&vd, 0, sizeof(vd));
            vd.format = (CUresourceViewFormat)v[0];
            vd.width = (size_t)*(const unsigned long long *)((const char *)pResViewDesc + 8);
            vd.height = (size_t)*(const unsigned long long *)((const char *)pResViewDesc + 16);
            vd.depth = (size_t)*(const unsigned long long *)((const char *)pResViewDesc + 24);
            vd.firstMipmapLevel = v[8];
            vd.lastMipmapLevel = v[9];
            vd.firstLayer = v[10];
            vd.lastLayer = v[11];
            pvd = &vd;
        }
        r = cuTexObjectCreate(&tex, &rd, &td, pvd);
    }
    if (r != CUDA_SUCCESS) return r;
    *(unsigned long long *)pTexObject = (unsigned long long)tex;
    acuda_dd9e98a4_store_tex_view((unsigned long long)tex, pResViewDesc);
    return CUDA_SUCCESS;
}
/* id 0xba: cudaDestroyTextureObject */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_destroy_texture_object(
        void *texObject, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!texObject) return CUDA_ERROR_INVALID_VALUE;
    acuda_dd9e98a4_store_tex_view((unsigned long long)(uintptr_t)texObject, NULL);
    return cuTexObjectDestroy((CUtexObject)(uintptr_t)texObject);
}
/* id 0xbb: cudaGetTextureObjectResourceDesc */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_texture_object_resource_desc(
        void *pResDesc, void *texObject, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_RESOURCE_DESC rd;
    CUresult r;
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pResDesc) return CUDA_ERROR_INVALID_VALUE;
    r = cuTexObjectGetResourceDesc(&rd, (CUtexObject)(uintptr_t)texObject);
    if (r != CUDA_SUCCESS) return r;
    memset(pResDesc, 0, 64);
    {
        typedef struct { int resType; int pad; union { struct { void *array; } array; struct { void *mipmap; } mipmap; char raw[64]; } res; } out_t;
        out_t *o = (out_t *)pResDesc;
        o->resType = (int)rd.resType;
        if (rd.resType == CU_RESOURCE_TYPE_ARRAY) o->res.array.array = (void *)rd.res.array.hArray;
        else if (rd.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) o->res.mipmap.mipmap = (void *)rd.res.mipmap.hMipmappedArray;
        else memcpy(&o->res, &rd.res, sizeof(o->res));
    }
    return CUDA_SUCCESS;
}
/* id 0xbc: cudaGetTextureObjectTextureDesc */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_texture_object_texture_desc(
        void *pTexDesc, void *texObject, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_TEXTURE_DESC td;
    CUresult r;
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pTexDesc) return CUDA_ERROR_INVALID_VALUE;
    r = cuTexObjectGetTextureDesc(&td, (CUtexObject)(uintptr_t)texObject);
    if (r != CUDA_SUCCESS) return r;
    memset(pTexDesc, 0, 64);
    {
        int *dst = (int *)pTexDesc;
        float *fdst = (float *)pTexDesc;
        for (int i = 0; i < 3; i++) dst[i] = (int)td.addressMode[i];
        dst[3] = (int)td.filterMode;
        dst[4] = 0; /* readMode */
        dst[5] = 0; /* sRGB */
        fdst[6] = td.borderColor[0]; fdst[7] = td.borderColor[1]; fdst[8] = td.borderColor[2]; fdst[9] = td.borderColor[3];
        dst[10] = (td.flags & 0x2u) ? 1 : 0;
        dst[11] = (int)td.maxAnisotropy;
        dst[12] = (int)td.mipmapFilterMode;
        fdst[13] = td.mipmapLevelBias; fdst[14] = td.minMipmapLevelClamp; fdst[15] = td.maxMipmapLevelClamp;
    }
    return CUDA_SUCCESS;
}
/* id 0xbd: cudaCreateSurfaceObject */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_create_surface_object(
        void *pSurfObject, void *pResDesc, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_RESOURCE_DESC rd;
    CUsurfObject surf = 0;
    CUresult r;
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pSurfObject || !pResDesc) return CUDA_ERROR_INVALID_VALUE;
    r = acuda_dd9e98a4_to_cu_resdesc(pResDesc, &rd);
    if (r != CUDA_SUCCESS) return r;
    r = cuSurfObjectCreate(&surf, &rd);
    if (r != CUDA_SUCCESS) return r;
    *(unsigned long long *)pSurfObject = (unsigned long long)surf;
    return CUDA_SUCCESS;
}
/* id 0xbe: cudaDestroySurfaceObject */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_destroy_surface_object(
        void *surfObject, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!surfObject) return CUDA_ERROR_INVALID_VALUE;
    return cuSurfObjectDestroy((CUsurfObject)(uintptr_t)surfObject);
}
/* id 0xbf: cudaGetSurfaceObjectResourceDesc */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_surface_object_resource_desc(
        void *pResDesc, void *surfObject, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_RESOURCE_DESC rd;
    CUresult r;
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pResDesc) return CUDA_ERROR_INVALID_VALUE;
    r = cuSurfObjectGetResourceDesc(&rd, (CUsurfObject)(uintptr_t)surfObject);
    if (r != CUDA_SUCCESS) return r;
    memset(pResDesc, 0, 64);
    {
        typedef struct { int resType; int pad; union { struct { void *array; } array; struct { void *mipmap; } mipmap; char raw[64]; } res; } out_t;
        out_t *o = (out_t *)pResDesc;
        o->resType = (int)rd.resType;
        if (rd.resType == CU_RESOURCE_TYPE_ARRAY) o->res.array.array = (void *)rd.res.array.hArray;
        else if (rd.resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) o->res.mipmap.mipmap = (void *)rd.res.mipmap.hMipmappedArray;
        else memcpy(&o->res, &rd.res, sizeof(o->res));
    }
    return CUDA_SUCCESS;
}
/* id 0xc7: cudaGetTextureObjectResourceViewDesc */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_texture_object_resource_view_desc(
        void *pResViewDesc, void *texObject, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUDA_RESOURCE_VIEW_DESC vd;
    CUresult r;
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!pResViewDesc) return CUDA_ERROR_INVALID_VALUE;
    memset(&vd, 0, sizeof(vd));
    r = cuTexObjectGetResourceViewDesc(&vd, (CUtexObject)(uintptr_t)texObject);
    if (r == CUDA_SUCCESS) {
        memset(pResViewDesc, 0, 64);
        memcpy(pResViewDesc, &vd, sizeof(vd) < 64 ? sizeof(vd) : 64);
        return CUDA_SUCCESS;
    }
    if (acuda_dd9e98a4_load_tex_view((unsigned long long)(uintptr_t)texObject, pResViewDesc))
        return CUDA_SUCCESS;
    memset(pResViewDesc, 0, 64);
    return r;
}

/* id 0x201-0x203: CUDA logs */
typedef struct { void *cb; void *userData; } acuda_log_cb_t;
static acuda_log_cb_t g_log_cbs[16];
#if defined(_WIN32)
static SRWLOCK g_log_cbs_lock = SRWLOCK_INIT;
#endif
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_logs_register_callback(
        void *cb, void *userData, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!cb) return CUDA_ERROR_INVALID_VALUE;
#if defined(_WIN32)
    AcquireSRWLockExclusive(&g_log_cbs_lock);
#endif
    for (int i = 0; i < 16; i++) {
        if (g_log_cbs[i].cb == cb && g_log_cbs[i].userData == userData) {
#if defined(_WIN32)
            ReleaseSRWLockExclusive(&g_log_cbs_lock);
#endif
            return CUDA_SUCCESS;
        }
    }
    for (int i = 0; i < 16; i++) {
        if (!g_log_cbs[i].cb) { g_log_cbs[i].cb = cb; g_log_cbs[i].userData = userData; break; }
    }
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&g_log_cbs_lock);
#endif
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_logs_unregister_callback(
        void *cb, void *userData, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)userData; (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!cb) return CUDA_ERROR_INVALID_VALUE;
#if defined(_WIN32)
    AcquireSRWLockExclusive(&g_log_cbs_lock);
#endif
    for (int i = 0; i < 16; i++) {
        if (g_log_cbs[i].cb == cb) { g_log_cbs[i].cb = NULL; g_log_cbs[i].userData = NULL; }
    }
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&g_log_cbs_lock);
#endif
    return CUDA_SUCCESS;
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_logs_current(
        void *log, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (log) *(void **)log = NULL;
    return CUDA_SUCCESS;
}



/* id 0x83: cudaStreamSynchronize(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_synchronize(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_synchronize");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    acuda_dd9e98a4_htrace("stream_sync", stream, NULL, 0, 0);
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL stream_synchronize stream=%p\n", stream);
    CUresult _r = cuStreamSynchronize((CUstream)stream);
    acuda_dd9e98a4_htrace("stream_sync", stream, NULL, (long long)_r, 1);
    return _r;
}

/* id 0x84: cudaStreamQuery(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_query(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_query");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    return cuStreamQuery((CUstream)stream);
}

/* id 0xc9: cudaStreamDestroy(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_destroy(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_destroy");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 CALL stream_destroy stream=%p\n", stream);
    if (!stream)
        return CUDA_SUCCESS;
    return cuStreamDestroy_v2((CUstream)stream);
}

/* id 0xcb: cudaStreamGetPriority(cudaStream_t stream, int *priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_priority(
        void *stream, void *priority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_priority");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!priority)
        return CUDA_ERROR_INVALID_VALUE;
    *(int *)priority = 0;
    return cuStreamGetPriority((CUstream)stream, (int *)priority);
}

/* id 0xcc: cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_flags(
        void *stream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_flags");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    *(unsigned int *)flags = 0;
    return cuStreamGetFlags((CUstream)stream, (unsigned int *)flags);
}

/* id 0xed: cudaStreamGetPriority_ptsz(cudaStream_t stream, int *priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_priority_ptsz(
        void *stream, void *priority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_priority_ptsz");
    return acuda_dd9e98a4_api_stream_get_priority(stream, priority, unused_c,
                                                  unused_d, unused_e, unused_f,
                                                  unused_g, unused_h);
}

/* id 0xee: cudaStreamGetFlags_ptsz(cudaStream_t stream, unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_flags_ptsz(
        void *stream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_flags_ptsz");
    return acuda_dd9e98a4_api_stream_get_flags(stream, flags, unused_c,
                                               unused_d, unused_e, unused_f,
                                               unused_g, unused_h);
}

/* id 0xef: cudaStreamSynchronize_ptsz(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_synchronize_ptsz(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_synchronize_ptsz");
    return acuda_dd9e98a4_api_stream_synchronize(stream, unused_b, unused_c,
                                                  unused_d, unused_e, unused_f,
                                                  unused_g, unused_h);
}

/* id 0xf0: cudaStreamQuery_ptsz(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_query_ptsz(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_query_ptsz");
    return acuda_dd9e98a4_api_stream_query(stream, unused_b, unused_c,
                                           unused_d, unused_e, unused_f,
                                           unused_g, unused_h);
}

/* id 0x1e0: cudaStreamGetDevice(cudaStream_t stream, int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_device(
        void *stream, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_device");
    (void)stream; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    *(int *)device = 0;
    return cuCtxGetDevice((CUdevice *)device);
}

/* id 0x1e1: cudaStreamGetDevice_ptsz(cudaStream_t stream, int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_device_ptsz(
        void *stream, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_device_ptsz");
    return acuda_dd9e98a4_api_stream_get_device(stream, device, unused_c,
                                                unused_d, unused_e, unused_f,
                                                unused_g, unused_h);
}



/* id 0xf2: cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record_ptsz(
        void *event, void *stream, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_event_record_ptsz");
    return acuda_dd9e98a4_api_event_record(event, stream, unused_c, unused_d,
                                           unused_e, unused_f, unused_g, unused_h);
}

/* id 0x13c: cudaStreamBeginCapture_ptsz(cudaStream_t stream, cudaStreamCaptureMode mode) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_begin_capture_ptsz(
        void *stream, void *mode, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_begin_capture_ptsz");
    (void)stream; (void)mode; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    /* Stream capture not implemented; report not supported to avoid
     * leaving capture state inconsistent. */
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x13e: cudaStreamIsCapturing_ptsz(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_is_capturing_ptsz(
        void *stream, void *captureStatus, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_is_capturing_ptsz");
    (void)stream; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!captureStatus)
        return CUDA_ERROR_INVALID_VALUE;
    /* cudaStreamCaptureStatusNone = 0 */
    *(int *)captureStatus = 0;
    return CUDA_SUCCESS;
}

/* id 0x140: cudaStreamEndCapture_ptsz(cudaStream_t stream, cudaGraph_t *pGraph) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_end_capture_ptsz(
        void *stream, void *pGraph, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_end_capture_ptsz");
    (void)stream; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (pGraph)
        *(void **)pGraph = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x147: cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_thread_exchange_stream_capture_mode(
        void *mode, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_thread_exchange_stream_capture_mode");
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    /* Keep mode unchanged (e.g. relaxed/global/thread-local as passed). */
    (void)mode;
    return CUDA_SUCCESS;
}

/* id 0x151: cudaCtxResetPersistingL2Cache(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ctx_reset_persisting_l2_cache(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ctx_reset_persisting_l2_cache");
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x156: cudaStreamCopyAttributes_ptsz(cudaStream_t dst, cudaStream_t src) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_copy_attributes_ptsz(
        void *dst, void *src, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_copy_attributes_ptsz");
    (void)dst; (void)src; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x174: cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_default_mempool(
        void *memPool, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_default_mempool");
    (void)device; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!memPool)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)memPool = (void *)(uintptr_t)1;
    return CUDA_SUCCESS;
}

/* id 0x181: cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_mempool(
        void *device, void *memPool, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_set_mempool");
    (void)device; (void)memPool; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x182: cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_mempool(
        void *memPool, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_mempool");
    (void)device; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!memPool)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)memPool = (void *)(uintptr_t)1;
    return CUDA_SUCCESS;
}

/* id 0x1ba: cudaStreamGetId_ptsz(cudaStream_t stream, unsigned long long *streamId) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_id_ptsz(
        void *stream, void *streamId, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_get_id_ptsz");
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!streamId)
        return CUDA_ERROR_INVALID_VALUE;
    *(unsigned long long *)streamId = (unsigned long long)(uintptr_t)stream;
    return CUDA_SUCCESS;
}



/* id 0x1bc: cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_init_device(
        void *device, void *deviceFlags, void *flags, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_init_device");
    (void)device; (void)deviceFlags; (void)flags;
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xff: cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr,
 *                                     int srcDevice, int dstDevice) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_p2p_attribute(
        void *value, void *attr, void *srcDevice, void *dstDevice,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_p2p_attribute");
    (void)attr; (void)srcDevice; (void)dstDevice;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    *(int *)value = 0;
    return CUDA_SUCCESS;
}

/* id 0x15b: cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
 *          const struct cudaChannelFormatDesc *fmtDesc, int numChannels, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_texture1d_linear_max_width(
        void *maxWidthInElements, void *fmtDesc, void *numChannels, void *device,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_texture1d_linear_max_width");
    (void)fmtDesc; (void)numChannels; (void)device;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!maxWidthInElements)
        return CUDA_ERROR_INVALID_VALUE;
    *(size_t *)maxWidthInElements = 0;
    return CUDA_SUCCESS;
}

/* id 0x195: cudaDeviceFlushGPUDirectRDMAWrites(target, scope) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_flush_gpudirect_rdma_writes(
        void *target, void *scope, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_flush_gpudirect_rdma_writes");
    (void)target; (void)scope; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x209: cudaDeviceGetHostAtomicCapabilities(unsigned int *capabilities,
 *          const enum cudaAtomicOperation *operations, unsigned int count, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_host_atomic_capabilities(
        void *capabilities, void *operations, void *count, void *device,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_host_atomic_capabilities");
    (void)operations; (void)count; (void)device;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (capabilities)
        *(unsigned int *)capabilities = 0;
    return CUDA_SUCCESS;
}

/* id 0x20a: cudaDeviceGetP2PAtomicCapabilities(unsigned int *capabilities,
 *          const enum cudaAtomicOperation *operations, unsigned int count,
 *          int srcDevice, int dstDevice) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_p2p_atomic_capabilities(
        void *capabilities, void *operations, void *count, void *srcDevice,
        void *dstDevice, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_get_p2p_atomic_capabilities");
    (void)operations; (void)count; (void)srcDevice; (void)dstDevice;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (capabilities)
        *(unsigned int *)capabilities = 0;
    return CUDA_SUCCESS;
}

/* id 0xb1: cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ipc_open_event_handle(
        void *event, void *handle, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ipc_open_event_handle");
    (void)handle; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (event)
        *(void **)event = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0xb3: cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_ipc_open_mem_handle(
        void *devPtr, void *handle, void *flags, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_ipc_open_mem_handle");
    (void)handle; (void)flags; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (devPtr)
        *(void **)devPtr = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x112: cudaImportExternalMemory(cudaExternalMemory_t *extMem_out,
 *           const struct cudaExternalMemoryHandleDesc *memHandleDesc) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_import_external_memory(
        void *extMem_out, void *memHandleDesc, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_import_external_memory");
    (void)memHandleDesc; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (extMem_out)
        *(void **)extMem_out = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x113: cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem,
 *           const struct cudaExternalMemoryBufferDesc *bufferDesc) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_external_memory_get_mapped_buffer(
        void *devPtr, void *extMem, void *bufferDesc, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_external_memory_get_mapped_buffer");
    (void)extMem; (void)bufferDesc; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (devPtr)
        *(void **)devPtr = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x114: cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap,
 *           cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_external_memory_get_mapped_mipmapped_array(
        void *mipmap, void *extMem, void *mipmapDesc, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_external_memory_get_mapped_mipmapped_array");
    (void)extMem; (void)mipmapDesc; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (mipmap)
        *(void **)mipmap = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x115: cudaDestroyExternalMemory(cudaExternalMemory_t extMem) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_destroy_external_memory(
        void *extMem, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_destroy_external_memory");
    (void)extMem; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x116: cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out,
 *           const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_import_external_semaphore(
        void *extSem_out, void *semHandleDesc, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_import_external_semaphore");
    (void)semHandleDesc; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (extSem_out)
        *(void **)extSem_out = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x124: cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
 * Used by statically linked cudart (CUDA 13.5). Synchronous backend: drain the
 * stream, then invoke the host function. */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_launch_host_func(
        void *stream, void *fn, void *userData, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_launch_host_func");
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!fn)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = cuStreamSynchronize((CUstream)stream);
    if (r != CUDA_SUCCESS)
        return r;
    ((void (*)(void *))fn)(userData);
    return CUDA_SUCCESS;
}

/* id 0x1d1: cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc,
 *           void* userData, cudaAsyncCallbackHandle_t* callback) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_register_async_notification(
        void *device, void *callbackFunc, void *userData, void *callback,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_register_async_notification");
    (void)device; (void)callbackFunc; (void)userData;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (callback)
        *(void **)callback = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x1d2: cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_unregister_async_notification(
        void *device, void *callback, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_device_unregister_async_notification");
    (void)device; (void)callback; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}


/* id 0x93: cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
 *                              unsigned int flags) (dd9e98a4 runtime interface) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_wait_event(
        void *stream, void *event, void *flags, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_wait_event");
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    return cuStreamWaitEvent((CUstream)stream, (CUevent)event,
                             (unsigned int)(uintptr_t)flags);
}

/* id 0xc5: cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t cb,
 *                                void *userData, unsigned int flags)
 * (dd9e98a4 runtime interface). Synchronous backend: drain stream, invoke. */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_add_callback(
        void *stream, void *callback, void *userData, void *flags,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_check_ready_canary("acuda_dd9e98a4_api_stream_add_callback");
    (void)flags; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!callback)
        return CUDA_ERROR_INVALID_VALUE;
    {
        CUresult r = cuStreamSynchronize((CUstream)stream);
        if (r != CUDA_SUCCESS)
            return r;
    }
    ((void (*)(void *, int, void *))callback)(stream, (int)CUDA_SUCCESS, userData);
    return CUDA_SUCCESS;
}

static void *acuda_dd9e98a4_get_id(unsigned int id) {
    switch (id) {
    case 0x1:  return (void *)&acuda_dd9e98a4_api_driver_get_version;
    case 0x2:  return (void *)&acuda_dd9e98a4_api_runtime_get_version;
    case 0x7:  return (void *)&acuda_dd9e98a4_api_create_channel_desc;
    case 0x4c: return (void *)&acuda_dd9e98a4_api_graphics_map_resources;
    case 0x4d: return (void *)&acuda_dd9e98a4_api_graphics_unmap_resources;
    case 0x4e: return (void *)&acuda_dd9e98a4_api_graphics_resource_get_mapped_pointer;
    case 0x4f: return (void *)&acuda_dd9e98a4_api_graphics_sub_resource_get_mapped_array;
    case 0xb9: return (void *)&acuda_dd9e98a4_api_create_texture_object;
    case 0xba: return (void *)&acuda_dd9e98a4_api_destroy_texture_object;
    case 0xbb: return (void *)&acuda_dd9e98a4_api_get_texture_object_resource_desc;
    case 0xbc: return (void *)&acuda_dd9e98a4_api_get_texture_object_texture_desc;
    case 0xbd: return (void *)&acuda_dd9e98a4_api_create_surface_object;
    case 0xbe: return (void *)&acuda_dd9e98a4_api_destroy_surface_object;
    case 0xbf: return (void *)&acuda_dd9e98a4_api_get_surface_object_resource_desc;
    case 0xc4: return (void *)&acuda_dd9e98a4_api_graphics_resource_get_mapped_mipmapped_array;
    case 0xc7: return (void *)&acuda_dd9e98a4_api_get_texture_object_resource_view_desc;
    case 0x201: return (void *)&acuda_dd9e98a4_api_logs_register_callback;
    case 0x202: return (void *)&acuda_dd9e98a4_api_logs_unregister_callback;
    case 0x203: return (void *)&acuda_dd9e98a4_api_logs_current;
    /* 0x03-0x31: Device selection, errors, memory management */
    case 0x03: return (void *)&acuda_dd9e98a4_api_device_get_count;
    case 0x05: return (void *)&acuda_dd9e98a4_api_choose_device;
    case 0x6:  return (void *)&acuda_dd9e98a4_api_get_channel_desc;
    case 0x0a: return (void *)&acuda_dd9e98a4_api_get_last_error;
    case 0x0b: return (void *)&acuda_dd9e98a4_api_peek_at_last_error;
    case 0x0c: return (void *)&acuda_dd9e98a4_api_get_error_string_runtime;
    case 0x10: return (void *)&acuda_dd9e98a4_api_set_device;
    case 0x11: return (void *)&acuda_dd9e98a4_api_get_device;
    case 0x12: return (void *)&acuda_dd9e98a4_api_set_valid_devices;
    case 0x13: return (void *)&acuda_dd9e98a4_api_set_device_flags;
    case 0x14: return (void *)&acuda_dd9e98a4_api_malloc;
    case 0x16: return (void *)&acuda_dd9e98a4_api_free;
    case 0x19: return (void *)&acuda_dd9e98a4_api_malloc_host;
    case 0x1a: return (void *)&acuda_dd9e98a4_api_free_host;
    case 0x1b: return (void *)&acuda_dd9e98a4_api_host_alloc;
    case 0x1f: return (void *)&acuda_dd9e98a4_api_memcpy;
    case 0x27: return (void *)&acuda_dd9e98a4_api_memcpy_to_symbol;
    case 0x28: return (void *)&acuda_dd9e98a4_api_memcpy_from_symbol;
    case 0x29: return (void *)&acuda_dd9e98a4_api_memcpy_async;
    case 0x31: return (void *)&acuda_dd9e98a4_api_memset;

    /* 0x81–0x8A: Stream and event creation/management */
    case 0x81: return (void *)&acuda_dd9e98a4_api_stream_create;
    case 0x83: return (void *)&acuda_dd9e98a4_api_stream_synchronize;
    case 0x84: return (void *)&acuda_dd9e98a4_api_stream_query;
    case 0x85: return (void *)&acuda_dd9e98a4_api_event_create;
    case 0x86: return (void *)&acuda_dd9e98a4_api_event_create_with_flags;
    case 0x87: return (void *)&acuda_dd9e98a4_api_event_record;
    case 0x88: return (void *)&acuda_dd9e98a4_api_event_destroy;
    case 0x89: return (void *)&acuda_dd9e98a4_api_event_synchronize;
    case 0x8a: return (void *)&acuda_dd9e98a4_api_event_query;

    /* 0xA4–0xAE: Device control, configuration, PCI information */
    case 0xa4: return (void *)&acuda_dd9e98a4_api_device_reset;
    case 0xa5: return (void *)&acuda_dd9e98a4_api_device_synchronize;
    case 0xa6: return (void *)&acuda_dd9e98a4_api_device_get_limit;
    case 0xa7: return (void *)&acuda_dd9e98a4_api_device_set_limit;
    case 0xa8: return (void *)&acuda_dd9e98a4_api_device_get_cache_config;
    case 0xa9: return (void *)&acuda_dd9e98a4_api_device_set_cache_config;
    case 0xab: return (void *)&acuda_dd9e98a4_api_profiler_noop;
    case 0xac: return (void *)&acuda_dd9e98a4_api_profiler_noop;
    case 0xad: return (void *)&acuda_dd9e98a4_api_device_get_by_pci_bus_id;
    case 0xae: return (void *)&acuda_dd9e98a4_api_device_get_pci_bus_id;

    /* 0xB0–0xB8: IPC and shared-memory configuration */
    case 0xb0: return (void *)&acuda_dd9e98a4_api_ipc_get_event_handle;
    case 0xb1: return (void *)&acuda_dd9e98a4_api_ipc_open_event_handle;
    case 0xb2: return (void *)&acuda_dd9e98a4_api_ipc_get_mem_handle;
    case 0xb3: return (void *)&acuda_dd9e98a4_api_ipc_open_mem_handle;
    case 0xb4: return (void *)&acuda_dd9e98a4_api_ipc_close_mem_handle;
    case 0xb7: return (void *)&acuda_dd9e98a4_api_device_get_shared_mem_config;
    case 0xb8: return (void *)&acuda_dd9e98a4_api_device_set_shared_mem_config;

    /* 0xC6–0xD4: Additional stream/device APIs and error handling */
    case 0xc6: return (void *)&acuda_dd9e98a4_api_stream_create_with_flags;
    case 0xc8: return (void *)&acuda_dd9e98a4_api_device_get_attribute;
    case 0xc9: return (void *)&acuda_dd9e98a4_api_stream_destroy;
    case 0xca: return (void *)&acuda_dd9e98a4_api_stream_create_with_priority;
    case 0xcb: return (void *)&acuda_dd9e98a4_api_stream_get_priority;
    case 0xcc: return (void *)&acuda_dd9e98a4_api_stream_get_flags;
    case 0xcd: return (void *)&acuda_dd9e98a4_api_device_get_stream_priority_range;
    case 0xd1: return (void *)&acuda_dd9e98a4_api_get_error_string;
    case 0xd4: return (void *)&acuda_dd9e98a4_api_get_device_flags;

    /* Stream wait/callback (dd9e98a4 runtime interface, CUDA 13.x). */
    case 0x93: return (void *)&acuda_dd9e98a4_api_stream_wait_event;
    case 0xc5: return (void *)&acuda_dd9e98a4_api_stream_add_callback;

    /* 0xED-0xF2: Per-thread stream/event APIs */
    case 0xed: return (void *)&acuda_dd9e98a4_api_stream_get_priority_ptsz;
    case 0xee: return (void *)&acuda_dd9e98a4_api_stream_get_flags_ptsz;
    case 0xef: return (void *)&acuda_dd9e98a4_api_stream_synchronize_ptsz;
    case 0xf0: return (void *)&acuda_dd9e98a4_api_stream_query_ptsz;
    case 0xf2: return (void *)&acuda_dd9e98a4_api_event_record_ptsz;

    /* 0xFF: Device P2P attribute */
    case 0xff: return (void *)&acuda_dd9e98a4_api_device_get_p2p_attribute;

    /* 0x112-0x116: External memory and semaphore APIs */
    case 0x112: return (void *)&acuda_dd9e98a4_api_import_external_memory;
    case 0x113: return (void *)&acuda_dd9e98a4_api_external_memory_get_mapped_buffer;
    case 0x114: return (void *)&acuda_dd9e98a4_api_external_memory_get_mapped_mipmapped_array;
    case 0x115: return (void *)&acuda_dd9e98a4_api_destroy_external_memory;
    case 0x116: return (void *)&acuda_dd9e98a4_api_import_external_semaphore;

    /* 0x11c, 0x124, 0x1dc, 0x1fa, 0x226: Host function launch (cudaLaunchHostFunc).
     * CUDA 13.5 statically linked cudart (simpleCallback) resolves 0x11c for
     * the cached cudaLaunchHostFunc entry used by the runtime wrapper. */
    case 0x11c:
    case 0x124:
    case 0x1dc:
    case 0x1fa:
    case 0x226:
        return (void *)&acuda_dd9e98a4_api_launch_host_func;

    /* 0x13C–0x147: Stream capture */
    case 0x13c: return (void *)&acuda_dd9e98a4_api_stream_begin_capture_ptsz;
    case 0x13e: return (void *)&acuda_dd9e98a4_api_stream_is_capturing_ptsz;
    case 0x140: return (void *)&acuda_dd9e98a4_api_stream_end_capture_ptsz;
    case 0x147: return (void *)&acuda_dd9e98a4_api_thread_exchange_stream_capture_mode;

    /* 0x151–0x174: L2 cache, stream attributes, memory pools */
    case 0x151: return (void *)&acuda_dd9e98a4_api_ctx_reset_persisting_l2_cache;
    case 0x156: return (void *)&acuda_dd9e98a4_api_stream_copy_attributes_ptsz;
    case 0x15b: return (void *)&acuda_dd9e98a4_api_device_get_texture1d_linear_max_width;
    case 0x174: return (void *)&acuda_dd9e98a4_api_device_get_default_mempool;

    /* 0x181–0x195: Memory-pool and GPUDirect RDMA APIs */
    case 0x181: return (void *)&acuda_dd9e98a4_api_device_set_mempool;
    case 0x182: return (void *)&acuda_dd9e98a4_api_device_get_mempool;
    case 0x195: return (void *)&acuda_dd9e98a4_api_device_flush_gpudirect_rdma_writes;

    /* 0x1B8–0x1E6: Device properties, streams and events */
    case 0x1b8: return (void *)&acuda_dd9e98a4_api_get_device_properties;
    case 0x1ba: return (void *)&acuda_dd9e98a4_api_stream_get_id_ptsz;
    case 0x1bc: return (void *)&acuda_dd9e98a4_api_init_device;
    case 0x1d1: return (void *)&acuda_dd9e98a4_api_device_register_async_notification;
    case 0x1d2: return (void *)&acuda_dd9e98a4_api_device_unregister_async_notification;
    case 0x1e0: return (void *)&acuda_dd9e98a4_api_stream_get_device;
    case 0x1e1: return (void *)&acuda_dd9e98a4_api_stream_get_device_ptsz;
    case 0x1e6: return (void *)&acuda_dd9e98a4_api_event_elapsed_time;

    /* 0x1EE–0x1F9: Runtime registration and kernel launching */
    case 0x1ee: return (void *)&acuda_dd9e98a4_api_register_fat_binary;
    case 0x1ef: return (void *)&acuda_dd9e98a4_api_register_fat_binary_end;
    case 0x1f1: return (void *)&acuda_dd9e98a4_api_register_var;
    case 0x1f3: return (void *)&acuda_dd9e98a4_api_register_function;
    case 0x1f6: return (void *)&acuda_dd9e98a4_api_push_config;
    case 0x1f7: return (void *)&acuda_dd9e98a4_api_pop_config;
    case 0x1f8: return (void *)&acuda_dd9e98a4_api_get_kernel;
    case 0x1f9: return (void *)&acuda_dd9e98a4_api_launch_kernel;

    /* 0x209–0x20A: Host atomic and peer-to-peer capabilities */
    case 0x209: return (void *)&acuda_dd9e98a4_api_device_get_host_atomic_capabilities;
    case 0x20a: return (void *)&acuda_dd9e98a4_api_device_get_p2p_atomic_capabilities;

    default:
        return acuda_dd9e98a4_get_generic_stub(id);
    }
}


static void *ACUDA_CUDAAPI acuda_dd9e98a4_dispatcher(
        void *request, void *arg_b, void *arg_c, void *arg_d, void *arg_e,
        void *arg_f, void *arg_g, void *arg_h) {
    static int n = 0;
    static FILE *trace_fp = NULL;
    static int trace_fp_init = 0;
#if defined(_WIN32)
    static SRWLOCK trace_lock = SRWLOCK_INIT;
#endif
    unsigned int id = 0xFFFFFFFFu, sub = 0;
    int cur_n;
    if (request) {
        const unsigned char *p = (const unsigned char *)request;
        id = (unsigned int)p[8] | ((unsigned int)p[9] << 8) |
             ((unsigned int)p[10] << 16) | ((unsigned int)p[11] << 24);
        sub = (unsigned int)p[12] | ((unsigned int)p[13] << 8) |
              ((unsigned int)p[14] << 16) | ((unsigned int)p[15] << 24);
    }
#if defined(_WIN32)
    AcquireSRWLockExclusive(&trace_lock);
#endif
    cur_n = n;
    if (cur_n < 80) {
        ACUDA_DEBUG_PRINT(
                "ACuda-debug: dd9e98a4_dispatch[%d] id=0x%x sub=%u a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n",
                cur_n, id, sub, request, arg_b, arg_c, arg_d, arg_e, arg_f, arg_g,
                arg_h);
    } else if (cur_n == 80) {
        ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4_dispatch ... (%d calls so far)\n", cur_n);
    }
    if (!trace_fp_init) {
        trace_fp_init = 1;
        const char *tp = getenv("ACUDA_DD9E98A4_TRACE_FILE");
        if (tp && tp[0])
            trace_fp = fopen(tp, "w");
    }
    if (trace_fp) {
        fprintf(trace_fp, "%d 0x%x %u\n", cur_n, id, sub);
        if ((cur_n % 64) == 0)
            fflush(trace_fp);
    }
    /* Catch corruption between dispatches (canonical ready state is g_init). */
    if (!g_init && g_dd9e98a4_was_ready) {
        ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4_dispatch[%d] id=0x%x saw not-ready (was=%d g_init=%d g_actx=%p)\n",
                          cur_n, id, g_dd9e98a4_was_ready, g_init, g_actx);
        if (trace_fp) {
            fprintf(trace_fp, "!CORRUPT %d 0x%x was=%d g_init=%d\n",
                    cur_n, id, g_dd9e98a4_was_ready, g_init);
            fflush(trace_fp);
        }
    }
    n = cur_n + 1;
#if defined(_WIN32)
    ReleaseSRWLockExclusive(&trace_lock);
#endif
    (void)arg_f;
    return acuda_dd9e98a4_get_id(id);
}
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_slot01(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 slot01 a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n", a,b,c,d,e,f,g,h);
    acuda_dd9e98a4_dump_mem("slot01[a]", a, 32);
    acuda_dd9e98a4_dump_mem("slot01[b]", b, 32);
    acuda_dd9e98a4_dump_mem("slot01[c]", c, 32);
    /* NVIDIA probe: slot01 sets *b to dispatcher pointer (dispatcher returns fn ptr in RAX) */
    if (b) *(void**)b = (void *)&acuda_dd9e98a4_dispatcher;
    return CUDA_SUCCESS;
}
#define ACUDA_DD9E98A4_STUB(n) \
    static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_slot##n(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) { \
        ACUDA_DEBUG_PRINT( "ACuda-debug: acuda_infoocial_cudart_interface_dd9e98a4 slot " #n " called a=%p b=%p c=%p d=%p\n", a,b,c,d); \
        return CUDA_SUCCESS; \
    }
ACUDA_DD9E98A4_STUB(02)
ACUDA_DD9E98A4_STUB(10)
ACUDA_DD9E98A4_STUB(11)
ACUDA_DD9E98A4_STUB(19)
ACUDA_DD9E98A4_STUB(20)
ACUDA_DD9E98A4_STUB(21)
ACUDA_DD9E98A4_STUB(22)
ACUDA_DD9E98A4_STUB(31)
#undef ACUDA_DD9E98A4_STUB


int acuda_export_table_is_dd9e98a4(const struct CUuuid_st *id) {
    const unsigned char *b;
    if (!id)
        return 0;
    b = (const unsigned char *)id->bytes;
    return b[0] == 0xdd && b[1] == 0x9e && b[2] == 0x98 && b[3] == 0xa4 &&
           b[4] == 0xf4 && b[5] == 0x10 && b[6] == 0x48 && b[7] == 0x44 &&
           b[8] == 0x99 && b[9] == 0xb0 && b[10] == 0xe2 && b[11] == 0xae &&
           b[12] == 0xb7 && b[13] == 0xb4 && b[14] == 0x32 && b[15] == 0xb5;
}

static const void *g_dd9e98a4_export_tbl[64];
#if defined(_WIN32)
static INIT_ONCE g_dd9e98a4_export_tbl_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK acuda_dd9e98a4_init_export_tbl_once(
        PINIT_ONCE InitOnce, PVOID Parameter, PVOID *Context) {
    (void)InitOnce; (void)Parameter; (void)Context;
#else
static int g_dd9e98a4_export_tbl_inited = 0;
static void acuda_dd9e98a4_init_export_tbl(void) {
    if (g_dd9e98a4_export_tbl_inited)
        return;
#endif
    /* Layout mirrored from NVIDIA nvcuda.dll probe (first 32 entries) */
    g_dd9e98a4_export_tbl[0]  = (const void *)(uintptr_t)0x18;
    g_dd9e98a4_export_tbl[1]  = (const void *)&acuda_dd9e98a4_slot01;
    g_dd9e98a4_export_tbl[2]  = (const void *)&acuda_dd9e98a4_slot02;
    g_dd9e98a4_export_tbl[3]  = (const void *)0;
    g_dd9e98a4_export_tbl[4]  = (const void *)(uintptr_t)0x1;
    g_dd9e98a4_export_tbl[5]  = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
    g_dd9e98a4_export_tbl[6]  = (const void *)(uintptr_t)0x781ef6113590168cULL;
    g_dd9e98a4_export_tbl[7]  = (const void *)(uintptr_t)0xffffffffffffffffULL;
    g_dd9e98a4_export_tbl[8]  = (const void *)(uintptr_t)0xffffffffULL;
    g_dd9e98a4_export_tbl[9]  = (const void *)(uintptr_t)0x18;
    g_dd9e98a4_export_tbl[10] = (const void *)&acuda_dd9e98a4_slot10;
    g_dd9e98a4_export_tbl[11] = (const void *)&acuda_dd9e98a4_slot11;
    g_dd9e98a4_export_tbl[12] = (const void *)0;
    g_dd9e98a4_export_tbl[13] = (const void *)(uintptr_t)0x1;
    g_dd9e98a4_export_tbl[14] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
    g_dd9e98a4_export_tbl[15] = (const void *)(uintptr_t)0x781ef6113590168cULL;
    g_dd9e98a4_export_tbl[16] = (const void *)(uintptr_t)0xffffffffffffffffULL;
    g_dd9e98a4_export_tbl[17] = (const void *)(uintptr_t)0xffffffffULL;
    g_dd9e98a4_export_tbl[18] = (const void *)(uintptr_t)0x28;
    g_dd9e98a4_export_tbl[19] = (const void *)&acuda_dd9e98a4_slot19;
    g_dd9e98a4_export_tbl[20] = (const void *)&acuda_dd9e98a4_slot20;
    g_dd9e98a4_export_tbl[21] = (const void *)&acuda_dd9e98a4_slot21;
    g_dd9e98a4_export_tbl[22] = (const void *)&acuda_dd9e98a4_slot22;
    g_dd9e98a4_export_tbl[23] = (const void *)0;
    g_dd9e98a4_export_tbl[24] = (const void *)(uintptr_t)0x1;
    g_dd9e98a4_export_tbl[25] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
    g_dd9e98a4_export_tbl[26] = (const void *)(uintptr_t)0x781ef6113590168cULL;
    g_dd9e98a4_export_tbl[27] = (const void *)(uintptr_t)0xffffffffffffffffULL;
    g_dd9e98a4_export_tbl[28] = (const void *)(uintptr_t)0xffffffffffffffffULL;
    g_dd9e98a4_export_tbl[29] = (const void *)(uintptr_t)0xffffffffULL;
    g_dd9e98a4_export_tbl[30] = (const void *)(uintptr_t)0x68;
    g_dd9e98a4_export_tbl[31] = (const void *)&acuda_dd9e98a4_slot31;
    for (int i = 32; i < 64; i++)
        g_dd9e98a4_export_tbl[i] = (const void *)0;
#if defined(_WIN32)
    return TRUE;
#else
    g_dd9e98a4_export_tbl_inited = 1;
#endif
}

CUresult acuda_export_table_get_dd9e98a4(const void **ppExportTable) {
    if (getenv("ACUDA_DD9E98A4_FALLBACK")) {
        ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> forcing NOT_SUPPORTED (fallback experiment)\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
#if defined(_WIN32)
    InitOnceExecuteOnce(&g_dd9e98a4_export_tbl_once,
                        acuda_dd9e98a4_init_export_tbl_once, NULL, NULL);
#else
    acuda_dd9e98a4_init_export_tbl();
#endif
    if (ppExportTable)
        *ppExportTable = (const void *)g_dd9e98a4_export_tbl;
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (acuda_infoocial_cudart_interface_dd9e98a4 traced table) table=%p\n",
                      ppExportTable ? *ppExportTable : NULL);
    return CUDA_SUCCESS;
}
