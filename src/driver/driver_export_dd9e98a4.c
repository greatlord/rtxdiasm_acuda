#include "acuda/driver/driver_export_dd9e98a4.h"
#include "acuda/driver/driver_export_dd9e98a4_types.h"
#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_stub.h"
#include "acuda/runtime_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- dd9e98a4 interface functions mapped to CUDA runtime API entry points ---
 * The CUDA 13.x hybrid runtime resolves these IDs through the dispatcher and
 * tail-calls them from its cuda* wrapper functions (see docs/unknown notes).
 * id 0x3 is cudaGetDeviceCount(). */
extern CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count);
extern CUresult ACUDA_CUDAAPI cuInit(unsigned int flags);

/* The CUDA 13.x hybrid runtime never calls cuInit() or cuCtxCreate(): it
 * expects the driver interface to be usable right away, so bring up the
 * primary context before serving interface requests. */
static int acuda_dd9e98a4_driver_ready(void) {
    static struct {
        unsigned int canary1;
        int state; /* 0 unknown, 1 ok, -1 failed */
        unsigned int canary2;
    } s = { 0xA5C0FFEEu, 0, 0x5EEDFACEu };
    if (s.canary1 != 0xA5C0FFEEu || s.canary2 != 0x5EEDFACEu ||
        (s.state != 0 && s.state != 1 && s.state != -1)) {
        ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4_driver_ready CORRUPTION "
                          "c1=%08x state=%d c2=%08x\n",
                          s.canary1, s.state, s.canary2);
        if (getenv("ACUDA_TRACE")) fflush(stderr);
    }
    if (s.state == 0) {
        CUresult r = cuInit(0);
        if (r == CUDA_SUCCESS) {
            CUcontext ctx = NULL;
            r = cuDevicePrimaryCtxRetain(&ctx, 0);
            if (r == CUDA_SUCCESS)
                r = cuCtxSetCurrent(ctx);
        }
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 implicit init -> %d\n", (int)r);
        if (getenv("ACUDA_TRACE")) fflush(stderr);
        s.state = (r == CUDA_SUCCESS) ? 1 : -1;
    }
    return s.state == 1;
}

static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_count(
        void *count, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    r = cuDeviceGetCount((int *)count);
    ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4 id=0x3 cudaGetDeviceCount -> %d (count=%d)\n",
                      (int)r, count ? *(int *)count : -1);
    return r;
}

/* id 0xc8: cudaDeviceGetAttribute(int *value, int attr, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_attribute(
        void *value, void *attr, void *device, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuDeviceGetAttribute((int *)value, (int)(intptr_t)attr,
                                (CUdevice)(intptr_t)device);
}

/* id 0x14: cudaMalloc(void **devPtr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc(
        void *devPtr, void *size, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuMemAlloc((CUdeviceptr *)devPtr, (size_t)size);
}

/* id 0x19: cudaMallocHost(void **ptr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc_host(
        void *ptr, void *size, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!ptr)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)ptr = NULL;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    *(void **)ptr = malloc((size_t)size ? (size_t)size : 1);
    return *(void **)ptr ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

/* id 0x10: cudaSetDevice(int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device(
        void *device, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUcontext ctx = NULL;
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    r = cuDevicePrimaryCtxRetain(&ctx, (CUdevice)(intptr_t)device);
    if (r != CUDA_SUCCESS)
        return r;
    return cuCtxSetCurrent(ctx);
}

/* id 0x11: cudaGetDevice(int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device(
        void *device, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    CUdevice d = 0;
    CUresult r;
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
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
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return acuda_query_device_properties((struct cudaDeviceProp *)prop,
                                         (int)(intptr_t)device);
}
/* id 0xd1: const char *cudaGetErrorString(cudaError_t error) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_error_string(
        void *error, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 memset(%p,%d,%zu)\n",
                      devPtr, (int)(intptr_t)value, (size_t)count);
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
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
    int copy_kind = (int)(intptr_t)kind;
    size_t bytes = (size_t)count;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
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
    int copy_kind = (int)(intptr_t)kind;
    size_t bytes = (size_t)count;
    (void)stream; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if ((!dst || !src) && bytes)
        return CUDA_ERROR_INVALID_VALUE;
    if (!bytes)
        return CUDA_SUCCESS;
    switch (copy_kind) {
    case 0: /* cudaMemcpyHostToHost */
        memmove(dst, src, bytes);
        return CUDA_SUCCESS;
    case 1: /* cudaMemcpyHostToDevice */
        return cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, bytes);
    case 2: /* cudaMemcpyDeviceToHost */
        return cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, bytes);
    case 3: /* cudaMemcpyDeviceToDevice */
        return cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst,
                            (CUdeviceptr)(uintptr_t)src, bytes);
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

/* id 0x16: cudaFree(void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free(
        void *devPtr, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!devPtr)
        return CUDA_SUCCESS;
    return cuMemFree((CUdeviceptr)(uintptr_t)devPtr);
}

/* id 0x1a: cudaFreeHost(void *ptr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free_host(
        void *ptr, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    free(ptr);
    return CUDA_SUCCESS;
}

/* id 0xa4: cudaDeviceReset(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_reset(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuCtxSynchronize();
}

/* id 0xa5: cudaDeviceSynchronize(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_synchronize(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuCtxSynchronize();
}

/* id 0x85: cudaEventCreate(cudaEvent_t *event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventCreate((CUevent *)event, 0);
}

/* id 0x86: cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create_with_flags(
        void *event, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    return cuEventCreate((CUevent *)event, (unsigned int)(uintptr_t)flags);
}

/* id 0x87: cudaEventRecord(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record(
        void *event, void *stream, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventRecord((CUevent)event, (CUstream)stream);
}

/* id 0x89: cudaEventSynchronize(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_synchronize(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    return cuEventSynchronize((CUevent)event);
}

/* id 0x8a: cudaEventQuery(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_query(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventQuery((CUevent)event);
}

/* id 0x1e6: cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_elapsed_time(
        void *ms, void *start, void *end, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventElapsedTime((float *)ms, (CUevent)start, (CUevent)end);
}

/* id 0x88: cudaEventDestroy(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_destroy(
        void *event, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventDestroy((CUevent)event);
}

/* id 0xab / 0xac: cudaProfilerStart() / cudaProfilerStop() */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_profiler_noop(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
static unsigned g_dd9e98a4_nfatbins;
static acuda_dd9e98a4_kernel g_dd9e98a4_kernels[ACUDA_DD9E98A4_MAX_FUNCS];
static unsigned g_dd9e98a4_nkernels;

/* id 0x1ee: __cudaRegisterFatBinary(void *fatCubin) -> handle */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary(
        void *fatCubin, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)fatCubinHandle; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x1f3: __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
 *          deviceName, thread_limit, tid, bid, bDim, gDim, wSize) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_function(
        void *fatCubinHandle, void *hostFun, void *deviceFun, void *deviceName,
        void *thread_limit, void *tid, void *bid, void *bDim) {
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

/* id 0x1f8: __cudaGetKernel(void **kernel, const void *func) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_kernel(
        void *kernel, void *func, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 get kernel slot=%p func=%p\n",
                      kernel, func);
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!kernel || !func)
        return CUDA_ERROR_INVALID_VALUE;
    *(const void **)kernel = func;
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
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 push config\n");
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!gridDim || !blockDim || g_dd9e98a4_config_depth >= 16)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[g_dd9e98a4_config_depth++];
    cfg->grid = *(const acuda_dd9e98a4_dim3 *)gridDim;
    cfg->block = *(const acuda_dd9e98a4_dim3 *)blockDim;
    cfg->shared = (size_t)sharedMem;
    cfg->stream = stream;
    return CUDA_SUCCESS;
}

/* id 0x1f7: __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
 *                                      size_t *sharedMem, cudaStream_t *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_pop_config(
        void *gridDim, void *blockDim, void *sharedMem, void *stream,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 pop config\n");
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (g_dd9e98a4_config_depth == 0)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[--g_dd9e98a4_config_depth];
    if (gridDim)
        *(acuda_dd9e98a4_dim3 *)gridDim = cfg->grid;
    if (blockDim)
        *(acuda_dd9e98a4_dim3 *)blockDim = cfg->block;
    if (sharedMem)
        *(size_t *)sharedMem = cfg->shared;
    if (stream)
        *(void **)stream = cfg->stream;
    return CUDA_SUCCESS;
}

/* id 0x1f9: cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
 *                            void **args, size_t sharedMem, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_launch_kernel(
        void *func, void *gridDim, void *blockDim, void *args, void *sharedMem,
        void *stream, void *unused_g, void *unused_h) {
    const acuda_dd9e98a4_dim3 *grid = (const acuda_dd9e98a4_dim3 *)gridDim;
    const acuda_dd9e98a4_dim3 *block = (const acuda_dd9e98a4_dim3 *)blockDim;
    acuda_dd9e98a4_kernel *k = NULL;
    acuda_dd9e98a4_fatbin *fb;
    CUfunction fn = NULL;
    CUresult r;
    unsigned i;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!func || !grid || !block)
        return CUDA_ERROR_INVALID_VALUE;
    for (i = 0; i < g_dd9e98a4_nkernels; i++) {
        if (g_dd9e98a4_kernels[i].host_fn == func) {
            k = &g_dd9e98a4_kernels[i];
            break;
        }
    }
    if (!k || !k->fatbin)
        return CUDA_ERROR_NOT_FOUND;
    fb = k->fatbin;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 launch %s grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                      k->device_name, grid->x, grid->y, grid->z, block->x,
                      block->y, block->z);
    if (fb->state == 0) {
        r = cuModuleLoadFatBinary(&fb->module, fb->blob);
        fb->state = (r == CUDA_SUCCESS) ? 1 : -1;
        ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 module load -> %d\n", (int)r);
    }
    if (fb->state != 1)
        return CUDA_ERROR_INVALID_IMAGE;
    r = cuModuleGetFunction(&fn, fb->module, k->device_name);
    if (r != CUDA_SUCCESS)
        return r;
    return cuLaunchKernel(fn, grid->x, grid->y, grid->z, block->x, block->y,
                          block->z, (unsigned int)(size_t)sharedMem,
                          (CUstream)stream, args, NULL);
}


/* id 0x5: cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_choose_device(
        void *device, void *prop, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)prop; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (device)
        *(int *)device = 0;
    return CUDA_SUCCESS;
}

/* id 0xa: cudaGetLastError(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_last_error(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xb: cudaPeekAtLastError(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_peek_at_last_error(
        void *unused_a, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xc: cudaGetErrorString(cudaError_t error) -> const char * */
static const char *ACUDA_CUDAAPI acuda_dd9e98a4_api_get_error_string_runtime(
        void *error, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)device_arr; (void)len; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

/* id 0x13: cudaSetDeviceFlags(unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device_flags(
        void *flags, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)flags; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

/* id 0xd4: cudaGetDeviceFlags(unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_device_flags(
        void *flags, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (flags)
        *(unsigned int *)flags = 0;
    return CUDA_SUCCESS;
}

/* id 0x81: cudaStreamCreate(cudaStream_t *pStream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create(
        void *pStream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, 0);
}

/* id 0xc6: cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create_with_flags(
        void *pStream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, (unsigned int)(uintptr_t)flags);
}

/* id 0xca: cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_create_with_priority(
        void *pStream, void *flags, void *priority, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)priority; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pStream)
        return CUDA_ERROR_INVALID_VALUE;
    *(CUstream *)pStream = NULL;
    return cuStreamCreate((CUstream *)pStream, (unsigned int)(uintptr_t)flags);
}

/* id 0xa6: cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_limit(
        void *pValue, void *limit, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!pValue)
        return CUDA_ERROR_INVALID_VALUE;
    *(size_t *)pValue = 0;
    return cuCtxGetLimit((size_t *)pValue, (int)(intptr_t)limit);
}

/* id 0xa7: cudaDeviceSetLimit(enum cudaLimit limit, size_t value) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_limit(
        void *limit, void *value, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuCtxSetLimit((int)(intptr_t)limit, (size_t)(uintptr_t)value);
}

/* id 0xa8: cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_cache_config(
        void *pCacheConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (pCacheConfig)
        *(int *)pCacheConfig = 0;
    return CUDA_SUCCESS;
}

/* id 0xa9: cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_cache_config(
        void *cacheConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)cacheConfig; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

/* id 0xad: cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_by_pci_bus_id(
        void *device, void *pciBusId, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)pciBusId; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (device)
        *(int *)device = 0;
    return CUDA_SUCCESS;
}

/* id 0xae: cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_pci_bus_id(
        void *pciBusId, void *len, void *device, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    int n = (int)(intptr_t)len;
    (void)device; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (pciBusId && n > 0) {
        const char *id = "0000:00:00.0";
        int i = 0;
        for (; id[i] && i < n - 1; i++)
            ((char *)pciBusId)[i] = id[i];
        ((char *)pciBusId)[i] = '\0';
    }
    return CUDA_SUCCESS;
}

/* id 0xcd: cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_stream_priority_range(
        void *leastPriority, void *greatestPriority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
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
    (void)devPtr; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0xb7: cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_shared_mem_config(
        void *pConfig, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (pConfig)
        *(int *)pConfig = 0;
    return CUDA_SUCCESS;
}

/* id 0xb8: cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_set_shared_mem_config(
        void *config, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)config; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}



/* id 0x83: cudaStreamSynchronize(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_synchronize(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuStreamSynchronize((CUstream)stream);
}

/* id 0x84: cudaStreamQuery(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_query(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuStreamQuery((CUstream)stream);
}

/* id 0xc9: cudaStreamDestroy(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_destroy(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_b; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!stream)
        return CUDA_SUCCESS;
    return cuStreamDestroy_v2((CUstream)stream);
}

/* id 0xcb: cudaStreamGetPriority(cudaStream_t stream, int *priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_priority(
        void *stream, void *priority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!priority)
        return CUDA_ERROR_INVALID_VALUE;
    *(int *)priority = 0;
    return cuStreamGetPriority((CUstream)stream, (int *)priority);
}

/* id 0xcc: cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_flags(
        void *stream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)unused_c; (void)unused_d; (void)unused_e; (void)unused_f;
    (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    *(unsigned int *)flags = 0;
    return cuStreamGetFlags((CUstream)stream, (unsigned int *)flags);
}

/* id 0xed: cudaStreamGetPriority_ptsz(cudaStream_t stream, int *priority) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_priority_ptsz(
        void *stream, void *priority, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_stream_get_priority(stream, priority, unused_c,
                                                  unused_d, unused_e, unused_f,
                                                  unused_g, unused_h);
}

/* id 0xee: cudaStreamGetFlags_ptsz(cudaStream_t stream, unsigned int *flags) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_flags_ptsz(
        void *stream, void *flags, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_stream_get_flags(stream, flags, unused_c,
                                               unused_d, unused_e, unused_f,
                                               unused_g, unused_h);
}

/* id 0xef: cudaStreamSynchronize_ptsz(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_synchronize_ptsz(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_stream_synchronize(stream, unused_b, unused_c,
                                                  unused_d, unused_e, unused_f,
                                                  unused_g, unused_h);
}

/* id 0xf0: cudaStreamQuery_ptsz(cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_query_ptsz(
        void *stream, void *unused_b, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_stream_query(stream, unused_b, unused_c,
                                           unused_d, unused_e, unused_f,
                                           unused_g, unused_h);
}

/* id 0x1e0: cudaStreamGetDevice(cudaStream_t stream, int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_device(
        void *stream, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)stream; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    *(int *)device = 0;
    return cuCtxGetDevice((CUdevice *)device);
}

/* id 0x1e1: cudaStreamGetDevice_ptsz(cudaStream_t stream, int *device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_get_device_ptsz(
        void *stream, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_stream_get_device(stream, device, unused_c,
                                                unused_d, unused_e, unused_f,
                                                unused_g, unused_h);
}



/* id 0xf2: cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record_ptsz(
        void *event, void *stream, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    return acuda_dd9e98a4_api_event_record(event, stream, unused_c, unused_d,
                                           unused_e, unused_f, unused_g, unused_h);
}

/* id 0x13c: cudaStreamBeginCapture_ptsz(cudaStream_t stream, cudaStreamCaptureMode mode) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_begin_capture_ptsz(
        void *stream, void *mode, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)unused_a; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x156: cudaStreamCopyAttributes_ptsz(cudaStream_t dst, cudaStream_t src) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_stream_copy_attributes_ptsz(
        void *dst, void *src, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)dst; (void)src; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x174: cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_default_mempool(
        void *memPool, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)device; (void)memPool; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x182: cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_mempool(
        void *memPool, void *device, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)device; (void)deviceFlags; (void)flags;
    (void)unused_d; (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

/* id 0xff: cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr,
 *                                     int srcDevice, int dstDevice) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_p2p_attribute(
        void *value, void *attr, void *srcDevice, void *dstDevice,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)target; (void)scope; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x209: cudaDeviceGetHostAtomicCapabilities(unsigned int *capabilities,
 *          const enum cudaAtomicOperation *operations, unsigned int count, int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_get_host_atomic_capabilities(
        void *capabilities, void *operations, void *count, void *device,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)extMem; (void)unused_b; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}

/* id 0x116: cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out,
 *           const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_import_external_semaphore(
        void *extSem_out, void *semHandleDesc, void *unused_c, void *unused_d,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
    (void)semHandleDesc; (void)unused_c; (void)unused_d; (void)unused_e;
    (void)unused_f; (void)unused_g; (void)unused_h;
    if (extSem_out)
        *(void **)extSem_out = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* id 0x1d1: cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc,
 *           void* userData, cudaAsyncCallbackHandle_t* callback) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_register_async_notification(
        void *device, void *callbackFunc, void *userData, void *callback,
        void *unused_e, void *unused_f, void *unused_g, void *unused_h) {
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
    (void)device; (void)callback; (void)unused_c; (void)unused_d;
    (void)unused_e; (void)unused_f; (void)unused_g; (void)unused_h;
    return CUDA_SUCCESS;
}


static void *acuda_dd9e98a4_get_id_stub(unsigned int id) {
    switch(id){
    case 0xb1: return (void*)&acuda_dd9e98a4_api_ipc_open_event_handle;
    case 0xb3: return (void*)&acuda_dd9e98a4_api_ipc_open_mem_handle;
    case 0xff: return (void*)&acuda_dd9e98a4_api_device_get_p2p_attribute;
    case 0x112: return (void*)&acuda_dd9e98a4_api_import_external_memory;
    case 0x113: return (void*)&acuda_dd9e98a4_api_external_memory_get_mapped_buffer;
    case 0x114: return (void*)&acuda_dd9e98a4_api_external_memory_get_mapped_mipmapped_array;
    case 0x115: return (void*)&acuda_dd9e98a4_api_destroy_external_memory;
    case 0x116: return (void*)&acuda_dd9e98a4_api_import_external_semaphore;
    case 0x15b: return (void*)&acuda_dd9e98a4_api_device_get_texture1d_linear_max_width;
    case 0x195: return (void*)&acuda_dd9e98a4_api_device_flush_gpudirect_rdma_writes;
    case 0x1bc: return (void*)&acuda_dd9e98a4_api_init_device;
    case 0x1d1: return (void*)&acuda_dd9e98a4_api_device_register_async_notification;
    case 0x1d2: return (void*)&acuda_dd9e98a4_api_device_unregister_async_notification;
    case 0x209: return (void*)&acuda_dd9e98a4_api_device_get_host_atomic_capabilities;
    case 0x20a: return (void*)&acuda_dd9e98a4_api_device_get_p2p_atomic_capabilities;
    case 0xf2: return (void*)&acuda_dd9e98a4_api_event_record_ptsz;
    case 0x13c: return (void*)&acuda_dd9e98a4_api_stream_begin_capture_ptsz;
    case 0x13e: return (void*)&acuda_dd9e98a4_api_stream_is_capturing_ptsz;
    case 0x140: return (void*)&acuda_dd9e98a4_api_stream_end_capture_ptsz;
    case 0x147: return (void*)&acuda_dd9e98a4_api_thread_exchange_stream_capture_mode;
    case 0x151: return (void*)&acuda_dd9e98a4_api_ctx_reset_persisting_l2_cache;
    case 0x156: return (void*)&acuda_dd9e98a4_api_stream_copy_attributes_ptsz;
    case 0x174: return (void*)&acuda_dd9e98a4_api_device_get_default_mempool;
    case 0x181: return (void*)&acuda_dd9e98a4_api_device_set_mempool;
    case 0x182: return (void*)&acuda_dd9e98a4_api_device_get_mempool;
    case 0x1ba: return (void*)&acuda_dd9e98a4_api_stream_get_id_ptsz;
    case 0x83: return (void*)&acuda_dd9e98a4_api_stream_synchronize;
    case 0x84: return (void*)&acuda_dd9e98a4_api_stream_query;
    case 0xc9: return (void*)&acuda_dd9e98a4_api_stream_destroy;
    case 0xcb: return (void*)&acuda_dd9e98a4_api_stream_get_priority;
    case 0xcc: return (void*)&acuda_dd9e98a4_api_stream_get_flags;
    case 0xed: return (void*)&acuda_dd9e98a4_api_stream_get_priority_ptsz;
    case 0xee: return (void*)&acuda_dd9e98a4_api_stream_get_flags_ptsz;
    case 0xef: return (void*)&acuda_dd9e98a4_api_stream_synchronize_ptsz;
    case 0xf0: return (void*)&acuda_dd9e98a4_api_stream_query_ptsz;
    case 0x1e0: return (void*)&acuda_dd9e98a4_api_stream_get_device;
    case 0x1e1: return (void*)&acuda_dd9e98a4_api_stream_get_device_ptsz;
    case 0x5: return (void*)&acuda_dd9e98a4_api_choose_device;
    case 0x1f: return (void*)&acuda_dd9e98a4_api_memcpy;
    case 0xa: return (void*)&acuda_dd9e98a4_api_get_last_error;
    case 0xb: return (void*)&acuda_dd9e98a4_api_peek_at_last_error;
    case 0xc: return (void*)&acuda_dd9e98a4_api_get_error_string_runtime;
    case 0x12: return (void*)&acuda_dd9e98a4_api_set_valid_devices;
    case 0x13: return (void*)&acuda_dd9e98a4_api_set_device_flags;
    case 0x81: return (void*)&acuda_dd9e98a4_api_stream_create;
    case 0xa6: return (void*)&acuda_dd9e98a4_api_device_get_limit;
    case 0xa7: return (void*)&acuda_dd9e98a4_api_device_set_limit;
    case 0xa8: return (void*)&acuda_dd9e98a4_api_device_get_cache_config;
    case 0xa9: return (void*)&acuda_dd9e98a4_api_device_set_cache_config;
    case 0xad: return (void*)&acuda_dd9e98a4_api_device_get_by_pci_bus_id;
    case 0xae: return (void*)&acuda_dd9e98a4_api_device_get_pci_bus_id;
    case 0xb0: return (void*)&acuda_dd9e98a4_api_ipc_get_event_handle;
    case 0xb2: return (void*)&acuda_dd9e98a4_api_ipc_get_mem_handle;
    case 0xb4: return (void*)&acuda_dd9e98a4_api_ipc_close_mem_handle;
    case 0xb7: return (void*)&acuda_dd9e98a4_api_device_get_shared_mem_config;
    case 0xb8: return (void*)&acuda_dd9e98a4_api_device_set_shared_mem_config;
    case 0xc6: return (void*)&acuda_dd9e98a4_api_stream_create_with_flags;
    case 0xca: return (void*)&acuda_dd9e98a4_api_stream_create_with_priority;
    case 0xcd: return (void*)&acuda_dd9e98a4_api_device_get_stream_priority_range;
    case 0xd4: return (void*)&acuda_dd9e98a4_api_get_device_flags;
    case 0x10: return (void*)&acuda_dd9e98a4_api_set_device;
    case 0x11: return (void*)&acuda_dd9e98a4_api_get_device;
    case 0x14: return (void*)&acuda_dd9e98a4_api_malloc;
    case 0x19: return (void*)&acuda_dd9e98a4_api_malloc_host;
    case 0x1b8: return (void*)&acuda_dd9e98a4_api_get_device_properties;
    case 0xc8: return (void*)&acuda_dd9e98a4_api_device_get_attribute;
    case 0xd1: return (void*)&acuda_dd9e98a4_api_get_error_string;
    case 0x3: return (void*)&acuda_dd9e98a4_api_device_get_count;
    case 0x16: return (void*)&acuda_dd9e98a4_api_free;
    case 0x1a: return (void*)&acuda_dd9e98a4_api_free_host;
    case 0x29: return (void*)&acuda_dd9e98a4_api_memcpy_async;
    case 0x31: return (void*)&acuda_dd9e98a4_api_memset;
    case 0x85: return (void*)&acuda_dd9e98a4_api_event_create;
    case 0x86: return (void*)&acuda_dd9e98a4_api_event_create_with_flags;
    case 0x87: return (void*)&acuda_dd9e98a4_api_event_record;
    case 0x88: return (void*)&acuda_dd9e98a4_api_event_destroy;
    case 0x89: return (void*)&acuda_dd9e98a4_api_event_synchronize;
    case 0x8a: return (void*)&acuda_dd9e98a4_api_event_query;
    case 0xa4: return (void*)&acuda_dd9e98a4_api_device_reset;
    case 0xa5: return (void*)&acuda_dd9e98a4_api_device_synchronize;
    case 0xab: return (void*)&acuda_dd9e98a4_api_profiler_noop;
    case 0xac: return (void*)&acuda_dd9e98a4_api_profiler_noop;
    case 0x1e6: return (void*)&acuda_dd9e98a4_api_event_elapsed_time;
    case 0x1ee: return (void*)&acuda_dd9e98a4_api_register_fat_binary;
    case 0x1ef: return (void*)&acuda_dd9e98a4_api_register_fat_binary_end;
    case 0x1f3: return (void*)&acuda_dd9e98a4_api_register_function;
    case 0x1f6: return (void*)&acuda_dd9e98a4_api_push_config;
    case 0x1f7: return (void*)&acuda_dd9e98a4_api_pop_config;
    case 0x1f8: return (void*)&acuda_dd9e98a4_api_get_kernel;
    case 0x1f9: return (void*)&acuda_dd9e98a4_api_launch_kernel;
    default: return acuda_dd9e98a4_get_generic_stub(id);
    }
}

static void *ACUDA_CUDAAPI acuda_dd9e98a4_dispatcher(
        void *request, void *arg_b, void *arg_c, void *arg_d, void *arg_e,
        void *arg_f, void *arg_g, void *arg_h) {
    static int n = 0;
    unsigned int id = 0xFFFFFFFFu, sub = 0;
    if (request) {
        const unsigned char *p = (const unsigned char *)request;
        id = (unsigned int)p[8] | ((unsigned int)p[9] << 8) |
             ((unsigned int)p[10] << 16) | ((unsigned int)p[11] << 24);
        sub = (unsigned int)p[12] | ((unsigned int)p[13] << 8) |
              ((unsigned int)p[14] << 16) | ((unsigned int)p[15] << 24);
    }
    if (n < 80) {
        ACUDA_DEBUG_PRINT(
                "ACuda-debug: dd9e98a4_dispatch[%d] id=0x%x sub=%u a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n",
                n, id, sub, request, arg_b, arg_c, arg_d, arg_e, arg_f, arg_g,
                arg_h);
    } else if (n == 80) {
        ACUDA_DEBUG_PRINT("ACuda-debug: dd9e98a4_dispatch ... (%d calls so far)\n", n);
    }
    n++;
    (void)arg_f;
    return acuda_dd9e98a4_get_id_stub(id);
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

CUresult acuda_export_table_get_dd9e98a4(const void **ppExportTable) {
    static const void *tbl[64];
    static int inited = 0;
    if (getenv("ACUDA_DD9E98A4_FALLBACK")) {
        ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> forcing NOT_SUPPORTED (fallback experiment)\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!inited) {
        /* Layout mirrored from NVIDIA nvcuda.dll probe (first 32 entries) */
        tbl[0]  = (const void *)(uintptr_t)0x18;
        tbl[1]  = (const void *)&acuda_dd9e98a4_slot01;
        tbl[2]  = (const void *)&acuda_dd9e98a4_slot02;
        tbl[3]  = (const void *)0;
        tbl[4]  = (const void *)(uintptr_t)0x1;
        tbl[5]  = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
        tbl[6]  = (const void *)(uintptr_t)0x781ef6113590168cULL;
        tbl[7]  = (const void *)(uintptr_t)0xffffffffffffffffULL;
        tbl[8]  = (const void *)(uintptr_t)0xffffffffULL;
        tbl[9]  = (const void *)(uintptr_t)0x18;
        tbl[10] = (const void *)&acuda_dd9e98a4_slot10;
        tbl[11] = (const void *)&acuda_dd9e98a4_slot11;
        tbl[12] = (const void *)0;
        tbl[13] = (const void *)(uintptr_t)0x1;
        tbl[14] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
        tbl[15] = (const void *)(uintptr_t)0x781ef6113590168cULL;
        tbl[16] = (const void *)(uintptr_t)0xffffffffffffffffULL;
        tbl[17] = (const void *)(uintptr_t)0xffffffffULL;
        tbl[18] = (const void *)(uintptr_t)0x28;
        tbl[19] = (const void *)&acuda_dd9e98a4_slot19;
        tbl[20] = (const void *)&acuda_dd9e98a4_slot20;
        tbl[21] = (const void *)&acuda_dd9e98a4_slot21;
        tbl[22] = (const void *)&acuda_dd9e98a4_slot22;
        tbl[23] = (const void *)0;
        tbl[24] = (const void *)(uintptr_t)0x1;
        tbl[25] = (const void *)(uintptr_t)0xe943fb0113fafa73ULL;
        tbl[26] = (const void *)(uintptr_t)0x781ef6113590168cULL;
        tbl[27] = (const void *)(uintptr_t)0xffffffffffffffffULL;
        tbl[28] = (const void *)(uintptr_t)0xffffffffffffffffULL;
        tbl[29] = (const void *)(uintptr_t)0xffffffffULL;
        tbl[30] = (const void *)(uintptr_t)0x68;
        tbl[31] = (const void *)&acuda_dd9e98a4_slot31;
        for (int i = 32; i < 64; i++)
            tbl[i] = (const void *)0;
        inited = 1;
    }
    if (ppExportTable)
        *ppExportTable = (const void *)tbl;
    ACUDA_DEBUG_PRINT("ACuda-debug: cuGetExportTable -> SUCCESS (acuda_infoocial_cudart_interface_dd9e98a4 traced table) table=%p\n",
                      ppExportTable ? *ppExportTable : NULL);
    return CUDA_SUCCESS;
}
