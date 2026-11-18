#include "acuda/driver/driver_export_dd9e98a4.h"
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
    static int state = 0; /* 0 unknown, 1 ok, -1 failed */
    if (state == 0) {
        CUresult r = cuInit(0);
        if (r == CUDA_SUCCESS) {
            CUcontext ctx = NULL;
            r = cuDevicePrimaryCtxRetain(&ctx, 0);
            if (r == CUDA_SUCCESS)
                r = cuCtxSetCurrent(ctx);
        }
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4 implicit init -> %d\n", (int)r);
        state = (r == CUDA_SUCCESS) ? 1 : -1;
    }
    return state == 1;
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
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuMemAlloc((CUdeviceptr *)a, (size_t)b);
}

/* id 0x19: cudaMallocHost(void **ptr, size_t size) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_malloc_host(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    size_t size = (size_t)b;
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    *(void **)a = NULL;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    *(void **)a = malloc(size ? size : 1);
    return *(void **)a ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

/* id 0x10: cudaSetDevice(int device) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_set_device(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    CUcontext ctx = NULL;
    CUresult r;
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    r = cuDevicePrimaryCtxRetain(&ctx, (CUdevice)(intptr_t)a);
    if (r != CUDA_SUCCESS)
        return r;
    return cuCtxSetCurrent(ctx);
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
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memset(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 memset(%p,%d,%zu)\n", a, (int)(intptr_t)b, (size_t)c);
    (void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    if (!c)
        return CUDA_SUCCESS;
    return cuMemsetD8((CUdeviceptr)(uintptr_t)a, (unsigned char)(uintptr_t)b,
                      (size_t)c);
}

/* id 0x29: cudaMemcpyAsync(void *dst, const void *src, size_t count,
 *                          cudaMemcpyKind kind, cudaStream_t stream)
 * ACuda's device memory is directly addressable host memory, so the async
 * copies are executed synchronously, exactly like ACuda's cudart does. */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_memcpy_async(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    int kind = (int)(intptr_t)d;
    size_t count = (size_t)c;
    (void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if ((!a || !b) && count)
        return CUDA_ERROR_INVALID_VALUE;
    if (!count)
        return CUDA_SUCCESS;
    switch (kind) {
    case 0: /* cudaMemcpyHostToHost */
        memmove(a, b, count);
        return CUDA_SUCCESS;
    case 1: /* cudaMemcpyHostToDevice */
        return cuMemcpyHtoD((CUdeviceptr)(uintptr_t)a, b, count);
    case 2: /* cudaMemcpyDeviceToHost */
        return cuMemcpyDtoH(a, (CUdeviceptr)(uintptr_t)b, count);
    case 3: /* cudaMemcpyDeviceToDevice */
        return cuMemcpyDtoD((CUdeviceptr)(uintptr_t)a, (CUdeviceptr)(uintptr_t)b,
                            count);
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

/* id 0x16: cudaFree(void *devPtr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a)
        return CUDA_SUCCESS;
    return cuMemFree((CUdeviceptr)(uintptr_t)a);
}

/* id 0x1a: cudaFreeHost(void *ptr) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_free_host(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    free(a);
    return CUDA_SUCCESS;
}

/* id 0xa5: cudaDeviceSynchronize(void) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_device_synchronize(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuCtxSynchronize();
}

/* id 0x85: cudaEventCreate(cudaEvent_t *event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_create(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventCreate((CUevent *)a, 0);
}

/* id 0x87: cudaEventRecord(cudaEvent_t event, cudaStream_t stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_record(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventRecord((CUevent)a, (CUstream)b);
}

/* id 0x8a: cudaEventQuery(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_query(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventQuery((CUevent)a);
}

/* id 0x1e6: cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_elapsed_time(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventElapsedTime((float *)a, (CUevent)b, (CUevent)c);
}

/* id 0x88: cudaEventDestroy(cudaEvent_t event) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_event_destroy(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    return cuEventDestroy((CUevent)a);
}

/* id 0xab / 0xac: cudaProfilerStart() / cudaProfilerStop() */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_profiler_noop(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    return CUDA_SUCCESS;
}

/* --- kernel launch -----------------------------------------------------
 * The hybrid runtime reports the executable's fatbin once at startup and
 * then every __global__ host stub that gets launched.  Launches arrive with
 * the host stub address as the function key, which is exactly what ACuda's
 * cudart does for statically linked executables. */

typedef struct {
    unsigned int x, y, z;
} acuda_dd9e98a4_dim3;

#define ACUDA_DD9E98A4_MAX_FATBINS 8
#define ACUDA_DD9E98A4_MAX_FUNCS 256

typedef struct {
    const void *blob;
    CUmodule module;
    int state; /* 0 = not loaded, 1 = loaded, -1 = failed */
} acuda_dd9e98a4_fatbin;

typedef struct {
    const void *host_fn;
    const char *device_name; /* owned by the executable image */
    acuda_dd9e98a4_fatbin *fatbin;
} acuda_dd9e98a4_kernel;

static acuda_dd9e98a4_fatbin g_dd9e98a4_fatbins[ACUDA_DD9E98A4_MAX_FATBINS];
static unsigned g_dd9e98a4_nfatbins;
static acuda_dd9e98a4_kernel g_dd9e98a4_kernels[ACUDA_DD9E98A4_MAX_FUNCS];
static unsigned g_dd9e98a4_nkernels;

/* id 0x1ee: __cudaRegisterFatBinary(void *fatCubin) -> handle */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    unsigned i;
    (void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a)
        return CUDA_ERROR_INVALID_VALUE;
    /* The hybrid runtime keeps the returned handle in a 32-bit field, so a
     * pointer cannot be handed back; use a 1-based index instead. */
    for (i = 0; i < g_dd9e98a4_nfatbins; i++) {
        if (g_dd9e98a4_fatbins[i].blob == a)
            return (CUresult)(i + 1);
    }
    if (g_dd9e98a4_nfatbins >= ACUDA_DD9E98A4_MAX_FATBINS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    i = g_dd9e98a4_nfatbins++;
    g_dd9e98a4_fatbins[i].blob = a;
    g_dd9e98a4_fatbins[i].module = NULL;
    g_dd9e98a4_fatbins[i].state = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register fatbin %p -> %u\n", a,
                      i + 1);
    return (CUresult)(i + 1);
}

/* id 0x1ef: __cudaRegisterFatBinaryEnd(void **handle) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_fat_binary_end(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    return CUDA_SUCCESS;
}

/* id 0x1f3: __cudaRegisterFunction(handle, hostFun, deviceFun, deviceName,
 *                                  threadLimit, tid, bid, bDim, gDim, wSize) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_register_function(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_kernel *k;
    (void)c;(void)e;(void)f;(void)g;(void)h;
    if (!a || !b || !d)
        return CUDA_ERROR_INVALID_VALUE;
    if (g_dd9e98a4_nkernels >= ACUDA_DD9E98A4_MAX_FUNCS)
        return CUDA_ERROR_OUT_OF_MEMORY;
    k = &g_dd9e98a4_kernels[g_dd9e98a4_nkernels++];
    k->host_fn = b;
    k->device_name = (const char *)d;
    k->fatbin = NULL;
    {
        unsigned h = (unsigned)(uintptr_t)a;
        if (h >= 1 && h <= g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[h - 1];
        else if (g_dd9e98a4_nfatbins)
            k->fatbin = &g_dd9e98a4_fatbins[g_dd9e98a4_nfatbins - 1];
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 register kernel %p -> %s\n", b,
                      k->device_name);
    return CUDA_SUCCESS;
}

/* id 0x1f8: __cudaGetKernel(void **kernel, const void *func) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_get_kernel(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 get kernel slot=%p func=%p\n", a, b);
    (void)c;(void)d;(void)e;(void)f;(void)g;(void)h;
    if (!a || !b)
        return CUDA_ERROR_INVALID_VALUE;
    *(const void **)a = b;
    return CUDA_SUCCESS;
}

/* Pushed <<<grid, block, shared, stream>>> configuration. */
typedef struct {
    acuda_dd9e98a4_dim3 grid;
    acuda_dd9e98a4_dim3 block;
    size_t shared;
    void *stream;
} acuda_dd9e98a4_launch_config;

#if defined(_MSC_VER)
#define ACUDA_DD9E98A4_TLS __declspec(thread)
#else
#define ACUDA_DD9E98A4_TLS __thread
#endif

static ACUDA_DD9E98A4_TLS acuda_dd9e98a4_launch_config
    g_dd9e98a4_configs[16];
static ACUDA_DD9E98A4_TLS unsigned g_dd9e98a4_config_depth;

/* id 0x1f6: __cudaPushCallConfiguration(dim3 *grid, dim3 *block,
 *                                       size_t sharedMem, void *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_push_config(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 push config\n");
    (void)e;(void)f;(void)g;(void)h;
    if (!a || !b || g_dd9e98a4_config_depth >= 16)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[g_dd9e98a4_config_depth++];
    cfg->grid = *(const acuda_dd9e98a4_dim3 *)a;
    cfg->block = *(const acuda_dd9e98a4_dim3 *)b;
    cfg->shared = (size_t)c;
    cfg->stream = d;
    return CUDA_SUCCESS;
}

/* id 0x1f7: __cudaPopCallConfiguration(dim3 *grid, dim3 *block,
 *                                      size_t *sharedMem, void **stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_pop_config(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    acuda_dd9e98a4_launch_config *cfg;
    ACUDA_DEBUG_PRINT("acuda-trace .. dd9e98a4 pop config\n");
    (void)e;(void)f;(void)g;(void)h;
    if (g_dd9e98a4_config_depth == 0)
        return CUDA_ERROR_INVALID_VALUE;
    cfg = &g_dd9e98a4_configs[--g_dd9e98a4_config_depth];
    if (a)
        *(acuda_dd9e98a4_dim3 *)a = cfg->grid;
    if (b)
        *(acuda_dd9e98a4_dim3 *)b = cfg->block;
    if (c)
        *(size_t *)c = cfg->shared;
    if (d)
        *(void **)d = cfg->stream;
    return CUDA_SUCCESS;
}

/* id 0x1f9: cudaLaunchKernel(const void *func, dim3 *grid, dim3 *block,
 *                            void **args, size_t sharedMem, void *stream) */
static CUresult ACUDA_CUDAAPI acuda_dd9e98a4_api_launch_kernel(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    const acuda_dd9e98a4_dim3 *grid = (const acuda_dd9e98a4_dim3 *)b;
    const acuda_dd9e98a4_dim3 *block = (const acuda_dd9e98a4_dim3 *)c;
    acuda_dd9e98a4_kernel *k = NULL;
    acuda_dd9e98a4_fatbin *fb;
    CUfunction fn = NULL;
    CUresult r;
    unsigned i;
    (void)g;(void)h;
    if (!acuda_dd9e98a4_driver_ready())
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!a || !grid || !block)
        return CUDA_ERROR_INVALID_VALUE;
    for (i = 0; i < g_dd9e98a4_nkernels; i++) {
        if (g_dd9e98a4_kernels[i].host_fn == a) {
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
                          block->z, (unsigned int)(size_t)e, (CUstream)f, d,
                          NULL);
}

static void *acuda_dd9e98a4_get_id_stub(unsigned int id) {
    switch(id){
    case 0x10: return (void*)&acuda_dd9e98a4_api_set_device;
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
    case 0x87: return (void*)&acuda_dd9e98a4_api_event_record;
    case 0x88: return (void*)&acuda_dd9e98a4_api_event_destroy;
    case 0x8a: return (void*)&acuda_dd9e98a4_api_event_query;
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

static void *ACUDA_CUDAAPI acuda_dd9e98a4_dispatcher(void *a, void *b, void *c, void *d, void *e, void *f, void *g, void *h) {
    static int n = 0;
    unsigned int id = 0xFFFFFFFFu, sub = 0;
    if (a) {
        const unsigned char *p = (const unsigned char *)a;
        id = (unsigned int)p[8] | ((unsigned int)p[9] << 8) | ((unsigned int)p[10] << 16) | ((unsigned int)p[11] << 24);
        sub = (unsigned int)p[12] | ((unsigned int)p[13] << 8) | ((unsigned int)p[14] << 16) | ((unsigned int)p[15] << 24);
    }
    if (n < 80) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4_dispatch[%d] id=0x%x sub=%u a=%p b=%p c=%p d=%p e=%p f=%p g=%p h=%p\n",
                n, id, sub, a,b,c,d,e,f,g,h);
    } else if (n == 80) {
        ACUDA_DEBUG_PRINT( "ACuda-debug: dd9e98a4_dispatch ... (%d calls so far)\n", n);
    }
    n++;
    (void)f;
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
