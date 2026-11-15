#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// CUDA Driver API over the ACuda runtime. Single-threaded.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rtxdiasm_fatbin.h"
#if defined(_M_IX86)
#include <intrin.h>
#endif
#include "internal.h" // stream internals for cuStreamQuery
#include "arena.h"
#include "rtxdisam_bridge.h"
#include "acuda/driver.h"
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/memory.h"
#include "acuda/stream.h"

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#define unsetenv(name) _putenv_s((name), "")
#else
#include <dlfcn.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define ACUDA_CC_MAJOR 8
#define ACUDA_CC_MINOR 6
#define ACUDA_DRIVER_VERSION 13040
#if defined(__i386__) || (defined(_WIN32) && !defined(_WIN64))
#define ACUDA_DEVICE_MEM (512ull << 20)
#else
#define ACUDA_DEVICE_MEM (8ull << 30)
#endif
/* Diagnostics: PhysX-era capability gates may key off total device memory
 * (the 32-bit default is 512 MiB; real cards of the era had 1+ GiB).
 * ACUDA_DEVICE_MEM_MB=<n> overrides the reported total without touching
 * the arenas.  Cached once per process. */
static unsigned long long acuda_device_mem_total(void) {
    static unsigned long long cached;
    static int cached_ok;
    if (!cached_ok) {
        const char *s = getenv("ACUDA_DEVICE_MEM_MB");
        unsigned long mb = s && *s ? strtoul(s, NULL, 0) : 0;
        cached = mb ? (unsigned long long)mb << 20 : ACUDA_DEVICE_MEM;
        cached_ok = 1;
    }
    return cached;
}
#define ACUDA_CTX_STACK 16
/* Full-module PTX parsing of large modules (multi-MB) OOMs in 32-bit and
 * costs seconds; per-function extraction/compile still validates what is
 * actually used, so skip the eager whole-module parse above this size. */
#define ACUDA_PTX_EAGER_VALIDATE_MAX (1u << 20)

#if defined(_WIN32)
static int acuda_host_heap_init(size_t need);
#endif

#if defined(_WIN32) && !defined(_WIN64)
// 32-bit host address-space probe: PhysX does not check malloc failures,
// so a late OOM shows up as an access violation in PhysXCore.  Logging the
// free/largest region around our large allocations makes that visible.
static void acuda_trace_va(const char *tag) {
    if (!getenv("ACUDA_TRACE"))
        return;
    MEMORY_BASIC_INFORMATION mbi;
    unsigned char *p = (unsigned char *)0x10000;
    size_t total_free = 0, largest = 0, committed = 0;
    while ((uintptr_t)p < (uintptr_t)0x80000000u) {
        if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi))
            break;
        if (mbi.State == MEM_FREE) {
            total_free += mbi.RegionSize;
            if (mbi.RegionSize > largest)
                largest = mbi.RegionSize;
        } else if (mbi.State == MEM_COMMIT) {
            committed += mbi.RegionSize;
        }
        unsigned char *next =
            (unsigned char *)mbi.BaseAddress + mbi.RegionSize;
        if (next <= p)
            break;
        p = next;
    }
    fprintf(stderr,
            "acuda-va %s: free=%zu largest=%zu committed=%zu\n",
            tag, total_free, largest, committed);
}
#endif

#if defined(_M_IX86)
/* PhysXCore crashes on an unchecked allocator return.  A vectored handler
 * lets us capture the registers at the fault without a debugger: if eax is
 * NULL and the fault address is 0x10, the allocator returned NULL. */
static HANDLE g_veh_log = INVALID_HANDLE_VALUE;

/* Probe whether [p, p+n) is committed readable memory.  Used by the
 * VEH so fault-context garbage can never crash the handler itself. */
static int acuda_can_read(const void *p, size_t n) {
    MEMORY_BASIC_INFORMATION mbi;
    uintptr_t addr = (uintptr_t)p;
    uintptr_t base, len;
    if (!p || n == 0 || addr + n < addr)
        return 0;
#if defined(_WIN32)
    if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi))
        return 0;
    if (mbi.State != MEM_COMMIT)
        return 0;
    if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))
        return 0;
    base = (uintptr_t)mbi.BaseAddress;
    len = (uintptr_t)mbi.RegionSize;
    if (addr < base || addr + n > base + len)
        return 0;
    return 1;
#else
    (void)addr;
    return 1;
#endif
}

static LONG CALLBACK acuda_veh(EXCEPTION_POINTERS *ep) {
    if (!g_veh_log || g_veh_log == INVALID_HANDLE_VALUE)
        return EXCEPTION_CONTINUE_SEARCH;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != 0xC0000409 /* STATUS_STACK_BUFFER_OVERRUN */)
        return EXCEPTION_CONTINUE_SEARCH;
    unsigned long fn = 0;
    unsigned long modbase = 0;
    unsigned int count16 = 0;
    unsigned long stk[8] = {0};
    /* The faulting context can carry garbage registers (e.g. Esi=0x200
     * after a bad user-mode read); validate every probe address with
     * VirtualQuery first so the diagnostic handler itself can never
     * fault.  A nested fault inside a VEH would otherwise terminate
     * the process with no log. */
    {
        HMODULE h = GetModuleHandleA("PhysXCore.dll");
        modbase = (unsigned long)(uintptr_t)h;
        if (h && acuda_can_read((const void *)(uintptr_t)(modbase +
                                                          0x2f4048u),
                                sizeof(fn)))
            fn = *(volatile unsigned long *)(modbase + 0x2f4048u);
        {
            const void *cp =
                (const void *)(uintptr_t)(ep->ContextRecord->Esi + 0x2b4u);
            if (acuda_can_read(cp, sizeof(unsigned short)))
                count16 = *(volatile unsigned short *)cp;
            else
                count16 = 0xffffffffu;
        }
        {
            const unsigned long *sp =
                (const unsigned long *)(uintptr_t)ep->ContextRecord->Esp;
            for (int i = 0; i < 8; i++)
                if (acuda_can_read(sp + i, sizeof(sp[i])))
                    stk[i] = sp[i];
                else
                    stk[i] = 0xddddddddul;
        }
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    char buf[512];
    ULONG_PTR info0 = ep->ExceptionRecord->NumberParameters > 0 ? ep->ExceptionRecord->ExceptionInformation[0] : 0;
    ULONG_PTR info1 = ep->ExceptionRecord->NumberParameters > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
    int n = _snprintf(
        buf, sizeof(buf),
        "acuda-veh %s(0x%08lx) at %p op=%llu addr=%p eax=%08lx ecx=%08lx "
        "esi=%08lx edi=%08lx ebp=%08lx esp=%08lx eip=%08lx "
        "mod=%08lx fn4048=%08lx cnt2b4=%u availcommit=%llu\r\n"
        "acuda-veh stack: %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx\r\n",
        code == EXCEPTION_ACCESS_VIOLATION ? "AV" : "EXC",
        (unsigned long)code,
        ep->ExceptionRecord->ExceptionAddress,
        (unsigned long long)info0,
        (void *)info1,
        (unsigned long)ep->ContextRecord->Eax,
        (unsigned long)ep->ContextRecord->Ecx,
        (unsigned long)ep->ContextRecord->Esi,
        (unsigned long)ep->ContextRecord->Edi,
        (unsigned long)ep->ContextRecord->Ebp,
        (unsigned long)ep->ContextRecord->Esp,
        (unsigned long)ep->ContextRecord->Eip, modbase, fn, count16,
        (unsigned long long)ms.ullAvailPageFile,
        stk[0], stk[1], stk[2], stk[3], stk[4], stk[5], stk[6], stk[7]);
    if (n > 0) {
        DWORD w = 0;
        WriteFile(g_veh_log, buf, (DWORD)n, &w, NULL);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static void acuda_install_veh(void) {
    static int done = 0;
    if (done)
        return;
    done = 1;
    const char *tp = getenv("ACUDA_TRACE_FILE");
    if (!tp || !*tp)
        return;
    g_veh_log = CreateFileA(tp, FILE_APPEND_DATA,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_veh_log != INVALID_HANDLE_VALUE)
        AddVectoredExceptionHandler(1, acuda_veh);
}
#endif

#if defined(_M_IX86)
#pragma comment(linker, "/export:cuInit=_cuInit@4")
#endif

struct CUctx_st {
    int device;
};
struct CUmod_st {
    CUcontext ctx;
    char *ptx;
    // Original SASS-only image. Kept so a driver module with multiple text
    // sections can be lifted lazily for each requested entry point.
    unsigned char *image;
    size_t image_size;
    // The 32-bit PTX parser OOMed on this module (whole-text or per-entry);
    // go straight to the 64-bit compile host for each requested entry.
    int ptx_oom;
    struct { char name[ACUDA_ENTRY_NAME_MAX]; acuda_kernel_t *k; } *funcs;
    size_t nfuncs, fcap;
    struct CUtexref_st **texrefs;
    size_t ntexrefs, texrefs_cap;
    // Synthetic module globals.  Runtime-registered __constant__ symbols
    // are resolved through cuModuleGetGlobal; legacy SASS-only modules
    // carry no symbol table of their own, so allocate per-name storage.
    struct {
        char name[ACUDA_ENTRY_NAME_MAX];
        CUdeviceptr dptr;
        size_t bytes;
    } *globals;
    size_t nglobals, globals_cap;
};

struct CUtexref_st {
    CUmodule mod;
    char name[ACUDA_ENTRY_NAME_MAX];
    CUdeviceptr addr;
    size_t bytes;
    CUarray array;
    int is_array;
    int fmt_set;
    CUarray_format fmt;
    int channels;
    unsigned int flags;
    CUfilter_mode filter;
    CUaddress_mode address_mode[3];
    unsigned int max_aniso;
    CUtexObject obj;
    // Last built snapshot state; rebuilds that change nothing are
    // skipped (the app re-sets filter/mipmap state every frame).
    CUdeviceptr built_addr;
    size_t built_bytes;
    CUarray_format built_fmt;
    int built_channels;
};
#define ACUDA_HANDLE_FUNC 0x46554e43u
#define ACUDA_HANDLE_KERNEL 0x4b45524eu

struct CUfunc_st {
    uint32_t magic;
    CUmodule mod;
    acuda_kernel_t *k;
    char name[ACUDA_ENTRY_NAME_MAX];
    int maxdyn_set;      // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    unsigned maxdyn;
};
struct CUkernel_st {
    uint32_t magic;
    CUmodule mod;
    char name[ACUDA_ENTRY_NAME_MAX];
    CUfunction func;
};
struct CUstream_st {
    CUcontext ctx;
    acuda_stream_t s;
};
struct CUevent_st {
    CUcontext ctx;
    acuda_event_t e;
};
struct CUlink_st {
    char **blobs;
    size_t *sizes;
    size_t n, cap;
    char *linked;
    size_t linked_size;
};

static int g_init;
static acuda_context_t g_actx;
static int g_actx_exit_registered;
static CUcontext g_stack[ACUDA_CTX_STACK];
static size_t g_depth;

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

// ---- texture arrays and texture objects ------------------------------

struct CUarray_st {
    acuda_buffer_t image; // kind 2 (owned unless parent != NULL)
    CUDA_ARRAY_DESCRIPTOR desc2;
    CUDA_ARRAY3D_DESCRIPTOR desc3;
    int is3d;
    unsigned int level;
    int own;
    struct CUarray_st *parent; // mip level view (borrowed image)
};

struct CUmipmappedArray_st {
    acuda_buffer_t image;
    CUDA_ARRAY3D_DESCRIPTOR desc;
    unsigned int levels;
    int is3d;
};

typedef struct {
    CUtexObject id;
    acuda_buffer_t tex;    // kind 3 combined image sampler
    acuda_buffer_t sampler; // kind 4
    acuda_buffer_t own_image; // linear/pitch2D snapshot (kind 2)
    CUDA_RESOURCE_DESC res;
    CUDA_TEXTURE_DESC td;
} driver_texobj_t;

static driver_texobj_t *g_texobjs;
static size_t g_ntexobjs, g_ctexobjs;
static CUtexObject g_next_texid = 1;

typedef struct {
    CUsurfObject id;
    uint32_t slot;
    acuda_buffer_t image; // borrowed storage-capable array image (kind 2)
    CUDA_RESOURCE_DESC res;
} driver_surfobj_t;

static driver_surfobj_t *g_surfobjs;
static size_t g_nsurfobjs, g_csurfobjs;
static acuda_buffer_t *g_surf_slots;
static uint32_t g_surf_slot_cap;

static driver_surfobj_t *driver_surf_find(CUsurfObject id) {
    for (size_t i = 0; i < g_nsurfobjs; i++)
        if (g_surfobjs[i].id == id)
            return &g_surfobjs[i];
    return NULL;
}

static void driver_img_type(acuda_buffer_t img, int *dim, int *arrayed,
                            int *isint) {
    *dim = img->dim3d ? 2 : (img->dim1d ? 0 : 1);
    *arrayed = !img->dim3d && img->layers > 1;
    *isint = img->format == 3 ? 1 : (img->format == 0 ? 0 : -1);
}

typedef struct {
    int used, dim, arrayed, isint;
    acuda_buffer_t img;
} driver_dummy_t;
static driver_dummy_t g_surf_dummies[ACUDA_SURF_TABLE_MAX_TYPES];

static acuda_buffer_t driver_surf_dummy(int dim, int arrayed, int isint) {
    for (int i = 0; i < ACUDA_SURF_TABLE_MAX_TYPES; i++)
        if (g_surf_dummies[i].used &&
            g_surf_dummies[i].dim == dim &&
            g_surf_dummies[i].arrayed == arrayed &&
            g_surf_dummies[i].isint == isint)
            return g_surf_dummies[i].img;
    int slot = -1;
    for (int i = 0; i < ACUDA_SURF_TABLE_MAX_TYPES; i++)
        if (!g_surf_dummies[i].used) {
            slot = i;
            break;
        }
    if (slot < 0)
        return NULL;
    acuda_image_desc d = {
        .width = 1,
        .height = 1,
        .depth = 1,
        .layers = arrayed ? 2u : 1u,
        .levels = 1,
        .samples = 1,
        .format = isint ? 3u : 0u,
        .dim = dim + 1,
        .cube = 0,
    };
    acuda_buffer_t img = NULL;
    if (acuda_surface_create(g_actx, &img, &d, NULL) != ACUDA_SUCCESS)
        return NULL;
    g_surf_dummies[slot].used = 1;
    g_surf_dummies[slot].dim = dim;
    g_surf_dummies[slot].arrayed = arrayed;
    g_surf_dummies[slot].isint = isint;
    g_surf_dummies[slot].img = img;
    return img;
}

// Six words consumed by surface suq (geometry, levels/samples,
// channel type/order); mirrors the surface branch of tex_meta_words.
static void driver_surf_meta(acuda_buffer_t h, uint32_t out[6]) {
    memset(out, 0, 6 * sizeof(uint32_t));
    if (!h)
        return;
    out[0] = h->width;
    out[1] = h->dim1d ? 1u : h->height;
    out[2] = h->dim3d ? h->depth : 1u;
    uint32_t asz = 0;
    if (!h->dim3d) {
        if (h->cube) {
            if (h->layers > 6)
                asz = h->layers / 6;
        } else if (h->layers > 1) {
            asz = h->layers;
        }
    }
    out[3] = asz;
    uint32_t cdt = 2, order = 1;
    switch (h->format) {
    case 0: cdt = 2; order = 3; break; // RGBA32F
    case 1: cdt = 1; order = 3; break; // RGBA8 UNORM
    case 2: cdt = 2; order = 1; break; // R32F
    case 3: cdt = 1; order = 1; break; // R32_UINT
    }
    out[4] = (cdt & 0xffu) | (order << 8);
}

static driver_texobj_t *driver_tex_find(CUtexObject id) {
    for (size_t i = 0; i < g_ntexobjs; i++)
        if (g_texobjs[i].id == id)
            return &g_texobjs[i];
    return NULL;
}

static CUresult map_status(acuda_status_t st) {
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

// Default host-call stub for the CUDA driver API.  The driver API has
// no callback-registration entry point, so unresolved .extern .func
// calls use this identity stub unless the application registered a
// real callback through acuda_host_call_register().  It copies the
// argument bytes into the return area (remaining return bytes zero).
static int driver_host_identity(void *user, const uint8_t *args,
                                size_t args_bytes, uint8_t *ret,
                                size_t ret_bytes) {
    (void)user;
    memset(ret, 0, ret_bytes);
    if (args_bytes > ret_bytes)
        args_bytes = ret_bytes;
    memcpy(ret, args, args_bytes);
    return 0;
}

CUresult ACUDA_CUDAAPI cuModuleGetLoadingMode(int *mode);
CUresult ACUDA_CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                             size_t alignment, CUdeviceptr addr,
                             unsigned long long flags);
CUresult ACUDA_CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult ACUDA_CUDAAPI cuMemCreate(void **handle, size_t size, const void *prop,
                     unsigned long long flags);
CUresult ACUDA_CUDAAPI cuMemRelease(void *handle);
CUresult ACUDA_CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  void *handle, unsigned long long flags);
CUresult ACUDA_CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult ACUDA_CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const void *desc, size_t count);
CUresult ACUDA_CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const void *prop,
                                       int option);
CUresult ACUDA_CUDAAPI cuMemGetAllocationPropertiesFromHandle(void *prop, void *handle);
CUresult ACUDA_CUDAAPI cuMemExportToShareableHandle(void *shareableHandle, void *handle,
                                      int handleType, unsigned long long flags);
CUresult ACUDA_CUDAAPI cuMemImportFromShareableHandle(void **handle, void *osHandle,
                                        int shHandleType);
CUresult ACUDA_CUDAAPI cuMemRetainAllocationHandle(void **handle, void *addr);
CUresult ACUDA_CUDAAPI cuMemsetD32Async(CUdeviceptr dst, unsigned int value,
                          size_t count, void *stream);
CUresult ACUDA_CUDAAPI cuMemsetD8Async(CUdeviceptr dst, unsigned char value,
                         size_t count, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src,
                              size_t bytes, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src,
                              size_t bytes, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                              size_t bytes, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src,
                           size_t bytes, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t bytes,
                           void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src,
                           size_t bytes, void *stream);
CUresult ACUDA_CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t bytes, int kind, void *stream);
CUresult ACUDA_CUDAAPI cuStreamWriteValue32(void *stream, CUdeviceptr addr,
                              unsigned int value, unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamWaitValue32(void *stream, CUdeviceptr addr,
                             unsigned int value, unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamWriteValue64(void *stream, CUdeviceptr addr,
                              unsigned long long value, unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamWaitValue64(void *stream, CUdeviceptr addr,
                             unsigned long long value, unsigned int flags);
CUresult ACUDA_CUDAAPI cuStreamBatchMemOp(void *stream, unsigned int count, void *params,
                            unsigned int flags);
CUresult ACUDA_CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                     int *greatestPriority);
CUresult ACUDA_CUDAAPI cuStreamIsCapturing(void *stream, int *captureStatus);
CUresult ACUDA_CUDAAPI cuStreamBeginCapture(void *stream, int mode);
CUresult ACUDA_CUDAAPI cuStreamEndCapture(void *stream, void **graph);
CUresult ACUDA_CUDAAPI cuThreadExchangeStreamCaptureMode(int *mode);
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo(void *stream, int *status,
                                unsigned long long *id);
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo_v2(void *stream, int *status,
                                   unsigned long long *id, void **graph,
                                   const void **deps, size_t *numDeps);
CUresult ACUDA_CUDAAPI cuStreamGetCaptureInfo_v3(void *stream, int *status,
                                   unsigned long long *id, void **graph,
                                   const void **deps, size_t *numDeps,
                                   void **edgeData);
CUresult ACUDA_CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                              CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize,
                                 CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuKernelGetFunction(void **pFunc, void *kernel);
CUresult ACUDA_CUDAAPI cuKernelGetLibrary(void **pLib, void *kernel);
CUresult ACUDA_CUDAAPI cuKernelGetAttribute(int *pi, int attrib, void *kernel, CUdevice dev);
CUresult ACUDA_CUDAAPI cuKernelSetAttribute(int attrib, int val, void *kernel, CUdevice dev);
CUresult ACUDA_CUDAAPI cuKernelSetCacheConfig(void *kernel, int config, CUdevice dev);
CUresult ACUDA_CUDAAPI cuKernelGetName(const char **name, void *kernel);
CUresult ACUDA_CUDAAPI cuKernelGetParamInfo(void *kernel, size_t paramIndex, size_t *paramOffset, size_t *paramSize);
CUresult ACUDA_CUDAAPI cuKernelGetParamCount(void *kernel, size_t *count);
CUresult ACUDA_CUDAAPI cuLibraryLoadData(void **library, const void *code, void *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, void *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult ACUDA_CUDAAPI cuLibraryLoadFromFile(void **library, const char *fileName, void *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, void *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult ACUDA_CUDAAPI cuLibraryUnload(void *library);
CUresult ACUDA_CUDAAPI cuLibraryGetModule(void **pMod, void *library);
CUresult ACUDA_CUDAAPI cuLibraryGetKernel(void **pKernel, void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, void *library, const char *name);
CUresult ACUDA_CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, void *library, const char *symbol);
CUresult ACUDA_CUDAAPI cuLibraryGetKernelCount(unsigned int *count, void *library);
CUresult ACUDA_CUDAAPI cuLibraryEnumerateKernels(void **kernels, unsigned int numKernels, void *library);
CUresult acuda_stub_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult acuda_stub_cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult acuda_stub_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *maxBlockSize, CUfunction func, size_t blockSizeLimit, size_t dynamicSMemSize);
CUresult acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *maxBlockSize, CUfunction func, size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags);
CUresult acuda_stub_cuTensorMapEncodeTiled(void *tensorMap, int tensorDataType, unsigned int tensorRank, void *globalAddress, const unsigned long long *globalDim, const unsigned long long *globalStrides, const unsigned int *boxDim, const unsigned int *elementStrides, int interleave, int swizzle, int l2Promotion, int oobFill);
/* CUDA 6.x-era compat entries (defined below lookup_symbol). */
CUresult ACUDA_CUDAAPI cuMemsetD16_v2(CUdeviceptr dst, unsigned short value, size_t count);
CUresult ACUDA_CUDAAPI cuMemsetD16(CUdeviceptr dst, unsigned short value, size_t count);
CUresult ACUDA_CUDAAPI cuMemsetD16Async(CUdeviceptr dst, unsigned short value, size_t count, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dst, size_t pitch, unsigned char value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D8(CUdeviceptr dst, size_t pitch, unsigned char value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D8Async(CUdeviceptr dst, size_t pitch, unsigned char value, size_t width, size_t height, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dst, size_t pitch, unsigned short value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D16(CUdeviceptr dst, size_t pitch, unsigned short value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D16Async(CUdeviceptr dst, size_t pitch, unsigned short value, size_t width, size_t height, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dst, size_t pitch, unsigned int value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D32(CUdeviceptr dst, size_t pitch, unsigned int value, size_t width, size_t height);
CUresult ACUDA_CUDAAPI cuMemsetD2D32Async(CUdeviceptr dst, size_t pitch, unsigned int value, size_t width, size_t height, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemcpy2D_v2(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy2DAsync_v2(const void *pCopy, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned_v2(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy3D_v2(const void *pCopy);
CUresult ACUDA_CUDAAPI cuMemcpy3DAsync_v2(const void *pCopy, CUstream hStream);
CUresult ACUDA_CUDAAPI cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t bytes);
CUresult ACUDA_CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t bytes, CUstream hStream);


CUresult ACUDA_CUDAAPI cuMemGetInfo_v2(size_t *freeBytes, size_t *totalBytes);
CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer_v2(void **pdst, void *p, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t widthInBytes, size_t height, unsigned int elemSize);

CUresult ACUDA_CUDAAPI cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pDesc, CUarray hArray);
CUresult ACUDA_CUDAAPI cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pDesc, CUarray hArray);



CUresult ACUDA_CUDAAPI cuTexRefCreate(CUtexref *pTexRef);
CUresult ACUDA_CUDAAPI cuTexRefDestroy(CUtexref hTexRef);











CUresult ACUDA_CUDAAPI cuProfilerStart(void);
CUresult ACUDA_CUDAAPI cuProfilerStop(void);
CUresult ACUDA_CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
typedef void(ACUDA_CUDAAPI *acuda_stream_cb_t)(CUstream hStream, CUresult status, void *userData);
CUresult ACUDA_CUDAAPI cuStreamAddCallback(CUstream hStream, acuda_stream_cb_t cb, void *userData, unsigned int flags);








CUresult ACUDA_CUDAAPI cuD3D10UnregisterResource(void *pResource);






CUresult ACUDA_CUDAAPI cuGLInit(void);














CUresult ACUDA_CUDAAPI cuGraphicsUnregisterResource(void *resource);





CUresult ACUDA_CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *handle, CUevent event);
CUresult ACUDA_CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *handle, CUdeviceptr dptr);
CUresult ACUDA_CUDAAPI cuIpcOpenEventHandle(CUevent *event, CUipcEventHandle handle);
CUresult ACUDA_CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *dptr, CUipcMemHandle handle, unsigned int flags);
CUresult ACUDA_CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult ACUDA_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *maxBlockSize, CUfunction func, size_t blockSizeLimit, size_t dynamicSMemSize);
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *maxBlockSize, CUfunction func, size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuTensorMapEncodeTiled(void *tensorMap, int tensorDataType, unsigned int tensorRank, void *globalAddress, const unsigned long long *globalDim, const unsigned long long *globalStrides, const unsigned int *boxDim, const unsigned int *elementStrides, int interleave, int swizzle, int l2Promotion, int oobFill);

static CUresult need_init(void) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? need_init g=%d @%p\n", g_init,
                (void *)&g_init);
    return g_init ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

// Lazily creates the one shared Vulkan context for the process.

/* Arity-only declarations for compatibility stubs whose definitions use
 * ABI-loose void* parameter lists. */
CUresult ACUDA_CUDAAPI cuCtxDisablePeerAccess();
CUresult ACUDA_CUDAAPI cuCtxEnablePeerAccess();
CUresult ACUDA_CUDAAPI cuD3D10CtxCreateOnDevice();
CUresult ACUDA_CUDAAPI cuD3D10GetDevice();
CUresult ACUDA_CUDAAPI cuD3D10GetDevices();
CUresult ACUDA_CUDAAPI cuD3D10GetDirect3DDevice();
CUresult ACUDA_CUDAAPI cuD3D10MapResources();
CUresult ACUDA_CUDAAPI cuD3D10RegisterResource();
CUresult ACUDA_CUDAAPI cuD3D10UnmapResources();
CUresult ACUDA_CUDAAPI cuD3D11CtxCreateOnDevice();
CUresult ACUDA_CUDAAPI cuD3D11GetDevice();
CUresult ACUDA_CUDAAPI cuD3D11GetDevices();
CUresult ACUDA_CUDAAPI cuD3D11GetDirect3DDevice();
CUresult ACUDA_CUDAAPI cuFuncSetCacheConfig();
CUresult ACUDA_CUDAAPI cuFuncSetSharedMemConfig();
CUresult ACUDA_CUDAAPI cuGLCtxCreate_v2();
CUresult ACUDA_CUDAAPI cuGLGetDevices();
CUresult ACUDA_CUDAAPI cuGLMapBufferObjectAsync_v2();
CUresult ACUDA_CUDAAPI cuGLMapBufferObject_v2();
CUresult ACUDA_CUDAAPI cuGLRegisterBufferObject();
CUresult ACUDA_CUDAAPI cuGLSetBufferObjectMapFlags();
CUresult ACUDA_CUDAAPI cuGLUnmapBufferObject();
CUresult ACUDA_CUDAAPI cuGLUnmapBufferObjectAsync();
CUresult ACUDA_CUDAAPI cuGLUnregisterBufferObject();
CUresult ACUDA_CUDAAPI cuGraphicsD3D10RegisterResource();
CUresult ACUDA_CUDAAPI cuGraphicsD3D11RegisterResource();
CUresult ACUDA_CUDAAPI cuGraphicsD3D9RegisterResource();
CUresult ACUDA_CUDAAPI cuGraphicsGLRegisterBuffer();
CUresult ACUDA_CUDAAPI cuGraphicsGLRegisterImage();
CUresult ACUDA_CUDAAPI cuGraphicsMapResources();
CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedMipmappedArray();
CUresult ACUDA_CUDAAPI cuGraphicsResourceGetMappedPointer_v2();
CUresult ACUDA_CUDAAPI cuGraphicsResourceSetMapFlags_v2();
CUresult ACUDA_CUDAAPI cuGraphicsSubResourceGetMappedArray();
CUresult ACUDA_CUDAAPI cuGraphicsUnmapResources();
CUresult ACUDA_CUDAAPI cuMemAllocManaged();
CUresult ACUDA_CUDAAPI cuMemcpy3DPeer();
CUresult ACUDA_CUDAAPI cuMemcpy3DPeerAsync();
CUresult ACUDA_CUDAAPI cuModuleGetSurfRef();
CUresult ACUDA_CUDAAPI cuModuleLoad();
CUresult ACUDA_CUDAAPI cuProfilerInitialize();
CUresult ACUDA_CUDAAPI cuStreamAttachMemAsync();
CUresult ACUDA_CUDAAPI cuSurfRefSetArray();
CUresult ACUDA_CUDAAPI cuTexObjectGetResourceViewDesc();
CUresult ACUDA_CUDAAPI cuTexRefSetAddress2D_v3();
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapFilterMode();
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapLevelBias();
CUresult ACUDA_CUDAAPI cuTexRefSetMipmapLevelClamp();
CUresult ACUDA_CUDAAPI cuTexRefSetMipmappedArray();
CUresult ACUDA_CUDAAPI cuWGLGetDevice();
static CUresult need_actx(void) {
    CUresult r = need_init();
    if (r != CUDA_SUCCESS)
        return r;
    if (!g_actx) {
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

static CUcontext current_ctx(void) {
    return g_depth > 0 ? g_stack[g_depth - 1] : NULL;
}

static CUresult need_ctx(CUcontext *out) {
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

// Device pointers are arena virtual addresses (base+offset), so caching
// allocators and sub-views work with plain arithmetic. Resolved to host
// memory (persistently mapped, coherent) or to (buffer, offset) binds.
static CUresult host_for(CUdeviceptr dptr, size_t bytes, void **host) {
    if (!dptr || !host)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_host((uint64_t)dptr, bytes, host) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

static CUresult bind_for(CUdeviceptr dptr, size_t bytes, VkBuffer *buf,
                         size_t *offset) {
    if (!dptr || !buf || !offset)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_bind((uint64_t)dptr, bytes, buf, offset) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuInit(unsigned int flags) {
    /* Allow trace capture from GUI apps whose stderr is not wired
     * anywhere (FluidMark): ACUDA_TRACE_FILE=<path> redirects stderr. */
    static int trace_file_done = 0;
    if (!trace_file_done) {
        trace_file_done = 1;
        const char *tp = getenv("ACUDA_TRACE_FILE");
        if (tp && *tp) {
            FILE *tf = freopen(tp, "a", stderr);
            (void)tf;
        }
        /* Unbuffered: a crash must not lose the trailing trace lines
         * that identify the faulting call. */
        setvbuf(stderr, NULL, _IONBF, 0);
#if defined(_M_IX86)
        acuda_install_veh();
#endif
    }
#if defined(_WIN32)
    /* Reserve the pinned-host heap before the device arena fragments the
     * 32-bit address space.  PhysX requests a 128 MB staging buffer after
     * the arena has grown past 1 GB; a late VirtualAlloc then fails. */
    acuda_host_heap_init(128u << 20);
#endif
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuInit(flags=%u) g@%p\n", flags,
                (void *)&g_init);
    if (flags != 0)
        return CUDA_ERROR_INVALID_VALUE;
    g_init = 1;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuInit done g=%d\n", g_init);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion)
        return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = ACUDA_DRIVER_VERSION;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount\n");
    if (!count)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    uint32_t n = 0;
    acuda_status_t st = acuda_device_count(g_actx, &n);
    if (st != ACUDA_SUCCESS)
        return map_status(st);
    // Expose only GPUs with compute queues to driver clients; software
    // rasterizers (llvmpipe) confuse client heuristics. Matches ZLUDA.
    // TODO: filter by queue support instead of clamping to 1.
    if (n > 1)
        n = 1;
    *count = (int)n;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount -> %d\n", (int)n);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGet(%d)\n", ordinal);
    if (!device)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (ordinal < 0 || ordinal >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuDeviceGetName\n");
    if (!name || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    acuda_device_info_t info;
    acuda_status_t st = acuda_device_get(g_actx, (uint32_t)dev, &info);
    if (st != ACUDA_SUCCESS)
        return CUDA_ERROR_INVALID_DEVICE;
    strncpy(name, info.name, (size_t)len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

static int cc_major(void) {
    const char *v = getenv("ACUDA_CC_MAJOR");
    return v && *v ? atoi(v) : ACUDA_CC_MAJOR;
}

static int cc_minor(void) {
    const char *v = getenv("ACUDA_CC_MINOR");
    return v && *v ? atoi(v) : ACUDA_CC_MINOR;
}

CUresult ACUDA_CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (!major || !minor)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *major = cc_major();
    *minor = cc_minor();
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetProperties(dev=%d)\n", dev);
    if (!prop)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    memset(prop, 0, sizeof(*prop));
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock = 49152;
    prop->totalConstantMemory = 65536;
    prop->SIMDWidth = 32;
    prop->memPitch = 2147483647;
    prop->regsPerBlock = 65536;
    prop->clockRate = 1290000;
    prop->textureAlign = 512;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    if (!bytes)
        return CUDA_ERROR_INVALID_VALUE;
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= n)
        return CUDA_ERROR_INVALID_DEVICE;
    *bytes = (size_t)acuda_device_mem_total();
    return CUDA_SUCCESS;
}

static int device_pci_info(CUdevice dev, acuda_device_info_t *info);

CUresult ACUDA_CUDAAPI cuDeviceGetAttribute(int *value, int attrib, CUdevice dev) {
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuDeviceGetAttribute(%d, dev=%d)\n",
                attrib, dev);
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace !! getattr inner count -> %d\n", r);
        return r;
    }
    if (dev < 0 || dev >= n) {
        ACUDA_DEBUG_PRINT("acuda-trace !! getattr ordinal %d/%d\n", dev, n);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
        *value = cc_major();
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
        *value = cc_minor();
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
        *value = 2147483647;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *value = 1024;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
        *value = 1024;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
        *value = 64;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
        *value = 65535;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        *value = 49152;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
        // ACuda lowers PTX warps as 32 lanes: shfl/vote use 32-bit
        // masks and SubgroupLocalInvocationId, and a 64-lane hardware
        // wave is split into two 32-lane segments.  Report 32 here to
        // match cudaGetDeviceProperties and the compile-time
        // C10_WARP_SIZE; a caller that sized a warp-level launch from
        // this attribute would otherwise spawn two logical lanes per
        // PTX lane (e.g. torch.sort's warpMergeSort kernels).
        *value = 32;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *value = 40; // RTX 3070 laptop GPU
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *value = 1290000; // kHz
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
        *value = 6001000; // kHz
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
        *value = 256; // TODO per-device
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
        *value = 4 << 20; // TODO per-device
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
        *value = 1536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
        *value = 65536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
        *value = 2147483647;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
        *value = 65536;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
    case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
        *value = 512;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT:
        *value = 32;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
        *value = 131072;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
        *value = 131072;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
        *value = 65536;
        return CUDA_SUCCESS;
    case 15: // legacy CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
        *value = 1;
        return CUDA_SUCCESS;
    case 24: // MAXIMUM_TEXTURE3D_WIDTH
    case 25: // MAXIMUM_TEXTURE3D_HEIGHT
    case 26: // MAXIMUM_TEXTURE3D_DEPTH
        *value = 16384;
        return CUDA_SUCCESS;
    case 27: // MAXIMUM_TEXTURE2D_LAYERED_WIDTH
    case 28: // MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
        *value = 32768;
        return CUDA_SUCCESS;
    case 29: // MAXIMUM_TEXTURE2D_LAYERED_LAYERS
        *value = 2048;
        return CUDA_SUCCESS;
    case 42: // MAXIMUM_TEXTURE1D_LAYERED_WIDTH
        *value = 32768;
        return CUDA_SUCCESS;
    case 43: // MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        *value = 2048;
        return CUDA_SUCCESS;
    case 44: // MAXIMUM_TEXTURE2D_GATHER_WIDTH
        *value = 1;
        return CUDA_SUCCESS;
    case 45: // MAXIMUM_TEXTURE2D_GATHER_HEIGHT
        *value = 32768;
        return CUDA_SUCCESS;
    case 46: // MAXIMUM_TEXTURE3D_WIDTH_ALT
        *value = 32768;
        return CUDA_SUCCESS;
    case 47: // MAXIMUM_TEXTURE3D_HEIGHT_ALT
        *value = 8192;
        return CUDA_SUCCESS;
    case 48: // MAXIMUM_TEXTURE3D_DEPTH_ALT
        *value = 8192;
        return CUDA_SUCCESS;
    case 49: // MAXIMUM_TEXTURE3D_DEPTH_ALT (newer)
        *value = 32768;
        return CUDA_SUCCESS;
    case 52: // MAXIMUM_TEXTURE1D_LAYERED_WIDTH (newer enum)
        *value = 32768;
        return CUDA_SUCCESS;
    case 53: // MAXIMUM_TEXTURE1D_LAYERED_LAYERS
        *value = 32768;
        return CUDA_SUCCESS;
    case 54: // MAXIMUM_TEXTURE2D_LINEAR_WIDTH
        *value = 2046;
        return CUDA_SUCCESS;
    case 55: // MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
        *value = 32768;
        return CUDA_SUCCESS;
    case 56: // MAXIMUM_TEXTURE2D_LINEAR_PITCH
        *value = 131072;
        return CUDA_SUCCESS;
    case 57: // MAXIMUM_TEXTURE2D_LINEAR_WIDTH (newer)
        *value = 65536;
        return CUDA_SUCCESS;
    case 58: // MAXIMUM_TEXTURE2D_LINEAR_HEIGHT (newer)
        *value = 16384;
        return CUDA_SUCCESS;
    case 59: // MAXIMUM_TEXTURE2D_LINEAR_PITCH (newer)
        *value = 16384;
        return CUDA_SUCCESS;
    case 60:
        *value = 16384;
        return CUDA_SUCCESS;
    case 61:
        *value = 32768;
        return CUDA_SUCCESS;
    case 62:
        *value = 2048;
        return CUDA_SUCCESS;
    case 63:
        *value = 32768;
        return CUDA_SUCCESS;
    case 64:
        *value = 32768;
        return CUDA_SUCCESS;
    case 65:
        *value = 2048;
        return CUDA_SUCCESS;
    case 66:
        *value = 32768;
        return CUDA_SUCCESS;
    case 67:
        *value = 32768;
        return CUDA_SUCCESS;
    case 68:
        *value = 2046;
        return CUDA_SUCCESS;
    case 69:
        *value = 268435456;
        return CUDA_SUCCESS;
    case 70:
        *value = 131072;
        return CUDA_SUCCESS;
    case 71:
        *value = 65000;
        return CUDA_SUCCESS;
    case 72:
        *value = 2097120;
        return CUDA_SUCCESS;
    case 73:
        *value = 32768;
        return CUDA_SUCCESS;
    case 74:
        *value = 32768;
        return CUDA_SUCCESS;
    case 77:
        *value = 32768;
        return CUDA_SUCCESS;
    case 78:
    case 79:
    case 80:
        *value = 1;
        return CUDA_SUCCESS;
    case 81: // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        *value = 102400;
        return CUDA_SUCCESS;
    case 82: // MAX_REGISTERS_PER_MULTIPROCESSOR
        *value = 65536;
        return CUDA_SUCCESS;
    case 87:
        *value = 64;
        return CUDA_SUCCESS;
    case 90:
        *value = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
        *value = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED:
        *value = 0;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
        *value = 1; // host-visible heaps
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
        *value = 0; // default
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
        *value = 1;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
        {
            const char *s = getenv("ACUDA_UNIFIED_ADDRESSING");
            *value = s && *s ? (int)strtol(s, NULL, 0) : 1;
        }
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: {
        acuda_device_info_t info;
        const char *s = getenv("ACUDA_PCI_BUS");
        if (s && *s) *value = (int)strtol(s, NULL, 0);
        else if (device_pci_info(dev, &info)) *value = (int)info.pci_bus;
        else *value = 0;
        return CUDA_SUCCESS;
    }
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: {
        acuda_device_info_t info;
        const char *s = getenv("ACUDA_PCI_DEVICE");
        if (s && *s) *value = (int)strtol(s, NULL, 0);
        else if (device_pci_info(dev, &info)) *value = (int)info.pci_device;
        else *value = 0;
        return CUDA_SUCCESS;
    }
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: {
        acuda_device_info_t info;
        const char *s = getenv("ACUDA_PCI_DOMAIN");
        if (s && *s) *value = (int)strtol(s, NULL, 0);
        else if (device_pci_info(dev, &info)) *value = (int)info.pci_domain;
        else *value = 0;
        return CUDA_SUCCESS;
    }
    case 0x20080402:
        /* PhysX uses this private attribute as the CUDA-side GPU key.
         * It is matched against the physical handle returned by NVAPI. */
        {
            const char *s = getenv("ACUDA_PPU_ATTR");
            if (s && *s) {
                *value = (int)strtol(s, NULL, 0);
            } else {
                acuda_device_info_t info;
                if (device_pci_info(dev, &info)) {
                    /* Stable per-device key; NVAPI must return the same handle.
                     * Keep the high bit clear: PhysX stores the CUDA-side key
                     * in a signed int and sign-extends it when comparing with
                     * the NVAPI physical handle. */
                    *value = (int)(0x4C000000u | (info.pci_domain << 16) |
                                   (info.pci_bus << 8) | info.pci_device);
                    if (*value == 0) *value = 1;
                } else {
                    *value = (int)(0x4C000000u | ((uint32_t)dev + 1u));
                }
                ACUDA_DEBUG_PRINT("acuda-trace .. ppu_attr dev=%d value=0x%x\n",
                            dev, (unsigned)*value);
            }
        }
        return CUDA_SUCCESS;
    default:
        // CUDA 13's runtime queries a long tail of optional limits during
        // device enumeration and treats any error as a failed
        // cudaGetDeviceCount.  Zero means "unsupported" for every one of
        // those attributes, which is the safe answer for a software
        // device; required limits are handled explicitly above.
        ACUDA_DEBUG_PRINT("acuda-trace ?? cuDeviceGetAttribute(%d) = 0 "
                    "(default)\n", attrib);
        *value = 0;
        return CUDA_SUCCESS;
    }
}

static void release_param_buffers(acuda_buffer_t *bufs, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (bufs[i]) {
            acuda_free(g_actx, bufs[i]);
            bufs[i] = NULL;
        }
    }
}

static CUresult ctx_push(CUcontext ctx) {
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxDestroy_v2\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxPushCurrent_v2\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxPopCurrent_v2\n");
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

static CUresult mem_alloc(CUdeviceptr *dptr, size_t bytesize) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAlloc(%zu)\n", bytesize);
    if (!dptr)
        return CUDA_ERROR_INVALID_VALUE;
    *dptr = 0;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint64_t virt = 0;
    VkBuffer buf = VK_NULL_HANDLE;
    (void)buf;
    int ar = acuda_arena_alloc(bytesize, &virt, &buf);
    if (ar != 0) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuMemAlloc(%zu) -> OOM (arena %d)\n", bytesize, ar);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *dptr = (CUdeviceptr)virt;
    if (getenv("ACUDA_TRACE")) {
        fprintf(stderr, "acuda-trace .. cuMemAlloc(%zu) -> 0x%llx\n", bytesize, (unsigned long long)virt);
#if defined(_WIN32) && !defined(_WIN64)
        acuda_trace_va("cuMemAlloc");
#endif
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    return mem_alloc(dptr, bytesize);
}

CUresult ACUDA_CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return mem_alloc(dptr, bytesize);
}

CUresult ACUDA_CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuMemFree_v2\n");
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

static CUresult memcpy_h2d(CUdeviceptr dst, const void *src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyHtoD(dst=%llx bytes=%zu)\n",
                (unsigned long long)dst, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (!src)
        return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    // Host-visible coherent: plain copy, ordered after default-stream work.
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memcpy(host, src, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t bytes) {
    return memcpy_h2d(dst, src, bytes);
}

static CUresult memcpy_d2h(void *dst, CUdeviceptr src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoH(dst=%p src=%llx bytes=%zu)\n",
                dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (!dst)
        return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(src, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memcpy(dst, host, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2h(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2h(dst, src, bytes);
}

static CUresult memcpy_d2d(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpyDtoD(dst=%llx src=%llx bytes=%zu)\n",
                (unsigned long long)dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *hd = NULL, *hs = NULL;
    r = host_for(dst, bytes, &hd);
    if (r != CUDA_SUCCESS)
        return r;
    r = host_for(src, bytes, &hs);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    memmove(hd, hs, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_d2d(dst, src, bytes);
}

// Generic copy used by CUDA 13 cudart for the scalar `.item()` path.
// Dispatch on which side is arena-backed; host/host stays a plain copy.
static CUresult memcpy_generic(CUdeviceptr dst, CUdeviceptr src,
                               size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemcpy(dst=%llx src=%llx bytes=%zu)\n",
                (unsigned long long)dst, (unsigned long long)src, bytes);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (bytes == 0)
        return CUDA_SUCCESS;
    if (!dst || !src)
        return CUDA_ERROR_INVALID_VALUE;
    size_t cap = 0;
    int ddev = acuda_arena_capacity(dst, &cap) == 0;
    int sdev = acuda_arena_capacity(src, &cap) == 0;
    if (ddev && sdev)
        return memcpy_d2d(dst, src, bytes);
    if (ddev)
        return memcpy_h2d(dst, (const void *)(uintptr_t)src, bytes);
    if (sdev)
        return memcpy_d2h((void *)(uintptr_t)dst, src, bytes);
    memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, bytes);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_generic(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpy_v2(CUdeviceptr dst, CUdeviceptr src, size_t bytes) {
    return memcpy_generic(dst, src, bytes);
}

// CUDA 13.4 cudart probes the attributed-copy entry point for scalar
// readback.  The attributes are placement hints; the copy itself is
// the same.
CUresult ACUDA_CUDAAPI cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src,
                                     size_t bytes, void *attr,
                                     void *stream) {
    (void)attr;
    (void)stream;
    return memcpy_generic(dst, src, bytes);
}

static CUresult memset_d(CUdeviceptr dst, unsigned long long value, size_t count,
                         int is32) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemsetD%s(dst=%llx value=%llu count=%zu)\n",
                is32 ? "32" : "8", (unsigned long long)dst, value, count);
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    size_t bytes = count * (is32 ? 4u : 1u);
    if (bytes == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, bytes, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    if (is32) {
        // Word fill (cuMemsetD32 semantics); offset assumed 4-aligned.
        uint32_t *w = host;
        for (size_t i = 0; i < count; i++)
            w[i] = (uint32_t)value;
    } else {
        memset(host, (int)(unsigned char)value, bytes);
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) {
    return memset_d(dst, value, count, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    return memset_d(dst, value, count, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t count) {
    return memset_d(dst, value, count, 0);
}

CUresult ACUDA_CUDAAPI cuMemsetD8(CUdeviceptr dst, unsigned char value, size_t count) {
    return memset_d(dst, value, count, 0);
}

static CUresult memset_d16(CUdeviceptr dst, unsigned short value, size_t count) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (count == 0)
        return CUDA_SUCCESS;
    void *host = NULL;
    r = host_for(dst, count * 2u, &host);
    if (r != CUDA_SUCCESS)
        return r;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    {
        uint16_t *w = host;
        for (size_t i = 0; i < count; i++)
            w[i] = value;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD16_v2(CUdeviceptr dst, unsigned short value,
                                      size_t count) {
    return memset_d16(dst, value, count);
}

CUresult ACUDA_CUDAAPI cuMemsetD16(CUdeviceptr dst, unsigned short value,
                                   size_t count) {
    return memset_d16(dst, value, count);
}

CUresult ACUDA_CUDAAPI cuMemsetD16Async(CUdeviceptr dst, unsigned short value,
                                       size_t count, CUstream hStream) {
    (void)hStream;
    return memset_d16(dst, value, count);
}

// 2D device memset used by cudaMemset2D.  Width is in elements (bytes for
// D8, 16-bit units for D16, 32-bit units for D32); the driver is
// synchronous so the Async variants run inline.
static CUresult memset_d2d(CUdeviceptr dst, size_t pitch,
                           unsigned long long value, size_t width,
                           size_t height, int elem_size) {
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    (void)c;
    if (r != CUDA_SUCCESS)
        return r;
    if (width == 0 || height == 0)
        return CUDA_SUCCESS;
    if (pitch < width * (size_t)elem_size)
        return CUDA_ERROR_INVALID_VALUE;
    r = map_status(acuda_stream_synchronize(g_actx, NULL));
    if (r != CUDA_SUCCESS)
        return r;
    for (size_t y = 0; y < height; y++) {
        void *host = NULL;
        r = host_for(dst + y * pitch, width * (size_t)elem_size, &host);
        if (r != CUDA_SUCCESS)
            return r;
        if (elem_size == 4) {
            uint32_t *w = host;
            for (size_t i = 0; i < width; i++)
                w[i] = (uint32_t)value;
        } else if (elem_size == 2) {
            uint16_t *w = host;
            for (size_t i = 0; i < width; i++)
                w[i] = (uint16_t)value;
        } else {
            memset(host, (int)(unsigned char)value, width);
        }
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned char value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8(CUdeviceptr dst, size_t pitch,
                                    unsigned char value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D8Async(CUdeviceptr dst, size_t pitch,
                                        unsigned char value, size_t width,
                                        size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 1);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned short value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16(CUdeviceptr dst, size_t pitch,
                                    unsigned short value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D16Async(CUdeviceptr dst, size_t pitch,
                                         unsigned short value, size_t width,
                                         size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 2);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dst, size_t pitch,
                                       unsigned int value, size_t width,
                                       size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 4);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32(CUdeviceptr dst, size_t pitch,
                                    unsigned int value, size_t width,
                                    size_t height) {
    return memset_d2d(dst, pitch, value, width, height, 4);
}

CUresult ACUDA_CUDAAPI cuMemsetD2D32Async(CUdeviceptr dst, size_t pitch,
                                         unsigned int value, size_t width,
                                         size_t height, CUstream hStream) {
    (void)hStream;
    return memset_d2d(dst, pitch, value, width, height, 4);
}

static uint32_t rd32le(const unsigned char *p);

static int looks_like_ptx(const char *text) {
    // Skips whitespace and comments; PTX starts with ".version".
    for (;;) {
        while (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r')
            text++;
        if (text[0] == '/' && text[1] == '/') {
            while (*text && *text != '\n')
                text++;
            continue;
        }
        if (text[0] == '/' && text[1] == '*') {
            text += 2;
            while (*text && !(text[0] == '*' && text[1] == '/'))
                text++;
            if (*text)
                text += 2;
            continue;
        }
        break;
    }
    return strncmp(text, ".version", 8) == 0;
}

static void reported_sm_target(char *buf, size_t len) {
    /* Diagnostics: the SASS lift target normally follows the reported
     * compute capability (old software may need an era CC to accept the
     * GPU, while the lifter still needs a supported arch).
     * ACUDA_SASS_TARGET=<sm_xx> pins the lift/cubin-selection target. */
    const char *t = getenv("ACUDA_SASS_TARGET");
    if (t && t[0]) {
        snprintf(buf, len, "%s", t);
        return;
    }
    snprintf(buf, len, "sm_%d%d", cc_major(), cc_minor());
}

#if defined(_WIN32)
#define ACUDA_CACHE_SEP '\\'
#else
#define ACUDA_CACHE_SEP '/'
#endif

/* Persistent per-entry kernel cache.  Large PhysX modules make the 32-bit
 * PTX parser OOM and fall back to spawning the 64-bit compile host for every
 * cuModuleGetFunction, which can take minutes.  Cache the serialized AKB
 * blob keyed by PTX+entry+target so repeated runs are fast.  Disable with
 * ACUDA_NO_CACHE=1, relocate with ACUDA_CACHE_DIR=<dir>. */
static int acuda_cache_dir(char *buf, size_t bufn) {
    const char *d = getenv("ACUDA_CACHE_DIR");
    if (d && d[0]) {
        snprintf(buf, bufn, "%s", d);
        return 1;
    }
#if defined(_WIN32)
    {
        const char *la = getenv("LOCALAPPDATA");
        if (la && la[0]) {
            snprintf(buf, bufn, "%s\\acuda\\kernel-cache", la);
            return 1;
        }
        {
            char tmp[MAX_PATH];
            DWORD n = GetTempPathA(MAX_PATH, tmp);
            if (n > 0 && n < MAX_PATH) {
                snprintf(buf, bufn, "%sacuda-kernel-cache", tmp);
                return 1;
            }
        }
    }
#else
    {
        const char *xdg = getenv("XDG_CACHE_HOME");
        const char *home = getenv("HOME");
        if (xdg && xdg[0]) {
            snprintf(buf, bufn, "%s/acuda/kernel-cache", xdg);
            return 1;
        }
        if (home && home[0]) {
            snprintf(buf, bufn, "%s/.cache/acuda/kernel-cache", home);
            return 1;
        }
        snprintf(buf, bufn, "/tmp/acuda-kernel-cache");
        return 1;
    }
#endif
    return 0;
}

static void acuda_cache_mkdir(const char *dir) {
    char tmp[1024];
    size_t n;
    snprintf(tmp, sizeof(tmp), "%s", dir);
    n = strlen(tmp);
    for (size_t i = 2; i < n; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            char c = tmp[i];
            tmp[i] = 0;
#if defined(_WIN32)
            _mkdir(tmp);
#else
            mkdir(tmp, 0700);
#endif
            tmp[i] = c;
        }
    }
#if defined(_WIN32)
    _mkdir(tmp);
#else
    mkdir(tmp, 0700);
#endif
}

/* Default namespace for the kernel cache when ACUDA_CACHE_TAG is unset.
 * Derived from this module's PE link timestamp: stable for every run of
 * one build, but automatically different after a rebuild, so a new
 * lifter never silently reuses kernels (or .failed markers) produced by
 * an older one.  ACUDA_CACHE_TAG overrides it. */
static const char *acuda_cache_default_tag(void) {
    static char tag[64];
    if (!tag[0]) {
        unsigned long stamp = 0;
#if defined(_WIN32)
        HMODULE self = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)&acuda_cache_default_tag, &self) &&
            self) {
            const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)self;
            if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
                const IMAGE_NT_HEADERS *nt =
                    (const IMAGE_NT_HEADERS *)((const char *)self +
                                               dos->e_lfanew);
                if (nt->Signature == IMAGE_NT_SIGNATURE)
                    stamp = nt->FileHeader.TimeDateStamp;
            }
        }
#endif
        snprintf(tag, sizeof(tag), "acuda-kernel-cache-%08lx", stamp);
    }
    return tag;
}

static uint64_t acuda_cache_hash(const char *ptx, const char *entry) {
    uint64_t h = 1469598103934665603ull;
    /* The tag is the cache namespace: ACUDA_CACHE_TAG wins when set
     * (a build id, a timestamp, ...); otherwise use the build stamp. */
    const char *tag = getenv("ACUDA_CACHE_TAG");
    if (!tag || !*tag)
        tag = acuda_cache_default_tag();
    const unsigned char *s;
    for (s = (const unsigned char *)tag; *s; s++) {
        h ^= *s;
        h *= 1099511628211ull;
    }
    if (ptx)
        for (s = (const unsigned char *)ptx; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    h ^= 0xffu;
    h *= 1099511628211ull;
    if (entry)
        for (s = (const unsigned char *)entry; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    h ^= 0xfeu;
    h *= 1099511628211ull;
    {
        const char *emu = getenv("ACUDA_EMU_F64");
        if (emu)
            for (s = (const unsigned char *)emu; *s; s++) {
                h ^= *s;
                h *= 1099511628211ull;
            }
    }
    h ^= 0xfdu;
    h *= 1099511628211ull;
    {
        char target[16];
        reported_sm_target(target, sizeof(target));
        for (s = (const unsigned char *)target; *s; s++) {
            h ^= *s;
            h *= 1099511628211ull;
        }
    }
    return h;
}

static acuda_status_t acuda_cache_load(const char *ptx, const char *entry,
                                       acuda_kernel_t **out) {
    char dir[1024], path[1200], tpath[1200];
    uint64_t h;
    FILE *src, *dst;
    char buf[65536];
    size_t r;
    static unsigned long long serial;
    acuda_status_t st;
    *out = NULL;
    if (!ptx || !entry || getenv("ACUDA_NO_CACHE"))
        return ACUDA_ERROR_INVALID_VALUE;
    if (!acuda_cache_dir(dir, sizeof(dir)))
        return ACUDA_ERROR_INVALID_VALUE;
    h = acuda_cache_hash(ptx, entry);
    snprintf(path, sizeof(path), "%s%c%016llx.akb", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
    src = fopen(path, "rb");
    if (!src)
        return ACUDA_ERROR_INVALID_VALUE;
    fclose(src);
#if defined(_WIN32)
    {
        char tmp[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp);
        snprintf(tpath, sizeof(tpath), "%sacuda_cache_%lu_%llu.akb", tmp,
                 (unsigned long)GetCurrentProcessId(), ++serial);
    }
#else
    snprintf(tpath, sizeof(tpath), "/tmp/acuda_cache_%ld_%llu.akb",
             (long)getpid(), ++serial);
#endif
    src = fopen(path, "rb");
    dst = fopen(tpath, "wb");
    if (!src || !dst) {
        if (src)
            fclose(src);
        if (dst)
            fclose(dst);
        remove(tpath);
        return ACUDA_ERROR_INVALID_VALUE;
    }
    while ((r = fread(buf, 1, sizeof(buf), src)) > 0) {
        if (fwrite(buf, 1, r, dst) != r)
            break;
    }
    fclose(src);
    fclose(dst);
    st = acuda_kernel_deserialize_file(tpath, out);
    if (st != ACUDA_SUCCESS)
        remove(tpath);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache load %016llx (%s) st=%d\n",
                (unsigned long long)h, entry, (int)st);
    return st;
}

static void acuda_cache_fail_path(const char *dir, uint64_t h, char *out,
                                  size_t outlen) {
    snprintf(out, outlen, "%s%c%016llx.failed", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
}

/* Tag-scoped negative cache: deterministic compile failures (unsupported
 * SASS, missing BDA types, ...) are re-attempted on every run, each
 * paying a full lift+compile roundtrip.  A .failed marker skips them.
 * The marker key includes the cache tag (via acuda_cache_hash), so a
 * compiler fix automatically re-attempts.  OOM/resource failures are
 * transient and never cached. */
static int acuda_cache_is_transient(const char *err) {
    static const char *words[] = {"oom", "out of memory", "out_of_memory",
                                  "no memory", "alloc fail", "resource"};
    if (!err || !*err)
        return 1;
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++)
        if (strstr(err, words[i]) != NULL)
            return 1;
    {
        // Case-insensitive OOM check without extra deps.
        char lower[128];
        size_t n = strlen(err);
        if (n >= sizeof(lower))
            n = sizeof(lower) - 1;
        for (size_t i = 0; i < n; i++) {
            char ch = err[i];
            lower[i] = (char)((ch >= 'A' && ch <= 'Z') ? ch + 32 : ch);
        }
        lower[n] = 0;
        if (strstr(lower, "oom") != NULL ||
            strstr(lower, "out of memory") != NULL)
            return 1;
    }
    return 0;
}

static int acuda_cache_failed(const char *dir, uint64_t h, int *code,
                              char *err, size_t errlen) {
    char path[1200];
    FILE *f;
    if (!dir || getenv("ACUDA_NO_CACHE"))
        return 0;
    acuda_cache_fail_path(dir, h, path, sizeof(path));
    f = fopen(path, "rb");
    if (!f)
        return 0;
    {
        char line[32];
        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 0;
        }
        if (code)
            *code = atoi(line);
    }
    if (err && errlen > 0) {
        size_t n = fread(err, 1, errlen - 1, f);
        err[n] = 0;
    }
    fclose(f);
    return 1;
}

static void acuda_cache_mark_failed(const char *dir, uint64_t h, int code,
                                    const char *err) {
    char path[1200];
    FILE *f;
    if (!dir || getenv("ACUDA_NO_CACHE") || !err || !*err)
        return;
    if (acuda_cache_is_transient(err))
        return;
    acuda_cache_fail_path(dir, h, path, sizeof(path));
    f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "%d\n%s", code, err);
    fclose(f);
}

static void acuda_cache_store(const char *ptx, const char *entry,
                              const acuda_kernel_t *k) {
    char dir[1024], path[1200], tpath[1200];
    uint64_t h;
    uint8_t *blob = NULL;
    size_t bsz = 0;
    static unsigned long long serial;
    FILE *f;
    if (!ptx || !entry || !k || getenv("ACUDA_NO_CACHE"))
        return;
    if (!acuda_cache_dir(dir, sizeof(dir)))
        return;
    acuda_cache_mkdir(dir);
    h = acuda_cache_hash(ptx, entry);
    snprintf(path, sizeof(path), "%s%c%016llx.akb", dir, ACUDA_CACHE_SEP,
             (unsigned long long)h);
    if (acuda_kernel_serialize(k, &blob, &bsz) != ACUDA_SUCCESS || !blob)
        return;
#if defined(_WIN32)
    {
        char tmp[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp);
        snprintf(tpath, sizeof(tpath), "%sacuda_cache_%lu_%llu.tmp", tmp,
                 (unsigned long)GetCurrentProcessId(), ++serial);
    }
#else
    snprintf(tpath, sizeof(tpath), "/tmp/acuda_cache_%ld_%llu.tmp",
             (long)getpid(), ++serial);
#endif
    f = fopen(tpath, "wb");
    if (f) {
        size_t wr = fwrite(blob, 1, bsz, f);
        fclose(f);
        if (wr == bsz) {
            remove(path);
            if (rename(tpath, path) != 0)
                remove(tpath);
        } else {
            remove(tpath);
        }
    }
    free(blob);
    ACUDA_DEBUG_PRINT("acuda-trace .. cache store %016llx (%s) %zu bytes\n",
                (unsigned long long)h, entry, bsz);
}

static CUresult attach_sass_image(CUmodule module, const unsigned char *data,
                                  size_t size) {
    if (!module || !data || size == 0)
        return CUDA_ERROR_INVALID_VALUE;
    unsigned char *copy = malloc(size);
    if (!copy)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(copy, data, size);
    free(module->image);
    module->image = copy;
    module->image_size = size;
    return CUDA_SUCCESS;
}

static CUresult module_from_sass_image(CUmodule *module,
                                       const unsigned char *data,
                                       size_t size) {
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    CUmodule m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    m->ctx = c;
    r = attach_sass_image(m, data, size);
    if (r != CUDA_SUCCESS) {
        free(m);
        return r;
    }
    *module = m;
    return CUDA_SUCCESS;
}

/* Select the cubin/fatbin backend (shared with rtxdiasm_guard/cudart):
 *   0 (default) - C fatbin parser when the image carries PTX, otherwise
 *                 lift SASS with rtxdisam (C#).
 *   1           - C fatbin parser only; never call the SASS lifter.
 *   2           - rtxdisam (C#) only; skip the C fatbin parser.
 * Unset, empty or invalid values mean 0. */
static int cubin_backend(void) {
    const char *s = getenv("ACUDA_CUBIN_BACKEND");
    if (s && *s) {
        long v = strtol(s, NULL, 0);
        if (v == 1 || v == 2)
            return (int)v;
    }
    return 0;
}

static CUresult load_sass_image(CUmodule *module, const unsigned char *data,
                                size_t size) {
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    if (cubin_backend() == 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. backend=1 (C parser only): "
                    "refusing the rtxdisam SASS lift\n");
        return CUDA_ERROR_INVALID_IMAGE;
    }
    char err[1024] = {0};
    char target[16];
    reported_sm_target(target, sizeof(target));
    char *ptx = NULL;
    acuda_rtxdisam_status_t status =
        acuda_rtxdisam_cubin_to_ptx_status_for_target(
            data, size, NULL, target, &ptx, err, sizeof(err));
    if (status == ACUDA_RTXDISAM_ENTRY_REQUIRED)
        return module_from_sass_image(module, data, size);
    if (status != ACUDA_RTXDISAM_OK || !ptx) {
        if (err[0])
            fprintf(stderr, "ACuda: %s\n", err);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    CUmodule m = NULL;
    CUresult r = cuModuleLoadData(&m, ptx);
    free(ptx);
    if (r != CUDA_SUCCESS)
        return r;
    // Keep the original image: a fatbin may contain several single-entry
    // cubins, so a later cuModuleGetFunction(name) may need to lift another
    // member even though the entry-less lift selected the first one.
    r = attach_sass_image(m, data, size);
    if (r != CUDA_SUCCESS) {
        cuModuleUnload(m);
        return r;
    }
    *module = m;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuModuleLoadDataEx\n");
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (getenv("ACUDA_TRACE") && image)
        fprintf(stderr, "acuda-trace .. cuModuleLoadDataEx magic=%08x\n",
                rd32le((const unsigned char *)image));
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!image)
        return CUDA_ERROR_INVALID_VALUE;
    const char *text = (const char *)image;
    const unsigned char *bytes = (const unsigned char *)image;
    // A __fatBinC_Wrapper_t is what cuModuleLoadData receives from old
    // PhysX/CUDA 4.x runtimes.  It points at the real 0xBA55ED50 blob.
    if (rd32le(bytes) == 0x466243B1u)
        return cuModuleLoadFatBinary(module, image);
    // A raw ELF cubin or fatbin blob has no PTX for ACuda to import. Ask the
    // optional rtxdisam native library to lift its SASS first.
    if ((bytes[0] == 0x7f && strncmp(text + 1, "ELF", 3) == 0) ||
        rd32le(bytes) == 0xBA55ED50u) {
        size_t size = acuda_image_size(bytes);
        if (!size)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_sass_image(module, bytes, size);
    }
    if (!looks_like_ptx(text))
        return CUDA_ERROR_INVALID_IMAGE;
    size_t tlen = strlen(text);
    int ptx_oom = 0;
    if (tlen > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        /* Large module: skip the eager whole-text parse. cuModuleGetFunction
         * extracts and compiles only the requested entry, which validates it
         * there; a full parse here just burns 32-bit address space and can
         * fail/crash before any function is ever needed. */
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: skip eager validate (%zu bytes)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
        // The per-entry 32-bit parser is also likely to OOM on a module this
        // large; use the 64-bit compile host directly.
        ptx_oom = 1;
#endif
    } else {
        char err[1024];
        if (acuda_ptx_validate(text, err, sizeof(err)) != ACUDA_SUCCESS) {
            /* 32-bit large modules: full parse can OOM during validation; store
             * text and let cuModuleGetFunction/compile handle per-function. */
            if (!(err[0] && (strstr(err, "out of memory") || strstr(err, "(no detail)"))))
                return CUDA_ERROR_INVALID_IMAGE;
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx: validate OOM, storing ptx text (%zu)\n", tlen);
#if defined(_WIN32) && !defined(_WIN64)
            // The same 32-bit parser will OOM again for every entry in this
            // module; skip straight to the 64-bit compile host.
            ptx_oom = 1;
#endif
        }
    }
    CUmodule m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    m->ctx = c;
    m->ptx_oom = ptx_oom;
    m->ptx = malloc(strlen(text) + 1);
    if (!m->ptx) {
        free(m);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    strcpy(m->ptx, text);
    *module = m;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadDataEx PTX ok mod=%p len=%zu ptx_oom=%d\n",
                      (void *)m, tlen, ptx_oom);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
    return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

// ---- NVIDIA fatbin container parsing ----
//
// Core PTX member extraction/decompression (raw/LZ4/ZSTD) is shared via
// rtxdiasm-fatbin-parse (rtxdiasm_fatbin_extract_first_ptx) so acuda and
// cudart use identical fatbin decode logic. Driver-specific wrapper handling,
// validation/OOM acceptance, and SASS fallback remain here.

static uint32_t rd32le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64le(const unsigned char *p) {
    uint64_t lo = rd32le(p), hi = rd32le(p + 4);
    return lo | (hi << 32);
}

static char *fatbin_first_ptx(const unsigned char *blob, size_t cap) {
    char *ptx = NULL;
    size_t ptx_len = 0;
    if (rtxdiasm_fatbin_extract_first_ptx(blob, cap, &ptx, &ptx_len) != RTXDIASM_FATBIN_OK)
        return NULL;
    /* Preserve driver behavior: accept large PTX without eager full validate,
     * and tolerate 32-bit validation OOM/internal failures (compile host revalidates). */
    if (ptx_len > ACUDA_PTX_EAGER_VALIDATE_MAX) {
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx large, accept without eager validate (%zu bytes)\n", ptx_len);
        return ptx;
    }
    {
        char err[256];
        err[0] = '\0';
        if (acuda_ptx_validate(ptx, err, sizeof(err)) == ACUDA_SUCCESS)
            return ptx;
        if (!err[0] || strstr(err, "out of memory") || strstr(err, "(no detail)")) {
            ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate OOM/internal, accepting as ptx (%zu bytes)\n", ptx_len);
            return ptx;
        }
        ACUDA_DEBUG_PRINT("acuda-trace .. extracted ptx validate failed: %s\n", err);
        {
            const char *dump_dir = getenv("ACUDA_DUMP_PTX_DIR");
            if (dump_dir && *dump_dir) {
                static unsigned long long dump_serial;
                char path[1024];
                snprintf(path, sizeof(path), "%s/extracted_invalid_%llu.ptx",
                         dump_dir, ++dump_serial);
                FILE *dfp = fopen(path, "wb");
                if (dfp) {
                    fwrite(ptx, 1, ptx_len, dfp);
                    fclose(dfp);
                    ACUDA_DEBUG_PRINT("acuda-trace .. dumped invalid extracted ptx to %s\n", path);
                }
            }
        }
    }
    free(ptx);
    return NULL;
}

static CUresult load_fatbin_member(CUmodule *module,
                                   const unsigned char *data, size_t size) {
    const char *dump = getenv("ACUDA_DUMP_FATBIN");
    if (dump && *dump) {
        // Numbered dumps: PhysX loads several fatbins per run and the
        // interesting one is rarely the last.
        static unsigned long dump_serial;
        char path[1024];
        size_t dl = strlen(dump);
        if (dl > 7 && strcmp(dump + dl - 7, ".fatbin") == 0)
            snprintf(path, sizeof(path), "%.*s_%lu.fatbin", (int)(dl - 7),
                     dump, ++dump_serial);
        else
            snprintf(path, sizeof(path), "%s_%lu", dump, ++dump_serial);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(data, 1, size, f);
            fclose(f);
            ACUDA_DEBUG_PRINT("acuda-trace .. dumped fatbin %zu bytes\n", size);
        }
    }
    char *ptx = cubin_backend() == 2 ? NULL : fatbin_first_ptx(data, size);
    if (!ptx)
        return load_sass_image(module, data, size);
    CUresult r = cuModuleLoadData(module, ptx);
    free(ptx);
    if (r == CUDA_SUCCESS && module && *module)
        attach_sass_image(*module, data, size);
    ACUDA_DEBUG_PRINT("acuda-trace .. load_fatbin_member -> %d\n", (int)r);
    return r;
}

CUresult ACUDA_CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleLoadFatBinary\n");
    if (!module)
        return CUDA_ERROR_INVALID_VALUE;
    *module = NULL;
    if (!fatCubin)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    // The argument is the __fatBinC_Wrapper_t control struct, not the
    // raw blob (verified field-by-field against real torch fatbins).
    const unsigned char *w = (const unsigned char *)fatCubin;
    if (rd32le(w) != 0x466243B1u)
        return CUDA_ERROR_INVALID_IMAGE;
    uint32_t ver = rd32le(w + 4);
    ACUDA_DEBUG_PRINT("acuda-trace .. fatbin wrapper ver=%u\n", ver);
    if (ver != 1 && ver != 2)
        return CUDA_ERROR_INVALID_IMAGE;
    if (ver == 1) {
        const unsigned char *data = NULL;
        memcpy(&data, w + 8, sizeof(data));
        if (!data)
            return CUDA_ERROR_INVALID_IMAGE;
        // Bound the walk by the blob's own declared size (+ header).
        if (rd32le(data) != 0xBA55ED50u)
            return CUDA_ERROR_INVALID_IMAGE;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            return CUDA_ERROR_INVALID_IMAGE;
        return load_fatbin_member(module, data,
                                  16 + (size_t)files_size);
    }
    // Version 2: NULL-terminated array of wrapper pointers (submodules).
    const void *const *arr = NULL;
    memcpy(&arr, w + 16, sizeof(arr));
    if (!arr)
        return CUDA_ERROR_INVALID_IMAGE;
    for (size_t i = 0; i < 4096; i++) {
        const unsigned char *sub = NULL;
        memcpy(&sub, &arr[i], sizeof(sub));
        if (!sub)
            break;
        if (rd32le(sub) != 0x466243B1u || rd32le(sub + 4) != 1)
            continue;
        const unsigned char *data = NULL;
        memcpy(&data, sub + 8, sizeof(data));
        if (!data || rd32le(data) != 0xBA55ED50u)
            continue;
        uint64_t files_size = rd64le(data + 8);
        if (files_size > (uint64_t)512 << 20)
            continue;
        CUresult r = load_fatbin_member(module, data,
                                        16 + (size_t)files_size);
        if (r == CUDA_SUCCESS)
            return r;
    }
    return CUDA_ERROR_INVALID_IMAGE;
}

CUresult ACUDA_CUDAAPI cuModuleUnload(CUmodule module) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuModuleUnload\n");
    if (!module)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    for (size_t i = 0; i < module->nfuncs; i++)
        acuda_kernel_free(module->funcs[i].k);
    for (size_t i = 0; i < module->ntexrefs; i++) {
        struct CUtexref_st *tr = module->texrefs[i];
        if (tr) {
            if (tr->obj)
                cuTexObjectDestroy(tr->obj);
            free(tr);
        }
    }
    for (size_t i = 0; i < module->nglobals; i++)
        if (module->globals[i].dptr)
            cuMemFree_v2(module->globals[i].dptr);
    free(module->globals);
    free(module->texrefs);
    free(module->funcs);
    free(module->ptx);
    free(module->image);
    free(module);
    return CUDA_SUCCESS;
}

static char *replace_text(const char *src, const char *from,
                         const char *to) {
    size_t fl = strlen(from), tl = strlen(to), n = 0;
    const char *p = src, *q;
    while ((q = strstr(p, from)) != NULL) {
        n++;
        p = q + fl;
    }
    size_t len = strlen(src) + n * (tl - fl) + 1;
    char *out = malloc(len);
    if (!out)
        return NULL;
    char *d = out;
    p = src;
    while ((q = strstr(p, from)) != NULL) {
        size_t k = (size_t)(q - p);
        memcpy(d, p, k);
        d += k;
        memcpy(d, to, tl);
        d += tl;
        p = q + fl;
    }
    strcpy(d, p);
    return out;
}

#if defined(_WIN32)
static const char *acuda_compile_host_exe(char *buf, size_t bufn) {
    const char *e = getenv("ACUDA_COMPILE_HOST");
    if (e && e[0]) return e;
    /* next to RTXDISAM_HOST if set */
    const char *rh = getenv("RTXDISAM_HOST");
    if (rh && rh[0]) {
        snprintf(buf, bufn, "%s", rh);
        char *sl = strrchr(buf, '\\'); char *sl2 = strrchr(buf, '/');
        char *s = sl > sl2 ? sl : sl2;
        if (s) {
            strcpy(s + 1, "rtxdiasm_ptx-to-spirv_compiler.exe");
            if (GetFileAttributesA(buf) != INVALID_FILE_ATTRIBUTES) return buf;
            strcpy(s + 1, "acuda_compile_host.exe");
            if (GetFileAttributesA(buf) != INVALID_FILE_ATTRIBUTES) return buf;
        }
    }
    if (GetFileAttributesA("rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES) return "rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES) return "acuda_compile_host.exe";
    if (GetFileAttributesA("out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES) return "out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("out\\win-x64\\acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES) return "out\\win-x64\\acuda_compile_host.exe";
    if (GetFileAttributesA("C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe") != INVALID_FILE_ATTRIBUTES)
        return "C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\rtxdiasm_ptx-to-spirv_compiler.exe";
    if (GetFileAttributesA("C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\acuda_compile_host.exe") != INVALID_FILE_ATTRIBUTES)
        return "C:\\Users\\Magnus\\source\\repos\\ConsoleApp2\\out\\win-x64\\acuda_compile_host.exe";
    return NULL;
}

static size_t acuda_append_caps_args(char *dst, size_t dstsz, const acuda_compiler_caps_t *caps) {
    if (!dst || dstsz == 0) return 0;
    dst[0] = '\0';
    acuda_compiler_caps_t def;
    if (!caps) {
        acuda_compiler_caps_init(&def);
        caps = &def;
    }
    return (size_t)snprintf(dst, dstsz,
        " --cap.shader_fma=%d --cap.shader_fma_f64=%d --cap.soft_fma=%d"
        " --cap.f16_native=%d --cap.coopmat_f16=%d --cap.coopmat_bf16=%d"
        " --cap.mixed_fdot=%d --cap.dot_product=%d --cap.dot_signed=%d"
        " --cap.dot_unsigned=%d --cap.dot_mixed=%d --cap.zero_init_wg=%d"
        " --cap.ssbo8=%d --cap.ssbo16=%d --cap.fc32=%d --cap.fc64=%d"
        " --cap.vmm=%d --cap.bda=%d --cap.atomic_float=%u"
        " --cap.int64_image=%d --cap.surf_alias_ok=%d",
        caps->shader_fma, caps->shader_fma_f64, caps->soft_fma,
        caps->f16_native, caps->coopmat_f16, caps->coopmat_bf16,
        caps->mixed_fdot, caps->dot_product, caps->dot_signed,
        caps->dot_unsigned, caps->dot_mixed, caps->zero_init_wg,
        caps->ssbo8, caps->ssbo16, caps->fc32, caps->fc64,
        caps->vmm, caps->bda, caps->atomic_float,
        caps->int64_image, caps->surf_alias_ok);
}

static acuda_status_t acuda_compile_via_host(const char *ptx, const char *entry,
                                             const acuda_compiler_caps_t *caps,
                                             acuda_kernel_t **out, char *err, size_t errlen) {
    *out = NULL;
    char exebuf[MAX_PATH];
    const char *exe = acuda_compile_host_exe(exebuf, sizeof(exebuf));
    if (!exe) { if (err && errlen) snprintf(err, errlen, "rtxdiasm_ptx-to-spirv_compiler executable not found"); return ACUDA_ERROR_NOT_SUPPORTED; }
    char tmp[MAX_PATH], inpath[MAX_PATH], outpath[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    snprintf(inpath, sizeof(inpath), "%sacuda_ch_%lu_%lu.ptx", tmp, GetCurrentProcessId(), GetTickCount());
    snprintf(outpath, sizeof(outpath), "%sacuda_ch_%lu_%lu.akb", tmp, GetCurrentProcessId(), GetTickCount()+1);
    FILE *fi = fopen(inpath, "wb");
    if (!fi) return ACUDA_ERROR_NOT_SUPPORTED;
    fwrite(ptx, 1, strlen(ptx), fi); fclose(fi);
    char capsargs[1024];
    acuda_append_caps_args(capsargs, sizeof(capsargs), caps);
    char cmd[8192];
    if (entry && *entry)
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\" \"%s\"%s", exe, inpath, outpath, entry, capsargs);
    else
        snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\" -%s", exe, inpath, outpath, capsargs);
    STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof(si)); si.cb=sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    SECURITY_ATTRIBUTES sa; memset(&sa,0,sizeof(sa)); sa.nLength=sizeof(sa); sa.bInheritHandle=TRUE;
    char logpath[MAX_PATH]; snprintf(logpath,sizeof(logpath),"%sacuda_ch_%lu.log",tmp,GetCurrentProcessId());
    HANDLE lh = CreateFileA(logpath, GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (lh != INVALID_HANDLE_VALUE) { si.hStdOutput=lh; si.hStdError=lh; si.hStdInput=GetStdHandle(STD_INPUT_HANDLE); }
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    if (lh != INVALID_HANDLE_VALUE) CloseHandle(lh);
    if (!ok) { DeleteFileA(inpath); if(err&&errlen) snprintf(err,errlen,"cannot start compiler host executable"); return ACUDA_ERROR_NOT_SUPPORTED; }
    WaitForSingleObject(pi.hProcess, 120000);
    DWORD ec=1; GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    DeleteFileA(inpath);
    ACUDA_DEBUG_PRINT("acuda-trace .. compile_host ec=%lu in=%s out=%s exists=%d\n",
                (unsigned long)ec, inpath, outpath, (int)(GetFileAttributesA(outpath)!=INVALID_FILE_ATTRIBUTES));
    if (ec != 0) {
        if (err && errlen) {
            FILE *lf=fopen(logpath,"rb");
            if (lf) { char msg[512]={0}; size_t rn=fread(msg,1,sizeof(msg)-1,lf); fclose(lf); msg[rn]=0;
                /* last non-empty line */
                char *last=msg; for(char *q=msg;*q;q++) if(*q=='\n'&&q[1]) last=q+1;
                snprintf(err,errlen,"compiler host exit %lu: %.400s",(unsigned long)ec,last);
            } else snprintf(err,errlen,"compiler host exit %lu",(unsigned long)ec);
        }
        DeleteFileA(logpath); DeleteFileA(outpath);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    DeleteFileA(logpath);
    acuda_status_t st = acuda_kernel_deserialize_file(outpath, out);
    ACUDA_DEBUG_PRINT("acuda-trace .. compile_host deserialize_file=%d\n", (int)st);
    /* deserialize_file removes AKB when not used as SPIR-V backing; if used as
     * backing, kernel owns the file and will remove it on free. */
    if (st != ACUDA_SUCCESS) {
        const char *dump_dir = getenv("ACUDA_DUMP_PTX_DIR");
        if (dump_dir && *dump_dir) {
            static unsigned long dump_akb_serial;
            char akbpath[1024];
            snprintf(akbpath, sizeof(akbpath), "%s/compile_host_fail_%lu.akb",
                     dump_dir, ++dump_akb_serial);
            if (!MoveFileA(outpath, akbpath))
                CopyFileA(outpath, akbpath, FALSE);
            ACUDA_DEBUG_PRINT("acuda-trace .. kept failed compile_host akb -> %s\n", akbpath);
        }
        DeleteFileA(outpath);
    }
    if (st != ACUDA_SUCCESS && err && errlen) snprintf(err,errlen,"compile_host deserialize_file status %d", (int)st);
    return st;
}
#endif

CUresult ACUDA_CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule module,
                             const char *name) {
    if (!hfunc)
        return CUDA_ERROR_INVALID_VALUE;
    *hfunc = NULL;
    if (!module || !name)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    ACUDA_DEBUG_PRINT("acuda-trace >> cuModuleGetFunction(%s)\n", name);
    for (size_t i = 0; i < module->nfuncs; i++) {
        if (strcmp(module->funcs[i].name, name) == 0) {
            CUfunction f = malloc(sizeof(*f));
            if (!f)
                return CUDA_ERROR_OUT_OF_MEMORY;
            f->magic = ACUDA_HANDLE_FUNC;
            f->mod = module;
            f->k = module->funcs[i].k;
            snprintf(f->name, sizeof(f->name), "%s", name);
            f->maxdyn_set = 0;
            f->maxdyn = 0;
            *hfunc = f;
            return CUDA_SUCCESS;
        }
    }
    char err[1024] = {0};
    acuda_kernel_t *k = NULL;
    // Auto-fallback: f64 code on a device without shaderFloat64 compiles
    // emulated (explicit ACUDA_EMU_F64, in either direction, always wins).
    int emu_auto = 0;
    if (!getenv("ACUDA_EMU_F64") && need_actx() == CUDA_SUCCESS && g_actx &&
        !g_actx->f64_ok) {
        setenv("ACUDA_EMU_F64", "1", 1);
        emu_auto = 1;
    }
    acuda_status_t cst = ACUDA_ERROR_INVALID_VALUE;
    // Tag-scoped negative cache: skip kernels whose compilation already
    // failed deterministically (unsupported SASS, missing BDA types...).
    // The key covers the entry and target; the tag is folded in by
    // acuda_cache_hash, so a compiler fix automatically re-attempts.
    char fail_key[2048];
    char fail_dir[1024];
    uint64_t fail_hash = 0;
    int have_fail_dir = 0;
    int skip_attempts = 0;
    {
        char tgt[16];
        reported_sm_target(tgt, sizeof(tgt));
        snprintf(fail_key, sizeof(fail_key), "%s|%s", name ? name : "", tgt);
        have_fail_dir = acuda_cache_dir(fail_dir, sizeof(fail_dir));
        if (have_fail_dir)
            fail_hash = acuda_cache_hash(NULL, fail_key);
    }
    if (have_fail_dir) {
        int fcode = 0;
        if (acuda_cache_failed(fail_dir, fail_hash, &fcode, err,
                               sizeof(err))) {
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): cached failure, skipping compile\n",
                        name);
            cst = (acuda_status_t)fcode;
            if (cst == ACUDA_SUCCESS)
                cst = ACUDA_ERROR_NOT_SUPPORTED;
            skip_attempts = 1;
        }
    }
    if (!skip_attempts && module->ptx) {
        acuda_kernel_t *ck = NULL;
        if (acuda_cache_load(module->ptx, name, &ck) == ACUDA_SUCCESS && ck) {
            k = ck;
            cst = ACUDA_SUCCESS;
            ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): cache hit\n", name);
        } else {
            if (!module->ptx_oom) {
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying ptx compile\n", name);
                cst = acuda_compile_ptx(module->ptx, name, &k, err, sizeof(err), acuda_context_get_compiler_caps(g_actx));
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): ptx compile cst=%d err=%s\n", name, (int)cst, err[0]?err:"(none)");
            } else if (getenv("ACUDA_TRACE")) {
                fprintf(stderr, "acuda-trace .. cuModuleGetFunction(%s): skip 32-bit ptx compile (module parse OOMed)\n", name);
            }
#if defined(_WIN32)
            if (cst != ACUDA_SUCCESS && sizeof(void *) <= 4) {
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying 64-bit compile_host fallback\n", name);
                acuda_kernel_t *hk = NULL; char herr[1024] = {0};
                acuda_status_t hcst = acuda_compile_via_host(module->ptx, name, acuda_context_get_compiler_caps(g_actx), &hk, herr, sizeof(herr));
                ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): compile_host cst=%d err=%s\n", name, (int)hcst, herr[0]?herr:"(none)");
                if (hcst == ACUDA_SUCCESS && hk) { k = hk; cst = ACUDA_SUCCESS; err[0] = 0; }
                else if (herr[0] && !err[0]) snprintf(err, sizeof(err), "%s", herr);
            }
#endif
            if (cst == ACUDA_SUCCESS && k)
                acuda_cache_store(module->ptx, name, k);
        }
    }
    if (!skip_attempts && cst != ACUDA_SUCCESS && module->image &&
        module->image_size && cubin_backend() == 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): "
                    "backend=1 (C parser only): no rtxdisam lift\n", name);
    }
    if (!skip_attempts && cst != ACUDA_SUCCESS && module->image &&
        module->image_size && cubin_backend() != 1) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuModuleGetFunction(%s): trying rtxdisam fallback\n", name);
        // The entry-less lift either could not choose a section (multi-entry
        // cubin) or chose the first member of a multi-cubin fatbin. Ask the
        // bridge for this specific entry and prefer the device's reported SM.
        char target[16];
        reported_sm_target(target, sizeof(target));
        char *lifted = NULL;
        if (acuda_rtxdisam_cubin_to_ptx_status_for_target(
                module->image, module->image_size, name, target, &lifted,
                err, sizeof(err)) == ACUDA_RTXDISAM_OK &&
            lifted) {
            // The PTX parser caps identifiers at 256 bytes; torch's
            // mangled kernel names are longer.  Compile under a short
            // synthetic entry and keep the public name in module->funcs.
            const char *compileName = name;
            char shortName[64];
            // Always compile under a short synthetic entry: mangled torch
            // names and the lifter's L__text__<name> labels can exceed the
            // parser's 256-byte identifier cap.
            static unsigned long long serial;
            snprintf(shortName, sizeof(shortName), "__acuda_sass_%llu",
                     ++serial);
            if (getenv("ACUDA_DUMP_PTX_DIR")) {
                static unsigned long long dump_serial;
                char path[512];
                snprintf(path, sizeof(path), "%s/%llu.ptx",
                         getenv("ACUDA_DUMP_PTX_DIR"), ++dump_serial);
                FILE *dfp = fopen(path, "wb");
                if (dfp) {
                    fwrite(lifted, 1, strlen(lifted), dfp);
                    fclose(dfp);
                }
                fprintf(stderr, "acuda-ptxdump %llu target=%s name=%s\n",
                        dump_serial, target, name ? name : "(null)");
            }
            char *renamed = replace_text(lifted, name, shortName);
            if (!renamed) {
                free(lifted);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            compileName = shortName;
            cst = acuda_compile_ptx(renamed, compileName, &k, err,
                                    sizeof(err), acuda_context_get_compiler_caps(g_actx));
            free(renamed);
            free(lifted);
        }
    }
    if (emu_auto)
        unsetenv("ACUDA_EMU_F64");
    if (cst != ACUDA_SUCCESS) {
        if (err[0])
            fprintf(stderr, "ACuda: %s\n", err);
        if (have_fail_dir && !skip_attempts)
            acuda_cache_mark_failed(fail_dir, fail_hash, (int)cst, err);
        // Diagnostic: keep the exact PTX that failed, whether it came
        // from the module image or from an entry-specific re-lift.
        const char *dump_dir = getenv("ACUDA_DUMP_PTX_DIR");
        const char *failed_ptx = (module && module->ptx) ? module->ptx : NULL;
        if (dump_dir && *dump_dir && failed_ptx) {
            static unsigned long dump_fail_serial;
            char path[1024];
            snprintf(path, sizeof(path), "%s/fail_%lu.ptx", dump_dir,
                     ++dump_fail_serial);
            FILE *dfp = fopen(path, "wb");
            if (dfp) {
                fwrite(failed_ptx, 1, strlen(failed_ptx), dfp);
                fclose(dfp);
                fprintf(stderr, "acuda-ptxdump fail_%lu name=%s\n",
                        dump_fail_serial, name ? name : "(null)");
            }
        }
        ACUDA_DEBUG_PRINT("acuda-trace !! cuModuleGetFunction(%s): "
                    "compile status %d err=%s\n", name, (int)cst, err);
        return CUDA_ERROR_NOT_FOUND;
    }
    if (module->nfuncs == module->fcap) {
        size_t ncap = module->fcap ? module->fcap * 2 : 4;
        void *np = malloc(ncap * sizeof(*module->funcs));
        if (!np) {
            acuda_kernel_free(k);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(np, module->funcs, module->nfuncs * sizeof(*module->funcs));
        free(module->funcs);
        module->funcs = np;
        module->fcap = ncap;
    }
    strncpy(module->funcs[module->nfuncs].name, name,
            sizeof(module->funcs[module->nfuncs].name) - 1);
    module->funcs[module->nfuncs].name[sizeof(module->funcs[module->nfuncs].name) - 1] =
        '\0';
    module->funcs[module->nfuncs].k = k;
    module->nfuncs++;
    CUfunction f = malloc(sizeof(*f));
    if (!f)
        return CUDA_ERROR_OUT_OF_MEMORY;
    f->magic = ACUDA_HANDLE_FUNC;
    f->mod = module;
    f->k = k;
    snprintf(f->name, sizeof(f->name), "%s", name);
    f->maxdyn_set = 0;
    f->maxdyn = 0;
    *hfunc = f;
    ACUDA_DEBUG_PRINT("acuda-mfunc f=%p mod=%p k=%p name=%s\n",
                (void *)f, (void *)f->mod, (void *)f->k, f->name);
    return CUDA_SUCCESS;
}

static CUresult handle_as_func(void *handle, CUfunction *out) {
    if (!handle || !out)
        return CUDA_ERROR_INVALID_HANDLE;
    uint32_t magic = *(const uint32_t *)handle;
    if (magic == ACUDA_HANDLE_FUNC) {
        *out = (CUfunction)handle;
        return CUDA_SUCCESS;
    }
    if (magic == ACUDA_HANDLE_KERNEL) {
        struct CUkernel_st *k = (struct CUkernel_st *)handle;
        if (!k->func) {
            CUfunction f = calloc(1, sizeof(*f));
            if (!f)
                return CUDA_ERROR_OUT_OF_MEMORY;
            f->magic = ACUDA_HANDLE_FUNC;
            f->mod = k->mod;
            snprintf(f->name, sizeof(f->name), "%s", k->name);
            k->func = f;
        }
        *out = k->func;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult ACUDA_CUDAAPI cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    if (!pi)
        return CUDA_ERROR_INVALID_VALUE;
    *pi = 0;
    if (!hfunc)
        return CUDA_ERROR_INVALID_HANDLE;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuFuncGetAttribute(%d, %s)\n",
                attrib, hfunc->name);
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)hfunc, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        hfunc = real;
    }
    if (!hfunc->k) {
        // Lazy library kernel: answer the probe without compiling.
        if (attrib == CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            *pi = ACUDA_MAX_BLOCK_X;
        else if (attrib == CU_FUNC_ATTRIBUTE_CACHE_MODE_CA)
            *pi = 1;
        else if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            *pi = hfunc->maxdyn_set ? (int)hfunc->maxdyn : 0;
        else if (attrib == CU_FUNC_ATTRIBUTE_PTX_VERSION ||
                 attrib == CU_FUNC_ATTRIBUTE_BINARY_VERSION)
            // cudart validates a library kernel's PTX/binary version
            // before launching; answering 0 makes it report
            // cudaErrorInvalidDeviceFunction (e.g. for CUB's
            // runtime-API kernels).
            *pi = cc_major() * 10 + cc_minor();
        ACUDA_DEBUG_PRINT("acuda-trace .. cuFuncGetAttribute -> %d\n", *pi);
        return CUDA_SUCCESS;
    }
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (attrib < 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (attrib > CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        // Newer attributes (cluster geometry and the like) are not
        // modelled.  Answer with a benign zero instead of an error: the
        // CUDA 12.8+ runtime caches a failed post-launch attribute query
        // per function, which then makes later
        // cudaFuncGetAttributes/cudaOccupancyMaxPotentialBlockSize calls
        // fail even though the kernel itself is fine.
        *pi = 0;
        return CUDA_SUCCESS;
    }
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(hfunc->k);
    if (!info)
        return CUDA_ERROR_UNKNOWN;
    switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = ACUDA_MAX_BLOCK_X; // any 1..1024 launches (specialized)
        break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: // static .shared total
        *pi = (int)info->shared_bytes;
        break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: // no .const: always 0
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: // no .local: always 0
        *pi = 0;
        break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: // declared registers (estimate)
        *pi = (int)info->num_regs;
        break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: // .version as major*10+minor
        *pi = (int)info->ptx_version;
        break;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION: // .target sm
        *pi = (int)info->sm_version;
        break;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: // always cached, report 1
        *pi = 1;
        break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
        *pi = hfunc->maxdyn_set ? (int)hfunc->maxdyn : 0;
        break;
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuFuncGetAttribute -> %d\n", *pi);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuFuncSetAttribute\n");
    if (!hfunc)
        return CUDA_ERROR_INVALID_HANDLE;
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)hfunc, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        hfunc = real;
    }
    if (!hfunc->k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (attrib < 0)
        return CUDA_ERROR_INVALID_VALUE;
    // Carveout/cluster hints and other newer attributes are advisory on
    // this backend.  Accept and ignore them: the CUDA 12.8+ runtime
    // caches a failure from a launch-time attribute probe and then
    // rejects later cudaFuncGetAttributes/cudaOccupancy queries.
    if (attrib > CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
        return CUDA_SUCCESS;
    // Single-path backend: most attributes are advisory only. Static
    // .shared is real (Workgroup storage, reported via
    // SHARED_SIZE_BYTES); dynamic .shared is sized per launch up to
    // the 64KB total cap (and MAX_DYNAMIC when raised here).
    // Negative values are rejected like the real driver.
    if (value < 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (attrib == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) {
        hfunc->maxdyn_set = 1;
        hfunc->maxdyn = (unsigned)value;
    }
    return CUDA_SUCCESS;
}

// CUDA 12.4 extended launch. Field offsets mirror NVIDIA's
// CUlaunchConfig exactly (verified against cuda.h in the cu126 pip
// wheel: 7x u32 dims+shared, ptr stream, ptr attrs, u32 numAttrs).
// Extended attributes are NOT supported: any attrs -> INVALID_VALUE.
// Plain configs forward to cuLaunchKernel (same 1D/sharedMem=0 rules).
CUresult ACUDA_CUDAAPI cuLaunchKernelEx(const void *config, CUfunction f,
                          void **kernelParams, void **extra) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLaunchKernelEx\n");
    if (!config || !f)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t grid[3], block[3], shm = 0, nattr = 0;
    void *stream = NULL;
    memcpy(grid, (const char *)config + 0, sizeof(grid));
    memcpy(block, (const char *)config + 12, sizeof(block));
    memcpy(&shm, (const char *)config + 24, sizeof(shm));
    memcpy(&stream, (const char *)config + 32, sizeof(stream));
    memcpy(&nattr, (const char *)config + 48, sizeof(nattr));
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (nattr != 0)
        return CUDA_ERROR_INVALID_VALUE; // cluster/etc. unsupported
    return cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1],
                          block[2], shm, (CUstream)stream, kernelParams,
                          extra);
}

static CUresult ensure_func(CUfunction f) {
    if (!f)
        return CUDA_ERROR_INVALID_HANDLE;
    if (f->k)
        return CUDA_SUCCESS;
    CUfunction tmp = NULL;
    CUresult r = cuModuleGetFunction(&tmp, f->mod, f->name);
    ACUDA_DEBUG_PRINT("acuda-lazy compile %s rc=%d tmp=%p k=%p\n",
                f->name, r, (void *)tmp, tmp ? (void *)tmp->k : NULL);
    if (r != CUDA_SUCCESS)
        return r;
    f->k = tmp->k;
    f->maxdyn_set = tmp->maxdyn_set;
    f->maxdyn = tmp->maxdyn;
    free(tmp);
    return CUDA_SUCCESS;
}

// A zeroed dummy buffer bound in place of a null buffer argument (for
// example layer-norm's optional gamma/beta).  The shader's null check
// still sees a zero device address; the descriptor only has to be
// valid for the pipeline layout.
#define ACUDA_NULL_PARAM_BYTES 65536u
static acuda_buffer_t g_null_param_buffer;

static acuda_status_t ensure_null_param_buffer(void) {
    if (g_null_param_buffer)
        return ACUDA_SUCCESS;
    acuda_status_t st = acuda_malloc_host(
        g_actx, &g_null_param_buffer, ACUDA_NULL_PARAM_BYTES);
    if (st != ACUDA_SUCCESS) {
        g_null_param_buffer = NULL;
        return st;
    }
    void *mapped = NULL;
    if (vkMapMemory(g_actx->device, g_null_param_buffer->memory, 0,
                    VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
        memset(mapped, 0, ACUDA_NULL_PARAM_BYTES);
        VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = g_null_param_buffer->memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkFlushMappedMemoryRanges(g_actx->device, 1, &range);
        vkUnmapMemory(g_actx->device, g_null_param_buffer->memory);
    }
    g_null_param_buffer->gpu_dirty = 1;
    return ACUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
    if (!f)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
    {
        CUfunction real = NULL;
        CUresult hr = handle_as_func((void *)f, &real);
        if (hr != CUDA_SUCCESS)
            return hr;
        f = real;
    }
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLaunchKernel %s\n", f->name);
    {
        // Diagnostic: skip named kernels entirely (pretend success) to
        // isolate a hanging launch. Comma-separated substring list.
        const char *skip = getenv("ACUDA_SKIP_KERNEL");
        if (skip && skip[0]) {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", skip);
            for (char *tok = strtok(buf, ","); tok;
                 tok = strtok(NULL, ",")) {
                if (strstr(f->name, tok) != NULL) {
                    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s SKIPPED\n",
                                f->name);
                    return CUDA_SUCCESS;
                }
            }
        }
    }
    r = ensure_func(f);
    if (r != CUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel ensure_func %s "
                    "rc=%d\n", f->name, (int)r);
        return r;
    }
    /* One parameter-model fetch serves the launch-info log, the packed
     * argument walk, and the launch below.  acuda_kernel_info_t is
     * ~283 KiB (per-parameter name tables); keep exactly one live copy
     * — worker threads that drive PhysX do not have stack to spare. */
    const acuda_kernel_info_t *info = acuda_kernel_info_ptr(f->k);
    if (!info)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_UNKNOWN, __LINE__); return CUDA_ERROR_UNKNOWN; }
    if (getenv("ACUDA_TRACE")) {
        unsigned stackleft = 0;
#if defined(_WIN32)
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        stackleft = (unsigned)((high - (ULONG_PTR)&low) / 1024u);
#endif
        fprintf(stderr, "acuda-launch-info %s params=%u ptr=%u bda=%u "
                "grid=%u,%u,%u block=%u,%u,%u shm=%u kparams=%p extra=%p "
                "stackleft=%uk\n",
                info->entry, info->num_params, info->num_ptr_params, info->bda,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes, (void *)kernelParams,
                (void *)extra, stackleft);
    }
    void **kparams = kernelParams;
    void *argv_buf[ACUDA_MAX_PARAMS]; // synthetic argv for packed buffers
    if (extra != NULL) {
        // cudart 6.5 passes kernel arguments as a packed buffer via
        // CU_LAUNCH_PARAM_* pairs (kernelParams == NULL).  Unpack it
        // into a synthetic argument vector using our per-parameter
        // model: scalars/pointers are 4 bytes in a 32-bit client,
        // 64-bit scalars 8 bytes, by-value structs wide_words*8 bytes
        // (8-byte stride, matching the launcher), mixed blobs 8 bytes.
        // Hidden texture params occupy no client bytes.  Pointer-word
        // groups (kind 5) ride inside the head's struct bytes.
        void **e = extra;
        void *pbuf = NULL;
        for (int guard = 0; guard < 66; guard++) {
            uintptr_t key = (uintptr_t)e[0];
            if (key == 0)
                break;
            if (key == 1)
                pbuf = e[1];
            else if (key != 2) {
                ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel "
                            "extra[%d] UNKNOWN key=%lu\n", guard,
                            (unsigned long)key);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            e += 2;
        }
        if (!pbuf) {
            ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel packed "
                        "buffer missing\n");
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
        }
        {
            size_t off = 0;
            for (uint32_t i = 0; i < info->num_params &&
                                i < ACUDA_MAX_PARAMS;
                 i++) {
                if (info->hidden_texref[i]) {
                    argv_buf[i] = NULL;
                    continue;
                }
                // Walk with declared strides (what cudart packed),
                // not kind-implied sizes: a .u64 pointer still
                // occupies 8 bytes even though only 4 are read.
                // A 64-bit client aligns 8-byte params to 8; a 32-bit
                // client packs everything tightly at 4 bytes (strides
                // are already 4-multiples there).
                size_t sz = info->param_stride[i];
#if !defined(_M_IX86) && !defined(__i386__)
                if (sz >= 8)
                    off = (off + 7u) & ~(size_t)7u;
#endif
                argv_buf[i] = (char *)pbuf + off;
                off += sz;
            }
            ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel packed walk "
                        "n=%u bytes=%zu\n", info->num_params, off);
            {
                const unsigned char *pb = (const unsigned char *)pbuf;
                for (size_t o = 0; o < off; o += 16) {
                    char line[80];
                    size_t n = 0;
                    for (size_t j = o; j < o + 16 && j < off && n < sizeof(line) - 4; j++) {
                        n += (size_t)snprintf(line + n, sizeof(line) - n,
                                              "%02x ", pb[j]);
                    }
                    ACUDA_DEBUG_PRINT("acuda-trace .. packed[%04zu] %s\n", o, line);
                }
            }
            kparams = argv_buf;
        }
    }
    if (!kparams) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel kernelParams=NULL\n");
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
    }
    if (gridDimX == 0 || gridDimY == 0 || gridDimZ == 0 ||
        blockDimX == 0 || blockDimY == 0 || blockDimZ == 0) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel zero dim "
                    "grid=%u,%u,%u block=%u,%u,%u\n", gridDimX, gridDimY,
                    gridDimZ, blockDimX, blockDimY, blockDimZ);
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
    }
    if (getenv("ACUDA_TRACE_PARAMS")) {
        // Diagnostic: dump by-value struct arguments word by word so a
        // lift/runtime mismatch can be correlated with the PTX offsets.
        fprintf(stderr, "acuda-params %s", info->entry);
        for (uint32_t i = 0; i < info->num_params && i < ACUDA_MAX_PARAMS; i++) {
            if (!info->wide_words[i])
                continue;
            const unsigned char *p = (const unsigned char *)kparams[i];
            if (!p)
                continue;
            uint32_t words = info->wide_words[i] * 8;
            fprintf(stderr, " [%u]", i);
            for (uint32_t w = 0; w + 4 <= words && w < 2048; w += 4) {
                uint32_t v;
                memcpy(&v, p + w, 4);
                fprintf(stderr, " %08x", v);
            }
        }
        fprintf(stderr, "\n");
    }
    ACUDA_DEBUG_PRINT("acuda-launch-info %s params=%u ptr=%u bda=%u "
                "grid=%u,%u,%u block=%u,%u,%u shm=%u\n",
                info->entry, info->num_params, info->num_ptr_params, info->bda,
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                blockDimZ, sharedMemBytes);
    // Give unresolved .extern .func calls a usable default when the
    // CUDA API is used without ACuda's host-call registration.  Keep a
    // callback the application already registered.
    for (uint32_t hi = 0; hi < info->host_call_count; hi++) {
        const char *hname = info->host_call_names[hi];
        if (!acuda_host_call_find(g_actx, hname))
            acuda_host_call_register(g_actx, hname, driver_host_identity,
                                     NULL);
    }
    // Static .shared lives in Workgroup storage; dynamic .shared is
    // sized by this launch (spec-constant extern array). Cap the
    // total against the advertised per-block limit, like hardware,
    // and honor a raised MAX_DYNAMIC cap when set.
    if (info->shared_bytes > 65536 ||
        (uint64_t)info->shared_bytes + (uint64_t)sharedMemBytes > 65536)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, __LINE__); return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES; }
    if (f->maxdyn_set && sharedMemBytes > f->maxdyn)
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, __LINE__); return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES; }
    uint32_t nptr = 0;
    for (uint32_t i = 0; i < info->num_params; i++)
        nptr += (info->is_ptr[i] == 1 || info->is_ptr[i] == 5 ||
                 info->is_ptr[i] == 6 ||
                 (info->is_ptr[i] == 4 && info->param_ssbo[i]))
                    ? 1u : 0u;
    VkBuffer *vkb = NULL;
    size_t *offs = NULL, *ranges = NULL;
    void **hostbufs = NULL;
    acuda_buffer_t *texh = NULL;
    if (nptr > 0) {
        vkb = malloc(sizeof(*vkb) * nptr);
        offs = malloc(sizeof(*offs) * nptr);
        ranges = malloc(sizeof(*ranges) * nptr);
        hostbufs = calloc(nptr, sizeof(*hostbufs));
        texh = calloc(nptr, sizeof(*texh));
        if (!vkb || !offs || !ranges || !hostbufs || !texh) {
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
        }
    }
    uint32_t *words = NULL;
    if (info->push_bytes > 0) {
        words = malloc(info->push_bytes);
        if (!words) {
            free(vkb);
            free(offs);
            free(ranges);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
        }
    }
    acuda_stream_t s = NULL;
    if (hStream) {
        if (hStream->ctx != c) {
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        s = hStream->s;
    }
    acuda_stream_t resolved_stream = NULL;
    if (acuda_resolve_stream(g_actx, s, &resolved_stream) != ACUDA_SUCCESS) {
        free(vkb);
        free(offs);
        free(ranges);
        free(hostbufs);
        free(texh);
        free(words);
        { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
    }
    acuda_buffer_t owned[ACUDA_MAX_PARAMS];
    memset(owned, 0, sizeof(owned));
    // Sizes per binding are unknown here; resolve each pointer to its full
    // allocation range via a zero-size probe first, then bind the range
    // actually needed. Simpler and correct: bind the whole allocation.
    uint32_t bi = 0, wi = 0, harg = 0, cont_off = 8;
    for (uint32_t i = 0; i < info->num_params; i++) {
        uint32_t pkind = info->is_ptr[i];
        void *karg;
        if (info->hidden_texref[i]) {
            CUtexref tr = NULL;
            for (size_t ti = 0; ti < f->mod->ntexrefs; ti++) {
                if (f->mod->texrefs[ti] &&
                    strcmp(f->mod->texrefs[ti]->name,
                           info->hidden_texref_name[i]) == 0) {
                    tr = f->mod->texrefs[ti];
                    break;
                }
            }
            driver_texobj_t *to = tr ? driver_tex_find(tr->obj) : NULL;
            ACUDA_DEBUG_PRINT("acuda-trace ?? hidden texref [%u] mod=%p want='%s' found=%d obj=%llx to=%p\n",
                    i, (void *)f->mod, info->hidden_texref_name[i], tr != NULL,
                    (unsigned long long)(tr ? (uint64_t)tr->obj : 0u),
                    (void *)to);
            if (!to) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = to->tex;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
            continue;
        }
        if (pkind == 5) {
            // Next pointer word of the previous by-value argument,
            // not a separate host arg.
            if (harg == 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            karg = (char *)kparams[harg - 1] + cont_off;
            cont_off += 8;
        } else {
            karg = kparams[harg++];
            cont_off = 8;
        }
        if (!karg) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
        }
        if (info->is_ptr[i] == 3) {
            continue; // unused param: no arg read, no words
        } else if (info->is_ptr[i] == 1 && info->is_tex[i] == 3) {
            // Surface parameter: the argument is an opaque CUsurfObject
            // id; bind its borrowed storage image.
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            driver_surfobj_t *so = driver_surf_find((CUsurfObject)dp);
            if (!so || !so->image) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = so->image;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
        } else if (info->is_ptr[i] == 1 && info->is_tex[i]) {
            // Texture/sampler parameter: the argument is an opaque
            // CUtexObject id, not a device address.
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            driver_texobj_t *to = driver_tex_find((CUtexObject)dp);
            if (!to) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
            }
            texh[bi] = info->is_tex[i] == 2 ? to->sampler : to->tex;
            vkb[bi] = VK_NULL_HANDLE;
            ranges[bi] = 0;
            bi++;
        } else if (info->is_ptr[i] == 1 || info->is_ptr[i] == 5) {
            CUdeviceptr dp = 0;
            memcpy(&dp, karg, sizeof(dp));
            if (dp == 0) {
                // Optional pointer argument (e.g. layer-norm
                // gamma/beta).  Bind a zeroed dummy so the descriptor
                // is valid, but push a zero device address: the
                // shader's null check must still see null.
                acuda_status_t nst = ensure_null_param_buffer();
                if (nst != ACUDA_SUCCESS) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    return map_status(nst);
                }
                vkb[bi] = g_null_param_buffer->buffer;
                offs[bi] = 0;
                ranges[bi] = ACUDA_NULL_PARAM_BYTES;
                hostbufs[bi] = NULL;
                texh[bi] = ACUDA_NULL_PARAM_HANDLE;
                bi++;
                continue;
            }
            // Range unknown: bind from dp to end of its allocation.
            // arena_bind validates containment of at least 1 byte.
            size_t cap = 0;
            ACUDA_DEBUG_PRINT("acuda-trace ?? bind ptr param [%u] dp=%llx stride=%u\n",
                        i, (unsigned long long)dp,
                        info->param_stride
                            ? (unsigned)info->param_stride[i] : 9999u);
            if (acuda_arena_capacity((uint64_t)dp, &cap) != 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            if (bind_for(dp, cap, &vkb[bi], &offs[bi]) != CUDA_SUCCESS ||
                acuda_arena_host((uint64_t)dp, cap, &hostbufs[bi]) != 0) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
            }
            ranges[bi] = cap;
            bi++;
        } else if (info->is_ptr[i] == 6) {
            // Mixed blob: bind word0 target like a plain pointer,
            // then push word1 as the scalar word.
            CUdeviceptr dp6 = 0;
            memcpy(&dp6, karg, sizeof(dp6));
            ACUDA_DEBUG_PRINT("acuda-trace ?? bind kind-6 param [%u] dp=%llx stride=%u\n",
                        i, (unsigned long long)dp6,
                        info->param_stride
                            ? (unsigned)info->param_stride[i] : 9999u);
            if (dp6 == 0) {
                acuda_status_t nst6 = ensure_null_param_buffer();
                if (nst6 != ACUDA_SUCCESS) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    return map_status(nst6);
                }
                vkb[bi] = g_null_param_buffer->buffer;
                offs[bi] = 0;
                ranges[bi] = ACUDA_NULL_PARAM_BYTES;
                hostbufs[bi] = NULL;
                texh[bi] = ACUDA_NULL_PARAM_HANDLE;
                bi++;
            } else {
                size_t cap6 = 0;
                if (acuda_arena_capacity((uint64_t)dp6, &cap6) != 0) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
                }
                if (bind_for(dp6, cap6, &vkb[bi], &offs[bi]) != CUDA_SUCCESS ||
                    acuda_arena_host((uint64_t)dp6, cap6, &hostbufs[bi]) != 0) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_VALUE, __LINE__); return CUDA_ERROR_INVALID_VALUE; }
                }
                ranges[bi] = cap6;
                bi++;
            }
            memcpy(&words[wi++], (const char *)karg + 4, 4);
        } else if (info->is_ptr[i] == 2) {
            // 64-bit scalars take two push words and are even-aligned
            // by the compiler; honor the same padding here.
            if (wi & 1u)
                wi++;
            uint64_t v = 0;
            memcpy(&v, karg, sizeof(v));
            // A u64 scalar can still be a device pointer that the
            // parameter analysis only tracked as a value (e.g. an
            // output pointer used solely as a store base).  Translate
            // an arena address to the real Vulkan device address, the
            // same way by-value struct words are translated below, so
            // the shader's PhysicalStorageBuffer access hits the buffer.
            if (v) {
                size_t vcap = 0;
                VkBuffer vbuf = VK_NULL_HANDLE;
                size_t voff = 0;
                if (acuda_arena_capacity(v, &vcap) == 0 &&
                    acuda_arena_bind(v, vcap, &vbuf, &voff) == 0 &&
                    vbuf != VK_NULL_HANDLE) {
                    VkBufferDeviceAddressInfo vbdai = {
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                        .buffer = vbuf,
                    };
                    VkDeviceAddress va = vkGetBufferDeviceAddress(
                        g_actx->device, &vbdai);
                    if (va)
                        v = (uint64_t)va + (uint64_t)voff;
                }
            }
            words[wi++] = (uint32_t)(v & 0xffffffffu);
            words[wi++] = (uint32_t)(v >> 32);
        } else if (info->is_ptr[i] == 4 && info->param_ssbo[i]) {
            // Large by-value struct: upload the raw argument bytes into
            // a storage buffer; the shader reads it by byte offset.
            uint32_t nq = info->wide_words[i];
            if (nq < 2)
                nq = 2;
            size_t bytes = (size_t)nq * 8u;
            acuda_buffer_t pb = NULL;
            acuda_status_t pst = acuda_malloc_host(g_actx, &pb, bytes);
            if (pst == ACUDA_SUCCESS) {
                void *mapped = NULL;
                VkResult vr = vkMapMemory(g_actx->device, pb->memory, 0,
                                          VK_WHOLE_SIZE, 0, &mapped);
                if (vr != VK_SUCCESS) {
                    pst = ACUDA_ERROR_INTERNAL;
                } else {
                    memcpy(mapped, karg, bytes);

                    // The argument is metadata whose pointer fields use
                    // ACuda's arena virtual addresses.  Translate any
                    // such 8-byte word to the real Vulkan device address
                    // so the shader's BDA loads hit the right buffer.
                    uint64_t *qw = (uint64_t *)mapped;
                    for (uint32_t q = 0; q < nq; q++) {
                        size_t qcap = 0;
                        VkBuffer qbuf = VK_NULL_HANDLE;
                        size_t qoff = 0;
                        if (acuda_arena_capacity(qw[q], &qcap) != 0 ||
                            acuda_arena_bind(qw[q], qcap, &qbuf, &qoff) != 0 ||
                            qbuf == VK_NULL_HANDLE)
                            continue;
                        VkBufferDeviceAddressInfo qbdai = {
                            .sType =
                                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                            .buffer = qbuf,
                        };
                        VkDeviceAddress qa = vkGetBufferDeviceAddress(
                            g_actx->device, &qbdai);
                        if (qa)
                            qw[q] = (uint64_t)qa + (uint64_t)qoff;
                    }
                    VkMappedMemoryRange range = {
                        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                        .memory = pb->memory,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                    };
                    VkResult fr = vkFlushMappedMemoryRanges(
                        g_actx->device, 1, &range);
                    vkUnmapMemory(g_actx->device, pb->memory);
                    if (fr != VK_SUCCESS) {
                        pst = ACUDA_ERROR_INTERNAL;
                    } else {
                        pb->gpu_dirty = 1;
                        pb->gpu_read = 0;
                    }
                }
            }
            if (pst != ACUDA_SUCCESS) {
                if (pb)
                    acuda_free(g_actx, pb);
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                return map_status(pst);
            }
            owned[bi] = pb;
            texh[bi] = pb;
            vkb[bi] = pb->buffer;
            offs[bi] = 0;
            ranges[bi] = bytes;
            bi++;
        } else if (info->is_ptr[i] == 4) {
            // By-value struct of scalar words: two push words per
            // 8-byte word, packed from one host argument.
            uint32_t nq = info->wide_words[i];
            if (nq < 2)
                nq = 2;
            if (wi & 1u)
                wi++;
            // A by-value struct can mix scalars and pointers (e.g. a
            // functor plus the output pointer).  Translate any qword
            // that lands inside the arena to its Vulkan device address.
            for (uint32_t q = 0; q < nq; q++) {
                uint64_t v = 0;
                memcpy(&v, (const char *)karg + (size_t)q * 8u, 8u);
                size_t qcap = 0;
                VkBuffer qbuf = VK_NULL_HANDLE;
                size_t qoff = 0;
                if (acuda_arena_capacity(v, &qcap) == 0 &&
                    acuda_arena_bind(v, qcap, &qbuf, &qoff) == 0 &&
                    qbuf != VK_NULL_HANDLE) {
                    VkBufferDeviceAddressInfo qbdai = {
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                        .buffer = qbuf,
                    };
                    VkDeviceAddress qa = vkGetBufferDeviceAddress(
                        g_actx->device, &qbdai);
                    if (qa)
                        v = (uint64_t)qa + (uint64_t)qoff;
                }
                words[wi++] = (uint32_t)(v & 0xffffffffu);
                words[wi++] = (uint32_t)(v >> 32);
            }
        } else {
            memcpy(&words[wi++], karg, 4);
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s bind done "
                "nptr=%u bi(words=%u)\n",
                f->name, nptr, wi);
    if (getenv("ACUDA_TRACE_PARAMS") && getenv("ACUDA_TRACE")) {
        // Scalar push words, in the same order the shader's
        // PushConstant block reads them.  Pair with acuda-params for
        // by-value structs when a launch reads a wrong value.
        fprintf(stderr, "acuda-push %s", f->name);
        for (uint32_t k = 0; k < wi; k++)
            fprintf(stderr, " %08x", words[k]);
        fprintf(stderr, "\n");
    }
    {
        // Headless-hang diagnosis: snapshot the exact launch inputs
        // (push words, grid/block, per-binding bytes) so a device-loss
        // dispatch can be replayed without the application.
        const char *fdir = getenv("ACUDA_DUMP_FORCE_INPUTS");
        if (fdir && *fdir &&
            (strstr(f->name, "KernelFluidForce") != NULL ||
             strstr(f->name, "KernelFluidDensity") != NULL)) {
            static int fserial = 0;
            char path[1024];
            snprintf(path, sizeof(path), "%s/forcedump_%d.bin", fdir,
                     fserial++);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                uint32_t hdr[16];
                memset(hdr, 0, sizeof(hdr));
                hdr[0] = 0x46445243u; // 'FDRC'
                hdr[1] = nptr;
                hdr[2] = wi;
                hdr[3] = gridDimX;
                hdr[4] = gridDimY;
                hdr[5] = gridDimZ;
                hdr[6] = blockDimX;
                hdr[7] = blockDimY;
                hdr[8] = blockDimZ;
                fwrite(hdr, sizeof(hdr), 1, fp);
                fwrite(words, sizeof(uint32_t), wi, fp);
                for (uint32_t b = 0; b < nptr; b++) {
                    uint64_t rb = (uint64_t)ranges[b];
                    fwrite(&rb, 8, 1, fp);
                    size_t take = ranges[b] < 262144 ? ranges[b] : 262144;
                    if (take && hostbufs && hostbufs[b])
                        fwrite(hostbufs[b], 1, take, fp);
                }
                fclose(fp);
                ACUDA_DEBUG_PRINT("acuda-trace .. forcedump %s nptr=%u words=%u\n",
                            path, nptr, wi);
            }
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s: post-bind\n", f->name);
    acuda_buffer_t *tables = NULL;
    acuda_buffer_t smeta = NULL;
    uint32_t surf_slots = 0;
    if (info->surf_table) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s: surf_table setup\n", f->name);
        acuda_buffer_t fallback = NULL;
        for (uint32_t i = 1; i < g_surf_slot_cap; i++)
            if (g_surf_slots[i]) {
                fallback = g_surf_slots[i];
                break;
            }
        if (!fallback) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        surf_slots = 0;
        for (uint32_t i = g_surf_slot_cap; i > 0; i--)
            if (g_surf_slots[i - 1]) {
                surf_slots = i;
                break;
            }
        if (surf_slots == 0) {
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_INVALID_HANDLE, __LINE__); return CUDA_ERROR_INVALID_HANDLE; }
        }
        uint32_t nt = info->surf_table_count;
        if (nt) {
            tables = calloc((size_t)nt * surf_slots, sizeof(*tables));
            if (!tables) {
                release_param_buffers(owned, ACUDA_MAX_PARAMS);
                free(vkb);
                free(offs);
                free(ranges);
                free(hostbufs);
                free(texh);
                free(words);
                { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
            }
        }
        for (uint32_t t = 0; t < nt; t++) {
            int dim = (int)info->surf_table_dim[t];
            int arrayed = (int)info->surf_table_arrayed[t];
            int isint = (int)info->surf_table_int[t];
            for (uint32_t i = 0; i < surf_slots; i++) {
                acuda_buffer_t img = g_surf_slots[i];
                int id = -1, ia = -1, ii = -1;
                if (img)
                    driver_img_type(img, &id, &ia, &ii);
                if (!img || id != dim || ia != arrayed || ii != isint)
                    img = driver_surf_dummy(dim, arrayed, isint);
                if (!img) {
                    release_param_buffers(owned, ACUDA_MAX_PARAMS);
                    free(tables);
                    free(vkb);
                    free(offs);
                    free(ranges);
                    free(hostbufs);
                    free(texh);
                    free(words);
                    { ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel fail %d at line %d\n", (int)CUDA_ERROR_OUT_OF_MEMORY, __LINE__); return CUDA_ERROR_OUT_OF_MEMORY; }
                }
                tables[(size_t)t * surf_slots + i] = img;
            }
        }
        size_t mbytes = surf_slots * 6u * sizeof(uint32_t);
        uint32_t *mw = calloc(1, mbytes);
        acuda_status_t mst = mw ? acuda_malloc(g_actx, &smeta, mbytes)
                                : ACUDA_ERROR_OUT_OF_MEMORY;
        if (mst == ACUDA_SUCCESS) {
            for (uint32_t i = 0; i < surf_slots; i++) {
                acuda_buffer_t img = g_surf_slots[i];
                driver_surf_meta(img ? img : fallback, mw + i * 6);
            }
            mst = acuda_memcpy_h2d(g_actx, smeta, mw, mbytes, NULL);
        }
        free(mw);
        if (mst != ACUDA_SUCCESS) {
            if (smeta)
                acuda_free(g_actx, smeta);
            release_param_buffers(owned, ACUDA_MAX_PARAMS);
            free(tables);
            free(vkb);
            free(offs);
            free(ranges);
            free(hostbufs);
            free(texh);
            free(words);
            return map_status(mst);
        }
    }
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s: calling raw_ex k=%p g_actx=%p info=%p\n", f->name, (void *)f->k, (void *)g_actx, (void *)info);
    if (!f->k || !info) {
        ACUDA_DEBUG_PRINT("acuda-trace !! cuLaunchKernel %s: null kernel/info\n", f->name);
        free(tables);
        if (smeta) acuda_free(g_actx, smeta);
        free(vkb); free(offs); free(ranges); free(hostbufs); free(texh); free(words);
        release_param_buffers(owned, ACUDA_MAX_PARAMS);
        return CUDA_ERROR_INVALID_HANDLE;
    }
    acuda_status_t st =
        acuda_launch_raw_ex(g_actx, f->k, vkb, offs, ranges, nptr,
                            hostbufs, texh, tables,
                            info->surf_table_count, surf_slots, smeta, words,
                            info->push_bytes,
                            gridDimX, gridDimY, gridDimZ,
                            blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, s);
    ACUDA_DEBUG_PRINT("acuda-trace .. cuLaunchKernel %s raw_ex st=%d\n",
                f->name, (int)st);
    free(tables);
    if (smeta && st != ACUDA_SUCCESS)
        acuda_free(g_actx, smeta);
    free(vkb);
    free(offs);
    free(ranges);
    free(hostbufs);
    free(texh);
    free(words);
    if (st != ACUDA_SUCCESS) {
        release_param_buffers(owned, ACUDA_MAX_PARAMS);
        return map_status(st);
    }
    // Storage-buffer parameter data must outlive the recorded launch.
    // Attach each temporary buffer to the stream; on the rare tracking
    // failure, synchronize before releasing it.
    for (uint32_t i = 0; i < nptr && i < ACUDA_MAX_PARAMS; i++) {
        if (!owned[i])
            continue;
        if (acuda_stream_track_buffer_pending(
                g_actx, resolved_stream, owned[i]) != ACUDA_SUCCESS) {
            acuda_stream_synchronize(g_actx, s);
            acuda_free(g_actx, owned[i]);
        }
        owned[i] = NULL;
    }
    // NOTE: the CUfunction shell stays owned by the caller (like real
    // CUDA, functions die with the module, not the launch). Freeing it
    // here used to turn every handle reuse into use-after-free.
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamCreate\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamSynchronize\n");
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

CUresult ACUDA_CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuEventCreate\n");
    if (!phEvent)
        return CUDA_ERROR_INVALID_VALUE;
    *phEvent = NULL;
    if (flags > 3)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    ACUDA_DEBUG_PRINT("acuda-trace .. cuEventCreate: need_ctx=%d c=%p g_actx=%p\n", (int)r, (void *)c, (void *)g_actx);
    if (r != CUDA_SUCCESS)
        return r;
    CUevent e = calloc(1, sizeof(*e));
    ACUDA_DEBUG_PRINT("acuda-trace .. cuEventCreate: calloc e=%p sizeof=%zu\n", (void *)e, sizeof(*e));
    if (!e)
        return CUDA_ERROR_OUT_OF_MEMORY;
    e->ctx = c;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuEventCreate: calling acuda_event_create\n");
    acuda_status_t st = acuda_event_create(g_actx, &e->e);
    ACUDA_DEBUG_PRINT("acuda-trace .. cuEventCreate: acuda_event_create st=%d\n", (int)st);
    if (st != ACUDA_SUCCESS) {
        free(e);
        return map_status(st);
    }
    *phEvent = e;
    ACUDA_DEBUG_PRINT("acuda-trace << cuEventCreate ok e=%p\n", (void *)e);
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuEventRecord e=%p s=%p\n", (void *)hEvent, (void *)hStream);
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuEventSynchronize\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuEventQuery\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuEventElapsedTime\n");
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

// PTX-passthrough linker: accepts PTX inputs, concatenates them, and
// hands the text to cuModuleLoadData as the "cubin". No SASS involved.
CUresult ACUDA_CUDAAPI cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                         void **optionValues, CUlinkState *stateOut) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLinkCreate_v2\n");
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (!stateOut)
        return CUDA_ERROR_INVALID_VALUE;
    *stateOut = NULL;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    CUlinkState s = calloc(1, sizeof(*s));
    if (!s)
        return CUDA_ERROR_OUT_OF_MEMORY;
    *stateOut = s;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                      void **optionValues, CUlinkState *stateOut) {
    return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

CUresult ACUDA_CUDAAPI cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data,
                          size_t size, const char *name,
                          unsigned int numOptions, CUjit_option *options,
                          void **optionValues) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLinkAddData_v2\n");
    (void)name;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!data)
        return CUDA_ERROR_INVALID_VALUE;
    if (type != CU_JIT_INPUT_PTX)
        return CUDA_ERROR_INVALID_IMAGE; // cubin/fatbin/object unsupported
    char *copy = malloc(size + 1);
    if (!copy)
        return CUDA_ERROR_OUT_OF_MEMORY;
    memcpy(copy, data, size);
    copy[size] = '\0';
    char err[1024];
    if (acuda_ptx_validate(copy, err, sizeof(err)) != ACUDA_SUCCESS) {
        free(copy);
        return CUDA_ERROR_INVALID_IMAGE;
    }
    if (state->n == state->cap) {
        size_t ncap = state->cap ? state->cap * 2 : 4;
        char **nb = malloc(sizeof(*nb) * ncap);
        size_t *ns = malloc(sizeof(*ns) * ncap);
        if (!nb || !ns) {
            free(nb);
            free(ns);
            free(copy);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(nb, state->blobs, sizeof(*nb) * state->n);
        memcpy(ns, state->sizes, sizeof(*ns) * state->n);
        free(state->blobs);
        free(state->sizes);
        state->blobs = nb;
        state->sizes = ns;
        state->cap = ncap;
    }
    state->blobs[state->n] = copy;
    state->sizes[state->n] = size;
    state->n++;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type, void *data,
                       size_t size, const char *name,
                       unsigned int numOptions, CUjit_option *options,
                       void **optionValues) {
    return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                            optionValues);
}

CUresult ACUDA_CUDAAPI cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                          const char *path, unsigned int numOptions,
                          CUjit_option *options, void **optionValues) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLinkAddFile_v2\n");
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (!path)
        return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(path, "rb");
    if (!f)
        return CUDA_ERROR_INVALID_IMAGE; // unreadable input, documented
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    CUresult r = CUDA_ERROR_INVALID_IMAGE;
    if (n >= 0) {
        char *buf = malloc((size_t)n ? (size_t)n : 1);
        if (buf && (n == 0 || fread(buf, 1, (size_t)n, f) == (size_t)n))
            r = cuLinkAddData_v2(state, type, buf, (size_t)n, path,
                                 numOptions, options, optionValues);
        free(buf);
    }
    fclose(f);
    return r;
}

CUresult ACUDA_CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                       const char *path, unsigned int numOptions,
                       CUjit_option *options, void **optionValues) {
    return cuLinkAddFile_v2(state, type, path, numOptions, options,
                            optionValues);
}

CUresult ACUDA_CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuLinkComplete\n");
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!cubinOut || !sizeOut)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    if (state->n == 0)
        return CUDA_ERROR_INVALID_IMAGE;
    size_t total = 0;
    for (size_t i = 0; i < state->n; i++)
        total += state->sizes[i] + 1;
    char *out = malloc(total + 1);
    if (!out)
        return CUDA_ERROR_OUT_OF_MEMORY;
    size_t pos = 0;
    for (size_t i = 0; i < state->n; i++) {
        memcpy(out + pos, state->blobs[i], state->sizes[i]);
        pos += state->sizes[i];
        out[pos++] = '\n';
    }
    out[pos] = '\0';
    free(state->linked);
    state->linked = out;
    state->linked_size = pos;
    *cubinOut = out;
    *sizeOut = pos;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuLinkDestroy(CUlinkState state) {
    if (!state)
        return CUDA_ERROR_INVALID_HANDLE;
    for (size_t i = 0; i < state->n; i++)
        free(state->blobs[i]);
    free(state->blobs);
    free(state->sizes);
    free(state->linked);
    free(state);
    return CUDA_SUCCESS;
}

static const struct { CUresult code; const char *name, *str; } err_table[] = {    { CUDA_SUCCESS, "CUDA_SUCCESS", "no error" },
    { CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument" },
    { CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory" },
    { CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED",
      "driver not initialized" },
    { CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down" },
    { CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device" },
    { CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal" },
    { CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE",
      "device program image invalid (cubin/SASS unsupported; PTX only)" },
    { CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid context" },
    { CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid handle" },
    { CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found" },
    { CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "operation not ready" },
    { CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported" },
    { CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error" },
};

CUresult ACUDA_CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(err_table) / sizeof(err_table[0]); i++) {
        if (err_table[i].code == error) {
            *pStr = err_table[i].str;
            return CUDA_SUCCESS;
        }
    }
    *pStr = "unknown error";
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
    if (!pStr)
        return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < sizeof(err_table) / sizeof(err_table[0]); i++) {
        if (err_table[i].code == error) {
            *pStr = err_table[i].name;
            return CUDA_SUCCESS;
        }
    }
    *pStr = "CUDA_ERROR_UNKNOWN";
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetDevice_v2(CUdevice *device) { return cuCtxGetDevice(device); }
CUresult ACUDA_CUDAAPI cuCtxSynchronize_v2(void) { return cuCtxSynchronize(); }
CUresult ACUDA_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem(bytes, dev);
}
CUresult ACUDA_CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
}
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    return cuDevicePrimaryCtxRelease(dev);
}
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    return cuDevicePrimaryCtxReset(dev);
}
CUresult ACUDA_CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

CUresult ACUDA_CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
CUresult ACUDA_CUDAAPI cuStreamGetCtx_v2(CUstream hStream, CUcontext *pctx,
                           void *pGreenCtx);
CUresult ACUDA_CUDAAPI cuStreamGetDevice(CUstream hStream, CUdevice *device);
CUresult ACUDA_CUDAAPI cuStreamGetId(CUstream hStream, unsigned long long *streamId);
CUresult ACUDA_CUDAAPI cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr);
CUresult ACUDA_CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                int *attributes, void **data,
                                CUdeviceptr ptr);
CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer(void **pdst, void *p, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemHostGetFlags(unsigned int *flags, void *p);
CUresult ACUDA_CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemHostAlloc_v2(void **pp, size_t bytesize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemFreeHost(void *p);
CUresult ACUDA_CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags);
CUresult ACUDA_CUDAAPI cuMemHostUnregister(void *p);

static CUresult lookup_symbol(const char *symbol, void **pfn) {
    if (!symbol || !pfn)
        return CUDA_ERROR_INVALID_VALUE;
    *pfn = NULL;
#define ENTRY(fn) \
    if (strcmp(symbol, #fn) == 0) { \
        *pfn = (void *)(uintptr_t)fn; \
        return CUDA_SUCCESS; \
    }
    ENTRY(cuInit)
    ENTRY(cuDriverGetVersion)
    ENTRY(cuDeviceGetCount)
    ENTRY(cuDeviceGet)
    ENTRY(cuDeviceGetName)
    ENTRY(cuDeviceComputeCapability)
    ENTRY(cuDeviceGetProperties)
    ENTRY(cuDeviceTotalMem)
    ENTRY(cuDeviceGetAttribute)
    ENTRY(cuCtxCreate_v2)
    ENTRY(cuCtxCreate)
    ENTRY(cuCtxDestroy_v2)
    ENTRY(cuCtxDestroy)
    ENTRY(cuCtxPushCurrent_v2)
    ENTRY(cuCtxPushCurrent)
    ENTRY(cuCtxPopCurrent_v2)
    ENTRY(cuCtxPopCurrent)
    ENTRY(cuCtxSynchronize)
    ENTRY(cuMemAlloc_v2)
    ENTRY(cuMemAlloc)
    ENTRY(cuMemFree_v2)
    ENTRY(cuMemFree)
    ENTRY(cuMemcpyHtoD_v2)
    ENTRY(cuMemcpyHtoD)
    ENTRY(cuMemcpyDtoH_v2)
    ENTRY(cuMemcpyDtoH)
    ENTRY(cuMemcpyDtoD_v2)
    ENTRY(cuMemcpyDtoD)
    ENTRY(cuMemcpy)
    ENTRY(cuMemcpy_v2)
    ENTRY(cuMemcpyWithAttributesAsync)
    ENTRY(cuMemsetD32_v2)
    ENTRY(cuMemsetD32)
    ENTRY(cuMemsetD8_v2)
    ENTRY(cuMemsetD8)
    ENTRY(cuModuleLoadDataEx)
    ENTRY(cuModuleLoadData)
    ENTRY(cuModuleLoadFatBinary)
    ENTRY(cuModuleUnload)
    ENTRY(cuModuleGetFunction)
    ENTRY(cuFuncGetAttribute)
    ENTRY(cuFuncSetAttribute)
    ENTRY(cuLaunchKernel)
    ENTRY(cuLaunchKernelEx)
    ENTRY(cuStreamCreate)
    ENTRY(cuStreamDestroy_v2)
    ENTRY(cuStreamDestroy)
    ENTRY(cuStreamSynchronize)
    ENTRY(cuEventCreate)
    ENTRY(cuEventDestroy_v2)
    ENTRY(cuEventDestroy)
    ENTRY(cuEventRecord)
    ENTRY(cuEventSynchronize)
    ENTRY(cuEventQuery)
    ENTRY(cuEventElapsedTime)
    ENTRY(cuLinkCreate_v2)
    ENTRY(cuLinkCreate)
    ENTRY(cuLinkAddData_v2)
    ENTRY(cuLinkAddData)
    ENTRY(cuLinkAddFile_v2)
    ENTRY(cuLinkAddFile)
    ENTRY(cuLinkComplete)
    ENTRY(cuLinkDestroy)
    ENTRY(cuGetErrorString)
    ENTRY(cuGetErrorName)
    ENTRY(cuGetProcAddress)
    ENTRY(cuGetProcAddress_v2)
    ENTRY(cuDeviceGetUuid)
    ENTRY(cuDeviceGetPCIBusId)
    ENTRY(cuDeviceGetByPCIBusId)
    ENTRY(cuArrayCreate)
    ENTRY(cuArrayCreate_v2)
    ENTRY(cuArray3DCreate)
    ENTRY(cuArray3DCreate_v2)
    ENTRY(cuArrayDestroy)
    ENTRY(cuArrayGetDescriptor)
    ENTRY(cuArray3DGetDescriptor)
    ENTRY(cuMipmappedArrayCreate)
    ENTRY(cuMipmappedArrayDestroy)
    ENTRY(cuMipmappedArrayGetLevel)
    ENTRY(cuModuleGetTexRef)
    ENTRY(cuTexRefSetAddress_v2)
    ENTRY(cuTexRefSetAddress)
    ENTRY(cuTexRefSetArray)
    ENTRY(cuTexRefSetFormat)
    ENTRY(cuTexRefSetFlags)
    ENTRY(cuTexRefSetFilterMode)
    ENTRY(cuTexRefSetAddressMode)
    ENTRY(cuTexRefSetMaxAnisotropy)
    ENTRY(cuTexRefGetAddress_v2)
    ENTRY(cuTexRefGetAddress)
    ENTRY(cuTexRefGetArray)
    ENTRY(cuTexRefGetFormat)
    ENTRY(cuTexRefGetFlags)
    ENTRY(cuTexRefGetFilterMode)
    ENTRY(cuTexRefGetAddressMode)
    ENTRY(cuTexRefGetMaxAnisotropy)
    ENTRY(cuTexObjectCreate)
    ENTRY(cuTexObjectDestroy)
    ENTRY(cuTexObjectGetResourceDesc)
    ENTRY(cuTexObjectGetTextureDesc)
    ENTRY(cuSurfObjectCreate)
    ENTRY(cuSurfObjectDestroy)
    ENTRY(cuSurfObjectGetResourceDesc)
    ENTRY(cuMemcpy2D)
    ENTRY(cuMemcpy2DAsync)
    ENTRY(cuMemcpy3D)
    ENTRY(cuMemcpy3DAsync)
    ENTRY(cuDeviceGetTexture1DLinearMaxWidth)
    ENTRY(cuDeviceGetP2PAttribute)
    ENTRY(cuDeviceCanAccessPeer)
    ENTRY(cuFlushGPUDirectRDMAWrites)
    ENTRY(cuDevicePrimaryCtxRetain)
    ENTRY(cuDevicePrimaryCtxRelease)
    ENTRY(cuDevicePrimaryCtxReset)
    ENTRY(cuDevicePrimaryCtxSetFlags)
    ENTRY(cuDevicePrimaryCtxGetState)
    ENTRY(cuCtxGetDevice_v2)
    ENTRY(cuCtxSynchronize_v2)
    ENTRY(cuDeviceTotalMem_v2)
    ENTRY(cuDeviceGetUuid_v2)
    ENTRY(cuDevicePrimaryCtxRelease_v2)
    ENTRY(cuDevicePrimaryCtxReset_v2)
    ENTRY(cuDevicePrimaryCtxSetFlags_v2)
    ENTRY(cuCtxGetCurrent)
    ENTRY(cuCtxSetCurrent)
    ENTRY(cuCtxGetDevice)
    ENTRY(cuCtxGetApiVersion)
    ENTRY(cuCtxGetLimit)
    ENTRY(cuCtxSetLimit)
    ENTRY(cuCtxSetCacheConfig)
    ENTRY(cuCtxGetCacheConfig)
    ENTRY(cuCtxDetach)
    ENTRY(cuMemGetInfo)
    ENTRY(cuDeviceGetDefaultMemPool)
    ENTRY(cuDeviceGetMemPool)
    ENTRY(cuDeviceSetMemPool)
    ENTRY(cuMemPoolTrimTo)
    ENTRY(cuStreamQuery)
    ENTRY(cuStreamWaitEvent)
    ENTRY(cuStreamGetFlags)
    ENTRY(cuStreamGetPriority)
    ENTRY(cuStreamGetCtx)
    ENTRY(cuStreamGetCtx_v2)
    ENTRY(cuStreamGetDevice)
    ENTRY(cuStreamGetId)
    ENTRY(cuPointerGetAttribute)
    ENTRY(cuPointerGetAttributes)
    ENTRY(cuMemHostGetDevicePointer)
    ENTRY(cuMemHostGetFlags)
    ENTRY(cuMemHostAlloc)
    ENTRY(cuMemHostAlloc_v2)
    ENTRY(cuMemFreeHost)
    ENTRY(cuMemHostRegister)
    ENTRY(cuMemHostRegister_v2)
    ENTRY(cuMemHostUnregister)
    ENTRY(cuModuleGetLoadingMode)
    ENTRY(cuMemAddressReserve)
    ENTRY(cuMemAddressFree)
    ENTRY(cuMemCreate)
    ENTRY(cuMemRelease)
    ENTRY(cuMemMap)
    ENTRY(cuMemUnmap)
    ENTRY(cuMemSetAccess)
    ENTRY(cuMemGetAllocationGranularity)
    ENTRY(cuMemGetAllocationPropertiesFromHandle)
    ENTRY(cuMemExportToShareableHandle)
    ENTRY(cuMemImportFromShareableHandle)
    ENTRY(cuMemRetainAllocationHandle)
    ENTRY(cuMemsetD32Async)
    ENTRY(cuMemsetD8Async)
    ENTRY(cuMemcpyHtoDAsync)
    ENTRY(cuMemcpyDtoHAsync)
    ENTRY(cuMemcpyDtoDAsync)
    ENTRY(cuMemcpyHtoDAsync_v2)
    ENTRY(cuMemcpyDtoHAsync_v2)
    ENTRY(cuMemcpyDtoDAsync_v2)
    ENTRY(cuMemcpyAsync)
    ENTRY(cuStreamWriteValue32)
    ENTRY(cuStreamWaitValue32)
    ENTRY(cuStreamWriteValue64)
    ENTRY(cuStreamWaitValue64)
    ENTRY(cuStreamBatchMemOp)
    ENTRY(cuCtxGetStreamPriorityRange)
    ENTRY(cuStreamIsCapturing)
    ENTRY(cuStreamBeginCapture)
    ENTRY(cuStreamEndCapture)
    ENTRY(cuThreadExchangeStreamCaptureMode)
    ENTRY(cuStreamGetCaptureInfo)
    ENTRY(cuStreamGetCaptureInfo_v2)
    ENTRY(cuStreamGetCaptureInfo_v3)
    ENTRY(cuMemGetAddressRange)
    ENTRY(cuMemGetAddressRange_v2)
    ENTRY(cuKernelGetFunction)
    ENTRY(cuKernelGetLibrary)
    ENTRY(cuKernelGetAttribute)
    ENTRY(cuKernelSetAttribute)
    ENTRY(cuKernelSetCacheConfig)
    ENTRY(cuKernelGetName)
    ENTRY(cuKernelGetParamInfo)
    ENTRY(cuKernelGetParamCount)
    ENTRY(cuLibraryLoadData)
    ENTRY(cuLibraryLoadFromFile)
    ENTRY(cuLibraryUnload)
    ENTRY(cuLibraryGetModule)
    ENTRY(cuLibraryGetKernel)
    ENTRY(cuLibraryGetGlobal)
    ENTRY(cuLibraryGetManaged)
    ENTRY(cuLibraryGetUnifiedFunction)
    ENTRY(cuLibraryGetKernelCount)
    ENTRY(cuLibraryEnumerateKernels)
    ENTRY(cuModuleGetGlobal)
    ENTRY(cuModuleGetGlobal_v2)
    ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessor)
    ENTRY(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    ENTRY(cuOccupancyMaxPotentialBlockSize)
    ENTRY(cuOccupancyMaxPotentialBlockSizeWithFlags)
    ENTRY(acuda_stub_cuModuleGetGlobal)
    ENTRY(acuda_stub_cuModuleGetGlobal_v2)
    ENTRY(acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor)
    ENTRY(acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    ENTRY(acuda_stub_cuOccupancyMaxPotentialBlockSize)
    ENTRY(acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags)
    ENTRY(cuGetExportTable)
    ENTRY(acuda_stub_cuTensorMapEncodeTiled)
    /* CUDA 6.x-era names the runtime binds by handle (see above). */
    ENTRY(cuMemsetD16_v2)
    ENTRY(cuMemsetD16)
    ENTRY(cuMemsetD16Async)
    ENTRY(cuMemsetD2D8_v2)
    ENTRY(cuMemsetD2D8)
    ENTRY(cuMemsetD2D8Async)
    ENTRY(cuMemsetD2D16_v2)
    ENTRY(cuMemsetD2D16)
    ENTRY(cuMemsetD2D16Async)
    ENTRY(cuMemsetD2D32_v2)
    ENTRY(cuMemsetD2D32)
    ENTRY(cuMemsetD2D32Async)
    ENTRY(cuMemcpy2D_v2)
    ENTRY(cuMemcpy2DAsync_v2)
    ENTRY(cuMemcpy2DUnaligned)
    ENTRY(cuMemcpy2DUnaligned_v2)
    ENTRY(cuMemcpy3D_v2)
    ENTRY(cuMemcpy3DAsync_v2)
    ENTRY(cuMemcpyPeer)
    ENTRY(cuMemcpyPeerAsync)
    ENTRY(cuMemcpy3DPeer)
    ENTRY(cuMemcpy3DPeerAsync)
    ENTRY(cuMemGetInfo_v2)
    ENTRY(cuMemHostGetDevicePointer_v2)
    ENTRY(cuMemAllocPitch_v2)
    ENTRY(cuMemAllocManaged)
    ENTRY(cuArrayGetDescriptor_v2)
    ENTRY(cuArray3DGetDescriptor_v2)
    ENTRY(cuLinkAddData)
    ENTRY(cuLinkAddFile)
    ENTRY(cuModuleLoad)
    ENTRY(cuModuleGetSurfRef)
    ENTRY(cuSurfRefSetArray)
    ENTRY(cuTexRefCreate)
    ENTRY(cuTexRefDestroy)
    ENTRY(cuTexRefSetAddress2D_v3)
    ENTRY(cuTexRefSetMipmapFilterMode)
    ENTRY(cuTexRefSetMipmapLevelBias)
    ENTRY(cuTexRefSetMipmapLevelClamp)
    ENTRY(cuTexRefSetMipmappedArray)
    ENTRY(cuTexObjectGetResourceViewDesc)
    ENTRY(cuFuncSetCacheConfig)
    ENTRY(cuFuncSetSharedMemConfig)
    ENTRY(cuCtxEnablePeerAccess)
    ENTRY(cuCtxDisablePeerAccess)
    ENTRY(cuProfilerInitialize)
    ENTRY(cuProfilerStart)
    ENTRY(cuProfilerStop)
    ENTRY(cuStreamCreateWithPriority)
    ENTRY(cuStreamAddCallback)
    ENTRY(cuStreamAttachMemAsync)
    ENTRY(cuD3D10CtxCreateOnDevice)
    ENTRY(cuD3D10GetDevice)
    ENTRY(cuD3D10GetDevices)
    ENTRY(cuD3D10GetDirect3DDevice)
    ENTRY(cuD3D10MapResources)
    ENTRY(cuD3D10RegisterResource)
    ENTRY(cuD3D10UnmapResources)
    ENTRY(cuD3D10UnregisterResource)
    ENTRY(cuD3D11CtxCreateOnDevice)
    ENTRY(cuD3D11GetDevice)
    ENTRY(cuD3D11GetDevices)
    ENTRY(cuD3D11GetDirect3DDevice)
    ENTRY(cuGLCtxCreate_v2)
    ENTRY(cuGLGetDevices)
    ENTRY(cuGLInit)
    ENTRY(cuGLMapBufferObject_v2)
    ENTRY(cuGLMapBufferObjectAsync_v2)
    ENTRY(cuGLRegisterBufferObject)
    ENTRY(cuGLSetBufferObjectMapFlags)
    ENTRY(cuGLUnmapBufferObject)
    ENTRY(cuGLUnmapBufferObjectAsync)
    ENTRY(cuGLUnregisterBufferObject)
    ENTRY(cuGraphicsD3D10RegisterResource)
    ENTRY(cuGraphicsD3D11RegisterResource)
    ENTRY(cuGraphicsD3D9RegisterResource)
    ENTRY(cuGraphicsGLRegisterBuffer)
    ENTRY(cuGraphicsGLRegisterImage)
    ENTRY(cuGraphicsMapResources)
    ENTRY(cuGraphicsUnmapResources)
    ENTRY(cuGraphicsUnregisterResource)
    ENTRY(cuGraphicsResourceGetMappedPointer_v2)
    ENTRY(cuGraphicsResourceSetMapFlags_v2)
    ENTRY(cuGraphicsSubResourceGetMappedArray)
    ENTRY(cuGraphicsResourceGetMappedMipmappedArray)
    ENTRY(cuWGLGetDevice)
    ENTRY(cuIpcCloseMemHandle)
    ENTRY(cuIpcGetEventHandle)
    ENTRY(cuIpcGetMemHandle)
    ENTRY(cuIpcOpenEventHandle)
    ENTRY(cuIpcOpenMemHandle)
#undef ENTRY
    return CUDA_ERROR_NOT_FOUND;
}

// Resolve a base driver symbol for the ABI version the caller was built
// against. CUDA 12+ clients request the base name and expect the _v2
// entry point when one exists. cuGetProcAddress itself is the clearest
// case: v1 takes four arguments, v2 adds the symbolStatus out-parameter.
static CUresult proc_address(const char *symbol, void **pfn, int cudaVersion) {
    if (!symbol || !pfn)
        return CUDA_ERROR_INVALID_VALUE;
    *pfn = NULL;
    size_t n = strlen(symbol);
    if (cudaVersion >= 12000 && n > 3 &&
        strcmp(symbol + n - 3, "_v2") != 0) {
        char versioned[160];
        if (n + 3 < sizeof(versioned)) {
            memcpy(versioned, symbol, n);
            memcpy(versioned + n, "_v2", 4);
            if (lookup_symbol(versioned, pfn) == CUDA_SUCCESS)
                return CUDA_SUCCESS;
        }
    }
    CUresult r = lookup_symbol(symbol, pfn);
    if (r != CUDA_SUCCESS && getenv("ACUDA_TRACE"))
        fprintf(stderr, "acuda-trace ?? cuGetProcAddress(%s) -> NOT_FOUND\n",
                symbol);
    return r;
}

CUresult ACUDA_CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          unsigned long long flags) {
    (void)flags;
    if (!pfn)
        return CUDA_ERROR_INVALID_VALUE;
    if (!symbol || !*symbol) {
        *pfn = NULL;
        return CUDA_ERROR_NOT_FOUND;
    }
    return proc_address(symbol, pfn, cudaVersion);
}

CUresult ACUDA_CUDAAPI cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             unsigned long long flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
    (void)flags;
    if (!pfn)
        return CUDA_ERROR_INVALID_VALUE;
    if (cudaVersion > ACUDA_DRIVER_VERSION)
        return CUDA_ERROR_INVALID_VALUE;
    if (!symbol || !*symbol) {
        *pfn = NULL;
        if (symbolStatus)
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        return CUDA_SUCCESS;
    }
    CUresult r = proc_address(symbol, pfn, cudaVersion);
    ACUDA_DEBUG_PRINT_PROC("acuda-proc %s v=%d found=%d pfn=%p\n",
                symbol ? symbol : "(null)", cudaVersion, r == CUDA_SUCCESS,
                pfn ? *pfn : NULL);
    if (r != CUDA_SUCCESS) {
        *pfn = NULL;
        if (symbolStatus)
            *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        // The v2 ABI reports lookup failures through symbolStatus and
        // still returns CUDA_SUCCESS; callers that need the legacy
        // error use the four-argument entry point.
        return CUDA_SUCCESS;
    }
    if (symbolStatus)
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    return CUDA_SUCCESS;
}

/* ---- Device characterization, primary contexts, misc (torch init path) ---- */

struct CUmemPool_st {
    int device;
};

#define ACUDA_MAX_DEVICES 8
static CUcontext primary_ctx[ACUDA_MAX_DEVICES];
static int primary_refs[ACUDA_MAX_DEVICES];
static CUmemPool default_pool[ACUDA_MAX_DEVICES];

static CUresult check_dev(int dev) {
    int n = 0;
    CUresult r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS)
        return r;
    return (dev >= 0 && dev < n) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

static int device_pci_info(CUdevice dev, acuda_device_info_t *info) {
    if (!g_actx || acuda_device_get(g_actx, (uint32_t)dev, info) != ACUDA_SUCCESS)
        return 0;
    return 1;
}

CUresult ACUDA_CUDAAPI cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);

CUresult ACUDA_CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!uuid)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    // Deterministic pseudo-UUID: "ACUD" + ordinal + FNV of nothing (stable).
    memset(uuid->bytes, 0, 16);
    uuid->bytes[0] = 'A';
    uuid->bytes[1] = 'C';
    uuid->bytes[2] = 'U';
    uuid->bytes[3] = 'D';
    uuid->bytes[4] = (char)dev;
    uint32_t h = 2166136261u + (uint32_t)dev * 16777619u;
    memcpy(uuid->bytes + 8, &h, 4);
    memcpy(uuid->bytes + 12, &h, 4);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    acuda_device_info_t info;
    const char *id = getenv("ACUDA_PCI_BUS_ID");
    char generated[32];
    if (!pciBusId || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (!id || !*id) {
        if (device_pci_info(dev, &info)) {
            snprintf(generated, sizeof(generated), "%04x:%02x:%02x.%x",
                     info.pci_domain, info.pci_bus, info.pci_device,
                     info.pci_function);
            id = generated;
        } else {
            id = (dev == 0) ? "0000:00:00.0" : "0000:00:00.0";
        }
    }
    strncpy(pciBusId, id, (size_t)len - 1);
    pciBusId[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuDeviceGetByPCIBusId\n");
    if (!device || !pciBusId)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_init();
    if (r != CUDA_SUCCESS)
        return r;
    {
        const char *id = getenv("ACUDA_PCI_BUS_ID");
        if (id && *id && strcmp(pciBusId, id) == 0) {
            *device = 0;
            return CUDA_SUCCESS;
        }
    }
    if (g_actx) {
        uint32_t n = 0;
        if (acuda_device_count(g_actx, &n) == ACUDA_SUCCESS) {
            for (uint32_t i = 0; i < n; i++) {
                char bdf[32];
                if (cuDeviceGetPCIBusId(bdf, sizeof(bdf), (CUdevice)i) == CUDA_SUCCESS &&
                    strcmp(pciBusId, bdf) == 0) {
                    *device = (CUdevice)i;
                    return CUDA_SUCCESS;
                }
            }
        }
    }
    return CUDA_ERROR_NOT_FOUND;
}

CUresult ACUDA_CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                            CUarray_format format,
                                            unsigned numChannels,
                                            CUdevice dev) {
    (void)format;
    (void)numChannels;
    if (!maxWidthInElements)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    *maxWidthInElements = (size_t)1 << 27;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetP2PAttribute(int *value, int attrib,
                                 CUdevice srcDevice, CUdevice dstDevice) {
    (void)attrib;
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(srcDevice);
    if (r != CUDA_SUCCESS)
        return r;
    r = check_dev(dstDevice);
    if (r != CUDA_SUCCESS)
        return r;
    *value = 0; // no peer mappings in this model
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev) {
    if (!canAccessPeer)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    r = check_dev(peerDev);
    if (r != CUDA_SUCCESS)
        return r;
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuFlushGPUDirectRDMAWrites(int target, int scope) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuFlushGPUDirectRDMAWrites\n");
    (void)target;
    (void)scope;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS; // no GPUDirect transports here
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuDevicePrimaryCtxRelease\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxGetApiVersion\n");
    (void)ctx;
    if (!version)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *version = ACUDA_DRIVER_VERSION;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetLimit(size_t *pvalue, int limit) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxGetLimit\n");
    (void)limit;
    if (!pvalue)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *pvalue = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxSetLimit(int limit, size_t value) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxSetLimit\n");
    (void)limit;
    (void)value;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxSetCacheConfig(int config) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxSetCacheConfig\n");
    (void)config;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuCtxGetCacheConfig(int *config) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuCtxGetCacheConfig\n");
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

CUresult ACUDA_CUDAAPI cuMemGetInfo(size_t *freeBytes, size_t *totalBytes) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemGetInfo\n");
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

CUresult ACUDA_CUDAAPI cuDeviceGetDefaultMemPool(CUmemPool *pool_out, CUdevice dev) {
    if (!pool_out)
        return CUDA_ERROR_INVALID_VALUE;
    *pool_out = NULL;
    CUresult r = check_dev(dev);
    if (r != CUDA_SUCCESS)
        return r;
    if (dev < 0 || dev >= ACUDA_MAX_DEVICES)
        return CUDA_ERROR_INVALID_DEVICE;
    if (!default_pool[dev]) {
        CUmemPool p = calloc(1, sizeof(*p));
        if (!p)
            return CUDA_ERROR_OUT_OF_MEMORY;
        p->device = dev;
        default_pool[dev] = p;
    }
    *pool_out = default_pool[dev];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetMemPool(CUmemPool *pool_out, CUdevice dev) {
    return cuDeviceGetDefaultMemPool(pool_out, dev);
}

CUresult ACUDA_CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemPool pool) {
    (void)pool;
    return check_dev(dev);
}

CUresult ACUDA_CUDAAPI cuMemPoolTrimTo(CUmemPool pool, size_t minBytesToKeep) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuMemPoolTrimTo\n");
    (void)minBytesToKeep;
    if (!pool)
        return CUDA_ERROR_INVALID_HANDLE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamQuery(CUstream hStream) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamQuery\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamWaitEvent\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamGetFlags\n");
    (void)hStream;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *flags = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamGetPriority\n");
    (void)hStream;
    if (!priority)
        return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    *priority = 0;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuPointerGetAttribute(void *data, int attribute, CUdeviceptr ptr) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuPointerGetAttribute(attr=%d ptr=%llx)\n", attribute, (unsigned long long)ptr);
    if (!data)
        return CUDA_ERROR_INVALID_VALUE;
    size_t cap = 0;
    int device = acuda_arena_capacity((uint64_t)ptr, &cap) == 0;
    switch (attribute) {
    case 1: { // CU_POINTER_ATTRIBUTE_CONTEXT
        CUcontext *out = (CUcontext *)data;
        return need_ctx(out) == CUDA_SUCCESS ? CUDA_SUCCESS
                                             : CUDA_ERROR_NOT_INITIALIZED;
    }
    case 2: // MEMORY_TYPE
        *(int *)data = device ? 2 : 1; // CU_MEMORYTYPE_DEVICE / HOST
        return CUDA_SUCCESS;
    case 3: // DEVICE_POINTER
        *(CUdeviceptr *)data = ptr;
        return CUDA_SUCCESS;
    case 4: // HOST_POINTER (arena memory is host-mapped)
        *(void **)data = (void *)(uintptr_t)ptr;
        return CUDA_SUCCESS;
    case 8:  // IS_MANAGED
    case 10: // IS_LEGACY_CUDA_IPC_CAPABLE
    case 15: // IS_GPU_DIRECT_RDMA_CAPABLE
    case 21: // IS_HW_DECOMPRESS_CAPABLE
        *(int *)data = 0;
        return CUDA_SUCCESS;
    case 7:  // BUFFER_ID
    case 20: // MEMORY_BLOCK_ID
        *(unsigned long long *)data = (unsigned long long)ptr;
        return CUDA_SUCCESS;
    case 9: // DEVICE_ORDINAL
        *(int *)data = 0;
        return CUDA_SUCCESS;
    case 16: // ACCESS_FLAGS
        *(unsigned int *)data = 3; // READWRITE
        return CUDA_SUCCESS;
    case 11: // RANGE_START_ADDR
        *(CUdeviceptr *)data = ptr;
        return CUDA_SUCCESS;
    case 12: // RANGE_SIZE
        *(size_t *)data = cap ? cap : 0;
        return CUDA_SUCCESS;
    case 13: // MAPPED
        *(int *)data = device ? 1 : 0;
        return CUDA_SUCCESS;
    default:
        return CUDA_ERROR_INVALID_VALUE;
    }
}

CUresult ACUDA_CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                int *attributes, void **data,
                                CUdeviceptr ptr) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuPointerGetAttributes(n=%u ptr=%llx)\n", numAttributes, (unsigned long long)ptr);
    if (numAttributes && (!attributes || !data))
        return CUDA_ERROR_INVALID_VALUE;
    for (unsigned int i = 0; i < numAttributes; i++) {
        CUresult r = cuPointerGetAttribute(data[i], attributes[i], ptr);
        if (r != CUDA_SUCCESS)
            return r;
    }
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer(void **pdst, void *p, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostGetDevicePointer(%p flags=%x)\n",
                p, flags);
    (void)flags;
    if (!pdst || !p)
        return CUDA_ERROR_INVALID_VALUE;
    *pdst = p; // unified addressing
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostGetFlags(unsigned int *flags, void *p) {
    (void)p;
    if (!flags)
        return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

#if defined(_WIN32)
/* The 32-bit process runs out of contiguous address space once the device
 * arena has grown to ~1.4 GB, so a later 128 MB VirtualAlloc fails and
 * PhysX dereferences the failure.  Reserve a host heap early (on the first
 * host allocation) and suballocate/decommit from it.  Address space is
 * reserved once; commit charge is released on cuMemFreeHost. */
#define ACUDA_HOST_HEAP_RESERVE (160u << 20)
/* Fallback fixed base if no above-2 GiB window exists.  A heap reserved at
 * the first free low address fragments the region the NVIDIA OpenGL driver
 * needs for its Optimus DxPresent surface, which then fails with OOM. */
#define ACUDA_HOST_HEAP_BASE ((void *)(uintptr_t)0x70000000u)
#define ACUDA_HOST_DESC_MAX 256
typedef struct acuda_host_desc {
    size_t off;
    size_t size;
    int free;
} acuda_host_desc_t;
static unsigned char *g_host_base;
static size_t g_host_size;
static acuda_host_desc_t g_host_desc[ACUDA_HOST_DESC_MAX];
static int g_host_desc_n;
static int g_host_init_done;

/* Find the highest free block that can hold the heap.  When `above_2gb`
 * is set, only blocks fully above 0x80000000 are considered (the LAA
 * region a large-address-aware 32-bit process has on 64-bit Windows);
 * otherwise only blocks fully below 0x80000000 qualify.  The host heap
 * is plain CPU staging memory with no GPU-visibility requirement, so
 * placing it above 2 GiB keeps the low region - where the NVIDIA OpenGL
 * driver and PhysX must allocate (depth FBOs, DxPresent surfaces, CRT
 * heap) - unfragmented.  On a non-LAA process no above-2 GiB block
 * exists and the caller falls back below. */
static void *acuda_host_heap_find_high(size_t reserve, int above_2gb) {
    MEMORY_BASIC_INFORMATION mbi;
    unsigned char *p = (unsigned char *)0x10000;
    uintptr_t best = 0;
    while ((uintptr_t)p < (uintptr_t)0xFFFFFFFFu) {
        uintptr_t base, end;
        unsigned char *next;
        if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi))
            break;
        if (mbi.State == MEM_FREE && mbi.RegionSize >= reserve) {
            base = ((uintptr_t)mbi.BaseAddress + 0xFFFFu) & ~(uintptr_t)0xFFFFu;
            end = (uintptr_t)mbi.BaseAddress + mbi.RegionSize;
            if (above_2gb) {
                if (base >= (uintptr_t)0x80000000u &&
                    base + reserve >= base && base > best)
                    best = base;
            } else if (base + reserve <= (uintptr_t)0x80000000u &&
                       base > best) {
                best = base;
            }
        }
        next = (unsigned char *)mbi.BaseAddress + mbi.RegionSize;
        if (next <= p)
            break;
        p = next;
    }
    return best ? (void *)best : NULL;
}

static int acuda_host_heap_init(size_t need) {
    size_t reserve;
    const char *noheap;
    const char *heap_mb;
    if (g_host_init_done)
        return g_host_base != NULL;
    g_host_init_done = 1;
    /* Diagnostics: a lower/no reservation leaves more of the 32-bit
     * address space to the NVIDIA OpenGL driver (depth FBOs). */
    noheap = getenv("ACUDA_NO_HOST_HEAP");
    if (noheap && *noheap)
        return 0;
    reserve = ACUDA_HOST_HEAP_RESERVE;
    heap_mb = getenv("ACUDA_HOST_HEAP_MB");
    if (heap_mb && *heap_mb) {
        unsigned long mb = strtoul(heap_mb, NULL, 0);
        reserve = mb ? (size_t)mb << 20 : 0;
        if (!reserve)
            return 0;
    }
    /* In 32-bit, growing the upfront reserved host heap to satisfy a single
     * large request can consume/fragment scarce low VA. If the user capped
     * the heap, respect it. Even when growing, avoid low-address fallbacks
     * (fixed base / low "anywhere") that destabilize the process - prefer
     * above-2GB or let oversized allocs use per-allocation VirtualAlloc. */
    int grew_for_need = 0;
    if (need > reserve) {
#if defined(_WIN32) && !defined(_WIN64)
        if (heap_mb && *heap_mb) {
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr,
                        "acuda-trace .. host_heap: need=%zu exceeds cap=%zu, no grow\n",
                        need, reserve);
            /* keep reserve as capped; alloc will return NULL and caller can
             * fall back to per-allocation VirtualAlloc */
        } else {
#endif
            reserve = ((need + 0xFFFFu) & ~(size_t)0xFFFFu) + (16u << 20);
            grew_for_need = 1;
#if defined(_WIN32) && !defined(_WIN64)
        }
#endif
    }
    /* Prefer the LAA region above 2 GiB so the low 2 GiB stays contiguous
     * for the NVIDIA driver and PhysX.  Fall back to the previous order
     * (fixed base, highest free below 2 GiB, anywhere) on failure or on
     * non-LAA processes where no above-2 GiB window exists. */
    {
        void *high = acuda_host_heap_find_high(reserve, 1);
        if (high) {
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
            if (g_host_base && getenv("ACUDA_TRACE"))
                fprintf(stderr,
                        "acuda-trace .. host_heap reserve %zu above 2GiB at %p\n",
                        reserve, (void *)g_host_base);
        }
    }
#if defined(_WIN32) && !defined(_WIN64)
    /* For oversized-growth reservations in 32-bit, avoid low fixed-base or
     * arbitrary low placements (can land at 0x10020000 and fragment/crash).
     * Prefer above-2GB or highest-below-2GB; otherwise skip heap init so
     * the caller can try a direct per-allocation VirtualAlloc. */
    if (!g_host_base) {
        void *high = acuda_host_heap_find_high(reserve, 0);
        /* Avoid very low placements (< 0x40000000) which destabilize 32-bit
         * processes (CRT/PhysX/nvogl). Prefer fixed base in that case. */
        if (high && (uintptr_t)high >= (uintptr_t)0x40000000u) {
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
            if (g_host_base && getenv("ACUDA_TRACE"))
                fprintf(stderr, "acuda-trace .. host_heap reserve %zu below-2GiB-high at %p\n",
                        reserve, (void *)g_host_base);
        }
    }
    if (!g_host_base && !grew_for_need) {
        g_host_base = (unsigned char *)VirtualAlloc(ACUDA_HOST_HEAP_BASE, reserve,
                                                    MEM_RESERVE, PAGE_NOACCESS);
        if (g_host_base && getenv("ACUDA_TRACE"))
            fprintf(stderr, "acuda-trace .. host_heap reserve %zu fixed-base at %p\n",
                    reserve, (void *)g_host_base);
    }
    if (!g_host_base) {
        void *high = acuda_host_heap_find_high(reserve, 0);
        if (high) {
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
            if (g_host_base && getenv("ACUDA_TRACE"))
                fprintf(stderr, "acuda-trace .. host_heap reserve %zu below-2GiB-high(low-ok) at %p\n",
                        reserve, (void *)g_host_base);
        }
    }
    if (!g_host_base && !grew_for_need) {
        g_host_base = (unsigned char *)VirtualAlloc(NULL, reserve, MEM_RESERVE,
                                                    PAGE_NOACCESS);
        if (g_host_base && getenv("ACUDA_TRACE"))
            fprintf(stderr, "acuda-trace .. host_heap reserve %zu anywhere at %p\n",
                    reserve, (void *)g_host_base);
    }
#else
    if (!g_host_base) {
        g_host_base = (unsigned char *)VirtualAlloc(ACUDA_HOST_HEAP_BASE, reserve,
                                                    MEM_RESERVE, PAGE_NOACCESS);
        if (g_host_base && getenv("ACUDA_TRACE"))
            fprintf(stderr, "acuda-trace .. host_heap reserve %zu fixed-base at %p\n",
                    reserve, (void *)g_host_base);
    }
    if (!g_host_base) {
        void *high = acuda_host_heap_find_high(reserve, 0);
        if (high) {
            g_host_base = (unsigned char *)VirtualAlloc(high, reserve,
                                                        MEM_RESERVE,
                                                        PAGE_NOACCESS);
            if (g_host_base && getenv("ACUDA_TRACE"))
                fprintf(stderr, "acuda-trace .. host_heap reserve %zu below-2GiB-high at %p\n",
                        reserve, (void *)g_host_base);
        }
    }
    if (!g_host_base) {
        g_host_base = (unsigned char *)VirtualAlloc(NULL, reserve, MEM_RESERVE,
                                                    PAGE_NOACCESS);
        if (g_host_base && getenv("ACUDA_TRACE"))
            fprintf(stderr, "acuda-trace .. host_heap reserve %zu anywhere at %p\n",
                    reserve, (void *)g_host_base);
    }
#endif
    if (!g_host_base) {
        if (getenv("ACUDA_TRACE"))
            fprintf(stderr, "acuda-trace .. host_heap reserve %zu failed\n", reserve);
        return 0;
    }
    g_host_size = reserve;
    g_host_desc[0].off = 0;
    g_host_desc[0].size = reserve;
    g_host_desc[0].free = 1;
    g_host_desc_n = 1;
    return 1;
}

static void *acuda_host_heap_alloc(size_t bytes) {
    size_t need = (bytes + 0xFFFu) & ~(size_t)0xFFFu;
    int best = -1;
    void *p;
    if (need == 0)
        return NULL;
    if (!acuda_host_heap_init(need))
        return NULL;
    for (int i = 0; i < g_host_desc_n; i++) {
        if (g_host_desc[i].free && g_host_desc[i].size >= need) {
            if (best < 0 || g_host_desc[i].size < g_host_desc[best].size)
                best = i;
            if (g_host_desc[i].size == need)
                break;
        }
    }
    if (best < 0)
        return NULL;
    p = g_host_base + g_host_desc[best].off;
    if (!VirtualAlloc(p, need, MEM_COMMIT, PAGE_READWRITE))
        return NULL;
    if (g_host_desc[best].size > need && g_host_desc_n < ACUDA_HOST_DESC_MAX) {
        for (int j = g_host_desc_n; j > best + 1; j--)
            g_host_desc[j] = g_host_desc[j - 1];
        g_host_desc_n++;
        g_host_desc[best + 1].off = g_host_desc[best].off + need;
        g_host_desc[best + 1].size = g_host_desc[best].size - need;
        g_host_desc[best + 1].free = 1;
        g_host_desc[best].size = need;
    }
    g_host_desc[best].free = 0;
    ACUDA_DEBUG_PRINT("acuda-trace .. host_heap alloc %zu at %p\n", bytes, p);
    return p;
}

static int acuda_host_heap_free(void *p) {
    size_t off;
    int i;
    if (!g_host_base || !p)
        return 0;
    if ((unsigned char *)p < g_host_base ||
        (unsigned char *)p >= g_host_base + g_host_size)
        return 0;
    off = (size_t)((unsigned char *)p - g_host_base);
    for (i = 0; i < g_host_desc_n; i++) {
        if (g_host_desc[i].off == off && !g_host_desc[i].free) {
            g_host_desc[i].free = 1;
            if (g_host_desc[i].size)
                VirtualFree(p, g_host_desc[i].size, MEM_DECOMMIT);
            if (i + 1 < g_host_desc_n && g_host_desc[i + 1].free) {
                g_host_desc[i].size += g_host_desc[i + 1].size;
                memmove(&g_host_desc[i + 1], &g_host_desc[i + 2],
                        (size_t)(g_host_desc_n - i - 2) * sizeof(g_host_desc[0]));
                g_host_desc_n--;
            }
            if (i > 0 && g_host_desc[i - 1].free) {
                g_host_desc[i - 1].size += g_host_desc[i].size;
                memmove(&g_host_desc[i], &g_host_desc[i + 1],
                        (size_t)(g_host_desc_n - i - 1) * sizeof(g_host_desc[0]));
                g_host_desc_n--;
            }
            ACUDA_DEBUG_PRINT("acuda-trace .. host_heap free %p\n", p);
            return 1;
        }
    }
    return 0;
}
#endif

// Pinned host memory.  The arena is persistently mapped and coherent, so
// pinned and pageable host memory are the same thing here; cudart's
// cudaHostAlloc used by torch's caching host allocator only needs a
// stable host pointer that can feed the copy paths.
CUresult ACUDA_CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostAlloc(%zu flags=%x)\n",
                bytesize, flags);
    if (!pp)
        return CUDA_ERROR_INVALID_VALUE;
    *pp = NULL;
    if (bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    // CUDA host allocations are page-aligned and pinned.  Use the OS
    // virtual allocator on Windows: it bypasses the (already fragmented)
    // CRT heap in this 32-bit process and gives PhysX a page-aligned
    // staging buffer.  The CRT's own later malloc failures are what turn
    // into unchecked-NULL crashes inside PhysXCore.
#if defined(_WIN32)
    void *p = acuda_host_heap_alloc(bytesize);
    if (!p) {
#if defined(_WIN32) && !defined(_WIN64)
        /* In 32-bit LAA processes, prefer placing large standalone pinned
         * allocations above 2 GiB to preserve low VA for app/driver/CRT. */
        {
            size_t need = (bytesize + 0xFFFFu) & ~(size_t)0xFFFFu;
            void *high = acuda_host_heap_find_high(need, 1);
            if (high)
                p = VirtualAlloc(high, need, MEM_COMMIT | MEM_RESERVE,
                                 PAGE_READWRITE);
        }
        if (!p)
#endif
        p = VirtualAlloc(NULL, bytesize, MEM_COMMIT | MEM_RESERVE,
                         PAGE_READWRITE);
    }
#else
    void *p = malloc(bytesize);
#endif
    if (!p) {
        ACUDA_DEBUG_PRINT("acuda-trace .. cuMemHostAlloc(%zu) -> OOM\n",
                    bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *pp = p;
#if defined(_WIN32) && !defined(_WIN64)
    acuda_trace_va("cuMemHostAlloc");
#endif
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostAlloc_v2(void **pp, size_t bytesize, unsigned int flags) {
    return cuMemHostAlloc(pp, bytesize, flags);
}

CUresult ACUDA_CUDAAPI cuMemFreeHost(void *p) {
#if defined(_WIN32)
    if (p) {
        if (!acuda_host_heap_free(p))
            VirtualFree(p, 0, MEM_RELEASE);
    }
#else
    free(p);
#endif
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int flags) {
    (void)p;
    (void)bytesize;
    (void)flags;
    // All host memory is already accessible to the arena copies.
    return p ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult ACUDA_CUDAAPI cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags) {
    return cuMemHostRegister(p, bytesize, flags);
}

CUresult ACUDA_CUDAAPI cuMemHostUnregister(void *p) {
    return p ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult ACUDA_CUDAAPI cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamGetCtx\n");
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
    ACUDA_DEBUG_PRINT("acuda-trace >> cuStreamGetDevice\n");
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

/* Stubs for newer driver APIs that may be linked by torch */
CUresult acuda_stub_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                      const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s)\n",
                name ? name : "(null)");
    // `bytes` may be NULL when caller does not request the size.
    if (!dptr || !hmod || !name) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> invalid dptr=%p bytes=%p hmod=%p name=%p\n",
                    name ? name : "(null)", (void *)dptr, (void *)bytes, (void *)hmod, (void *)name);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dptr = 0;
    if (bytes) *bytes = 0;
    for (size_t i = 0; i < hmod->nglobals; i++) {
        if (strcmp(hmod->globals[i].name, name) == 0) {
            *dptr = hmod->globals[i].dptr;
            if (bytes) *bytes = hmod->globals[i].bytes;
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> cached 0x%llx %zu\n",
                        name, (unsigned long long)*dptr, hmod->globals[i].bytes);
            return CUDA_SUCCESS;
        }
    }
    // Legacy PhysX kernels keep their parameter block in a 10 KiB
    // __constant__ symbol.  Other names keep a generous default.
    size_t gsize = strcmp(name, "parameters_cmem") == 0 ? 10240 : 65536;
    CUdeviceptr p = 0;
    CUresult r = mem_alloc(&p, gsize);
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) alloc %zu -> r=%d p=0x%llx\n",
                name, gsize, (int)r, (unsigned long long)p);
    if (r != CUDA_SUCCESS)
        return r;
    if (hmod->nglobals == hmod->globals_cap) {
        size_t cap = hmod->globals_cap ? hmod->globals_cap * 2 : 4;
        void *ng = realloc(hmod->globals, cap * sizeof(*hmod->globals));
        if (!ng) {
            cuMemFree_v2(p);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        hmod->globals = ng;
        hmod->globals_cap = cap;
    }
    snprintf(hmod->globals[hmod->nglobals].name,
             sizeof(hmod->globals[hmod->nglobals].name), "%s", name);
    hmod->globals[hmod->nglobals].dptr = p;
    hmod->globals[hmod->nglobals].bytes = gsize;
    hmod->nglobals++;
    *dptr = p;
    if (bytes) *bytes = gsize;
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetGlobal(%s) -> 0x%llx %zu\n",
                name, (unsigned long long)*dptr, gsize);
    return CUDA_SUCCESS;
}
CUresult acuda_stub_cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                         const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}
CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                                CUfunction func,
                                                                int blockSize,
                                                                size_t dynamicSMemSize) {
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (!numBlocks) return CUDA_ERROR_INVALID_VALUE;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    *numBlocks = 1;
    return CUDA_SUCCESS;
}
CUresult acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
    (void)flags;
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func,
                                                                  blockSize,
                                                                  dynamicSMemSize);
}
CUresult acuda_stub_cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *maxBlockSize,
                                                     CUfunction func, size_t blockSizeLimit,
                                                     size_t dynamicSMemSize) {
    (void)func; (void)blockSizeLimit; (void)dynamicSMemSize;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    if (minGridSize) *minGridSize = 1;
    if (maxBlockSize) *maxBlockSize = 256;
    return CUDA_SUCCESS;
}
CUresult acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags) {
    (void)flags;
    return acuda_stub_cuOccupancyMaxPotentialBlockSize(minGridSize, maxBlockSize, func,
                                                       blockSizeLimit, dynamicSMemSize);
}
// CUDA 13's runtime fetches this small interface table with
// cuGetExportTable instead of resolving the corresponding entry points
// one by one.  It is queried during device enumeration; slot 12 is the
// "load compilers" callback, which is a no-op for our software path.
static CUresult ACUDA_CUDAAPI cudart_iface_load_compilers(void) {
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI cudart_iface_primary_ctx(CUcontext *pctx, CUdevice dev) {
    return cuDevicePrimaryCtxRetain(pctx, dev);
}

static CUresult ACUDA_CUDAAPI cudart_iface_primary_ctx_flags(CUdevice dev,
                                                              unsigned int flags) {
    CUcontext ctx;
    CUresult r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS)
        return r;
    return cuDevicePrimaryCtxSetFlags(dev, flags);
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_from_cubin(
    CUmodule *pmod, const void *fatbin) {
    if (!pmod || !fatbin)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData(pmod, fatbin);
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_ext1(
    CUmodule *pmod, const void *fatbin_wrapper, void *arg3, void *arg4,
    unsigned int arg5) {
    (void)arg3; (void)arg4; (void)arg5;
    if (!pmod || !fatbin_wrapper)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadFatBinary(pmod, fatbin_wrapper);
}

static void ACUDA_CUDAAPI cudart_iface_fn7(size_t arg) {
    (void)arg;
}

static CUresult ACUDA_CUDAAPI cudart_iface_module_ext2(
    const void *fatbin_header, CUmodule *pmod, void *arg3, void *arg4,
    unsigned int arg5) {
    (void)arg3; (void)arg4; (void)arg5;
    if (!pmod || !fatbin_header)
        return CUDA_ERROR_INVALID_VALUE;
    return cuModuleLoadData(pmod, fatbin_header);
}

// Opaque scratch buffers requested by the tools runtime hooks.  The
// caller treats them as private storage; zeroed memory is enough.
static unsigned int g_tools_buffer1[1024];
static unsigned int g_tools_buffer2[14];

static void ACUDA_CUDAAPI tools_runtime_buffer1(void **ptr, size_t *size) {
    if (ptr)
        *ptr = g_tools_buffer1;
    if (size)
        *size = sizeof(g_tools_buffer1) / sizeof(g_tools_buffer1[0]);
}

static void ACUDA_CUDAAPI tools_runtime_buffer2(void **ptr, size_t *size) {
    if (ptr)
        *ptr = g_tools_buffer2;
    if (size)
        *size = sizeof(g_tools_buffer2) / sizeof(g_tools_buffer2[0]);
}

// CUDA 4.x cudart calls this slot from cudaMallocHost with
// (event_id, event_data).  Nothing in the data is consumed by our
// software path, so a success return is enough.
static int ACUDA_CUDAAPI tools_runtime_event(unsigned int code, void *data) {
    (void)code;
    (void)data;
    return 0;
}

static const void *g_tools_tls[4] = {
    (const void *)(uintptr_t)(4 * sizeof(void *)),   /* [0] table size */
    NULL, NULL, NULL
};

static const void *g_tools_runtime_hooks[7] = {
    (const void *)(uintptr_t)(7 * sizeof(void *)),   /* [0] table size */
    (const void *)(uintptr_t)tools_runtime_event,    /* [1] */
    (const void *)(uintptr_t)tools_runtime_buffer1,  /* [2] */
    NULL, NULL, NULL,
    (const void *)(uintptr_t)tools_runtime_buffer2   /* [6] */
};

/* Legacy 32-bit CUDA 4.x tools table.  CUDA 4.x runtimes query this
 * interface during context setup and expect six callable entries.  The
 * calls only carry bookkeeping/query data for tools; safe zeroed replies
 * are enough for our non-instrumented runtime. */
static int ACUDA_CUDAAPI tools_legacy_query_index(unsigned index, void *out) {
    (void)index;
    if (out)
        *(unsigned int *)out = 0;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_pair_a(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0x400;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_ctx(void *ctx, void *out, void *extra) {
    (void)ctx;
    (void)extra;
    if (out)
        *(unsigned int *)out = 0;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_ctx2(void *ctx, void *out) {
    (void)ctx;
    if (out) {
        ((unsigned int *)out)[0] = 0;
        ((unsigned int *)out)[1] = 0;
    }
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_pair_b(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0xE;
    return 0;
}
static int ACUDA_CUDAAPI tools_legacy_zero(void *a, void *b) {
    if (a)
        *(unsigned int *)a = 0;
    if (b)
        *(unsigned int *)b = 0;
    return 0;
}
#if defined(__GNUC__)
__attribute__((unused))
#endif
static void *g_tools_runtime_hooks_legacy[7] = {
    (void *)(uintptr_t)(7 * sizeof(void *)),
    (void *)(uintptr_t)tools_legacy_query_index,
    (void *)(uintptr_t)tools_legacy_pair_a,
    (void *)(uintptr_t)tools_legacy_ctx,
    (void *)(uintptr_t)tools_legacy_ctx2,
    (void *)(uintptr_t)tools_legacy_pair_b,
    (void *)(uintptr_t)tools_legacy_zero
};

// Context-local storage (C693336E), used by the tools layer to attach
// small records to a context.  A fixed table is enough for our use.
#define ACUDA_CTX_STORAGE_MAX 64
typedef struct {
    CUcontext ctx;
    void *key;
    void *value;
    void (*dtor)(CUcontext, void *, void *);
} acuda_ctx_slot_t;
static acuda_ctx_slot_t g_ctx_slots[ACUDA_CTX_STORAGE_MAX];

static CUresult ACUDA_CUDAAPI ctx_storage_put(CUcontext ctx, void *key, void *value,
                                void (*dtor)(CUcontext, void *, void *)) {
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            g_ctx_slots[i].value = value;
            g_ctx_slots[i].dtor = dtor;
            return CUDA_SUCCESS;
        }
    }
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].key == NULL) {
            g_ctx_slots[i].ctx = ctx;
            g_ctx_slots[i].key = key;
            g_ctx_slots[i].value = value;
            g_ctx_slots[i].dtor = dtor;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

static CUresult ACUDA_CUDAAPI ctx_storage_delete(CUcontext ctx, void *key) {
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            g_ctx_slots[i].key = NULL;
            g_ctx_slots[i].ctx = NULL;
            g_ctx_slots[i].value = NULL;
            g_ctx_slots[i].dtor = NULL;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI ctx_storage_get(void **value, CUcontext ctx,
                                              void *key) {
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    for (int i = 0; i < ACUDA_CTX_STORAGE_MAX; i++) {
        if (g_ctx_slots[i].ctx == ctx && g_ctx_slots[i].key == key) {
            *value = g_ctx_slots[i].value;
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

// The context-storage export table has no size slot: [0] put (4 args),
// [1] delete (2 args), [2] get (3 args).  The real driver's table is
// exactly three function pointers.
static const void *g_ctx_storage[3] = {
    (const void *)(uintptr_t)ctx_storage_put,
    (const void *)(uintptr_t)ctx_storage_delete,
    (const void *)(uintptr_t)ctx_storage_get
};

// CTX_CREATE_BYPASS: the CUDA 4.x runtime uses this interface to create
// the context through the driver without its own thread-local wrapper.
static CUresult ACUDA_CUDAAPI ctx_create_v2_bypass(CUcontext *pctx,
                                                   unsigned int flags,
                                                   CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

static const void *g_ctx_create_bypass[2] = {
    (const void *)(uintptr_t)(2 * sizeof(void *)),
    (const void *)(uintptr_t)ctx_create_v2_bypass
};

// CONTEXT_CHECKS (263E8860): both probes are advisory; cudart only
// records the answers.
static CUresult ACUDA_CUDAAPI context_check(CUcontext ctx, unsigned int *result1,
                              const void **result2) {
    (void)ctx;
    (void)result2;
    if (result1)
        *result1 = 0;
    return CUDA_SUCCESS;
}

static unsigned int ACUDA_CUDAAPI check_fn3(void) { return 0; }

static const void *g_context_checks[4] = {
    (const void *)(uintptr_t)(4 * sizeof(void *)),  /* [0] table size */
    NULL,
    (const void *)(uintptr_t)context_check,         /* [2] */
    (const void *)(uintptr_t)check_fn3              /* [3] */
};

// HEAP_ACCESS (195BCBF4-D67D-024A-ACC5-1D29CEA631AE): cudart 6.x queries
// this table during context setup and fails every allocation with
// "unknown error" (30) when it is missing.  The driver allocates an
// opaque per-context heap record; cudart stores the pointer and hands
// it back to heap_free at teardown.  Contents are driver-private, so a
// zeroed block is enough.
static CUresult ACUDA_CUDAAPI heap_alloc(void **rec, void *a, void *b) {
    (void)a;
    (void)b;
    if (!rec)
        return CUDA_ERROR_INVALID_VALUE;
    void *p = calloc(1, 64);
    if (!p)
        return CUDA_ERROR_OUT_OF_MEMORY;
    *rec = p;
    return CUDA_SUCCESS;
}

static CUresult ACUDA_CUDAAPI heap_free(void *rec, void *extra) {
    (void)extra;
    free(rec);
    return CUDA_SUCCESS;
}

static const void *g_heap_access[3] = {
    (const void *)(uintptr_t)(3 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)heap_alloc,            /* [1] */
    (const void *)(uintptr_t)heap_free              /* [2] */
};

static const void *g_cudart_interface[13];
static const void *g_integrity_check[3];

// INTEGRITY_CHECK (D4082055): CUDA 13 recomputes this checksum to
// confirm that the driver and runtime interface tables agree.  The
// mixing table and round function are the interoperability contract; the
// C implementation below is independent, cross-checked against the
// ZLUDA reference (dark_api/src/lib.rs) and its test vectors.
static const unsigned char acuda_integrity_mix[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13,
    0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca,
    0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12,
    0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a,
    0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b, 0x22, 0x5f, 0x21,
    0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03,
    0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6,
    0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1,
    0x45, 0x9d, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02,
    0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f,
    0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81, 0x4d, 0x52,
    0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a,
    0x78, 0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39,
    0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a,
    0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

static void acuda_integrity_round(unsigned char state[66], unsigned char v) {
    unsigned t1 = state[0x40];
    state[t1 + 0x10] = v;
    unsigned t2 = t1;
    unsigned t3 = (t1 + 1) & 0xf;
    state[t2 + 0x20] = state[t2] ^ v;
    unsigned t4 = acuda_integrity_mix[(v ^ state[0x41]) & 0xff];
    unsigned t5 = state[t2 + 0x30];
    state[t2 + 0x30] = (unsigned char)(t4 ^ t5);
    state[0x41] = (unsigned char)(t4 ^ t5);
    state[0x40] = (unsigned char)t3;
    if (t3 != 0)
        return;
    unsigned char a = 0x29;
    unsigned char c = 0;
    for (;;) {
        a ^= state[0];
        state[0] = a;
        for (unsigned i = 1; i < 0x30; i++) {
            a = (unsigned char)(state[i] ^ acuda_integrity_mix[a]);
            state[i] = a;
        }
        a = (unsigned char)(a + c);
        c = (unsigned char)(c + 1);
        if (c == 0x12)
            break;
        a = acuda_integrity_mix[a];
    }
}

static void acuda_integrity_hash(unsigned char state[66], const void *mix,
                                 size_t len, unsigned char x) {
    const unsigned char *p = (const unsigned char *)mix;
    for (size_t i = 0; i < len; i++)
        acuda_integrity_round(state, (unsigned char)(p[i] ^ x));
}

static void acuda_integrity_finish(unsigned char state[66],
                                   unsigned long long out[2]) {
    unsigned char t = (unsigned char)(16 - state[64]);
    for (unsigned i = 0; i < t; i++)
        acuda_integrity_round(state, t);
    for (unsigned i = 0x30; i < 0x40; i++)
        acuda_integrity_round(state, state[i]);
    memcpy(&out[0], state, sizeof(out[0]));
    memcpy(&out[1], state + 8, sizeof(out[1]));
}

typedef struct {
    uint32_t driver_version;
    uint32_t version;
    uint32_t current_process;
    uint32_t current_thread;
    const void *cudart_table;
    const void *integrity_table;
    const void *fn_address;
    uint64_t unix_seconds;
} acuda_integrity_pass3_t;

typedef struct {
    CUuuid guid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} acuda_integrity_dev_t;

static CUresult ACUDA_CUDAAPI integrity_check(unsigned int version,
                                unsigned long long unix_seconds,
                                unsigned long long result[2]) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? integrity_check(version=%u)\n",
                version);
    if (!result)
        return CUDA_ERROR_INVALID_VALUE;
    switch (version % 10) {
    case 0:
        result[0] = 0x3341181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    case 1:
        result[0] = 0x1841181C03CB675CULL;
        result[1] = 0x8ED383AA1F4CD1E8ULL;
        return CUDA_SUCCESS;
    default:
        break;
    }

    static const unsigned char pass1[16] = {
        0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
        0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f
    };
    unsigned char state[66];
    memset(state, 0, sizeof(state));
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x36);

#if defined(_WIN32)
    uint32_t pid = (uint32_t)GetCurrentProcessId();
    uint32_t tid = (uint32_t)GetCurrentThreadId();
#else
    uint32_t pid = (uint32_t)getpid();
    uint32_t tid = (uint32_t)(uintptr_t)pthread_self();
#endif
    acuda_integrity_pass3_t p3;
    memset(&p3, 0, sizeof(p3));
    p3.driver_version = ACUDA_DRIVER_VERSION;
    p3.version = version;
    p3.current_process = pid;
    p3.current_thread = tid;
    p3.cudart_table = (const void *)g_cudart_interface;
    p3.integrity_table = (const void *)g_integrity_check;
    p3.fn_address = (const void *)(uintptr_t)integrity_check;
    p3.unix_seconds = unix_seconds;
    acuda_integrity_hash(state, &p3, sizeof(p3), 0);

    int ndev = 0;
    if (cuDeviceGetCount(&ndev) == CUDA_SUCCESS) {
        for (int d = 0; d < ndev; d++) {
            acuda_integrity_dev_t dev;
            memset(&dev, 0, sizeof(dev));
            cuDeviceGetUuid(&dev.guid, d);
            int v = 0;
            if (cuDeviceGetAttribute(&v, 50, d) == CUDA_SUCCESS)
                dev.pci_domain = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 33, d) == CUDA_SUCCESS)
                dev.pci_bus = v;
            v = 0;
            if (cuDeviceGetAttribute(&v, 34, d) == CUDA_SUCCESS)
                dev.pci_device = v;
            acuda_integrity_hash(state, &dev, sizeof(dev), 0);
        }
    }

    acuda_integrity_finish(state, result);   /* pass5 #1 */
    unsigned long long pass5_1[2] = { result[0], result[1] };
    memset(state, 0, 16);
    memset(state + 48, 0, 66 - 48);
    acuda_integrity_hash(state, pass1, sizeof(pass1), 0x5c);
    acuda_integrity_hash(state, pass5_1, sizeof(pass5_1), 0);
    acuda_integrity_finish(state, result);
    return CUDA_SUCCESS;
}

static const void *g_integrity_check[3] = {
    (const void *)(uintptr_t)(3 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)integrity_check        /* [1] */
};
// CUDA 13.4 asks for this large table during stream setup.  Only the
// size slot is needed for initialisation; unknown entries stay NULL and
// the optional tool paths that use them are not reached by our runtime.
static const void *g_cuda134_tools[64] = {
    (const void *)(uintptr_t)(64 * sizeof(void *)),
};

/* Slot 5 is the driver module handle: the 32-bit CUDA 6.5 runtime passes
 * table[5] as the HMODULE to resolve its ~140 driver entries by name.
 * NULL here fails every resolution, leaving NULL call targets (EIP=0).
 * Filled lazily with our own base; name-independent. */
static const void *g_cudart_interface[13] = {
    (const void *)(uintptr_t)(13 * sizeof(void *)),  /* [0] table size */
    (const void *)(uintptr_t)cudart_iface_module_from_cubin, /* [1] */
    (const void *)(uintptr_t)cudart_iface_primary_ctx,/* [2] primary ctx */
    NULL,                                             /* [3] */
    (const void *)(uintptr_t)cudart_iface_primary_ctx_flags, /* [4] */
    NULL,                                             /* [5] */
    (const void *)(uintptr_t)cudart_iface_module_ext1, /* [6] */
    (const void *)(uintptr_t)cudart_iface_fn7,         /* [7] */
    (const void *)(uintptr_t)cudart_iface_module_ext2, /* [8] */
    NULL,                                             /* [9] */
    NULL,                                             /* [10] */
    NULL,                                             /* [11] */
    (const void *)(uintptr_t)cudart_iface_load_compilers /* [12] */
};

/* See the slot-5 note above. */
static void ensure_cudart_iface(void) {
    if (g_cudart_interface[5])
        return;
#if defined(_WIN32)
    {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery((LPCVOID)(uintptr_t)&cuGetExportTable, &mbi,
                         sizeof(mbi)) &&
            mbi.AllocationBase)
            g_cudart_interface[5] = mbi.AllocationBase;
    }
#else
    {
        Dl_info info;
        if (dladdr((const void *)(uintptr_t)&cuGetExportTable, &info) &&
            info.dli_fbase)
            g_cudart_interface[5] = info.dli_fbase;
    }
#endif
    ACUDA_DEBUG_PRINT("acuda-trace .. cudart_iface slot5=%p\n",
                g_cudart_interface[5]);
}

CUresult ACUDA_CUDAAPI cuGetExportTable(const void **ppExportTable,
                          const struct CUuuid_st *pExportTableId) {
    // {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}
    static const char cudart_guid[16] = {
        (char)0x6b, (char)0xd5, (char)0xfb, (char)0x6c,
        (char)0x5b, (char)0xf4, (char)0xe7, (char)0x4a,
        (char)0x89, (char)0x87, (char)0xd9, (char)0x39,
        (char)0x12, (char)0xfd, (char)0x9d, (char)0xf9
    };
    if (!ppExportTable || !pExportTableId)
        return CUDA_ERROR_INVALID_VALUE;
    if (getenv("ACUDA_TRACE")) {
#if defined(_M_IX86)
        void *retaddr = _ReturnAddress();
        MEMORY_BASIC_INFORMATION mbi;
        char modpath[MAX_PATH] = "?";
        if (VirtualQuery(retaddr, &mbi, sizeof(mbi)))
            GetModuleFileNameA((HMODULE)mbi.AllocationBase, modpath, MAX_PATH);
        fprintf(stderr, "acuda-trace .. cuGetExportTable ret=%p rva=%Ix mod=%s pp=%p\n",
                retaddr, (uintptr_t)retaddr - (uintptr_t)mbi.AllocationBase,
                modpath, (const void *)ppExportTable);
#endif
        fprintf(stderr, "acuda-trace ?? cuGetExportTable(%02x%02x%02x%02x-"
                "%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x)\n",
                (unsigned char)pExportTableId->bytes[0],
                (unsigned char)pExportTableId->bytes[1],
                (unsigned char)pExportTableId->bytes[2],
                (unsigned char)pExportTableId->bytes[3],
                (unsigned char)pExportTableId->bytes[4],
                (unsigned char)pExportTableId->bytes[5],
                (unsigned char)pExportTableId->bytes[6],
                (unsigned char)pExportTableId->bytes[7],
                (unsigned char)pExportTableId->bytes[8],
                (unsigned char)pExportTableId->bytes[9],
                (unsigned char)pExportTableId->bytes[10],
                (unsigned char)pExportTableId->bytes[11],
                (unsigned char)pExportTableId->bytes[12],
                (unsigned char)pExportTableId->bytes[13],
                (unsigned char)pExportTableId->bytes[14],
                (unsigned char)pExportTableId->bytes[15]);
    }
    // {A094798C-2E74-2E74-93F2-0800200C0A66}
    static const char tools_guid[16] = {
        (char)0xa0, (char)0x94, (char)0x79, (char)0x8c,
        (char)0x2e, (char)0x74, (char)0x2e, (char)0x74,
        (char)0x93, (char)0xf2, (char)0x08, (char)0x00,
        (char)0x20, (char)0x0c, (char)0x0a, (char)0x66
    };
    if (memcmp(pExportTableId->bytes, cudart_guid, sizeof(cudart_guid)) == 0) {
        ensure_cudart_iface();
        *ppExportTable = (const void *)g_cudart_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, tools_guid, sizeof(tools_guid)) == 0) {
        ACUDA_DEBUG_PRINT("acuda-trace .. tools branch pp=%p\n",
                    (const void *)ppExportTable);
        *ppExportTable = (const void *)g_tools_runtime_hooks;
        return CUDA_SUCCESS;
    }
    // {42D85A81-23F6-CB47-8298-F6E78A3AECDC}
    static const char tls_guid[16] = {
        (char)0x42, (char)0xd8, (char)0x5a, (char)0x81,
        (char)0x23, (char)0xf6, (char)0xcb, (char)0x47,
        (char)0x82, (char)0x98, (char)0xf6, (char)0xe7,
        (char)0x8a, (char)0x3a, (char)0xec, (char)0xdc
    };
    if (memcmp(pExportTableId->bytes, tls_guid, sizeof(tls_guid)) == 0) {
        *ppExportTable = (const void *)g_tools_tls;
        return CUDA_SUCCESS;
    }
    // {C693336E-1121-DF11-A8C3-68F355D89593}
    static const char ctx_storage_guid[16] = {
        (char)0xc6, (char)0x93, (char)0x33, (char)0x6e,
        (char)0x11, (char)0x21, (char)0xdf, (char)0x11,
        (char)0xa8, (char)0xc3, (char)0x68, (char)0xf3,
        (char)0x55, (char)0xd8, (char)0x95, (char)0x93
    };
    if (memcmp(pExportTableId->bytes, ctx_storage_guid,
               sizeof(ctx_storage_guid)) == 0) {
        *ppExportTable = (const void *)g_ctx_storage;
        return CUDA_SUCCESS;
    }
    // {0CA50B8C-1004-929A-89A7-D0DF10E77286}
    static const char ctx_create_guid[16] = {
        (char)0x0c, (char)0xa5, (char)0x0b, (char)0x8c,
        (char)0x10, (char)0x04, (char)0x92, (char)0x9a,
        (char)0x89, (char)0xa7, (char)0xd0, (char)0xdf,
        (char)0x10, (char)0xe7, (char)0x72, (char)0x86
    };
    if (memcmp(pExportTableId->bytes, ctx_create_guid,
               sizeof(ctx_create_guid)) == 0) {
        *ppExportTable = (const void *)g_ctx_create_bypass;
        return CUDA_SUCCESS;
    }
    // {263E8860-7CD2-6143-92F6-BBD5006DFA7E}
    static const char ctx_checks_guid[16] = {
        (char)0x26, (char)0x3e, (char)0x88, (char)0x60,
        (char)0x7c, (char)0xd2, (char)0x61, (char)0x43,
        (char)0x92, (char)0xf6, (char)0xbb, (char)0xd5,
        (char)0x00, (char)0x6d, (char)0xfa, (char)0x7e
    };
    if (memcmp(pExportTableId->bytes, ctx_checks_guid,
               sizeof(ctx_checks_guid)) == 0) {
        *ppExportTable = (const void *)g_context_checks;
        return CUDA_SUCCESS;
    }
    // {D4082055-BDE6-704B-8D34-BA123C66E1F2}
    static const char integrity_guid[16] = {
        (char)0xd4, (char)0x08, (char)0x20, (char)0x55,
        (char)0xbd, (char)0xe6, (char)0x70, (char)0x4b,
        (char)0x8d, (char)0x34, (char)0xba, (char)0x12,
        (char)0x3c, (char)0x66, (char)0xe1, (char)0xf2
    };
    if (memcmp(pExportTableId->bytes, integrity_guid,
               sizeof(integrity_guid)) == 0) {
        *ppExportTable = (const void *)g_integrity_check;
        return CUDA_SUCCESS;
    }
    // {21318C60-9714-3248-8CA6-41FF7324C8F2} (CUDA 13.4)
    static const char cuda134_guid[16] = {
        (char)0x21, (char)0x31, (char)0x8c, (char)0x60,
        (char)0x97, (char)0x14, (char)0x32, (char)0x48,
        (char)0x8c, (char)0xa6, (char)0x41, (char)0xff,
        (char)0x73, (char)0x24, (char)0xc8, (char)0xf2
    };
    if (memcmp(pExportTableId->bytes, cuda134_guid,
               sizeof(cuda134_guid)) == 0) {
        *ppExportTable = (const void *)g_cuda134_tools;
        return CUDA_SUCCESS;
    }
    // {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} (heap access, cudart 6.x allocs)
    static const char heap_guid[16] = {
        (char)0x19, (char)0x5b, (char)0xcb, (char)0xf4,
        (char)0xd6, (char)0x7d, (char)0x02, (char)0x4a,
        (char)0xac, (char)0xc5, (char)0x1d, (char)0x29,
        (char)0xce, (char)0xa6, (char)0x31, (char)0xae
    };
    if (memcmp(pExportTableId->bytes, heap_guid,
               sizeof(heap_guid)) == 0) {
        *ppExportTable = (const void *)g_heap_access;
        return CUDA_SUCCESS;
    }
    *ppExportTable = NULL;
    return CUDA_ERROR_NOT_FOUND;
}
typedef struct CUtensorMap_st *CUtensorMap_t;
typedef enum { CU_TENSOR_MAP_DATA_TYPE_UINT8_s = 0 } CUtensorMapDataType_s;
typedef enum { CU_TENSOR_MAP_INTERLEAVE_NONE_s = 0 } CUtensorMapInterleave_s;
typedef enum { CU_TENSOR_MAP_SWIZZLE_NONE_s = 0 } CUtensorMapSwizzle_s;
typedef enum { CU_TENSOR_MAP_L2_PROMOTION_NONE_s = 0 } CUtensorMapL2promotion_s;
typedef enum { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE_s = 0 } CUtensorMapFloatOOBfill_s;
CUresult acuda_stub_cuTensorMapEncodeTiled(void *tensorMap, int tensorDataType,
                                           unsigned int tensorRank, void *globalAddress,
                                           const unsigned long long *globalDim, const unsigned long long *globalStrides,
                                           const unsigned int *boxDim, const unsigned int *elementStrides,
                                           int interleave, int swizzle,
                                           int l2Promotion, int oobFill) {
    (void)tensorMap;(void)tensorDataType;(void)tensorRank;(void)globalAddress;(void)globalDim;(void)globalStrides;(void)boxDim;(void)elementStrides;(void)interleave;(void)swizzle;(void)l2Promotion;(void)oobFill;
    if (need_init() != CUDA_SUCCESS) return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ---- CUDA texture arrays and texture objects -------------------------

static int driver_elem_bytes(uint32_t fmt) {
    switch (fmt) {
    case 0: return 16;
    case 1: return 4;
    case 2: return 4;
    case 3: return 4;
    case 4: return 4;
    }
    return 0;
}

static int driver_array_format(CUarray_format f, unsigned nchans,
                               unsigned flags, uint32_t *fmt) {
    if (flags & CUDA_ARRAY3D_DEPTH_TEXTURE) {
        if (f == CU_AD_FORMAT_FLOAT && nchans == 1) {
            *fmt = 4;
            return 0;
        }
        return -1;
    }
    switch (f) {
    case CU_AD_FORMAT_FLOAT:
        if (nchans == 4) { *fmt = 0; return 0; }
        if (nchans == 1) { *fmt = 2; return 0; }
        return -1;
    case CU_AD_FORMAT_UNSIGNED_INT32:
        if (nchans == 1) { *fmt = 3; return 0; }
        return -1;
    case CU_AD_FORMAT_UNSIGNED_INT8:
        if (nchans == 4) { *fmt = 1; return 0; }
        return -1;
    default:
        return -1;
    }
}

static void driver_array_shape(const CUDA_ARRAY3D_DESCRIPTOR *d,
                               uint32_t *dim, uint32_t *layers,
                               uint32_t *cube) {
    *cube = (d->Flags & CUDA_ARRAY3D_CUBEMAP) ? 1u : 0u;
    if (d->Flags & CUDA_ARRAY3D_LAYERED) {
        *dim = d->Height <= 1 ? 1u : 2u;
        *layers = (uint32_t)d->Depth;
    } else if (d->Depth > 1) {
        *dim = 3;
        *layers = 1;
    } else if (d->Height <= 1) {
        *dim = 1;
        *layers = 1;
    } else {
        *dim = 2;
        *layers = 1;
    }
    if (*cube && *layers < 6)
        *layers = 6; // CUDA cube arrays are six layers per cube
}

static CUresult driver_array_alloc(CUarray *out,
                                   const CUDA_ARRAY3D_DESCRIPTOR *d,
                                   unsigned levels, acuda_buffer_t shared) {
    if (!out || !d || d->Width == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t fmt;
    if (driver_array_format(d->Format, d->NumChannels, d->Flags, &fmt) != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    uint32_t dim, layers, cube;
    driver_array_shape(d, &dim, &layers, &cube);
    if (cube && (layers % 6) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t img = shared;
    if (!img) {
        acuda_image_desc id = {
            .width = (uint32_t)d->Width,
            .height = dim == 1 ? 1u : (uint32_t)d->Height,
            .depth = dim == 3 ? (uint32_t)d->Depth : 1u,
            .layers = layers,
            .levels = levels ? levels : 1,
            .samples = 1,
            .format = fmt,
            .dim = dim,
            .cube = cube,
        };
        if (acuda_image_create(g_actx, &img, &id, NULL) != ACUDA_SUCCESS)
            return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUarray a = calloc(1, sizeof(*a));
    if (!a) {
        if (!shared)
            acuda_free(g_actx, img);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    a->image = img;
    a->own = shared ? 0 : 1;
    a->level = 0;
    a->desc3 = *d;
    a->is3d = 1;
    a->desc2.Width = d->Width;
    a->desc2.Height = d->Height;
    a->desc2.Format = d->Format;
    a->desc2.NumChannels = d->NumChannels;
    *out = a;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuArrayCreate_v2\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pHandle || !pAllocateArray)
        return CUDA_ERROR_INVALID_VALUE;
    CUDA_ARRAY3D_DESCRIPTOR d = {
        .Width = pAllocateArray->Width,
        .Height = pAllocateArray->Height,
        .Depth = 1,
        .Format = pAllocateArray->Format,
        .NumChannels = pAllocateArray->NumChannels,
        .Flags = 0,
    };
    return driver_array_alloc(pHandle, &d, 1, NULL);
}

CUresult ACUDA_CUDAAPI cuArrayCreate(CUarray *pHandle,
                       const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    return cuArrayCreate_v2(pHandle, pAllocateArray);
}

CUresult ACUDA_CUDAAPI cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuArray3DCreate_v2\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    return driver_array_alloc(pHandle, pAllocateArray, 1, NULL);
}

CUresult ACUDA_CUDAAPI cuArray3DCreate(CUarray *pHandle,
                         const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    return cuArray3DCreate_v2(pHandle, pAllocateArray);
}

CUresult ACUDA_CUDAAPI cuArrayDestroy(CUarray hArray) {
    if (!hArray)
        return CUDA_ERROR_INVALID_VALUE;
    if (hArray->own && hArray->image)
        acuda_free(g_actx, hArray->image);
    free(hArray);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                              CUarray hArray) {
    if (!pArrayDescriptor || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    *pArrayDescriptor = hArray->desc2;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                CUarray hArray) {
    if (!pArrayDescriptor || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    *pArrayDescriptor = hArray->desc3;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayCreate(
    CUmipmappedArray *pHandle,
    const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
    unsigned int numMipmapLevels) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuMipmappedArrayCreate\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pHandle || !pMipmappedArrayDesc || numMipmapLevels == 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t fmt;
    if (driver_array_format(pMipmappedArrayDesc->Format,
                            pMipmappedArrayDesc->NumChannels,
                            pMipmappedArrayDesc->Flags, &fmt) != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    uint32_t dim, layers, cube;
    driver_array_shape(pMipmappedArrayDesc, &dim, &layers, &cube);
    acuda_image_desc id = {
        .width = (uint32_t)pMipmappedArrayDesc->Width,
        .height = dim == 1 ? 1u : (uint32_t)pMipmappedArrayDesc->Height,
        .depth = dim == 3 ? (uint32_t)pMipmappedArrayDesc->Depth : 1u,
        .layers = layers,
        .levels = numMipmapLevels,
        .samples = 1,
        .format = fmt,
        .dim = dim,
        .cube = cube,
    };
    CUmipmappedArray m = calloc(1, sizeof(*m));
    if (!m)
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (acuda_image_create(g_actx, &m->image, &id, NULL) != ACUDA_SUCCESS) {
        free(m);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    m->desc = *pMipmappedArrayDesc;
    m->levels = numMipmapLevels;
    m->is3d = dim == 3;
    *pHandle = m;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    if (!hMipmappedArray)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_free(g_actx, hMipmappedArray->image);
    free(hMipmappedArray);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                  CUmipmappedArray hMipmappedArray,
                                  unsigned int level) {
    if (!pLevelArray || !hMipmappedArray || level >= hMipmappedArray->levels)
        return CUDA_ERROR_INVALID_VALUE;
    CUarray a = calloc(1, sizeof(*a));
    if (!a)
        return CUDA_ERROR_OUT_OF_MEMORY;
    a->image = hMipmappedArray->image;
    a->own = 0;
    a->parent = NULL;
    a->level = level;
    a->desc3 = hMipmappedArray->desc;
    a->desc2.Width = hMipmappedArray->desc.Width;
    a->desc2.Height = hMipmappedArray->desc.Height;
    a->desc2.Format = hMipmappedArray->desc.Format;
    a->desc2.NumChannels = hMipmappedArray->desc.NumChannels;
    a->is3d = hMipmappedArray->is3d;
    *pLevelArray = a;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject,
                           const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuTexObjectCreate\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pTexObject || !pResDesc)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t image = NULL;
    acuda_buffer_t own_img = NULL;
    int depth_tex = 0;
    int is_array = 0;
    if (pResDesc->resType == CU_RESOURCE_TYPE_ARRAY) {
        CUarray a = pResDesc->res.array.hArray;
        if (!a)
            return CUDA_ERROR_INVALID_VALUE;
        image = a->image;
        depth_tex = (a->desc3.Flags & CUDA_ARRAY3D_DEPTH_TEXTURE) != 0;
        is_array = 1;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
        CUmipmappedArray m = pResDesc->res.mipmap.hMipmappedArray;
        if (!m)
            return CUDA_ERROR_INVALID_VALUE;
        image = m->image;
        depth_tex = (m->desc.Flags & CUDA_ARRAY3D_DEPTH_TEXTURE) != 0;
        is_array = 1;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_LINEAR) {
        // Linear textures snapshot the backing memory into a 1D image.
        uint32_t fmt = 0;
        if (driver_array_format(pResDesc->res.linear.format,
                                pResDesc->res.linear.numChannels, 0,
                                &fmt) != 0)
            return CUDA_ERROR_NOT_SUPPORTED;
        size_t elem = (size_t)driver_elem_bytes(fmt);
        size_t bytes = pResDesc->res.linear.sizeInBytes;
        if (elem == 0 || bytes < elem)
            return CUDA_ERROR_INVALID_VALUE;
        void *host = NULL;
        if (host_for(pResDesc->res.linear.devPtr, bytes, &host) !=
            CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        acuda_image_desc id = {
            .width = (uint32_t)(bytes / elem),
            .height = 1,
            .depth = 1,
            .layers = 1,
            .levels = 1,
            .samples = 1,
            .format = fmt,
            .dim = 1,
            .cube = 0,
        };
        acuda_status_t icst = acuda_image_create(g_actx, &own_img, &id,
                                                         host);
        if (icst != ACUDA_SUCCESS) {
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexObjectCreate(linear): image create failed st=%d w=%u fmt=%d bytes=%zu\n",
                        (int)icst, (unsigned)id.width, (int)fmt, bytes);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        image = own_img;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_PITCH2D) {
        uint32_t fmt = 0;
        if (driver_array_format(pResDesc->res.pitch2D.format,
                                pResDesc->res.pitch2D.numChannels, 0,
                                &fmt) != 0)
            return CUDA_ERROR_NOT_SUPPORTED;
        size_t elem = (size_t)driver_elem_bytes(fmt);
        size_t w = pResDesc->res.pitch2D.width;
        size_t h = pResDesc->res.pitch2D.height;
        size_t pitch = pResDesc->res.pitch2D.pitchInBytes;
        size_t row = w * elem;
        if (elem == 0 || w == 0 || h == 0 || row > pitch)
            return CUDA_ERROR_INVALID_VALUE;
        void *host = NULL;
        if (host_for(pResDesc->res.pitch2D.devPtr, pitch * h, &host) !=
            CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *tight = malloc(row * h);
        if (!tight)
            return CUDA_ERROR_OUT_OF_MEMORY;
        for (size_t y = 0; y < h; y++)
            memcpy(tight + y * row,
                   (const unsigned char *)host + y * pitch, row);
        acuda_image_desc id = {
            .width = (uint32_t)w,
            .height = (uint32_t)h,
            .depth = 1,
            .layers = 1,
            .levels = 1,
            .samples = 1,
            .format = fmt,
            .dim = 2,
            .cube = 0,
        };
        acuda_status_t ist = acuda_image_create(g_actx, &own_img, &id,
                                                tight);
        free(tight);
        if (ist != ACUDA_SUCCESS)
            return CUDA_ERROR_OUT_OF_MEMORY;
        image = own_img;
    } else {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    (void)pResViewDesc;
    acuda_sampler_desc sd;
    memset(&sd, 0, sizeof(sd));
    CUDA_TEXTURE_DESC td;
    if (pTexDesc)
        td = *pTexDesc;
    else
        memset(&td, 0, sizeof(td));
    sd.filter = td.filterMode == CU_TR_FILTER_MODE_LINEAR ? 1u : 0u;
    sd.mip_filter = td.mipmapFilterMode == CU_TR_FILTER_MODE_LINEAR ? 1u : 0u;
    for (int i = 0; i < 3; i++) {
        switch (td.addressMode[i]) {
        case CU_TR_ADDRESS_MODE_CLAMP: sd.addr[i] = 2; break;
        case CU_TR_ADDRESS_MODE_MIRROR: sd.addr[i] = 1; break;
        case CU_TR_ADDRESS_MODE_BORDER: sd.addr[i] = 3; break;
        default: sd.addr[i] = 0; break;
        }
    }
    // Arrays require normalized coordinates; linear/pitch2D honour the
    // CU_TRSF_NORMALIZED_COORDINATES flag (bit 1).
    sd.normalized = is_array ? 1u : ((td.flags & 0x2u) ? 1u : 0u);
    sd.compare = depth_tex ? 1u : 0u;
    sd.compare_op = 3; // LESS_OR_EQUAL for depth texture references
    for (int i = 0; i < 4; i++)
        sd.border[i] = td.borderColor[i];

    acuda_buffer_t smp = NULL, tex = NULL;
    if (acuda_sampler_create(g_actx, &smp, &sd) != ACUDA_SUCCESS) {
        if (own_img)
            acuda_free(g_actx, own_img);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (acuda_texture_create_image(g_actx, &tex, image, smp) !=
        ACUDA_SUCCESS) {
        acuda_free(g_actx, smp);
        if (own_img)
            acuda_free(g_actx, own_img);
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (g_ntexobjs == g_ctexobjs) {
        size_t ncap = g_ctexobjs ? g_ctexobjs * 2 : 8;
        driver_texobj_t *n = realloc(g_texobjs, sizeof(*n) * ncap);
        if (!n) {
            acuda_free(g_actx, tex);
            acuda_free(g_actx, smp);
            if (own_img)
                acuda_free(g_actx, own_img);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        g_texobjs = n;
        g_ctexobjs = ncap;
    }
    driver_texobj_t *e = &g_texobjs[g_ntexobjs++];
    memset(e, 0, sizeof(*e));
    e->id = g_next_texid++;
    if (e->id == 0)
        e->id = g_next_texid++;
    e->tex = tex;
    e->sampler = smp;
    e->own_image = own_img;
    e->res = *pResDesc;
    e->td = td;
    *pTexObject = e->id;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
    for (size_t i = 0; i < g_ntexobjs; i++) {
        if (g_texobjs[i].id != texObject)
            continue;
        acuda_free(g_actx, g_texobjs[i].tex);
        acuda_free(g_actx, g_texobjs[i].sampler);
        if (g_texobjs[i].own_image)
            acuda_free(g_actx, g_texobjs[i].own_image);
        g_texobjs[i] = g_texobjs[--g_ntexobjs];
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult ACUDA_CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject) {
    driver_texobj_t *e = driver_tex_find(texObject);
    if (!pResDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = e->res;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject) {
    driver_texobj_t *e = driver_tex_find(texObject);
    if (!pTexDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pTexDesc = e->td;
    return CUDA_SUCCESS;
}

// ---- legacy texture references (CUDA 4.x / PhysX 2.8) ----------------

static CUresult texref_rebuild(CUtexref h, int force) {
    if (!h)
        return CUDA_ERROR_INVALID_VALUE;
    if (h->is_array) {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
    } else if (h->bytes == 0 || !h->fmt_set) {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
        return CUDA_SUCCESS;
    } else if (!force && h->obj && h->built_addr == h->addr &&
               h->built_bytes == h->bytes && h->built_fmt == h->fmt &&
               h->built_channels == h->channels) {
        // Bind state unchanged (filter/flag/mipmap re-set): keep the
        // existing snapshot instead of re-uploading every frame.
        return CUDA_SUCCESS;
    } else {
        if (h->obj) {
            cuTexObjectDestroy(h->obj);
            h->obj = 0;
        }
        h->built_addr = 0;
        h->built_bytes = 0;
    }
    /* The linear snapshot below copies from the host-visible mirror.
       Make sure previously submitted work (e.g. a sort that just filled
       this buffer) has completed first; otherwise the image captures
       stale bytes and kernels hang on the degenerate data. Only needed
       when actually (re)building; unchanged binds keep their snapshot. */
    if (!h->is_array && h->bytes != 0 && h->addr != 0) {
        CUcontext cc = NULL;
        if (need_ctx(&cc) == CUDA_SUCCESS)
            acuda_stream_synchronize(g_actx, NULL);
    }
    CUDA_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    if (h->is_array) {
        if (!h->array)
            return CUDA_SUCCESS;
        rd.resType = CU_RESOURCE_TYPE_ARRAY;
        rd.res.array.hArray = h->array;
    } else {
        rd.resType = CU_RESOURCE_TYPE_LINEAR;
        rd.res.linear.devPtr = h->addr;
        rd.res.linear.format = h->fmt;
        rd.res.linear.numChannels = (unsigned int)h->channels;
        rd.res.linear.sizeInBytes = h->bytes;
    }
    CUDA_TEXTURE_DESC td;
    memset(&td, 0, sizeof(td));
    td.addressMode[0] = h->address_mode[0];
    td.addressMode[1] = h->address_mode[1];
    td.addressMode[2] = h->address_mode[2];
    td.filterMode = h->filter;
    td.flags = h->flags;
    td.maxAnisotropy = h->max_aniso ? h->max_aniso : 1;
    td.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    td.borderColor[0] = td.borderColor[1] = td.borderColor[2] = td.borderColor[3] = 0.0f;
    CUresult tcr = cuTexObjectCreate(&h->obj, &rd, &td, NULL);
    ACUDA_DEBUG_PRINT("acuda-trace ?? texref_rebuild(%s) create rc=%d obj=%llx\n",
                h->name, (int)tcr, (unsigned long long)(uint64_t)h->obj);
    if (tcr == CUDA_SUCCESS && !h->is_array) {
        h->built_addr = h->addr;
        h->built_bytes = h->bytes;
        h->built_fmt = h->fmt;
        h->built_channels = h->channels;
    }
    return tcr;
}

CUresult ACUDA_CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                          const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuModuleGetTexRef(%s) mod=%p ntexrefs=%zu\n",
                name ? name : "(null)", (void *)hmod,
                hmod ? hmod->ntexrefs : 0);
    if (!pTexRef || !hmod || !name)
        return CUDA_ERROR_INVALID_VALUE;
    *pTexRef = NULL;
    for (size_t i = 0; i < hmod->ntexrefs; i++) {
        if (hmod->texrefs[i] &&
            strcmp(hmod->texrefs[i]->name, name) == 0) {
            *pTexRef = hmod->texrefs[i];
            return CUDA_SUCCESS;
        }
    }
    CUtexref tr = calloc(1, sizeof(*tr));
    if (!tr)
        return CUDA_ERROR_OUT_OF_MEMORY;
    tr->mod = hmod;
    strncpy(tr->name, name, sizeof(tr->name) - 1);
    tr->filter = CU_TR_FILTER_MODE_POINT;
    tr->address_mode[0] = tr->address_mode[1] = tr->address_mode[2] =
        CU_TR_ADDRESS_MODE_WRAP;
    tr->max_aniso = 1;
    if (hmod->ntexrefs == hmod->texrefs_cap) {
        size_t ncap = hmod->texrefs_cap ? hmod->texrefs_cap * 2 : 4;
        void *np = realloc(hmod->texrefs, ncap * sizeof(*hmod->texrefs));
        if (!np) {
            free(tr);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        hmod->texrefs = np;
        hmod->texrefs_cap = ncap;
    }
    hmod->texrefs[hmod->ntexrefs++] = tr;
    *pTexRef = tr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddress_v2(size_t *ByteOffset,
                                             CUtexref hTexRef,
                                             CUdeviceptr dptr, size_t bytes) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetAddress(%s mod=%p dptr=%llx bytes=%zu)\n",
                hTexRef ? hTexRef->name : "(null)",
                (void *)(hTexRef ? hTexRef->mod : NULL),
                (unsigned long long)dptr, bytes);
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    if (ByteOffset)
        *ByteOffset = 0;
    if (bytes == 0 && dptr != 0) {
        // cudart's texture bind path passes size 0; real CUDA binds to the
        // end of the allocation. Probe the arena for the same extent, but
        // cap the snapshot: PhysX sub-allocates particle buffers from one
        // 256 MiB heap, so the raw remainder would build a ~250 MiB image
        // per texture and exhaust device memory. 16 MiB covers ~1M float4
        // particles, far above FluidMark's range.
        size_t cap = 0;
        if (acuda_arena_capacity((uint64_t)dptr, &cap) == 0 && cap != 0) {
            size_t want = cap > (16u << 20) ? (16u << 20) : cap;
            ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetAddress(%s) size0 -> cap=%zu use=%zu\n",
                        hTexRef->name, cap, want);
            bytes = want;
        }
    }
    hTexRef->addr = dptr;
    hTexRef->bytes = bytes;
    hTexRef->is_array = 0;
    hTexRef->array = NULL;
    return texref_rebuild(hTexRef, 1);
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddress(size_t *ByteOffset,
                                          CUtexref hTexRef,
                                          CUdeviceptr dptr, size_t bytes) {
    return cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);
}

CUresult ACUDA_CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                        unsigned int Flags) {
    (void)Flags;
    if (!hTexRef || !hArray)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->array = hArray;
    hTexRef->is_array = 1;
    hTexRef->addr = 0;
    hTexRef->bytes = 0;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                         int NumPackedComponents) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuTexRefSetFormat(%s fmt=%d nch=%d)\n",
                hTexRef ? hTexRef->name : "(null)", (int)fmt,
                NumPackedComponents);
    if (!hTexRef || NumPackedComponents < 1 || NumPackedComponents > 4)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->fmt = fmt;
    hTexRef->channels = NumPackedComponents;
    hTexRef->fmt_set = 1;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuTexRefSetFlags(%s flags=%u)\n",
                hTexRef ? hTexRef->name : "(null)", Flags);
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    hTexRef->flags = Flags;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuTexRefSetFilterMode(%s fm=%d)\n",
                hTexRef ? hTexRef->name : "(null)", (int)fm);
    if (!hTexRef || (fm != CU_TR_FILTER_MODE_POINT &&
                     fm != CU_TR_FILTER_MODE_LINEAR))
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->filter = fm;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                              CUaddress_mode am) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuTexRefSetAddressMode(%s dim=%d am=%d)\n",
                hTexRef ? hTexRef->name : "(null)", dim, (int)am);
    if (!hTexRef || dim < 0 || dim > 2 || am > CU_TR_ADDRESS_MODE_BORDER)
        return CUDA_ERROR_INVALID_VALUE;
    hTexRef->address_mode[dim] = am;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                                unsigned int maxAniso) {
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    hTexRef->max_aniso = maxAniso;
    return texref_rebuild(hTexRef, 0);
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddress_v2(CUdeviceptr *pdptr,
                                             CUtexref hTexRef) {
    if (!pdptr || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pdptr = hTexRef->addr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr,
                                          CUtexref hTexRef) {
    return cuTexRefGetAddress_v2(pdptr, hTexRef);
}

CUresult ACUDA_CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    if (!phArray || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *phArray = hTexRef->array;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat,
                                         int *pNumChannels, CUtexref hTexRef) {
    if (!pFormat || !pNumChannels || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pFormat = hTexRef->fmt;
    *pNumChannels = hTexRef->channels;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFlags(unsigned int *pFlags,
                                        CUtexref hTexRef) {
    if (!pFlags || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pFlags = hTexRef->flags;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
    if (!pfm || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pfm = hTexRef->filter;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam,
                                              CUtexref hTexRef, int dim) {
    if (!pam || !hTexRef || dim < 0 || dim > 2)
        return CUDA_ERROR_INVALID_VALUE;
    *pam = hTexRef->address_mode[dim];
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso,
                                                CUtexref hTexRef) {
    if (!pmaxAniso || !hTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    *pmaxAniso = (int)hTexRef->max_aniso;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuSurfObjectCreate\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    if (!pSurfObject || !pResDesc)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t image = NULL;
    if (pResDesc->resType == CU_RESOURCE_TYPE_ARRAY) {
        CUarray a = pResDesc->res.array.hArray;
        if (!a || !a->image)
            return CUDA_ERROR_INVALID_VALUE;
        image = a->image;
    } else if (pResDesc->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
        CUmipmappedArray m = pResDesc->res.mipmap.hMipmappedArray;
        if (!m || !m->image)
            return CUDA_ERROR_INVALID_VALUE;
        image = m->image;
    } else {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    // Only color arrays get VK_IMAGE_USAGE_STORAGE_BIT; depth, cube and
    // multisampled images cannot back a surface object.
    if (image->kind != 2 || !image->is_surface)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (g_nsurfobjs == g_csurfobjs) {
        size_t ncap = g_csurfobjs ? g_csurfobjs * 2 : 8;
        driver_surfobj_t *n = realloc(g_surfobjs, sizeof(*n) * ncap);
        if (!n)
            return CUDA_ERROR_OUT_OF_MEMORY;
        g_surfobjs = n;
        g_csurfobjs = ncap;
    }
    uint32_t slot = 0;
    for (uint32_t i = 1; i < g_surf_slot_cap; i++)
        if (!g_surf_slots[i]) {
            slot = i;
            break;
        }
    if (slot == 0) {
        uint32_t ncap = g_surf_slot_cap ? g_surf_slot_cap * 2 : 16;
        acuda_buffer_t *ns =
            realloc(g_surf_slots, sizeof(*ns) * (size_t)ncap);
        if (!ns)
            return CUDA_ERROR_OUT_OF_MEMORY;
        memset(ns + g_surf_slot_cap, 0,
               sizeof(*ns) * (size_t)(ncap - g_surf_slot_cap));
        slot = g_surf_slot_cap ? g_surf_slot_cap : 1;
        g_surf_slots = ns;
        g_surf_slot_cap = ncap;
    }
    driver_surfobj_t *e = &g_surfobjs[g_nsurfobjs++];
    memset(e, 0, sizeof(*e));
    e->id = slot;
    e->slot = slot;
    e->image = image;
    e->res = *pResDesc;
    g_surf_slots[slot] = image;
    *pSurfObject = e->id;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
    for (size_t i = 0; i < g_nsurfobjs; i++) {
        if (g_surfobjs[i].id != surfObject)
            continue;
        // The array owns the image; surface objects are borrowed views.
        if (g_surfobjs[i].slot < g_surf_slot_cap)
            g_surf_slots[g_surfobjs[i].slot] = NULL;
        g_surfobjs[i] = g_surfobjs[--g_nsurfobjs];
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult ACUDA_CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject) {
    driver_surfobj_t *e = driver_surf_find(surfObject);
    if (!pResDesc || !e)
        return CUDA_ERROR_INVALID_VALUE;
    *pResDesc = e->res;
    return CUDA_SUCCESS;
}

// Copy a rectangle between host memory and an array level/layer.
static CUresult driver_array_rw(CUarray a, int write, size_t xb, size_t y,
                                size_t z, void *host, size_t wbytes,
                                size_t h, size_t depth, size_t pitch) {
    if (!a || !a->image || !host)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_buffer_t img = a->image;
    uint32_t level = a->level;
    uint32_t lw = img->width >> level;
    if (!lw) lw = 1;
    uint32_t lh = img->dim1d ? 1u : (img->height >> level);
    if (!lh) lh = 1;
    uint32_t ld = img->dim3d ? (img->depth >> level) : 1u;
    if (!ld) ld = 1;
    size_t elem = (size_t)driver_elem_bytes(img->format);
    if (elem == 0 || xb + wbytes > (size_t)lw * elem || y + h > lh ||
        z + depth > (img->dim3d ? ld : img->layers))
        return CUDA_ERROR_INVALID_VALUE;
    size_t layer_bytes = (size_t)lw * lh * elem;
    size_t total = img->dim3d ? layer_bytes * ld : layer_bytes;
    void *tmp = malloc(total);
    if (!tmp)
        return CUDA_ERROR_OUT_OF_MEMORY;
    acuda_status_t st;
    if (img->dim3d) {
        st = acuda_image_download(g_actx, img, level, 0, tmp);
        if (st != ACUDA_SUCCESS) {
            free(tmp);
            return map_status(st);
        }
    }
    for (size_t dz = 0; dz < depth; dz++) {
        void *lb;
        uint32_t layer = 0;
        if (img->dim3d) {
            lb = tmp;
        } else {
            layer = (uint32_t)(z + dz);
            lb = tmp;
            st = acuda_image_download(g_actx, img, level, layer, lb);
            if (st != ACUDA_SUCCESS) {
                free(tmp);
                return map_status(st);
            }
        }
        for (size_t row = 0; row < h; row++) {
            size_t ry = y + row;
            size_t rz = img->dim3d ? (z + dz) : 0;
            unsigned char *lp = (unsigned char *)lb +
                                ((size_t)rz * lh + ry) * lw * elem + xb;
            unsigned char *hp = (unsigned char *)host +
                                (dz * h + row) * pitch;
            if (write)
                memcpy(lp, hp, wbytes);
            else
                memcpy(hp, lp, wbytes);
        }
        if (!img->dim3d) {
            st = acuda_image_upload(g_actx, img, level, layer, lb);
            if (st != ACUDA_SUCCESS) {
                free(tmp);
                return map_status(st);
            }
        }
    }
    if (img->dim3d) {
        st = acuda_image_upload(g_actx, img, level, 0, tmp);
        if (st != ACUDA_SUCCESS) {
            free(tmp);
            return map_status(st);
        }
    }
    free(tmp);
    return CUDA_SUCCESS;
}

static CUresult driver_memcpy_common(size_t src_xb, size_t src_y, size_t src_z,
                                     CUmemorytype src_type, const void *src_host,
                                     CUdeviceptr src_dev, CUarray src_array,
                                     size_t src_pitch, size_t src_height,
                                     size_t dst_xb, size_t dst_y, size_t dst_z,
                                     CUmemorytype dst_type, void *dst_host,
                                     CUdeviceptr dst_dev, CUarray dst_array,
                                     size_t dst_pitch, size_t dst_height,
                                     size_t wbytes, size_t h, size_t depth) {
    // Host-to-array.
    if (src_type == CU_MEMORYTYPE_HOST && dst_type == CU_MEMORYTYPE_ARRAY) {
        unsigned char *hp = (unsigned char *)src_host +
                            src_y * src_pitch + src_xb;
        if (depth > 1)
            hp += src_z * src_pitch * (src_height ? src_height : h);
        return driver_array_rw(dst_array, 1, dst_xb, dst_y, dst_z, hp,
                               wbytes, h, depth, src_pitch);
    }
    // Array-to-host.
    if (src_type == CU_MEMORYTYPE_ARRAY && dst_type == CU_MEMORYTYPE_HOST) {
        unsigned char *hp = (unsigned char *)dst_host +
                            dst_y * dst_pitch + dst_xb;
        if (depth > 1)
            hp += dst_z * dst_pitch * (dst_height ? dst_height : h);
        return driver_array_rw(src_array, 0, src_xb, src_y, src_z, hp,
                               wbytes, h, depth, dst_pitch);
    }
    // Device-to-array / array-to-device via the host-visible arena.
    if (src_type == CU_MEMORYTYPE_DEVICE && dst_type == CU_MEMORYTYPE_ARRAY) {
        void *base = NULL;
        if (host_for(src_dev, src_pitch * h * depth, &base) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *hp = (unsigned char *)base +
                            src_y * src_pitch + src_xb;
        if (depth > 1)
            hp += src_z * src_pitch * (src_height ? src_height : h);
        return driver_array_rw(dst_array, 1, dst_xb, dst_y, dst_z, hp,
                               wbytes, h, depth, src_pitch);
    }
    if (src_type == CU_MEMORYTYPE_ARRAY && dst_type == CU_MEMORYTYPE_DEVICE) {
        void *base = NULL;
        if (host_for(dst_dev, dst_pitch * h * depth, &base) != CUDA_SUCCESS)
            return CUDA_ERROR_INVALID_VALUE;
        unsigned char *hp = (unsigned char *)base +
                            dst_y * dst_pitch + dst_xb;
        if (depth > 1)
            hp += dst_z * dst_pitch * (dst_height ? dst_height : h);
        return driver_array_rw(src_array, 0, src_xb, src_y, src_z, hp,
                               wbytes, h, depth, dst_pitch);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult ACUDA_CUDAAPI cuMemcpy2D(const void *pCopy) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuMemcpy2D\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    const CUDA_MEMCPY2D *m = pCopy;
    if (!m)
        return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(
        m->srcXInBytes, m->srcY, 0, m->srcMemoryType, m->srcHost,
        m->srcDevice, m->srcArray, m->srcPitch, 0, m->dstXInBytes, m->dstY,
        0, m->dstMemoryType, m->dstHost, m->dstDevice, m->dstArray,
        m->dstPitch, 0, m->WidthInBytes, m->Height, 1);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DAsync(const void *pCopy, CUstream hStream) {
    (void)hStream;
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3D(const void *pCopy) {
    ACUDA_DEBUG_PRINT("acuda-trace >> cuMemcpy3D\n");
    CUcontext c = NULL;
    CUresult r = need_ctx(&c);
    if (r != CUDA_SUCCESS)
        return r;
    const CUDA_MEMCPY3D *m = pCopy;
    if (!m)
        return CUDA_ERROR_INVALID_VALUE;
    return driver_memcpy_common(
        m->srcXInBytes, m->srcY, m->srcZ, m->srcMemoryType, m->srcHost,
        m->srcDevice, m->srcArray, m->srcPitch, m->srcHeight,
        m->dstXInBytes, m->dstY, m->dstZ, m->dstMemoryType, m->dstHost,
        m->dstDevice, m->dstArray, m->dstPitch, m->dstHeight,
        m->WidthInBytes, m->Height, m->Depth);
}

CUresult ACUDA_CUDAAPI cuMemcpy3DAsync(const void *pCopy, CUstream hStream) {
    (void)hStream;
    return cuMemcpy3D(pCopy);
}

/* CUDA 6.x-era compatibility: entries the 32-bit CUDA 6.5 runtime binds by
 * name at init through the module handle in export-table slot 5.  Every
 * probed name must resolve or the call dispatches through NULL (EIP=0
 * execute fault).  Real behavior where meaningful, success no-ops for
 * inapplicable concepts (single device, no interop), NOT_SUPPORTED for
 * unmodelled APIs.  All prototypes are arity-exact: the driver is
 * __stdcall on 32-bit Windows, so the callee must clean exactly the
 * pushed bytes. */
static CUresult compat_unimpl(const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? %s -> NOT_SUPPORTED (compat stub)\n",
                name);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult compat_noop(const char *name) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? %s -> SUCCESS (compat no-op)\n",
                name);
    return CUDA_SUCCESS;
}

#define ACUDA_COMPAT_UNIMPL0(fn) \
    CUresult ACUDA_CUDAAPI fn(void) { return compat_unimpl(#fn); }
#define ACUDA_COMPAT_UNIMPL1(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a) { \
        (void)a; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL2(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b) { \
        (void)a; (void)b; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL3(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c) { \
        (void)a; (void)b; (void)c; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL4(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d) { \
        (void)a; (void)b; (void)c; (void)d; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL5(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d, \
                              void *e) { \
        (void)a; (void)b; (void)c; (void)d; (void)e; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_UNIMPL6(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d, void *e, \
                              void *f) { \
        (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; \
        return compat_unimpl(#fn); \
    }
#define ACUDA_COMPAT_NOOP0(fn) \
    CUresult ACUDA_CUDAAPI fn(void) { return compat_noop(#fn); }
#define ACUDA_COMPAT_NOOP1(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a) { \
        (void)a; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP2(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b) { \
        (void)a; (void)b; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP3(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c) { \
        (void)a; (void)b; (void)c; \
        return compat_noop(#fn); \
    }
#define ACUDA_COMPAT_NOOP4(fn) \
    CUresult ACUDA_CUDAAPI fn(void *a, void *b, void *c, void *d) { \
        (void)a; (void)b; (void)c; (void)d; \
        return compat_noop(#fn); \
    }

/* 2D/3D copy variants the runtime probes for. */
CUresult ACUDA_CUDAAPI cuMemcpy2D_v2(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DAsync_v2(const void *pCopy,
                                          CUstream hStream) {
    return cuMemcpy2DAsync(pCopy, hStream);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy2DUnaligned_v2(const void *pCopy) {
    return cuMemcpy2D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3D_v2(const void *pCopy) {
    return cuMemcpy3D(pCopy);
}

CUresult ACUDA_CUDAAPI cuMemcpy3DAsync_v2(const void *pCopy,
                                          CUstream hStream) {
    return cuMemcpy3DAsync(pCopy, hStream);
}

/* Single-device peer copies are plain device-to-device copies. */
CUresult ACUDA_CUDAAPI cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx,
                                    CUdeviceptr src, CUcontext srcCtx,
                                    size_t bytes) {
    (void)dstCtx;
    (void)srcCtx;
    return memcpy_d2d(dst, src, bytes);
}

CUresult ACUDA_CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dstCtx,
                                         CUdeviceptr src, CUcontext srcCtx,
                                         size_t bytes, CUstream hStream) {
    (void)hStream;
    return cuMemcpyPeer(dst, dstCtx, src, srcCtx, bytes);
}

/* Multi-device / IPC / graphics-interop copies: no second device here. */
ACUDA_COMPAT_UNIMPL1(cuMemcpy3DPeer)
ACUDA_COMPAT_UNIMPL2(cuMemcpy3DPeerAsync)

/* Allocation queries. */
CUresult ACUDA_CUDAAPI cuMemGetInfo_v2(size_t *freeBytes, size_t *totalBytes) {
    return cuMemGetInfo(freeBytes, totalBytes);
}

CUresult ACUDA_CUDAAPI cuMemHostGetDevicePointer_v2(void **pdst, void *p,
                                                    unsigned int flags) {
    return cuMemHostGetDevicePointer(pdst, p, flags);
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

ACUDA_COMPAT_UNIMPL3(cuMemAllocManaged)

/* Array descriptors: forward the versioned queries. */
CUresult ACUDA_CUDAAPI cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pDesc,
                                               CUarray hArray) {
    return cuArrayGetDescriptor(pDesc, hArray);
}

CUresult ACUDA_CUDAAPI
cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pDesc, CUarray hArray) {
    return cuArray3DGetDescriptor(pDesc, hArray);
}

/* Legacy link entry points are defined alongside the _v2 variants above. */

/* File-based module load: the runtime uses the fat-binary path instead. */
ACUDA_COMPAT_UNIMPL2(cuModuleLoad)

/* Surfaces are unused by the fluid kernels; texref orphans are real. */
ACUDA_COMPAT_UNIMPL3(cuModuleGetSurfRef)
ACUDA_COMPAT_UNIMPL3(cuSurfRefSetArray)

CUresult ACUDA_CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
    if (!pTexRef)
        return CUDA_ERROR_INVALID_VALUE;
    CUtexref tr = calloc(1, sizeof(*tr));
    if (!tr)
        return CUDA_ERROR_OUT_OF_MEMORY;
    snprintf(tr->name, sizeof(tr->name), "orphan");
    *pTexRef = tr;
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
    if (!hTexRef)
        return CUDA_ERROR_INVALID_HANDLE;
    free(hTexRef);
    return CUDA_SUCCESS;
}

ACUDA_COMPAT_UNIMPL4(cuTexRefSetAddress2D_v3)
ACUDA_COMPAT_NOOP2(cuTexRefSetMipmapFilterMode)
ACUDA_COMPAT_NOOP2(cuTexRefSetMipmapLevelBias)
ACUDA_COMPAT_NOOP3(cuTexRefSetMipmapLevelClamp)
ACUDA_COMPAT_NOOP3(cuTexRefSetMipmappedArray)
ACUDA_COMPAT_UNIMPL2(cuTexObjectGetResourceViewDesc)

/* Cache / shared-memory configuration: accepted, no separate hardware. */
ACUDA_COMPAT_NOOP2(cuFuncSetCacheConfig)
ACUDA_COMPAT_NOOP2(cuFuncSetSharedMemConfig)

/* Peer access: single device, nothing to enable. */
ACUDA_COMPAT_NOOP2(cuCtxEnablePeerAccess)
ACUDA_COMPAT_NOOP1(cuCtxDisablePeerAccess)

/* Profiler hooks: accepted, nothing recorded. */
ACUDA_COMPAT_NOOP3(cuProfilerInitialize)
ACUDA_COMPAT_NOOP0(cuProfilerStart)
ACUDA_COMPAT_NOOP0(cuProfilerStop)

/* Stream extensions. */
CUresult ACUDA_CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                                  unsigned int flags,
                                                  int priority) {
    (void)priority;
    return cuStreamCreate(phStream, flags);
}

typedef void(ACUDA_CUDAAPI *acuda_stream_cb_t)(CUstream hStream,
                                               CUresult status,
                                               void *userData);

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

ACUDA_COMPAT_NOOP4(cuStreamAttachMemAsync)

/* Direct3D / OpenGL / WGL / graphics interop: outside our model. */
ACUDA_COMPAT_UNIMPL4(cuD3D10CtxCreateOnDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D10GetDevice)
ACUDA_COMPAT_UNIMPL3(cuD3D10GetDevices)
ACUDA_COMPAT_UNIMPL2(cuD3D10GetDirect3DDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D10MapResources)
ACUDA_COMPAT_UNIMPL3(cuD3D10RegisterResource)
ACUDA_COMPAT_UNIMPL2(cuD3D10UnmapResources)
ACUDA_COMPAT_UNIMPL1(cuD3D10UnregisterResource)
ACUDA_COMPAT_UNIMPL4(cuD3D11CtxCreateOnDevice)
ACUDA_COMPAT_UNIMPL2(cuD3D11GetDevice)
ACUDA_COMPAT_UNIMPL3(cuD3D11GetDevices)
ACUDA_COMPAT_UNIMPL2(cuD3D11GetDirect3DDevice)
ACUDA_COMPAT_UNIMPL3(cuGLCtxCreate_v2)
ACUDA_COMPAT_UNIMPL3(cuGLGetDevices)
ACUDA_COMPAT_UNIMPL0(cuGLInit)
ACUDA_COMPAT_UNIMPL3(cuGLMapBufferObject_v2)
ACUDA_COMPAT_UNIMPL4(cuGLMapBufferObjectAsync_v2)
ACUDA_COMPAT_UNIMPL1(cuGLRegisterBufferObject)
ACUDA_COMPAT_UNIMPL2(cuGLSetBufferObjectMapFlags)
ACUDA_COMPAT_UNIMPL1(cuGLUnmapBufferObject)
ACUDA_COMPAT_UNIMPL2(cuGLUnmapBufferObjectAsync)
ACUDA_COMPAT_UNIMPL1(cuGLUnregisterBufferObject)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D10RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D11RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsD3D9RegisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsGLRegisterBuffer)
ACUDA_COMPAT_UNIMPL4(cuGraphicsGLRegisterImage)
ACUDA_COMPAT_UNIMPL3(cuGraphicsMapResources)
ACUDA_COMPAT_UNIMPL3(cuGraphicsUnmapResources)
ACUDA_COMPAT_UNIMPL1(cuGraphicsUnregisterResource)
ACUDA_COMPAT_UNIMPL3(cuGraphicsResourceGetMappedPointer_v2)
ACUDA_COMPAT_UNIMPL2(cuGraphicsResourceSetMapFlags_v2)
ACUDA_COMPAT_UNIMPL4(cuGraphicsSubResourceGetMappedArray)
ACUDA_COMPAT_UNIMPL4(cuGraphicsResourceGetMappedMipmappedArray)
ACUDA_COMPAT_UNIMPL2(cuWGLGetDevice)

/* IPC handles: 64-byte structs passed by value, like the real driver. */
CUresult ACUDA_CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    return compat_unimpl("cuIpcCloseMemHandle");
}

CUresult ACUDA_CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *handle,
                                           CUevent event) {
    (void)handle;
    (void)event;
    return compat_unimpl("cuIpcGetEventHandle");
}

CUresult ACUDA_CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *handle,
                                        CUdeviceptr dptr) {
    (void)handle;
    (void)dptr;
    return compat_unimpl("cuIpcGetMemHandle");
}

CUresult ACUDA_CUDAAPI cuIpcOpenEventHandle(CUevent *event,
                                           CUipcEventHandle handle) {
    (void)event;
    (void)handle;
    return compat_unimpl("cuIpcOpenEventHandle");
}

CUresult ACUDA_CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *dptr,
                                         CUipcMemHandle handle,
                                         unsigned int flags) {
    (void)dptr;
    (void)handle;
    (void)flags;
    return compat_unimpl("cuIpcOpenMemHandle");
}

/* Exported aliases for clients that bind these names directly. */
CUresult ACUDA_CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                           const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}
CUresult ACUDA_CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name) {
    return acuda_stub_cuModuleGetGlobal(dptr, bytes, hmod, name);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, func, blockSize, dynamicSMemSize);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
    return acuda_stub_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, flags);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize) {
    return acuda_stub_cuOccupancyMaxPotentialBlockSize(
        minGridSize, maxBlockSize, func, blockSizeLimit, dynamicSMemSize);
}
CUresult ACUDA_CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *maxBlockSize, CUfunction func,
    size_t blockSizeLimit, size_t dynamicSMemSize, unsigned int flags) {
    return acuda_stub_cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize, maxBlockSize, func, blockSizeLimit, dynamicSMemSize,
        flags);
}
CUresult ACUDA_CUDAAPI cuTensorMapEncodeTiled(
    void *tensorMap, int tensorDataType, unsigned int tensorRank,
    void *globalAddress, const unsigned long long *globalDim,
    const unsigned long long *globalStrides, const unsigned int *boxDim,
    const unsigned int *elementStrides, int interleave, int swizzle,
    int l2Promotion, int oobFill) {
    return acuda_stub_cuTensorMapEncodeTiled(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, interleave, swizzle,
        l2Promotion, oobFill);
}

CUresult ACUDA_CUDAAPI cuModuleGetLoadingMode(int *mode) {
    if (!mode)
        return CUDA_ERROR_INVALID_VALUE;
    *mode = 0x1; /* CU_MODULE_EAGER_LOADING */
    return CUDA_SUCCESS;
}
/* CUDA 12/13 library API.  A CUlibrary is a CUmodule and a CUkernel is a
 * CUfunction; the module loader already knows how to lift SASS-only
 * cubins/fatbins through the optional rtxdisam bridge. */
CUresult ACUDA_CUDAAPI cuLibraryLoadData(void **library, const void *code, void *jitOptions,
                           void **jitOptionsValues, unsigned int numJitOptions,
                           void *libraryOptions, void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
    (void)jitOptions; (void)jitOptionsValues; (void)numJitOptions;
    (void)libraryOptions; (void)libraryOptionValues; (void)numLibraryOptions;
    if (!library || !code)
        return CUDA_ERROR_INVALID_VALUE;
    *library = NULL;
    const unsigned char *b = (const unsigned char *)code;
    ACUDA_DEBUG_PRINT("acuda-lib magic=%08x v=%08x\n",
                rd32le(b), rd32le(b + 4));
    const unsigned char *image = b;
    // Library blobs are ELF cubins, raw fatbins, or (rarely) the
    // __fatBinC_Wrapper_t control structure.  Store the image without
    // lifting it: cuLibraryGetKernel() lifts only the requested entry.
    if (rd32le(b) == 0x466243B1u) {
        uint32_t ver = rd32le(b + 4);
        const unsigned char *data = NULL;
        if (ver == 1) {
            memcpy(&data, b + 8, sizeof(data));
        } else if (ver == 2) {
            const void *const *arr = NULL;
            memcpy(&arr, b + 16, sizeof(arr));
            for (size_t i = 0; arr && i < 4096; i++) {
                const unsigned char *sub = NULL;
                memcpy(&sub, &arr[i], sizeof(sub));
                if (!sub)
                    break;
                ACUDA_DEBUG_PRINT("acuda-lib v2 sub[%zu]=%p magic=%08x "
                            "v=%08x\n", i, (const void *)sub,
                            rd32le(sub), rd32le(sub + 4));
                if (rd32le(sub) == 0xBA55ED50u) {
                    data = sub;
                    break;
                }
                if (rd32le(sub) == 0x466243B1u && rd32le(sub + 4) == 1) {
                    memcpy(&data, sub + 8, sizeof(data));
                    break;
                }
            }
        }
        if (!data || rd32le(data) != 0xBA55ED50u)
            return CUDA_ERROR_INVALID_IMAGE;
        image = data;
    }
    if (!(image[0] == 0x7f && image[1] == 'E' && image[2] == 'L' &&
          image[3] == 'F') &&
        rd32le(image) != 0xBA55ED50u)
        return cuModuleLoadData((CUmodule *)library, code);
    size_t size = acuda_image_size(image);
    ACUDA_DEBUG_PRINT("acuda-lib image size=%zu\n", size);
    if (!size)
        return CUDA_ERROR_INVALID_IMAGE;
    // Library loads stay lazy: cuLibraryGetKernel lifts only the entry
    // that is actually used.  backend=1 (C parser only) is the exception:
    // it has to import a PTX member now, and has nothing for SASS-only
    // containers.
    if (rd32le(image) == 0xBA55ED50u && cubin_backend() == 1) {
        char *ptx = fatbin_first_ptx(image, size);
        if (!ptx)
            return CUDA_ERROR_INVALID_IMAGE;
        CUresult r = cuModuleLoadData((CUmodule *)library, ptx);
        free(ptx);
        if (r == CUDA_SUCCESS && *(CUmodule *)library)
            attach_sass_image(*(CUmodule *)library, image, size);
        return r;
    }
    return module_from_sass_image((CUmodule *)library, image, size);
}
CUresult ACUDA_CUDAAPI cuLibraryLoadFromFile(void **library, const char *fileName,
                               void *jitOptions, void **jitOptionsValues,
                               unsigned int numJitOptions, void *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions) {
    if (!library || !fileName)
        return CUDA_ERROR_INVALID_VALUE;
    FILE *f = fopen(fileName, "rb");
    if (!f)
        return CUDA_ERROR_NOT_FOUND;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    long len = ftell(f);
    if (len <= 0 || len > (long)(512u << 20)) {
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    rewind(f);
    void *data = malloc((size_t)len);
    if (!data) {
        fclose(f);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (fread(data, 1, (size_t)len, f) != (size_t)len) {
        free(data);
        fclose(f);
        return CUDA_ERROR_INVALID_VALUE;
    }
    fclose(f);
    CUresult r = cuLibraryLoadData(library, data, jitOptions,
                                   jitOptionsValues, numJitOptions,
                                   libraryOptions, libraryOptionValues,
                                   numLibraryOptions);
    free(data);
    return r;
}
CUresult ACUDA_CUDAAPI cuLibraryUnload(void *library) {
    if (!library)
        return CUDA_ERROR_INVALID_HANDLE;
    return cuModuleUnload((CUmodule)library);
}
CUresult ACUDA_CUDAAPI cuLibraryGetModule(void **pMod, void *library) {
    if (!pMod)
        return CUDA_ERROR_INVALID_VALUE;
    *pMod = library;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetKernel(void **pKernel, void *library, const char *name) {
    if (!pKernel)
        return CUDA_ERROR_INVALID_VALUE;
    *pKernel = NULL;
    if (!library || !name)
        return CUDA_ERROR_INVALID_HANDLE;
    // cudart enumerates every kernel while bringing up a library, but
    // only asks for a launchable CUfunction when that kernel is used.
    // Defer the (expensive) SASS lift until then.
    struct CUkernel_st *k = calloc(1, sizeof(*k));
    if (!k)
        return CUDA_ERROR_OUT_OF_MEMORY;
    k->magic = ACUDA_HANDLE_KERNEL;
    k->mod = (CUmodule)library;
    snprintf(k->name, sizeof(k->name), "%s", name);
    *pKernel = (void *)k;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetFunction(void **pFunc, void *kernel) {
    if (!pFunc)
        return CUDA_ERROR_INVALID_VALUE;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!k->func) {
        CUfunction f = calloc(1, sizeof(*f));
        if (!f)
            return CUDA_ERROR_OUT_OF_MEMORY;
        f->magic = ACUDA_HANDLE_FUNC;
        f->mod = k->mod;
        snprintf(f->name, sizeof(f->name), "%s", k->name);
        k->func = f;
        ACUDA_DEBUG_PRINT("acuda-kfunc f=%p mod=%p k=%p name=%s\n",
                    (void *)f, (void *)f->mod, (void *)f->k, f->name);
    }
    *pFunc = (void *)k->func;
    return CUDA_SUCCESS;
}
// CUDA 13 places the attribute before the kernel handle:
//   cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
//                        CUkernel kernel, CUdevice dev)
CUresult ACUDA_CUDAAPI cuKernelGetAttribute(int *pi, int attrib, void *kernel, CUdevice dev) {
    (void)dev;
    if (!pi)
        return CUDA_ERROR_INVALID_VALUE;
    if (!kernel)
        return CUDA_ERROR_INVALID_HANDLE;
    // Accept either the CUkernel or the CUfunction returned by
    // cuKernelGetFunction.
    CUfunction f = NULL;
    CUresult r = handle_as_func(kernel, &f);
    if (r != CUDA_SUCCESS)
        return r;
    return cuFuncGetAttribute(pi, attrib, f);
}
CUresult ACUDA_CUDAAPI cuKernelGetLibrary(void **pLib, void *kernel) {
    if (!pLib)
        return CUDA_ERROR_INVALID_VALUE;
    *pLib = NULL;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    *pLib = (void *)k->mod;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelSetAttribute(int attrib, int val, void *kernel, CUdevice dev) {
    (void)dev;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    if (!k->func)
        return CUDA_SUCCESS;
    return cuFuncSetAttribute(k->func, attrib, val);
}
CUresult ACUDA_CUDAAPI cuKernelSetCacheConfig(void *kernel, int config, CUdevice dev) {
    (void)kernel; (void)config; (void)dev;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetName(const char **name, void *kernel) {
    if (!name)
        return CUDA_ERROR_INVALID_VALUE;
    struct CUkernel_st *k = (struct CUkernel_st *)kernel;
    if (!k)
        return CUDA_ERROR_INVALID_HANDLE;
    *name = k->name;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetParamInfo(void *kernel, size_t paramIndex,
                              size_t *paramOffset, size_t *paramSize) {
    (void)kernel; (void)paramIndex;
    if (paramOffset) *paramOffset = 0;
    if (paramSize) *paramSize = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuKernelGetParamCount(void *kernel, size_t *count) {
    (void)kernel;
    if (count) *count = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                            void *library, const char *name) {
    (void)library; (void)name;
    if (dptr) *dptr = 0;
    if (bytes) *bytes = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             void *library, const char *name) {
    return cuLibraryGetGlobal(dptr, bytes, library, name);
}
CUresult ACUDA_CUDAAPI cuLibraryGetUnifiedFunction(void **fptr, void *library,
                                     const char *symbol) {
    (void)library; (void)symbol;
    if (fptr) *fptr = NULL;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryGetKernelCount(unsigned int *count, void *library) {
    (void)library;
    if (count) *count = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuLibraryEnumerateKernels(void **kernels, unsigned int numKernels,
                                   void *library) {
    (void)kernels; (void)numKernels; (void)library;
    return CUDA_SUCCESS;
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
CUresult ACUDA_CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size,
                             size_t alignment, CUdeviceptr addr,
                             unsigned long long flags) {
    (void)alignment; (void)addr; (void)flags;
    if (!ptr)
        return CUDA_ERROR_INVALID_VALUE;
    return cuMemAlloc_v2(ptr, size);
}
CUresult ACUDA_CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)size;
    return cuMemFree_v2(ptr);
}
/* Minimal virtual-memory model: reserve/create allocate real arena
 * memory, map/unmap/access are no-ops, so probes during device
 * initialisation succeed without exposing a full VMM. */
CUresult ACUDA_CUDAAPI cuMemCreate(void **handle, size_t size, const void *prop,
                     unsigned long long flags) {
    (void)prop; (void)flags;
    if (!handle)
        return CUDA_ERROR_INVALID_VALUE;
    {
        CUdeviceptr dptr = 0;
        CUresult r = cuMemAlloc_v2(&dptr, size);
        if (r != CUDA_SUCCESS)
            return r;
        *handle = (void *)(uintptr_t)dptr;
        return CUDA_SUCCESS;
    }
}
CUresult ACUDA_CUDAAPI cuMemRelease(void *handle) {
    return cuMemFree_v2((CUdeviceptr)(uintptr_t)handle);
}
CUresult ACUDA_CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  void *handle, unsigned long long flags) {
    (void)ptr; (void)size; (void)offset; (void)handle; (void)flags;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size) {
    (void)ptr; (void)size;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const void *desc, size_t count) {
    (void)ptr; (void)size; (void)desc; (void)count;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const void *prop,
                                       int option) {
    (void)prop; (void)option;
    if (!granularity)
        return CUDA_ERROR_INVALID_VALUE;
    *granularity = 4096;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemGetAllocationPropertiesFromHandle(void *prop, void *handle) {
    (void)handle;
    if (prop)
        memset(prop, 0, 64);
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemExportToShareableHandle(void *shareableHandle, void *handle,
                                      int handleType,
                                      unsigned long long flags) {
    (void)handle; (void)handleType; (void)flags;
    if (shareableHandle)
        *(unsigned long long *)shareableHandle = 0;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemImportFromShareableHandle(void **handle, void *osHandle,
                                        int shHandleType) {
    (void)osHandle; (void)shHandleType;
    if (handle)
        *handle = NULL;
    return CUDA_SUCCESS;
}
CUresult ACUDA_CUDAAPI cuMemRetainAllocationHandle(void **handle, void *addr) {
    (void)addr;
    if (handle)
        *handle = NULL;
    return CUDA_SUCCESS;
}
/* Async entry points run synchronously on the default stream. */
CUresult ACUDA_CUDAAPI cuMemsetD32Async(CUdeviceptr dst, unsigned int value,
                          size_t count, void *stream) {
    (void)stream;
    return cuMemsetD32_v2(dst, value, count);
}
CUresult ACUDA_CUDAAPI cuMemsetD8Async(CUdeviceptr dst, unsigned char value,
                         size_t count, void *stream) {
    (void)stream;
    return cuMemsetD8_v2(dst, value, count);
}
CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyHtoD_v2(dst, src, bytes);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyDtoH_v2(dst, src, bytes);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dst, CUdeviceptr src,
                              size_t bytes, void *stream) {
    (void)stream;
    return cuMemcpyDtoD_v2(dst, src, bytes);
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
CUresult ACUDA_CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src,
                           size_t bytes, void *stream) {
    return cuMemcpyHtoDAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t bytes,
                           void *stream) {
    return cuMemcpyDtoHAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src,
                           size_t bytes, void *stream) {
    return cuMemcpyDtoDAsync_v2(dst, src, bytes, stream);
}
CUresult ACUDA_CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t bytes, int kind, void *stream) {
    (void)stream;
    switch (kind) {
    case 1: /* HtoD */
        return cuMemcpyHtoD_v2(dst, (const void *)(uintptr_t)src, bytes);
    case 2: /* DtoH */
        return cuMemcpyDtoH_v2((void *)(uintptr_t)dst, src, bytes);
    case 3: /* DtoD */
        return cuMemcpyDtoD_v2(dst, src, bytes);
    default:
        if (kind == 0) { /* HtoH */
            memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src,
                   bytes);
            return CUDA_SUCCESS;
        }
        return CUDA_ERROR_INVALID_VALUE;
    }
}
