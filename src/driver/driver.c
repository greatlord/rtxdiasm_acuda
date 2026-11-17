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
#include "driver_internal.h"
#include "driver_stub.h"

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
unsigned long long acuda_device_mem_total(void) {
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
int acuda_host_heap_init(size_t need);
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
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
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
    int n = _snprintf(
        buf, sizeof(buf),
        "acuda-veh AV at %p op=%llu addr=%p eax=%08lx ecx=%08lx "
        "esi=%08lx edi=%08lx ebp=%08lx esp=%08lx eip=%08lx "
        "mod=%08lx fn4048=%08lx cnt2b4=%u availcommit=%llu\r\n"
        "acuda-veh stack: %08lx %08lx %08lx %08lx %08lx %08lx %08lx %08lx\r\n",
        ep->ExceptionRecord->ExceptionAddress,
        (unsigned long long)ep->ExceptionRecord->ExceptionInformation[0],
        (void *)ep->ExceptionRecord->ExceptionInformation[1],
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

void acuda_install_veh(void) {
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

// ---- texture arrays and texture objects ------------------------------




driver_texobj_t *g_texobjs;
size_t g_ntexobjs, g_ctexobjs;
CUtexObject g_next_texid = 1;


driver_surfobj_t *g_surfobjs;
size_t g_nsurfobjs, g_csurfobjs;
acuda_buffer_t *g_surf_slots;
uint32_t g_surf_slot_cap;

driver_surfobj_t *driver_surf_find(CUsurfObject id) {
    for (size_t i = 0; i < g_nsurfobjs; i++)
        if (g_surfobjs[i].id == id)
            return &g_surfobjs[i];
    return NULL;
}

void driver_img_type(acuda_buffer_t img, int *dim, int *arrayed,
                            int *isint) {
    *dim = img->dim3d ? 2 : (img->dim1d ? 0 : 1);
    *arrayed = !img->dim3d && img->layers > 1;
    *isint = img->format == 3 ? 1 : (img->format == 0 ? 0 : -1);
}


driver_texobj_t *driver_tex_find(CUtexObject id) {
    for (size_t i = 0; i < g_ntexobjs; i++)
        if (g_texobjs[i].id == id)
            return &g_texobjs[i];
    return NULL;
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

// Default host-call stub for the CUDA driver API.  The driver API has
// no callback-registration entry point, so unresolved .extern .func
// calls use this identity stub unless the application registered a
// real callback through acuda_host_call_register().  It copies the
// argument bytes into the return area (remaining return bytes zero).

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

CUresult need_init(void) {
    ACUDA_DEBUG_PRINT("acuda-trace ?? need_init g=%d @%p\n", g_init,
                (void *)&g_init);
    return g_init ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
}

// Lazily creates the one shared Vulkan context for the process.

CUresult need_actx(void) {
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

// Device pointers are arena virtual addresses (base+offset), so caching
// allocators and sub-views work with plain arithmetic. Resolved to host
// memory (persistently mapped, coherent) or to (buffer, offset) binds.
CUresult host_for(CUdeviceptr dptr, size_t bytes, void **host) {
    if (!dptr || !host)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_host((uint64_t)dptr, bytes, host) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

CUresult bind_for(CUdeviceptr dptr, size_t bytes, VkBuffer *buf,
                         size_t *offset) {
    if (!dptr || !buf || !offset)
        return CUDA_ERROR_INVALID_VALUE;
    if (acuda_arena_bind((uint64_t)dptr, bytes, buf, offset) != 0)
        return CUDA_ERROR_INVALID_VALUE;
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



CUresult mem_alloc(CUdeviceptr *dptr, size_t bytesize) {
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




uint32_t rd32le(const unsigned char *p);

int looks_like_ptx(const char *text) {
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

CUresult attach_sass_image(CUmodule module, const unsigned char *data,
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

CUresult module_from_sass_image(CUmodule *module,
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
int cubin_backend(void) {
    const char *s = getenv("ACUDA_CUBIN_BACKEND");
    if (s && *s) {
        long v = strtol(s, NULL, 0);
        if (v == 1 || v == 2)
            return (int)v;
    }
    return 0;
}

CUresult load_sass_image(CUmodule *module, const unsigned char *data,
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
    if (st != ACUDA_SUCCESS) DeleteFileA(outpath);
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

CUresult handle_as_func(void *handle, CUfunction *out) {
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





CUresult ACUDA_CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
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


/* ---- Device characterization, primary contexts, misc (torch init path) ---- */


#define ACUDA_MAX_DEVICES 8
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
    (void)target;
    (void)scope;
    if (need_init() != CUDA_SUCCESS)
        return CUDA_ERROR_NOT_INITIALIZED;
    return CUDA_SUCCESS; // no GPUDirect transports here
}







/* Stubs for newer driver APIs that may be linked by torch */
typedef struct CUtensorMap_st *CUtensorMap_t;
typedef enum { CU_TENSOR_MAP_DATA_TYPE_UINT8_s = 0 } CUtensorMapDataType_s;
typedef enum { CU_TENSOR_MAP_INTERLEAVE_NONE_s = 0 } CUtensorMapInterleave_s;
typedef enum { CU_TENSOR_MAP_SWIZZLE_NONE_s = 0 } CUtensorMapSwizzle_s;
typedef enum { CU_TENSOR_MAP_L2_PROMOTION_NONE_s = 0 } CUtensorMapL2promotion_s;
typedef enum { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE_s = 0 } CUtensorMapFloatOOBfill_s;


// Copy a rectangle between host memory and an array level/layer.
CUresult driver_array_rw(CUarray a, int write, size_t xb, size_t y,
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


/* Multi-device / IPC / graphics-interop copies: no second device here. */

/* Allocation queries. */




/* Array descriptors: forward the versioned queries. */

/* Legacy link entry points are defined alongside the _v2 variants above. */

/* File-based module load: the runtime uses the fat-binary path instead. */

/* Surfaces are unused by the fluid kernels; texref orphans are real. */



/* Cache / shared-memory configuration: accepted, no separate hardware. */

/* Peer access: single device, nothing to enable. */

/* Profiler hooks: accepted, nothing recorded. */

/* Stream extensions. */


/* Direct3D / OpenGL / WGL / graphics interop: outside our model. */

/* IPC handles: 64-byte structs passed by value, like the real driver. */











