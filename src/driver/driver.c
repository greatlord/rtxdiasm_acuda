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


 int device_pci_info(CUdevice dev, acuda_device_info_t *info);




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



// CUDA 12.4 extended launch. Field offsets mirror NVIDIA's
// CUlaunchConfig exactly (verified against cuda.h in the cu126 pip
// wheel: 7x u32 dims+shared, ptr stream, ptr attrs, u32 numAttrs).
// Extended attributes are NOT supported: any attrs -> INVALID_VALUE.
// Plain configs forward to cuLaunchKernel (same 1D/sharedMem=0 rules).






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











