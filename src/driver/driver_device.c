#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_dd9e98a4.h"
#include "acuda/runtime_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    acuda_driver_mark_ready();
    ACUDA_DEBUG_PRINT("acuda-trace ?? cuInit done g=%d\n", g_init);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDriverGetVersion(int *driverVersion) {
    if (!driverVersion)
        return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = ACUDA_DRIVER_VERSION;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDriverGetVersion -> %d\n", *driverVersion);
    return CUDA_SUCCESS;
}

CUresult ACUDA_CUDAAPI cuDeviceGetCount(int *count) {
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount\n");
    { const char *_tp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"DRV cuDeviceGetCount in\n"); fflush(_f); fclose(_f);} } }
    if (!count)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS) {
        { const char *_tp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"DRV cuDeviceGetCount out r=%d\n",(int)r); fflush(_f); fclose(_f);} } }
        return r;
    }
    uint32_t n = 0;
    acuda_status_t st = acuda_device_count(g_actx, &n);
    if (st != ACUDA_SUCCESS) {
        CUresult _mr = map_status(st);
        { const char *_tp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"DRV cuDeviceGetCount out r=%d\n",(int)_mr); fflush(_f); fclose(_f);} } }
        return _mr;
    }
    // Expose only GPUs with compute queues to driver clients; software
    // rasterizers (llvmpipe) confuse client heuristics. Matches ZLUDA.
    // TODO: filter by queue support instead of clamping to 1.
    if (n > 1)
        n = 1;
    *count = (int)n;
    ACUDA_DEBUG_PRINT("acuda-trace .. cuDeviceGetCount -> %d\n", (int)n);
    { const char *_tp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"DRV cuDeviceGetCount out r=0 n=%d\n",(int)n); fflush(_f); fclose(_f);} } }
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
    if (!name || len <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult r = need_actx();
    if (r != CUDA_SUCCESS)
        return r;
    /* Legacy PhysX (2.8.4) parses the CUDA device name to choose its
     * hardware version (NX_HW_VERSION).  Allow a name override so an
     * unrecognised modern GPU can present a name that legacy clients
     * accept. */
    {
        const char *ov = getenv("ACUDA_DEVICE_NAME");
        if (ov && *ov) {
            strncpy(name, ov, (size_t)len - 1);
            name[len - 1] = '\0';
            return CUDA_SUCCESS;
        }
    }
    acuda_device_info_t info;
    acuda_status_t st = acuda_device_get(g_actx, (uint32_t)dev, &info);
    if (st != ACUDA_SUCCESS)
        return CUDA_ERROR_INVALID_DEVICE;
    strncpy(name, info.name, (size_t)len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

int cc_major(void) {
    const char *v = getenv("ACUDA_CC_MAJOR");
    return v && *v ? atoi(v) : ACUDA_CC_MAJOR;
}

int cc_minor(void) {
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
    {
        int clock = 1290000;
        if (g_actx && g_actx->core_clock_khz > 0)
            clock = g_actx->core_clock_khz;
        prop->clockRate = clock;
    }
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

CUresult ACUDA_CUDAAPI cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem(bytes, dev);
}

CUresult ACUDA_CUDAAPI cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    return cuDeviceGetUuid(uuid, dev);
}

 int device_pci_info(CUdevice dev, acuda_device_info_t *info) {
    if (!g_actx || acuda_device_get(g_actx, (uint32_t)dev, info) != ACUDA_SUCCESS)
        return 0;
    return 1;
}



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

CUresult ACUDA_CUDAAPI cuDeviceGetAttribute(int *value, int attrib, CUdevice dev) {
    { const char *_tp = getenv("ACUDA_DRIVER_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"DRV cuDeviceGetAttribute in attr=%d dev=%d\n",attrib,dev); fflush(_f); fclose(_f);} } }
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    acuda_dd9e98a4_check_ready_canary("cuDeviceGetAttribute");
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
        if (g_actx)
            acuda_context_refresh_clocks(g_actx);
        *value = (g_actx && g_actx->core_clock_khz > 0)
                     ? g_actx->core_clock_khz
                     : 1290000;
        return CUDA_SUCCESS;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
        if (g_actx)
            acuda_context_refresh_clocks(g_actx);
        *value = (g_actx && g_actx->mem_clock_khz > 0)
                     ? g_actx->mem_clock_khz
                     : 6001000;
        return CUDA_SUCCESS;
    case 0x20080403:
        /* Private: live GPU utilisation percentage from ADL2/hwinfo/NVML,
         * so the NVAPI shim can report a real load value. */
        if (g_actx)
            acuda_context_refresh_clocks(g_actx);
        *value = g_actx ? g_actx->gpu_load_percent : 0;
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

static void acuda_query_attr_int(int *out, int attr, CUdevice dev) {
    int v = 0;
    if (cuDeviceGetAttribute(&v, attr, dev) == CUDA_SUCCESS)
        *out = v;
}

CUresult ACUDA_CUDAAPI acuda_query_device_properties(struct cudaDeviceProp *prop, int device) {
    if (!prop)
        return CUDA_ERROR_INVALID_VALUE;
    int count = 0;
    CUresult r = cuDeviceGetCount(&count);
    if (r != CUDA_SUCCESS)
        return r;
    if (device < 0 || device >= count)
        return CUDA_ERROR_INVALID_DEVICE;
    memset(prop, 0, sizeof(*prop));

    {
        char name[256] = {0};
        if (cuDeviceGetName(name, (int)sizeof(name), (CUdevice)device) == CUDA_SUCCESS) {
            strncpy(prop->name, name, sizeof(prop->name) - 1);
            prop->name[sizeof(prop->name) - 1] = '\0';
        }
    }
    cuDeviceComputeCapability(&prop->major, &prop->minor, (CUdevice)device);
    cuDeviceTotalMem(&prop->totalGlobalMem, (CUdevice)device);

    /* UUID (deterministic, matches cuDeviceGetUuid) */
    {
        CUuuid uuid;
        if (cuDeviceGetUuid(&uuid, (CUdevice)device) == CUDA_SUCCESS)
            memcpy(&prop->uuid, &uuid, sizeof(prop->uuid));
    }
    prop->luidDeviceNodeMask = 0;

    acuda_query_attr_int(&prop->maxThreadsPerBlock, 1, (CUdevice)device);
    acuda_query_attr_int(&prop->maxThreadsDim[0], 2, (CUdevice)device);
    acuda_query_attr_int(&prop->maxThreadsDim[1], 3, (CUdevice)device);
    acuda_query_attr_int(&prop->maxThreadsDim[2], 4, (CUdevice)device);
    acuda_query_attr_int(&prop->maxGridSize[0], 5, (CUdevice)device);
    acuda_query_attr_int(&prop->maxGridSize[1], 6, (CUdevice)device);
    acuda_query_attr_int(&prop->maxGridSize[2], 7, (CUdevice)device);
    {
        int v = 0;
        if (cuDeviceGetAttribute(&v, 8, (CUdevice)device) == CUDA_SUCCESS)
            prop->sharedMemPerBlock = (size_t)v;
        v = 0;
        if (cuDeviceGetAttribute(&v, 9, (CUdevice)device) == CUDA_SUCCESS)
            prop->totalConstMem = (size_t)v;
    }
    acuda_query_attr_int(&prop->warpSize, 10, (CUdevice)device);
    {
        int v = 0;
        if (cuDeviceGetAttribute(&v, 11, (CUdevice)device) == CUDA_SUCCESS)
            prop->memPitch = (size_t)v;
    }
    acuda_query_attr_int(&prop->regsPerBlock, 12, (CUdevice)device);
    {
        int v = 0;
        if (cuDeviceGetAttribute(&v, 14, (CUdevice)device) == CUDA_SUCCESS)
            prop->textureAlignment = (size_t)v;
        v = 0;
        if (cuDeviceGetAttribute(&v, 51, (CUdevice)device) == CUDA_SUCCESS)
            prop->texturePitchAlignment = (size_t)v;
    }
    acuda_query_attr_int(&prop->multiProcessorCount, 16, (CUdevice)device);
    acuda_query_attr_int(&prop->integrated, 18, (CUdevice)device);
    acuda_query_attr_int(&prop->canMapHostMemory, 19, (CUdevice)device);
    acuda_query_attr_int(&prop->concurrentKernels, 31, (CUdevice)device);
    acuda_query_attr_int(&prop->pciBusID, 33, (CUdevice)device);
    acuda_query_attr_int(&prop->pciDeviceID, 34, (CUdevice)device);
    acuda_query_attr_int(&prop->memoryBusWidth, 37, (CUdevice)device);
    acuda_query_attr_int(&prop->l2CacheSize, 38, (CUdevice)device);
    acuda_query_attr_int(&prop->maxThreadsPerMultiProcessor, 39, (CUdevice)device);
    acuda_query_attr_int(&prop->asyncEngineCount, 40, (CUdevice)device);
    acuda_query_attr_int(&prop->unifiedAddressing, 41, (CUdevice)device);
    acuda_query_attr_int(&prop->pciDomainID, 50, (CUdevice)device);
    {
        int v = 0;
        if (cuDeviceGetAttribute(&v, 81, (CUdevice)device) == CUDA_SUCCESS)
            prop->sharedMemPerMultiprocessor = (size_t)v;
    }
    acuda_query_attr_int(&prop->regsPerMultiprocessor, 82, (CUdevice)device);

    /* Defaults / capabilities (reasonable values consistent with ACuda driver) */
    prop->textureAlignment = prop->textureAlignment ? prop->textureAlignment : 512;
    prop->surfaceAlignment = 512;
    prop->ECCEnabled = 0;
    prop->tccDriver = 0;
    prop->streamPrioritiesSupported = 0;
    prop->globalL1CacheSupported = 1;
    prop->localL1CacheSupported = 1;
    prop->managedMemory = 0;
    prop->isMultiGpuBoard = 0;
    prop->multiGpuBoardGroupID = 0;
    prop->hostNativeAtomicSupported = 0;
    prop->pageableMemoryAccess = 1;
    prop->concurrentManagedAccess = 0;
    prop->computePreemptionSupported = 0;
    prop->canUseHostPointerForRegisteredMem = 0;
    prop->cooperativeLaunch = 0;
    prop->sharedMemPerBlockOptin = prop->sharedMemPerBlock;
    prop->pageableMemoryAccessUsesHostPageTables = 1;
    prop->directManagedMemAccessFromHost = 0;
    prop->maxBlocksPerMultiProcessor = 32;
    prop->accessPolicyMaxWindowSize = 0;
    prop->reservedSharedMemPerBlock = 0;
    prop->hostRegisterSupported = 1;
    prop->sparseCudaArraySupported = 0;
    prop->hostRegisterReadOnlySupported = 0;
    prop->timelineSemaphoreInteropSupported = 1;
    prop->memoryPoolsSupported = 1;
    prop->gpuDirectRDMASupported = 0;
    prop->gpuDirectRDMAFlushWritesOptions = 0;
    prop->gpuDirectRDMAWritesOrdering = 0;
    prop->memoryPoolSupportedHandleTypes = 0;
    prop->deferredMappingCudaArraySupported = 0;
    prop->ipcEventSupported = 0;
    prop->clusterLaunch = 0;
    prop->unifiedFunctionPointers = 0;
    prop->deviceNumaConfig = 0;
    prop->deviceNumaId = -1;
    prop->mpsEnabled = 0;
    prop->hostNumaId = -1;
    prop->gpuPciDeviceID = 0;
    prop->gpuPciSubsystemID = 0;
    prop->hostNumaMultinodeIpcSupported = 0;

    /* Texture/surface limits (sensible defaults) */
    prop->maxTexture1D = 131072;
    prop->maxTexture1DMipmap = 32768;
    prop->maxTexture2D[0] = 131072; prop->maxTexture2D[1] = 65536;
    prop->maxTexture2DMipmap[0] = 32768; prop->maxTexture2DMipmap[1] = 32768;
    prop->maxTexture2DLinear[0] = 131072; prop->maxTexture2DLinear[1] = 65000; prop->maxTexture2DLinear[2] = 2097120;
    prop->maxTexture2DGather[0] = 32768; prop->maxTexture2DGather[1] = 32768;
    prop->maxTexture3D[0] = 16384; prop->maxTexture3D[1] = 16384; prop->maxTexture3D[2] = 16384;
    prop->maxTexture3DAlt[0] = 8192; prop->maxTexture3DAlt[1] = 8192; prop->maxTexture3DAlt[2] = 32768;
    prop->maxTextureCubemap = 32768;
    prop->maxTexture1DLayered[0] = 32768; prop->maxTexture1DLayered[1] = 2048;
    prop->maxTexture2DLayered[0] = 32768; prop->maxTexture2DLayered[1] = 32768; prop->maxTexture2DLayered[2] = 2048;
    prop->maxTextureCubemapLayered[0] = 32768; prop->maxTextureCubemapLayered[1] = 2046;
    prop->maxSurface1D = 32768;
    prop->maxSurface2D[0] = 32768; prop->maxSurface2D[1] = 32768;
    prop->maxSurface3D[0] = 16384; prop->maxSurface3D[1] = 16384; prop->maxSurface3D[2] = 16384;
    prop->maxSurface1DLayered[0] = 32768; prop->maxSurface1DLayered[1] = 2048;
    prop->maxSurface2DLayered[0] = 32768; prop->maxSurface2DLayered[1] = 32768; prop->maxSurface2DLayered[2] = 2048;
    prop->maxSurfaceCubemap = 32768;
    prop->maxSurfaceCubemapLayered[0] = 32768; prop->maxSurfaceCubemapLayered[1] = 2046;

    if (prop->warpSize == 0) prop->warpSize = 32;
    if (prop->maxThreadsPerBlock == 0) prop->maxThreadsPerBlock = 1024;
    if (prop->multiProcessorCount == 0) prop->multiProcessorCount = 40;
    return CUDA_SUCCESS;
}
