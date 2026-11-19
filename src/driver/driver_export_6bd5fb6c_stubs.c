#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_6bd5fb6c.h"

/* {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9} slot implementations.
 *
 * CUDA 4.x/6.x cudart fetches this small interface table with
 * cuGetExportTable instead of resolving the corresponding driver entry
 * points one by one.  It is queried during device enumeration; slot 12 is
 * the "load compilers" callback, which is a no-op for our software path. */

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

const void *acuda_export_6bd5fb6c_table(void) {
    ensure_cudart_iface();
    return (const void *)g_cudart_interface;
}
