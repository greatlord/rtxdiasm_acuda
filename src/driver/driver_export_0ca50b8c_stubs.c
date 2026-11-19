#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_0ca50b8c.h"

/* {0CA50B8C-1004-929A-89A7-D0DF10E77286} slot implementations.
 *
 * The CUDA 4.x runtime uses this interface to create the context through
 * the driver without its own thread-local wrapper. */

static CUresult ACUDA_CUDAAPI ctx_create_v2_bypass(CUcontext *pctx,
                                                   unsigned int flags,
                                                   CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

static const void *g_ctx_create_bypass[2] = {
    (const void *)(uintptr_t)(2 * sizeof(void *)),
    (const void *)(uintptr_t)ctx_create_v2_bypass
};

const void *acuda_export_0ca50b8c_table(void) {
    return (const void *)g_ctx_create_bypass;
}
