#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_195bcbf4.h"

/* {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} slot implementations.
 *
 * CUDA 6.x cudart queries this table during context setup and fails every
 * allocation with "unknown error" (30) when it is missing.  The driver
 * allocates an opaque per-context heap record; cudart stores the pointer
 * and hands it back to heap_free at teardown.  Contents are
 * driver-private, so a zeroed block is enough. */

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

const void *acuda_export_195bcbf4_table(void) {
    return (const void *)g_heap_access;
}
