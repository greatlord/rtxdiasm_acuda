#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_263e8860.h"

/* {263E8860-7CD2-6143-92F6-BBD5006DFA7E} slot implementations.
 *
 * Both probes are advisory; cudart only records the answers. */

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

const void *acuda_export_263e8860_table(void) {
    return (const void *)g_context_checks;
}
