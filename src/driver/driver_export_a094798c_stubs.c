#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_a094798c.h"

/* {A094798C-2E74-2E74-93F2-0800200C0A66} slot implementations.
 *
 * Opaque scratch buffers requested by the tools runtime hooks.  The caller
 * treats them as private storage; zeroed memory is enough. */

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

/* CUDA 4.x cudart calls this slot from cudaMallocHost with
 * (event_id, event_data).  Nothing in the data is consumed by our
 * software path, so a success return is enough. */
static int ACUDA_CUDAAPI tools_runtime_event(unsigned int code, void *data) {
    (void)code;
    (void)data;
    return 0;
}

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

const void *acuda_export_a094798c_table(void) {
    return (const void *)g_tools_runtime_hooks;
}
