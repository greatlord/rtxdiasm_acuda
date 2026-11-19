#include "acuda/driver/driver_internal.h"
#include "acuda/driver/driver_export_c693336e.h"

/* {C693336E-1121-DF11-A8C3-68F355D89593} slot implementations.
 *
 * Context-local storage, used by the tools layer to attach small records
 * to a context.  A fixed table is enough for our use. */

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

/* The context-storage export table has no size slot: [0] put (4 args),
 * [1] delete (2 args), [2] get (3 args).  The real driver's table is
 * exactly three function pointers. */
static const void *g_ctx_storage[3] = {
    (const void *)(uintptr_t)ctx_storage_put,
    (const void *)(uintptr_t)ctx_storage_delete,
    (const void *)(uintptr_t)ctx_storage_get
};

const void *acuda_export_c693336e_table(void) {
    return (const void *)g_ctx_storage;
}
