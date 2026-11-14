// Host-call callback registry (per context).
#include <stdio.h>
#include <string.h>
#include "acuda/hostcall.h"
#include "internal.h"

acuda_host_entry_t *acuda_host_call_find(acuda_context_t ctx,
                                        const char *name) {
    if (!ctx || !name)
        return NULL;
    for (uint32_t i = 0; i < ctx->nhost_calls; i++)
        if (strcmp(ctx->host_calls[i].name, name) == 0)
            return &ctx->host_calls[i];
    return NULL;
}

acuda_status_t acuda_host_call_register(acuda_context_t ctx,
                                        const char *name,
                                        acuda_host_call_fn fn,
                                        void *user) {
    if (!ctx || !name || !name[0] || !fn)
        return ACUDA_ERROR_INVALID_VALUE;
    if (strlen(name) >= ACUDA_HOST_NAME_MAX)
        return ACUDA_ERROR_INVALID_VALUE;
    acuda_host_entry_t *e = acuda_host_call_find(ctx, name);
    if (!e) {
        if (ctx->nhost_calls >= ACUDA_MAX_HOST_ENTRIES)
            return ACUDA_ERROR_OUT_OF_MEMORY;
        e = &ctx->host_calls[ctx->nhost_calls++];
        memset(e, 0, sizeof(*e));
        snprintf(e->name, sizeof(e->name), "%s", name);
    }
    e->fn = fn;
    e->user = user;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_host_call_unregister(acuda_context_t ctx,
                                          const char *name) {
    if (!ctx || !name)
        return ACUDA_ERROR_INVALID_VALUE;
    for (uint32_t i = 0; i < ctx->nhost_calls; i++) {
        if (strcmp(ctx->host_calls[i].name, name) == 0) {
            ctx->host_calls[i] = ctx->host_calls[ctx->nhost_calls - 1];
            ctx->nhost_calls--;
            return ACUDA_SUCCESS;
        }
    }
    return ACUDA_ERROR_INVALID_VALUE;
}
