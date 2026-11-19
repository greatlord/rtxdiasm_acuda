#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/stream.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/time_compat.h"

acuda_canary_hook_t acuda_runtime_canary_hook;

acuda_status_t acuda_resolve_stream(acuda_context_t ctx, acuda_stream_t stream,
                                    acuda_stream_t *out) {
    if (!ctx || !out)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!stream)
        stream = ctx->default_stream;
    if (!stream)
        return ACUDA_ERROR_INVALID_HANDLE;
    *out = stream;
    return ACUDA_SUCCESS;
}

int stream_push_free_cmd(acuda_stream_t s, VkCommandBuffer cmd) {
    if (s->free_cmd_count == s->free_cmd_cap) {
        size_t ncap = s->free_cmd_cap ? s->free_cmd_cap * 2 : 8;
        VkCommandBuffer *nc = realloc(s->free_cmds, sizeof(*nc) * ncap);
        if (!nc)
            return -1;
        s->free_cmds = nc;
        s->free_cmd_cap = ncap;
    }
    s->free_cmds[s->free_cmd_count++] = cmd;
    return 0;
}

int stream_push_free_dpool(acuda_stream_t s, VkDescriptorPool pool) {
    if (s->free_dpool_count == s->free_dpool_cap) {
        size_t ncap = s->free_dpool_cap ? s->free_dpool_cap * 2 : 8;
        VkDescriptorPool *np =
            realloc(s->free_dpools, sizeof(*np) * ncap);
        if (!np)
            return -1;
        s->free_dpools = np;
        s->free_dpool_cap = ncap;
    }
    s->free_dpools[s->free_dpool_count++] = pool;
    return 0;
}
