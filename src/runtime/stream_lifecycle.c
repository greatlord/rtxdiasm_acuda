#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

static uint64_t stream_id_counter;

static acuda_status_t stream_create_impl(acuda_context_t ctx, VkQueue queue,
                                         uint32_t queue_family, int blocking,
                                         int link,
                                         acuda_stream_t *out_stream) {
    if (!ctx || !out_stream)
        return ACUDA_ERROR_INVALID_HANDLE;
    *out_stream = NULL;
    acuda_stream_t s = calloc(1, sizeof(*s));
    if (!s)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    s->ctx = ctx;
    s->queue = queue ? queue : ctx->queue;
    s->queue_family = queue_family;
    s->blocking = blocking;
    s->id = ++stream_id_counter;
    if (s->id == 0)
        s->id = ++stream_id_counter;
    if (getenv("ACUDA_DEBUG_STREAMS"))
        fprintf(stderr, "ACuda stream %llu: queue=%p family=%u blocking=%d\n",
                (unsigned long long)s->id, (void *)s->queue, queue_family,
                blocking);
    s->batch_max = 32;
    if (getenv("ACUDA_NO_BATCH"))
        s->batch_max = 1;
    else if (getenv("ACUDA_BATCH")) {
        int v = atoi(getenv("ACUDA_BATCH"));
        if (v > 0)
            s->batch_max = v > 4096 ? 4096 : v;
    }
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family,
    };
    VkResult cvr = vkCreateCommandPool(ctx->device, &pci, NULL, &s->cmd_pool);
    if (cvr != VK_SUCCESS) {
        free(s);
        return acuda_from_vk(cvr);
    }
    VkSemaphoreTypeCreateInfo tci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
    };
    VkSemaphoreCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &tci,
    };
    VkResult vr = vkCreateSemaphore(ctx->device, &sci, NULL, &s->timeline);
    if (vr != VK_SUCCESS) {
        vkDestroyCommandPool(ctx->device, s->cmd_pool, NULL);
        free(s);
        return acuda_from_vk(vr);
    }
    // Internal streams (the keep-busy queue) are deliberately not linked
    // into ctx->streams: acuda_stream_flush_all / acuda_free run on the
    // user thread and must never touch a stream another thread owns.
    if (link) {
        s->next = ctx->streams;
        ctx->streams = s;
    }
    *out_stream = s;
    return ACUDA_SUCCESS;
}

static VkQueue stream_pick_queue(acuda_context_t ctx) {
    if (ctx->nuser_queues == 0)
        return ctx->queue;
    uint32_t i = ctx->next_user_queue % ctx->nuser_queues;
    ctx->next_user_queue++;
    return ctx->user_queues[i];
}

acuda_status_t acuda_stream_create_flags(acuda_context_t ctx, unsigned flags,
                                         acuda_stream_t *out_stream) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (flags & ~1u)
        return ACUDA_ERROR_INVALID_VALUE;
    VkQueue q = stream_pick_queue(ctx);
    return stream_create_impl(ctx, q, ctx->queue_family, !(flags & 1u), 1,
                              out_stream);
}

acuda_status_t acuda_stream_create(acuda_context_t ctx, acuda_stream_t *out_stream) {
    return acuda_stream_create_flags(ctx, 0, out_stream);
}

acuda_status_t acuda_stream_create_primary(acuda_context_t ctx,
                                           acuda_stream_t *out_stream) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    return stream_create_impl(ctx, ctx->queue, ctx->queue_family, 1, 1,
                              out_stream);
}

acuda_status_t acuda_stream_create_on_queue(acuda_context_t ctx, VkQueue queue,
                                            uint32_t queue_family,
                                            acuda_stream_t *out_stream) {
    return stream_create_impl(ctx, queue, queue_family, 0, 0, out_stream);
}

acuda_status_t acuda_stream_destroy(acuda_context_t ctx, acuda_stream_t stream) {
    if (!ctx || !stream)
        return ACUDA_ERROR_INVALID_HANDLE;
    stream_flush_impl(ctx, stream, NULL, NULL, 0, NULL);
    if (stream->last_submitted > 0)
        stream_wait_value(ctx, stream, stream->last_submitted, ACUDA_SYNC_TIMEOUT_NS);
    // Unlink from the context list.
    acuda_stream_t *link = &ctx->streams;
    while (*link && *link != stream)
        link = &(*link)->next;
    if (*link)
        *link = stream->next;
    // A failed flush can leave pending resources behind.
    for (size_t i = 0; i < stream->pend_dpool_count; i++)
        vkDestroyDescriptorPool(ctx->device, stream->pend_dpools[i], NULL);
    for (size_t i = 0; i < stream->pend_buf_count; i++)
        acuda_free(ctx, stream->pend_bufs[i]);
    free(stream->pend_dpools);
    free(stream->pend_bufs);
    if (stream->batch_cmd) {
        vkResetCommandBuffer(stream->batch_cmd, 0);
        stream_push_free_cmd(stream, stream->batch_cmd);
        stream->batch_cmd = VK_NULL_HANDLE;
    }
    vkDestroyCommandPool(ctx->device, stream->cmd_pool, NULL);
    free(stream->cmds);
    free(stream->cmd_values);
    free(stream->free_cmds);
    for (size_t i = 0; i < stream->dpool_count; i++)
        vkDestroyDescriptorPool(ctx->device, stream->dpools[i], NULL);
    free(stream->dpools);
    free(stream->dpool_values);
    for (size_t i = 0; i < stream->free_dpool_count; i++)
        vkDestroyDescriptorPool(ctx->device, stream->free_dpools[i], NULL);
    free(stream->free_dpools);
    for (size_t i = 0; i < stream->tmp_count; i++)
        acuda_free(ctx, stream->tmp_bufs[i]);
    free(stream->tmp_bufs);
    free(stream->tmp_values);
    free(stream->wait_sems);
    free(stream->wait_vals);
    vkDestroySemaphore(ctx->device, stream->timeline, NULL);
    free(stream);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_synchronize(acuda_context_t ctx, acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize in s=%p last=%llu\n",(void*)stream,(unsigned long long)s->last_submitted); fflush(_f); fclose(_f);} } }
    st = stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize flushed st=%d last=%llu\n",(int)st,(unsigned long long)s->last_submitted); fflush(_f); fclose(_f);} } }
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: flush failed st=%d\n",
                    (int)st);
        return st;
    }
    if (s->last_submitted == 0) {
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize out st=0 (nothing)\n"); fflush(_f); fclose(_f);} } }
        return ACUDA_SUCCESS;
    }
    st = stream_wait_value(ctx, s, s->last_submitted, ACUDA_SYNC_TIMEOUT_NS);
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize waited st=%d\n",(int)st); fflush(_f); fclose(_f);} } }
    if (st != ACUDA_SUCCESS) {
        uint64_t done = 0;
        VkResult gvr = vkGetSemaphoreCounterValue(ctx->device, s->timeline,
                                                  &done);
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: wait_value failed st=%d want=%llu have=%llu getvr=%d\n",
                    (int)st, (unsigned long long)s->last_submitted,
                    (unsigned long long)done, (int)gvr);
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize out st=%d (waitfail)\n",(int)st); fflush(_f); fclose(_f);} } }
        return st;
    }
    st = stream_reclaim(ctx, s);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: reclaim failed st=%d\n",
                    (int)st);
    }
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT acuda_stream_synchronize out st=%d\n",(int)st); fflush(_f); fclose(_f);} } }
    return st;
}
