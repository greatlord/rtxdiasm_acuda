#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

int stream_pending_dpool(acuda_stream_t s, VkDescriptorPool pool) {
    if (s->pend_dpool_count == s->pend_dpool_cap) {
        size_t ncap = s->pend_dpool_cap ? s->pend_dpool_cap * 2 : 8;
        VkDescriptorPool *np =
            realloc(s->pend_dpools, sizeof(*np) * ncap);
        if (!np)
            return -1;
        s->pend_dpools = np;
        s->pend_dpool_cap = ncap;
    }
    s->pend_dpools[s->pend_dpool_count++] = pool;
    return 0;
}

int stream_pending_buffer(acuda_stream_t s, acuda_buffer_t buf) {
    if (s->pend_buf_count == s->pend_buf_cap) {
        size_t ncap = s->pend_buf_cap ? s->pend_buf_cap * 2 : 8;
        acuda_buffer_t *nb =
            realloc(s->pend_bufs, sizeof(*nb) * ncap);
        if (!nb)
            return -1;
        s->pend_bufs = nb;
        s->pend_buf_cap = ncap;
    }
    s->pend_bufs[s->pend_buf_count++] = buf;
    return 0;
}

acuda_status_t acuda_stream_track_dpool_pending(acuda_context_t ctx,
                                                acuda_stream_t stream,
                                                VkDescriptorPool pool) {
    (void)ctx;
    if (!stream || pool == VK_NULL_HANDLE)
        return ACUDA_ERROR_INVALID_VALUE;
    return stream_pending_dpool(stream, pool) == 0
               ? ACUDA_SUCCESS
               : ACUDA_ERROR_OUT_OF_MEMORY;
}

acuda_status_t acuda_stream_track_buffer_pending(acuda_context_t ctx,
                                                 acuda_stream_t stream,
                                                 acuda_buffer_t buf) {
    (void)ctx;
    if (!stream || !buf)
        return ACUDA_ERROR_INVALID_VALUE;
    return stream_pending_buffer(stream, buf) == 0
               ? ACUDA_SUCCESS
               : ACUDA_ERROR_OUT_OF_MEMORY;
}

void stream_discard_batch(acuda_context_t ctx, acuda_stream_t s) {
    if (s->batch_cmd) {
        vkResetCommandBuffer(s->batch_cmd, 0);
        stream_push_free_cmd(s, s->batch_cmd);
        s->batch_cmd = VK_NULL_HANDLE;
    }
    for (size_t i = 0; i < s->pend_dpool_count; i++)
        vkDestroyDescriptorPool(ctx->device, s->pend_dpools[i], NULL);
    s->pend_dpool_count = 0;
    for (size_t i = 0; i < s->pend_buf_count; i++)
        acuda_free(ctx, s->pend_bufs[i]);
    s->pend_buf_count = 0;
    s->batch_count = 0;
    s->batch_serialize = 0;
}

acuda_status_t stream_begin_batch(acuda_context_t ctx,
                                         acuda_stream_t s) {
    if (s->batch_cmd)
        return ACUDA_SUCCESS;
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT begin_batch in s=%p ls=%llu\n",(void*)s,(unsigned long long)s->last_submitted); fflush(_f); fclose(_f);} } }
    // Recycle completed batches before recording a new one, so long
    // streaming runs do not accumulate command buffers/descriptor pools.
    acuda_status_t rst = stream_reclaim(ctx, s);
    if (rst != ACUDA_SUCCESS) {
        { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT begin_batch reclaim_err st=%d\n",(int)rst); fflush(_f); fclose(_f);} } }
        return rst;
    }
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (s->free_cmd_count > 0) {
        cmd = s->free_cmds[--s->free_cmd_count];
    } else {
        VkCommandBufferAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = s->cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkResult avr = vkAllocateCommandBuffers(ctx->device, &ai, &cmd);
        if (avr != VK_SUCCESS)
            return acuda_from_vk(avr);
    }
    VkResult vr = vkResetCommandBuffer(cmd, 0);
    if (vr != VK_SUCCESS) {
        stream_push_free_cmd(s, cmd);
        return acuda_from_vk(vr);
    }
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vr = vkBeginCommandBuffer(cmd, &bi);
    if (vr != VK_SUCCESS) {
        stream_push_free_cmd(s, cmd);
        return acuda_from_vk(vr);
    }
    s->batch_cmd = cmd;
    s->batch_count = 0;
    s->batch_gen++;
    s->batch_serialize = 0;
    // A fresh command buffer has no bound pipeline/descriptors/push data.
    s->bind_pipeline = VK_NULL_HANDLE;
    s->bind_layout = VK_NULL_HANDLE;
    s->bind_set_count = 0;
    s->bind_push_bytes = 0;
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT begin_batch out st=0 cmd=%p\n",(void*)cmd); fflush(_f); fclose(_f);} } }
    return ACUDA_SUCCESS;
}
