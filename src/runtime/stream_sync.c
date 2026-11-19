#include <stdlib.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

acuda_status_t stream_reclaim(acuda_context_t ctx, acuda_stream_t s) {
    acuda_runtime_canary_check("rt_stream_reclaim_in");
    const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE");
    uint64_t done = 0;
    if (s->last_submitted > 0) {
        VkResult vr = vkGetSemaphoreCounterValue(ctx->device, s->timeline, &done);
        if (vr != VK_SUCCESS) {
            if (_tp && *_tp) { FILE *_f=fopen(_tp,"a"); if(_f){fprintf(_f,"RT reclaim in ls=%llu vr=%d ->err\n",(unsigned long long)s->last_submitted,(int)vr);fflush(_f);fclose(_f);} }
            return acuda_from_vk(vr);
        }
    }
    if (_tp && *_tp) { FILE *_f=fopen(_tp,"a"); if(_f){fprintf(_f,"RT reclaim in ls=%llu done=%llu cmds=%zu dp=%zu tmp=%zu\n",(unsigned long long)s->last_submitted,(unsigned long long)done,(size_t)s->cmd_count,(size_t)s->dpool_count,(size_t)s->tmp_count);fflush(_f);fclose(_f);} }
    size_t keep = 0;
    for (size_t i = 0; i < s->cmd_count; i++) {
        if (s->cmd_values[i] <= done) {
            if (vkResetCommandBuffer(s->cmds[i], 0) == VK_SUCCESS)
                stream_push_free_cmd(s, s->cmds[i]);
        } else {
            s->cmds[keep] = s->cmds[i];
            s->cmd_values[keep] = s->cmd_values[i];
            keep++;
        }
    }
    s->cmd_count = keep;
    keep = 0;
    for (size_t i = 0; i < s->dpool_count; i++) {
        if (s->dpool_values[i] <= done) {
            vkResetDescriptorPool(ctx->device, s->dpools[i], 0);
            stream_push_free_dpool(s, s->dpools[i]);
        } else {
            s->dpools[keep] = s->dpools[i];
            s->dpool_values[keep] = s->dpool_values[i];
            keep++;
        }
    }
    s->dpool_count = keep;
    keep = 0;
    for (size_t i = 0; i < s->tmp_count; i++) {
        if (s->tmp_values[i] <= done) {
            acuda_free(ctx, s->tmp_bufs[i]);
        } else {
            s->tmp_bufs[keep] = s->tmp_bufs[i];
            s->tmp_values[keep] = s->tmp_values[i];
            keep++;
        }
    }
    s->tmp_count = keep;
    if (_tp && *_tp) { FILE *_f=fopen(_tp,"a"); if(_f){fprintf(_f,"RT reclaim out cmds=%zu dp=%zu tmp=%zu\n",(size_t)s->cmd_count,(size_t)s->dpool_count,(size_t)s->tmp_count);fflush(_f);fclose(_f);} }
    return ACUDA_SUCCESS;
}

acuda_status_t stream_wait_value(acuda_context_t ctx, acuda_stream_t s,
                                        uint64_t value, uint64_t timeout_ns) {
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &s->timeline,
        .pValues = &value,
    };
    VkResult vr = vkWaitSemaphores(ctx->device, &wi, timeout_ns);
    return acuda_from_vk(vr);
}

// ---- Open-batch recording -------------------------------------------
// Small kernels can be recorded into one command buffer and submitted
// together; a batch is flushed by its size threshold, stream sync,
// events, or direct-queue work. Pending resources stay attached to the
// batch until it is submitted and are then tracked against its timeline
// value.
