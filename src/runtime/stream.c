#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/stream.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/time_compat.h"

// ACUDA_PROFILE_LAUNCH: aggregate recording/submission timings.
int stream_prof_on = -1;
uint64_t stream_prof_cmds, stream_prof_flushes, stream_prof_record, stream_prof_submit;
double stream_prof_now_ns(void) {
    return (double)acuda_now_ns();
}
void stream_prof_dump(void) {
    if (stream_prof_flushes == 0)
        return;
    double c = (double)(stream_prof_cmds ? stream_prof_cmds : 1);
    double f = (double)stream_prof_flushes;
    fprintf(stderr,
            "ACuda launch profile: %llu commands / %llu flushes, "
            "record %.2f us/command, submit %.2f us/flush\n",
            (unsigned long long)stream_prof_cmds, (unsigned long long)stream_prof_flushes,
            stream_prof_record / c / 1e3, stream_prof_submit / f / 1e3);
}

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


acuda_status_t stream_reclaim(acuda_context_t ctx, acuda_stream_t s) {
    uint64_t done = 0;
    if (s->last_submitted > 0) {
        VkResult vr = vkGetSemaphoreCounterValue(ctx->device, s->timeline, &done);
        if (vr != VK_SUCCESS)
            return acuda_from_vk(vr);
    }
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
