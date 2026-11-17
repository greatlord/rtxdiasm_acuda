#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

acuda_status_t acuda_stream_append(acuda_context_t ctx, acuda_stream_t stream,
                                   acuda_record_fn record, void *user,
                                   int serialize) {
    if (!ctx || !stream || !record)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (stream->batch_cmd && stream->batch_count >= stream->batch_max) {
        acuda_status_t fst = stream_flush_impl(ctx, stream, NULL, NULL, 0,
                                               NULL);
        if (fst != ACUDA_SUCCESS)
            return fst;
    }
    if (stream_prof_on < 0) {
        stream_prof_on = getenv("ACUDA_PROFILE_LAUNCH") != NULL;
        if (stream_prof_on)
            atexit(stream_prof_dump);
    }
    acuda_status_t st = stream_begin_batch(ctx, stream);
    if (st != ACUDA_SUCCESS)
        return st;
    double rt0 = stream_prof_on ? stream_prof_now_ns() : 0;
    st = record(stream->batch_cmd, user);
    if (stream_prof_on) {
        stream_prof_record += (uint64_t)(stream_prof_now_ns() - rt0);
        stream_prof_cmds++;
    }
    if (st != ACUDA_SUCCESS) {
        stream_discard_batch(ctx, stream);
        return st;
    }
    stream->batch_count++;
    if (serialize)
        stream->batch_serialize = 1;
    return ACUDA_SUCCESS;
}

// True when a cached command buffer can be submitted or re-recorded: no
// open batch, and no submission still using the command buffer/resources.
acuda_status_t acuda_stream_cached_ready(acuda_context_t ctx,
                                         acuda_stream_t stream) {
    if (!ctx || !stream)
        return ACUDA_ERROR_INVALID_HANDLE;
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (s->blocking && ctx->nuser_queues > 0)
        return ACUDA_ERROR_NOT_READY;
    if (s->batch_cmd || s->batch_count != 0)
        return ACUDA_ERROR_NOT_READY;
    if (s->last_submitted > 0) {
        uint64_t done = 0;
        VkResult qr = vkGetSemaphoreCounterValue(ctx->device, s->timeline,
                                                 &done);
        if (qr != VK_SUCCESS)
            return acuda_from_vk(qr);
        if (done < s->last_submitted)
            return ACUDA_ERROR_NOT_READY;
    }
    return ACUDA_SUCCESS;
}

// Submits a command buffer that was recorded earlier and is being reused
// verbatim. Only safe when the stream has no open batch and the previous
// submission has completed, so the command buffer and the resources it
// references cannot still be in flight. Returns ACUDA_ERROR_NOT_READY when
// the caller should fall back to normal batch recording.
acuda_status_t acuda_stream_submit_cached(acuda_context_t ctx,
                                          acuda_stream_t stream,
                                          VkCommandBuffer cmd,
                                          uint64_t *out_value) {
    if (!ctx || !stream || !cmd)
        return ACUDA_ERROR_INVALID_HANDLE;
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (s->batch_cmd || s->batch_count != 0)
        return ACUDA_ERROR_NOT_READY;
    if (s->last_submitted > 0) {
        uint64_t done = 0;
        VkResult qr = vkGetSemaphoreCounterValue(ctx->device, s->timeline,
                                                 &done);
        if (qr != VK_SUCCESS)
            return acuda_from_vk(qr);
        if (done < s->last_submitted)
            return ACUDA_ERROR_NOT_READY;
    }
    st = stream_reclaim(ctx, s);
    if (st != ACUDA_SUCCESS)
        return st;

    uint32_t nwait = (uint32_t)s->wait_count +
                     stream_sync_wait_count(ctx, s);
    VkSemaphoreSubmitInfo *winfos = NULL;
    if (nwait > 0) {
        winfos = calloc(nwait, sizeof(*winfos));
        if (!winfos)
            return ACUDA_ERROR_OUT_OF_MEMORY;
        uint32_t k = 0;
        for (uint32_t i = 0; i < s->wait_count; i++, k++) {
            winfos[k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            winfos[k].semaphore = s->wait_sems[i];
            winfos[k].value = s->wait_vals[i];
            winfos[k].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        stream_fill_sync_waits(ctx, s, winfos, &k);
    }
    uint64_t sigval = ++s->next_value;
    VkSemaphoreSubmitInfo sinfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = s->timeline,
        .value = sigval,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    VkCommandBufferSubmitInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
    VkSubmitInfo2 si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = nwait,
        .pWaitSemaphoreInfos = winfos,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cbi,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &sinfo,
    };
    VkResult vr = vkQueueSubmit2(stream->queue ? stream->queue : ctx->queue, 1, &si, VK_NULL_HANDLE);
    free(winfos);
    s->wait_count = 0;
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    s->last_submitted = sigval;
    // A later normal batch must not overlap this standalone submission.
    s->last_serialized = 1;
    if (out_value)
        *out_value = sigval;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_maybe_flush(acuda_context_t ctx,
                                        acuda_stream_t stream) {
    if (!ctx || !stream)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (stream->batch_cmd && stream->batch_count >= stream->batch_max)
        return stream_flush_impl(ctx, stream, NULL, NULL, 0, NULL);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_flush(acuda_context_t ctx, acuda_stream_t stream) {
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    return stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
}
