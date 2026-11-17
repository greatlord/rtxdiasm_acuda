#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

acuda_status_t stream_flush_impl(acuda_context_t ctx,
                                 acuda_stream_t stream,
                                 const VkSemaphore *extra_sems,
                                 const uint64_t *extra_vals,
                                 uint32_t extra_count,
                                 uint64_t *out_value);

// Submit any open batches on other blocking streams before this stream
// crosses a queue boundary.  The in_flush guard prevents cycles when two
// blocking streams submit at nearly the same time.
acuda_status_t stream_flush_blocking_peers(acuda_context_t ctx,
                                                  acuda_stream_t stream) {
    if (!stream->blocking || ctx->nuser_queues == 0 || stream->in_flush)
        return ACUDA_SUCCESS;
    stream->in_flush = 1;
    for (acuda_stream_t s = ctx->streams; s; s = s->next) {
        if (s == stream || !s->blocking || s->in_flush ||
            s->batch_count == 0)
            continue;
        acuda_status_t st = stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
        if (st != ACUDA_SUCCESS) {
            stream->in_flush = 0;
            return st;
        }
    }
    stream->in_flush = 0;
    return ACUDA_SUCCESS;
}

// CUDA's legacy default stream synchronizes with every blocking stream.
// Non-blocking streams (CU_STREAM_NON_BLOCKING) do not participate.
uint32_t stream_sync_wait_count(acuda_context_t ctx,
                                       acuda_stream_t stream) {
    if (!stream->blocking || ctx->nuser_queues == 0)
        return 0;
    if (stream == ctx->default_stream) {
        uint32_t n = 0;
        for (acuda_stream_t s = ctx->streams; s; s = s->next)
            if (s != stream && s->blocking && s->last_submitted > 0)
                n++;
        return n;
    }
    return (ctx->default_stream && ctx->default_stream != stream &&
            ctx->default_stream->last_submitted > 0)
               ? 1u
               : 0u;
}

void stream_fill_sync_waits(acuda_context_t ctx, acuda_stream_t stream,
                                   VkSemaphoreSubmitInfo *w, uint32_t *k) {
    if (!stream->blocking || ctx->nuser_queues == 0)
        return;
    if (stream == ctx->default_stream) {
        for (acuda_stream_t s = ctx->streams; s; s = s->next) {
            if (s == stream || !s->blocking || s->last_submitted == 0)
                continue;
            w[*k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            w[*k].semaphore = s->timeline;
            w[*k].value = s->last_submitted;
            w[*k].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            (*k)++;
        }
    } else if (ctx->default_stream &&
               ctx->default_stream->last_submitted > 0) {
        w[*k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        w[*k].semaphore = ctx->default_stream->timeline;
        w[*k].value = ctx->default_stream->last_submitted;
        w[*k].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        (*k)++;
    }
}

acuda_status_t stream_flush_impl(acuda_context_t ctx,
                                 acuda_stream_t stream,
                                 const VkSemaphore *extra_sems,
                                 const uint64_t *extra_vals,
                                 uint32_t extra_count,
                                 uint64_t *out_value) {
    if (!stream->batch_cmd || stream->batch_count == 0) {
        if (out_value)
            *out_value = stream->last_submitted;
        return ACUDA_SUCCESS;
    }
    acuda_status_t peer_st = stream_flush_blocking_peers(ctx, stream);
    if (peer_st != ACUDA_SUCCESS) {
        if (out_value)
            *out_value = 0;
        return peer_st;
    }
    VkCommandBuffer cmd = stream->batch_cmd;
    VkResult vr = vkEndCommandBuffer(cmd);
    if (vr != VK_SUCCESS) {
        stream_discard_batch(ctx, stream);
        if (out_value)
            *out_value = 0;
        return acuda_from_vk(vr);
    }

    int serialize = stream->batch_serialize;
    int wait_prev = stream->last_submitted > 0 &&
                    (serialize || stream->last_serialized);
    if (getenv("ACUDA_FORCE_SERIAL"))
        wait_prev = stream->last_submitted > 0;
    uint32_t nwait = (wait_prev ? 1u : 0u) +
                     (uint32_t)stream->wait_count +
                     stream_sync_wait_count(ctx, stream);
    VkSemaphoreSubmitInfo *winfos = NULL;
    if (nwait > 0) {
        winfos = calloc(nwait, sizeof(*winfos));
        if (!winfos) {
            stream_discard_batch(ctx, stream);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        uint32_t k = 0;
        if (wait_prev) {
            winfos[k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            winfos[k].semaphore = stream->timeline;
            winfos[k].value = stream->last_submitted;
            winfos[k].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            k++;
        }
        for (size_t i = 0; i < stream->wait_count; i++, k++) {
            winfos[k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            winfos[k].semaphore = stream->wait_sems[i];
            winfos[k].value = stream->wait_vals[i];
            winfos[k].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        stream_fill_sync_waits(ctx, stream, winfos, &k);
    }

    uint64_t sigval = ++stream->next_value;
    uint32_t nsig = 1 + extra_count;
    VkSemaphoreSubmitInfo *sinfos = calloc(nsig, sizeof(*sinfos));
    if (!sinfos) {
        free(winfos);
        stream_discard_batch(ctx, stream);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    sinfos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    sinfos[0].semaphore = stream->timeline;
    sinfos[0].value = sigval;
    sinfos[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    for (uint32_t i = 0; i < extra_count; i++) {
        sinfos[1 + i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        sinfos[1 + i].semaphore = extra_sems[i];
        sinfos[1 + i].value = extra_vals[i];
        sinfos[1 + i].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
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
        .signalSemaphoreInfoCount = nsig,
        .pSignalSemaphoreInfos = sinfos,
    };
    double st0 = stream_prof_on ? stream_prof_now_ns() : 0;
    vr = vkQueueSubmit2(stream->queue ? stream->queue : ctx->queue, 1, &si, VK_NULL_HANDLE);
    if (stream_prof_on) {
        stream_prof_submit += (uint64_t)(stream_prof_now_ns() - st0);
        stream_prof_flushes++;
    }
    free(winfos);
    free(sinfos);
    stream->wait_count = 0; // consumed
    if (vr != VK_SUCCESS) {
        stream_discard_batch(ctx, stream);
        if (out_value)
            *out_value = 0;
        return acuda_from_vk(vr);
    }

    if (stream->cmd_count == stream->cmd_cap) {
        size_t ncap = stream->cmd_cap ? stream->cmd_cap * 2 : 8;
        VkCommandBuffer *nc = malloc(sizeof(*nc) * ncap);
        uint64_t *nv = malloc(sizeof(*nv) * ncap);
        if (!nc || !nv) {
            free(nc);
            free(nv);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(nc, stream->cmds, sizeof(*nc) * stream->cmd_count);
        memcpy(nv, stream->cmd_values, sizeof(*nv) * stream->cmd_count);
        free(stream->cmds);
        free(stream->cmd_values);
        stream->cmds = nc;
        stream->cmd_values = nv;
        stream->cmd_cap = ncap;
    }
    stream->cmds[stream->cmd_count] = cmd;
    stream->cmd_values[stream->cmd_count] = sigval;
    stream->cmd_count++;
    stream->last_submitted = sigval;
    stream->last_serialized = serialize;
    stream->batch_cmd = VK_NULL_HANDLE;
    stream->batch_count = 0;
    stream->batch_serialize = 0;

    // Attach the resources used by this batch to its timeline value.
    for (size_t i = 0; i < stream->pend_dpool_count; i++)
        acuda_stream_track_dpool(ctx, stream, stream->pend_dpools[i],
                                 sigval);
    stream->pend_dpool_count = 0;
    for (size_t i = 0; i < stream->pend_buf_count; i++)
        acuda_stream_track_buffer(ctx, stream, stream->pend_bufs[i],
                                  sigval);
    stream->pend_buf_count = 0;
    if (out_value)
        *out_value = sigval;
    return ACUDA_SUCCESS;
}
