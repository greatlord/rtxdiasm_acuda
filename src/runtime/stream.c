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
    // Recycle completed batches before recording a new one, so long
    // streaming runs do not accumulate command buffers/descriptor pools.
    acuda_status_t rst = stream_reclaim(ctx, s);
    if (rst != ACUDA_SUCCESS)
        return rst;
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
    return ACUDA_SUCCESS;
}

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
