#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acuda/stream.h"
#include "acuda/runtime/internal.h"
#include "acuda/runtime/time_compat.h"

// ACUDA_PROFILE_LAUNCH: aggregate recording/submission timings.
static int prof_on = -1;
static uint64_t prof_cmds, prof_flushes, prof_record, prof_submit;
static double prof_now_ns(void) {
    return (double)acuda_now_ns();
}
static void prof_dump(void) {
    if (prof_flushes == 0)
        return;
    double c = (double)(prof_cmds ? prof_cmds : 1);
    double f = (double)prof_flushes;
    fprintf(stderr,
            "ACuda launch profile: %llu commands / %llu flushes, "
            "record %.2f us/command, submit %.2f us/flush\n",
            (unsigned long long)prof_cmds, (unsigned long long)prof_flushes,
            prof_record / c / 1e3, prof_submit / f / 1e3);
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

static int stream_push_free_cmd(acuda_stream_t s, VkCommandBuffer cmd) {
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

static int stream_push_free_dpool(acuda_stream_t s, VkDescriptorPool pool) {
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

static uint64_t stream_id_counter;

static acuda_status_t stream_reclaim(acuda_context_t ctx, acuda_stream_t s) {
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

static acuda_status_t stream_wait_value(acuda_context_t ctx, acuda_stream_t s,
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

static int stream_pending_dpool(acuda_stream_t s, VkDescriptorPool pool) {
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

static int stream_pending_buffer(acuda_stream_t s, acuda_buffer_t buf) {
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

static void stream_discard_batch(acuda_context_t ctx, acuda_stream_t s) {
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

static acuda_status_t stream_begin_batch(acuda_context_t ctx,
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
static acuda_status_t stream_flush_blocking_peers(acuda_context_t ctx,
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
static uint32_t stream_sync_wait_count(acuda_context_t ctx,
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

static void stream_fill_sync_waits(acuda_context_t ctx, acuda_stream_t stream,
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
    double st0 = prof_on ? prof_now_ns() : 0;
    vr = vkQueueSubmit2(stream->queue ? stream->queue : ctx->queue, 1, &si, VK_NULL_HANDLE);
    if (prof_on) {
        prof_submit += (uint64_t)(prof_now_ns() - st0);
        prof_flushes++;
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
    if (prof_on < 0) {
        prof_on = getenv("ACUDA_PROFILE_LAUNCH") != NULL;
        if (prof_on)
            atexit(prof_dump);
    }
    acuda_status_t st = stream_begin_batch(ctx, stream);
    if (st != ACUDA_SUCCESS)
        return st;
    double rt0 = prof_on ? prof_now_ns() : 0;
    st = record(stream->batch_cmd, user);
    if (prof_on) {
        prof_record += (uint64_t)(prof_now_ns() - rt0);
        prof_cmds++;
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

acuda_status_t acuda_stream_flush_all(acuda_context_t ctx) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    for (acuda_stream_t s = ctx->streams; s; s = s->next) {
        acuda_status_t st = stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
        if (st != ACUDA_SUCCESS)
            return st;
    }
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_submit(acuda_context_t ctx, acuda_stream_t stream,
                                   acuda_record_fn record, void *user,
                                   const VkSemaphore *extra_sems,
                                   const uint64_t *extra_vals,
                                   uint32_t extra_count,
                                   uint64_t *out_value, int serialize) {
    if (!ctx || !stream || !record)
        return ACUDA_ERROR_INVALID_HANDLE;
    acuda_status_t st =
        acuda_stream_append(ctx, stream, record, user, serialize);
    if (st != ACUDA_SUCCESS)
        return st;
    return stream_flush_impl(ctx, stream, extra_sems, extra_vals,
                             extra_count, out_value);
}

acuda_status_t acuda_stream_track_dpool(acuda_context_t ctx,
                                        acuda_stream_t stream,
                                        VkDescriptorPool pool,
                                        uint64_t value) {
    (void)ctx;
    if (!stream || pool == VK_NULL_HANDLE || value == 0)
        return ACUDA_ERROR_INVALID_VALUE;
    if (stream->dpool_count == stream->dpool_cap) {
        size_t ncap = stream->dpool_cap ? stream->dpool_cap * 2 : 4;
        VkDescriptorPool *np = malloc(sizeof(*np) * ncap);
        uint64_t *nv = malloc(sizeof(*nv) * ncap);
        if (!np || !nv) {
            free(np);
            free(nv);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(np, stream->dpools, sizeof(*np) * stream->dpool_count);
        memcpy(nv, stream->dpool_values, sizeof(*nv) * stream->dpool_count);
        free(stream->dpools);
        free(stream->dpool_values);
        stream->dpools = np;
        stream->dpool_values = nv;
        stream->dpool_cap = ncap;
    }
    stream->dpools[stream->dpool_count] = pool;
    stream->dpool_values[stream->dpool_count] = value;
    stream->dpool_count++;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_track_buffer(acuda_context_t ctx,
                                         acuda_stream_t stream,
                                         acuda_buffer_t buf,
                                         uint64_t value) {
    (void)ctx;
    if (!stream || !buf || value == 0)
        return ACUDA_ERROR_INVALID_VALUE;
    if (stream->tmp_count == stream->tmp_cap) {
        size_t ncap = stream->tmp_cap ? stream->tmp_cap * 2 : 4;
        acuda_buffer_t *nb = malloc(sizeof(*nb) * ncap);
        uint64_t *nv = malloc(sizeof(*nv) * ncap);
        if (!nb || !nv) {
            free(nb);
            free(nv);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(nb, stream->tmp_bufs, sizeof(*nb) * stream->tmp_count);
        memcpy(nv, stream->tmp_values, sizeof(*nv) * stream->tmp_count);
        free(stream->tmp_bufs);
        free(stream->tmp_values);
        stream->tmp_bufs = nb;
        stream->tmp_values = nv;
        stream->tmp_cap = ncap;
    }
    stream->tmp_bufs[stream->tmp_count] = buf;
    stream->tmp_values[stream->tmp_count] = value;
    stream->tmp_count++;
    return ACUDA_SUCCESS;
}

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
    st = stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: flush failed st=%d\n",
                    (int)st);
        return st;
    }
    if (s->last_submitted == 0)
        return ACUDA_SUCCESS;
    st = stream_wait_value(ctx, s, s->last_submitted, ACUDA_SYNC_TIMEOUT_NS);
    if (st != ACUDA_SUCCESS) {
        uint64_t done = 0;
        VkResult gvr = vkGetSemaphoreCounterValue(ctx->device, s->timeline,
                                                  &done);
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: wait_value failed st=%d want=%llu have=%llu getvr=%d\n",
                    (int)st, (unsigned long long)s->last_submitted,
                    (unsigned long long)done, (int)gvr);
        return st;
    }
    st = stream_reclaim(ctx, s);
    if (st != ACUDA_SUCCESS) {
        ACUDA_DEBUG_PRINT("acuda-trace ?? stream_synchronize: reclaim failed st=%d\n",
                    (int)st);
    }
    return st;
}
