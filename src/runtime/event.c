#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

acuda_status_t acuda_event_create(acuda_context_t ctx, acuda_event_t *out_event) {
    if (!ctx || !out_event)
        return ACUDA_ERROR_INVALID_HANDLE;
    *out_event = NULL;
    acuda_event_t e = calloc(1, sizeof(*e));
    if (!e)
        return ACUDA_ERROR_OUT_OF_MEMORY;
    e->ctx = ctx;
    VkSemaphoreTypeCreateInfo tci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
    };
    VkSemaphoreCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &tci,
    };
    VkResult vr = vkCreateSemaphore(ctx->device, &sci, NULL, &e->timeline);
    if (vr != VK_SUCCESS) {
        free(e);
        return acuda_from_vk(vr);
    }
    if (ctx->timestamp_bits) {
        VkQueryPoolCreateInfo qci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 1,
        };
        vr = vkCreateQueryPool(ctx->device, &qci, NULL, &e->ts_pool);
        if (vr != VK_SUCCESS) {
            vkDestroySemaphore(ctx->device, e->timeline, NULL);
            free(e);
            return acuda_from_vk(vr);
        }
    }
    *out_event = e;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_event_destroy(acuda_context_t ctx, acuda_event_t event) {
    if (!ctx || !event)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (event->ts_pool)
        vkDestroyQueryPool(ctx->device, event->ts_pool, NULL);
    vkDestroySemaphore(ctx->device, event->timeline, NULL);
    free(event);
    return ACUDA_SUCCESS;
}

// Event record: write a GPU timestamp at this point in stream order. The
// empty command buffer is submitted with the event's timeline signal, so
// the timestamp covers all previously submitted stream work.
static acuda_status_t record_event(VkCommandBuffer cmd, void *user) {
    acuda_event_t e = user;
    if (e->ts_pool) {
        vkCmdResetQueryPool(cmd, e->ts_pool, 0, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                             e->ts_pool, 0);
    }
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_event_record(acuda_context_t ctx, acuda_event_t event,
                                  acuda_stream_t stream) {
    if (!ctx || !event)
        return ACUDA_ERROR_INVALID_HANDLE;
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    uint64_t val = event->value + 1;
    st = acuda_stream_submit(ctx, s, record_event, event, &event->timeline,
                             &val, 1, NULL, 1);
    if (st != ACUDA_SUCCESS)
        return st;
    event->value = val;
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_event_synchronize(acuda_context_t ctx, acuda_event_t event) {
    if (!ctx || !event)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (event->value == 0)
        return ACUDA_ERROR_UNINITIALIZED;
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &event->timeline,
        .pValues = &event->value,
    };
    VkResult vr = vkWaitSemaphores(ctx->device, &wi, ACUDA_SYNC_TIMEOUT_NS);
    return acuda_from_vk(vr);
}

acuda_status_t acuda_event_query(acuda_context_t ctx, acuda_event_t event) {
    if (!ctx || !event)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (event->value == 0)
        return ACUDA_ERROR_UNINITIALIZED;
    VkSemaphoreWaitInfo wi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &event->timeline,
        .pValues = &event->value,
    };
    VkResult vr = vkWaitSemaphores(ctx->device, &wi, 0);
    if (vr == VK_TIMEOUT)
        return ACUDA_ERROR_NOT_READY;
    return acuda_from_vk(vr);
}

// Elapsed GPU time between two recorded events, in nanoseconds. Both
// events must have been recorded; this waits for their completion.
acuda_status_t acuda_event_elapsed(acuda_context_t ctx, acuda_event_t start,
                                   acuda_event_t stop, uint64_t *out_ns) {
    if (!ctx || !start || !stop || !out_ns)
        return ACUDA_ERROR_INVALID_HANDLE;
    if (!start->ts_pool || !stop->ts_pool || start->value == 0 ||
        stop->value == 0)
        return ACUDA_ERROR_NOT_SUPPORTED;
    acuda_status_t st = acuda_event_synchronize(ctx, start);
    if (st != ACUDA_SUCCESS)
        return st;
    st = acuda_event_synchronize(ctx, stop);
    if (st != ACUDA_SUCCESS)
        return st;
    uint64_t a = 0, b = 0;
    VkResult vr = vkGetQueryPoolResults(
        ctx->device, start->ts_pool, 0, 1, sizeof(a), &a, sizeof(a),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    vr = vkGetQueryPoolResults(ctx->device, stop->ts_pool, 0, 1, sizeof(b),
                               &b, sizeof(b),
                               VK_QUERY_RESULT_64_BIT |
                                   VK_QUERY_RESULT_WAIT_BIT);
    if (vr != VK_SUCCESS)
        return acuda_from_vk(vr);
    // Timestamp counters may be narrower than 64 bits (the device
    // reports timestampValidBits); a span that crosses a wrap must be
    // reduced modulo 2^bits instead of clamping.
    uint64_t diff;
    if (ctx->timestamp_bits >= 64) {
        diff = b - a;
    } else {
        uint64_t mask = (1ull << ctx->timestamp_bits) - 1ull;
        diff = (b - a) & mask;
    }
    double period = (double)ctx->props.limits.timestampPeriod;
    *out_ns = (uint64_t)((double)diff * period + 0.5);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_stream_wait_event(acuda_context_t ctx, acuda_stream_t stream,
                                       acuda_event_t event) {
    if (!ctx || !event)
        return ACUDA_ERROR_INVALID_HANDLE;
    acuda_stream_t s = NULL;
    acuda_status_t st = acuda_resolve_stream(ctx, stream, &s);
    if (st != ACUDA_SUCCESS)
        return st;
    if (event->value == 0)
        return ACUDA_ERROR_UNINITIALIZED;
    st = stream_flush_impl(ctx, s, NULL, NULL, 0, NULL);
    if (st != ACUDA_SUCCESS)
        return st;
    if (s->wait_count == s->wait_cap) {
        size_t ncap = s->wait_cap ? s->wait_cap * 2 : 4;
        VkSemaphore *ns = malloc(sizeof(*ns) * ncap);
        uint64_t *nv = malloc(sizeof(*nv) * ncap);
        if (!ns || !nv) {
            free(ns);
            free(nv);
            return ACUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(ns, s->wait_sems, sizeof(*ns) * s->wait_count);
        memcpy(nv, s->wait_vals, sizeof(*nv) * s->wait_count);
        free(s->wait_sems);
        free(s->wait_vals);
        s->wait_sems = ns;
        s->wait_vals = nv;
        s->wait_cap = ncap;
    }
    s->wait_sems[s->wait_count] = event->timeline;
    s->wait_vals[s->wait_count] = event->value;
    s->wait_count++;
    return ACUDA_SUCCESS;
}
