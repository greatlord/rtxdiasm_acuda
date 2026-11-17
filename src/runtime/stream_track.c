#include <stdlib.h>
#include <string.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/stream_internal.h"

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
