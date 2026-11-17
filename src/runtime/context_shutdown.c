#include <stdlib.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"

acuda_status_t acuda_shutdown(acuda_context_t ctx) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    // Stop the keep-busy thread first: it owns a live stream whose command
    // buffer must not be touched by the shutdown path on this thread.
    acuda_stop_high_perf(ctx);
    acuda_stream_flush_all(ctx);
    if (ctx->lock_buf)
        acuda_free(ctx, ctx->lock_buf);
    if (ctx->default_stream)
        acuda_stream_destroy(ctx, ctx->default_stream);
    acuda_launch_cache_clear_ctx(ctx);
    if (ctx->device)
        vkDestroyDevice(ctx->device, NULL);
    free(ctx->devices);
    free(ctx->device_original);
    if (ctx->instance)
        destroy_messenger(ctx);
    if (ctx->instance)
        vkDestroyInstance(ctx->instance, NULL);
    free(ctx->buffers);
    free(ctx);
    return ACUDA_SUCCESS;
}
