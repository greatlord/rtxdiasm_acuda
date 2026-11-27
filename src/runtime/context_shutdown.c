#include <stdlib.h>
#include <stdio.h>
#include "acuda/runtime/internal.h"
#include "acuda/runtime/context_internal.h"

acuda_status_t acuda_shutdown(acuda_context_t ctx) {
    if (!ctx)
        return ACUDA_ERROR_INVALID_HANDLE;
    // Stop the keep-busy thread first: it owns a live stream whose command
    // buffer must not be touched by the shutdown path on this thread.
    acuda_stop_high_perf(ctx);
    { const char *_tp = getenv("ACUDA_RUNTIME_TRACE_FILE"); if (_tp && *_tp) { FILE *_f = fopen(_tp,"a"); if(_f){ fprintf(_f,"RT shutdown->flush_all\n"); fflush(_f); fclose(_f);} } }
    acuda_stream_flush_all(ctx);
    if (ctx->lock_buf)
        acuda_free(ctx, ctx->lock_buf);
    if (ctx->default_stream)
        acuda_stream_destroy(ctx, ctx->default_stream);
    acuda_launch_cache_clear_ctx(ctx);
    // Vulkan requires all queues to be idle before the device is
    // destroyed.  flush_all() only submits; the default stream is
    // waited by stream_destroy below, but any other stream is not.
    // An idle wait here also covers a teardown that races the last
    // dispatch (e.g. PhysX recreating its context between presets).
    if (ctx->device)
        vkDeviceWaitIdle(ctx->device);
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
