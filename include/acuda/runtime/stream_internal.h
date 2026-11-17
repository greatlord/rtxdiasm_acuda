#ifndef ACUDA_STREAM_INTERNAL_H
#define ACUDA_STREAM_INTERNAL_H

#include "acuda/runtime/internal.h"

acuda_status_t stream_flush_impl(acuda_context_t ctx,
                                 acuda_stream_t stream,
                                 const VkSemaphore *extra_sems,
                                 const uint64_t *extra_vals,
                                 uint32_t extra_count,
                                 uint64_t *out_value);
acuda_status_t stream_reclaim(acuda_context_t ctx, acuda_stream_t s);
acuda_status_t stream_wait_value(acuda_context_t ctx, acuda_stream_t s,
                                 uint64_t value, uint64_t timeout_ns);
int stream_push_free_cmd(acuda_stream_t s, VkCommandBuffer cmd);
int stream_push_free_dpool(acuda_stream_t s, VkDescriptorPool pool);

#endif
