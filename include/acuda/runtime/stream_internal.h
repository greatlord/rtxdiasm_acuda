#ifndef ACUDA_STREAM_INTERNAL_H
#define ACUDA_STREAM_INTERNAL_H

#include "acuda/runtime/internal.h"

acuda_status_t stream_flush_impl(acuda_context_t ctx,
                                 acuda_stream_t stream,
                                 const VkSemaphore *extra_sems,
                                 const uint64_t *extra_vals,
                                 uint32_t extra_count,
                                 uint64_t *out_value);

#endif
