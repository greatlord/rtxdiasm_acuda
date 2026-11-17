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

int stream_pending_dpool(acuda_stream_t s, VkDescriptorPool pool);
int stream_pending_buffer(acuda_stream_t s, acuda_buffer_t buf);
void stream_discard_batch(acuda_context_t ctx, acuda_stream_t s);
acuda_status_t stream_begin_batch(acuda_context_t ctx, acuda_stream_t s);
acuda_status_t stream_flush_blocking_peers(acuda_context_t ctx,
                                           acuda_stream_t stream);
uint32_t stream_sync_wait_count(acuda_context_t ctx, acuda_stream_t stream);
void stream_fill_sync_waits(acuda_context_t ctx, acuda_stream_t stream,
                            VkSemaphoreSubmitInfo *w, uint32_t *k);

extern int stream_prof_on;
extern uint64_t stream_prof_cmds, stream_prof_flushes, stream_prof_record, stream_prof_submit;
double stream_prof_now_ns(void);
void stream_prof_dump(void);

#endif
