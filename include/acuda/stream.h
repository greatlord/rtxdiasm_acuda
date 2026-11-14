// ACuda streams (in-order queues) and events. C ABI.
//
// Streams serialize submissions in order; different streams may overlap.
// Sync is real Vulkan timeline-semaphore sync, not mocked.
// NULL stream = context default stream. Single-threaded (see docs).
#ifndef ACUDA_STREAM_H
#define ACUDA_STREAM_H

#include "acuda/context.h"

#ifdef __cplusplus
extern "C" {
#endif

acuda_status_t acuda_stream_create(acuda_context_t ctx, acuda_stream_t *out_stream);
acuda_status_t acuda_stream_destroy(acuda_context_t ctx, acuda_stream_t stream);
// NULL stream synchronizes the default stream.
acuda_status_t acuda_stream_synchronize(acuda_context_t ctx, acuda_stream_t stream);

acuda_status_t acuda_event_create(acuda_context_t ctx, acuda_event_t *out_event);
acuda_status_t acuda_event_destroy(acuda_context_t ctx, acuda_event_t event);
// Records a marker on stream (or default). May be re-recorded.
acuda_status_t acuda_event_record(acuda_context_t ctx, acuda_event_t event,
                                  acuda_stream_t stream);
acuda_status_t acuda_event_synchronize(acuda_context_t ctx, acuda_event_t event);
// ACUDA_SUCCESS if last record completed, ACUDA_ERROR_NOT_READY if pending,
// ACUDA_ERROR_UNINITIALIZED if never recorded.
acuda_status_t acuda_event_query(acuda_context_t ctx, acuda_event_t event);
// GPU time between two recorded events, in nanoseconds. Waits for both
// events when needed. Returns ACUDA_ERROR_NOT_SUPPORTED when the device
// or queue family has no valid timestamp counter.
acuda_status_t acuda_event_elapsed(acuda_context_t ctx, acuda_event_t start,
                                   acuda_event_t stop, uint64_t *out_ns);
// Makes later work on stream wait for event's last record.
acuda_status_t acuda_stream_wait_event(acuda_context_t ctx, acuda_stream_t stream,
                                       acuda_event_t event);

#ifdef __cplusplus
}
#endif

#endif
