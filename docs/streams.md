# Streams and queue concurrency

CUDA applications may use several streams for independent work. ACuda now
maps user streams onto a small pool of queues from the primary compute
family instead of serializing every stream through one queue.

## Queue selection

- Queue 0 remains the default stream's queue.
- Extra queues are handed out round-robin by `acuda_stream_create()` and
  `cuStreamCreate()`.
- Async multi-queue is the default: when the primary family exposes more
  than one queue, the runtime automatically reserves up to four spare
  queues for user streams. No configuration is required.
- `ACUDA_STREAM_QUEUES=0..8` caps the number of extra user queues.
- `ACUDA_ASYNC_STREAMS=0` disables the queue pool entirely and forces the
  legacy serialized mode: every stream submits to one Vulkan queue. This
  also disables the keep-busy queue and overrides `ACUDA_STREAM_QUEUES`.
- `ACUDA_QUEUE_FAMILY=N` forces a Vulkan queue family (for example a
  compute-only family). The value must support compute; otherwise the
  normal automatic selection is used.
- If the queue family exposes only one queue, all streams share it and the
  old serialized behavior remains. This is common on some iGPUs.

`ACUDA_DEBUG_STREAMS=1` prints the selected queue count and whether the
keep-busy queue is active.

## CUDA ordering semantics

- Operations within one stream remain ordered.
- `cuStreamCreate(..., CU_STREAM_NON_BLOCKING)` streams do not participate
  in legacy default-stream synchronization. Use events for dependencies.
- Blocking streams and the default stream follow CUDA's legacy rule: the
  default stream orders against every blocking stream. This is implemented
  with timeline-semaphore waits; open blocking batches are flushed first.
- Non-blocking streams can overlap with each other and with the default
  stream, subject to the application's own event ordering.

## Keep-busy

The `ACUDA_HIGH_PERF` keep-busy thread only gets a queue when a spare queue
remains after the user request. Stream concurrency has priority; when all
queues are user queues, keep-busy is disabled (`ACUDA_DEBUG_STREAMS=1`
reports this).

## Limitations

- Queues from the same family are used, so no Vulkan ownership transfers
  are needed. Cross-family compute queues are not used.
- Independent queues are best-effort: Vulkan does not guarantee that two
  queues execute concurrently on the hardware.
- Buffer hazards across streams are the application's responsibility unless
  it uses CUDA events; this matches CUDA's stream model.
