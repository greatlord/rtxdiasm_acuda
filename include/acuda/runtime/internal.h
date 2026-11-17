// Internal (non-public) runtime structures and helpers.
#ifndef ACUDA_INTERNAL_H
#define ACUDA_INTERNAL_H

#include <vulkan/vulkan.h>
#include "acuda/compiler.h"
#include "acuda/context.h"
#include "acuda/stream.h"
#include "acuda/memory.h"
#include "acuda/hostcall.h"

// Conditional trace logging.  Use instead of open-coding
// `if (getenv("ACUDA_TRACE")) fprintf(stderr, ...)` at every site.
#define ACUDA_DEBUG_PRINT(...) \
    do { \
        if (getenv("ACUDA_TRACE")) \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)
#define ACUDA_DEBUG_PRINT_PARAMS(...) \
    do { \
        if (getenv("ACUDA_TRACE_PARAMS")) \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)
#define ACUDA_DEBUG_PRINT_PROC(...) \
    do { \
        if (getenv("ACUDA_TRACE_PROC")) \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

// Opaque kernel handle (defined in the compiler library).
typedef struct acuda_kernel acuda_kernel_t;

// Sentinel in the tex-handle array for a null pointer argument.  A
// descriptor still needs a valid dummy buffer, but the device-address
// push must remain zero so the shader's null check takes the right path.
#define ACUDA_NULL_PARAM_HANDLE ((acuda_buffer_t)(uintptr_t)1)

#define ACUDA_SYNC_TIMEOUT_NS (30000000000ull)

// Queue pool used by non-default streams.  The primary queue stays reserved
// for the default stream; extra queues let CUDA-level streams overlap.
#define ACUDA_MAX_USER_QUEUES 8

struct acuda_stream_impl {
    acuda_context_t ctx;
    // Queue this stream submits to.  NULL means the context's primary
    // queue; the high-perf keep-busy stream uses the secondary queue so
    // it never submits concurrently with user work on the primary one.
    VkQueue queue;
    uint32_t queue_family;    // family index that queue belongs to
    int blocking;             // legacy default-stream synchronization
    int in_flush;             // guards cross-stream peer flushing
    VkSemaphore timeline;
    uint64_t next_value;      // signal value for the next submission
    uint64_t last_submitted;  // 0 = nothing submitted yet
    // Command buffers are allocated from one persistent pool and
    // recycled (vkResetCommandBuffer) instead of being reallocated per
    // launch: vkAllocateCommandBuffers is expensive on some drivers.
    VkCommandPool cmd_pool;
    VkCommandBuffer *cmds;    // outstanding command buffers
    uint64_t *cmd_values;
    size_t cmd_count;
    size_t cmd_cap;
    VkCommandBuffer *free_cmds;
    size_t free_cmd_count;
    size_t free_cmd_cap;
    // Descriptor pools are recycled the same way. Any pool can serve any
    // layout as long as its descriptor counts suffice; a failed
    // allocation falls back to a fresh, exactly-sized pool.
    VkDescriptorPool *dpools; // outstanding per-launch pools
    uint64_t *dpool_values;
    size_t dpool_count;
    size_t dpool_cap;
    VkDescriptorPool *free_dpools;
    size_t free_dpool_count;
    size_t free_dpool_cap;
    acuda_buffer_t *tmp_bufs; // per-launch temporary buffers (surface meta)
    uint64_t *tmp_values;
    size_t tmp_count;
    size_t tmp_cap;
    int last_serialized;      // previous submission touched images/textures
    // Open batch: commands recorded but not submitted yet. Flushed by a
    // batch-size threshold, events, transfers that synchronize, image
    // work, or stream sync. Amortizes vkQueueSubmit2 for small kernels.
    uint32_t batch_max;
    VkCommandBuffer batch_cmd;
    uint32_t batch_count;
    uint64_t batch_gen;   // increments for each new command buffer
    int batch_serialize;
    uint64_t id; // globally unique (cached bindings survive stream frees)
    VkDescriptorPool *pend_dpools; // resources owned by the open batch
    size_t pend_dpool_count;
    size_t pend_dpool_cap;
    acuda_buffer_t *pend_bufs;
    size_t pend_buf_count;
    size_t pend_buf_cap;
    struct acuda_stream_impl *next; // context stream list
    // Command-buffer bind state for the open batch: skip redundant
    // pipeline/descriptor/push-constant commands between launches.
    VkPipeline bind_pipeline;
    VkPipelineLayout bind_layout;
    uint32_t bind_set_count;
    VkDescriptorSet bind_sets[1 + ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t bind_push_bytes;
    uint8_t bind_push[512];
    VkSemaphore *wait_sems;   // pending cross-stream waits (from events)
    uint64_t *wait_vals;
    size_t wait_count;
    size_t wait_cap;
};

struct acuda_event_impl {
    acuda_context_t ctx;
    VkSemaphore timeline; // owned; value k = k-th record generation
    uint64_t value;       // 0 = never recorded
    VkQueryPool ts_pool;  // optional GPU timestamp (1 query)
};

struct acuda_buffer_impl {
    int kind; // 0 = buffer, 1 = buffer texture, 2 = image, 3 = texture, 4 = sampler
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    VkDeviceAddress addr; // bufferDeviceAddress value (0 when unavailable)
    // kind 2/3: image-backed texture storage.
    VkImage image;
    VkImageView view;
    VkImageView uint_view; // R32_UINT alias for f32 surface CAS fallback
    VkSampler sampler;
    uint32_t width, height, depth, layers, levels, samples;
    uint32_t format; // acuda_image_desc.format
    int normalized;
    int dim1d, dim2d, dim3d; // image dimensionality (cube uses dim2d)
    int cube;
    int owns_storage; // kind 3: destroy image/view/memory/sampler on free
    // Sampler state mirrored for shader metadata queries.
    uint32_t sfilter, smip, saddr[3], snorm, scompare;
    // Multi-sample storage: `image`/`view` are the MS target and these
    // are the single-sample source used for uploads/downloads.
    VkImage ms_src_image;
    VkImageView ms_src_view;
    VkDeviceMemory ms_src_memory;
    int is_ms;
    int is_surface; // storage-capable image, kept in GENERAL layout
    // 1D textures may be tiled into a 2D image to satisfy
    // maxImageDimension1D limits (e.g. large linear texrefs). tile_w is
    // the 2D image width/stride; logical_width is the original 1D extent.
    int tiled1d;
    uint32_t tile_w;
    uint32_t logical_width;
    // Dependency state used to elide read-after-read launch barriers:
    // written since the last acquire, and read since the last write
    // (WAR ordering when submissions are allowed to overlap).
    int gpu_dirty;
    int gpu_read;
    int host_visible; // memory can be mapped on the host
};

#define ACUDA_MAX_HOST_ENTRIES 16

typedef struct {
    char name[ACUDA_HOST_NAME_MAX];
    acuda_host_call_fn fn;
    void *user;
} acuda_host_entry_t;

struct acuda_context_impl {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
    int validation_on;
    uint32_t validation_errors;
    VkPhysicalDevice *devices;
    // Original Vulkan enumeration index for each exposed device slot.
    // acuda_init moves the selected device to slot 0; this preserves the
    // index clients use for ACUDA_DEVICE_ONLY.
    uint32_t *device_original;
    uint32_t device_total;
    uint32_t selected;
    VkPhysicalDevice phys;
    VkPhysicalDeviceProperties props;
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
    VkDevice device;
    VkQueue queue;
    VkQueue user_queues[ACUDA_MAX_USER_QUEUES];
    uint32_t nuser_queues;
    uint32_t next_user_queue;
    VkQueue bg_queue;          // secondary queue used for ACUDA_HIGH_PERF keep-busy
    uint32_t bg_queue_family;  // family index for bg_queue (may differ)
    uint32_t queue_family;
    uint32_t mem_type;     // host-visible coherent type
    uint32_t mem_type_dev; // device-local type (== mem_type if none)
    int f64_ok; // shaderFloat64 available + enabled (f64 kernels need it)
    int i64_atomics_ok; // 64-bit buffer+shared atomics available + enabled
    int desc_idx_ok; // runtime descriptor arrays + variable counts
    int subgroup_size;   // default subgroup size
    int subgroup32_ok;   // subgroupSizeControl can require 32 for compute
    uint32_t subgroup_min, subgroup_max; // control range
    uint32_t pref_subgroup_size; // ACUDA_SUBGROUP_SIZE override (0 = default)
    uint32_t subgroup_ops; // VkSubgroupFeatureFlags supportedOperations
    uint32_t subgroup_stages; // VkShaderStageFlags supportedStages
    uint32_t max_workgroup_subgroups; // Vulkan13 maxComputeWorkgroupSubgroups
    uint32_t timestamp_bits; // valid timestamp bits on the compute queue
    VkDeviceSize max_buffer_size; // maintenance4 maxBufferSize (0 = unknown)
    uint32_t max_image_dimension1d;
    uint32_t max_image_dimension2d;
    int dynamic_rendering_ok; // 1.3 dynamicRendering available + enabled
    int full_subgroups_ok;    // 1.3 computeFullSubgroups available + enabled
    int dozen;                // Mesa Dozen (WSL D3D12) compatibility mode
    int maintenance4_ok;      // LocalSizeId / maxBufferSize support
    uint32_t max_block_x;     // device maxComputeWorkGroupSize[0]
    // VK_KHR_pipeline_executable_properties (optional; ISA introspection).
    int pipe_exec_ok;
    PFN_vkGetPipelineExecutablePropertiesKHR get_pipe_exec_props;
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR get_pipe_exec_reprs;
    int bda_ok;          // bufferDeviceAddress available + enabled
    acuda_stream_t default_stream;
    acuda_stream_t streams; // all live streams (for flush-all)
    acuda_buffer_t lock_buf; // hidden global surface min/max lock
    acuda_host_entry_t host_calls[ACUDA_MAX_HOST_ENTRIES];
    uint32_t nhost_calls;
    int host_servicing; // re-entrancy guard for the callback loop
    // Buffers allocated by acuda_malloc, so raw launches can map their
    // memory from a VkBuffer alone (host pointer callback arguments).
    acuda_buffer_t *buffers;
    size_t nbufs;
    size_t capbufs;
    // Per-device compiler capabilities/config (thread-safe per compile).
    acuda_compiler_caps_t compiler_caps;
    // Keep-busy / clock-ramp resources (ACUDA_HIGH_PERF).
    VkBuffer bg_buffer;
    VkDeviceMemory bg_memory;
    VkCommandPool bg_pool;
    VkCommandBuffer bg_cmd;
    VkFence bg_fence;
    volatile int bg_stop;
    void *bg_thread;          // HANDLE on Windows, heap pthread_t* on POSIX
    int bg_thread_created;
    int bg_high_perf;
    // PTX-based high-perf (preferred when available).
    void *bg_kernel;          // acuda_kernel_t*
    void *bg_buf;             // acuda_buffer_t*
    uint32_t bg_reps;
    acuda_stream_t bg_stream;
    struct acuda_context_impl *bg_ctx; // isolated unvalidated keep-busy context
};

static inline acuda_status_t acuda_from_vk(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return ACUDA_SUCCESS;
    case VK_TIMEOUT: return ACUDA_ERROR_TIMEOUT;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return ACUDA_ERROR_OUT_OF_MEMORY;
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_FEATURE_NOT_PRESENT: return ACUDA_ERROR_NOT_SUPPORTED;
    default: return ACUDA_ERROR_INTERNAL;
    }
}

// Record callback used by acuda_stream_submit.
typedef acuda_status_t (*acuda_record_fn)(VkCommandBuffer cmd, void *user);

// Resolves NULL stream to the default stream. Rejects NULL ctx.
acuda_status_t acuda_resolve_stream(acuda_context_t ctx, acuda_stream_t stream,
                                    acuda_stream_t *out);

// Creates a stream bound to a specific queue/family. Used for the
// ACUDA_HIGH_PERF keep-busy queue when it lives in another family.
acuda_status_t acuda_stream_create_on_queue(acuda_context_t ctx, VkQueue queue,
                                            uint32_t queue_family,
                                            acuda_stream_t *out_stream);

// User-facing stream creation with CU_STREAM_NON_BLOCKING-style flags.
acuda_status_t acuda_stream_create_flags(acuda_context_t ctx, unsigned flags,
                                         acuda_stream_t *out_stream);

// Default stream: always the primary queue, linked for flush-all and
// legacy default-stream synchronization.
acuda_status_t acuda_stream_create_primary(acuda_context_t ctx,
                                           acuda_stream_t *out_stream);

// Stops the ACUDA_HIGH_PERF keep-busy thread (idempotent). Safe to call
// without tearing down the device, e.g. from an atexit handler.
void acuda_high_perf_shutdown(acuda_context_t ctx);

// Records one command buffer (fresh pool) and submits it on the stream,
// waiting for stream order + pending event waits. Signals the stream
// timeline and any extra semaphores. Never blocks on completion.
// The signaled stream value is returned in out_value (may be NULL).
// serialize=1 forces this submission to wait for the previous one; used
// for image/texture work and event markers, which are not covered by the
// per-buffer dependency tracking.
acuda_status_t acuda_stream_submit(acuda_context_t ctx, acuda_stream_t stream,
                                   acuda_record_fn record, void *user,
                                   const VkSemaphore *extra_sems,
                                   const uint64_t *extra_vals,
                                   uint32_t extra_count,
                                   uint64_t *out_value, int serialize);
// Append to the stream's open command-buffer batch (no submit yet).
acuda_status_t acuda_malloc_host(acuda_context_t ctx, acuda_buffer_t *out,
                                 size_t bytes);
acuda_status_t acuda_stream_append(acuda_context_t ctx, acuda_stream_t stream,
                                   acuda_record_fn record, void *user,
                                   int serialize);
// Submit the open batch if it reached the configured size.
acuda_status_t acuda_stream_maybe_flush(acuda_context_t ctx,
                                        acuda_stream_t stream);
// Check whether a cached command buffer can be reused now.
acuda_status_t acuda_stream_cached_ready(acuda_context_t ctx,
                                         acuda_stream_t stream);
// Submit a previously recorded command buffer verbatim. Returns
// ACUDA_ERROR_NOT_READY when the stream is not idle enough to reuse it.
acuda_status_t acuda_stream_submit_cached(acuda_context_t ctx,
                                          acuda_stream_t stream,
                                          VkCommandBuffer cmd,
                                          uint64_t *out_value);
// Submit the open batch unconditionally.
acuda_status_t acuda_stream_flush(acuda_context_t ctx, acuda_stream_t stream);
// Flush every live stream (used before direct-queue image work).
acuda_status_t acuda_stream_flush_all(acuda_context_t ctx);
// Attach resources to the open batch; ownership transfers on success.
acuda_status_t acuda_stream_track_dpool_pending(acuda_context_t ctx,
                                                acuda_stream_t stream,
                                                VkDescriptorPool pool);
acuda_status_t acuda_stream_track_buffer_pending(acuda_context_t ctx,
                                                 acuda_stream_t stream,
                                                 acuda_buffer_t buf);

// Probe whether an R32_SFLOAT surface can expose the R32_UINT alias view
// used by the portable f32 min/max CAS fallback.
int acuda_surface_alias_supported(acuda_context_t ctx);

// Conservative full-buffer barrier (see barrier.c).
void acuda_buf_barrier(VkCommandBuffer cmd, VkBuffer buf,
                       VkDeviceSize offset, VkDeviceSize size);

// Retires a per-launch descriptor pool once submission value completes.
acuda_status_t acuda_stream_track_dpool(acuda_context_t ctx,
                                        acuda_stream_t stream,
                                        VkDescriptorPool pool,
                                        uint64_t value);
// Retires a per-launch temporary buffer once submission completes.
acuda_status_t acuda_stream_track_buffer(acuda_context_t ctx,
                                         acuda_stream_t stream,
                                         acuda_buffer_t buf,
                                         uint64_t value);

// Frees cached pipelines for a context (called from acuda_shutdown).
void acuda_launch_cache_clear_ctx(acuda_context_t ctx);

// Max threads per block (Vulkan workgroup x): pipelines specialize
// the workgroup-size spec constants per launch, so any 1..MAX works.
#define ACUDA_MAX_BLOCK_X 1024

// Host-call registry lookup (hostcall.c). NULL when not registered.
acuda_host_entry_t *acuda_host_call_find(acuda_context_t ctx,
                                        const char *name);

// Finds a buffer allocated by acuda_malloc on this context.
acuda_buffer_t acuda_buffer_find(acuda_context_t ctx, VkBuffer buffer);

// Stops tracking a handle (used when ownership moves to another handle).
void acuda_buffer_unregister(acuda_context_t ctx, acuda_buffer_t b);

// Raw launch over (buffer, offset, range) arrays (arena-backed driver).
acuda_status_t acuda_launch_raw(acuda_context_t ctx, acuda_kernel_t *kernel,
                               VkBuffer *bufs, const size_t *offsets,
                               const size_t *ranges, uint32_t num_bufs,
                               const void *push_data, size_t push_bytes,
                               uint32_t grid_dim_x, uint32_t block_dim_x,
                               uint32_t dyn_shared_bytes,
                               acuda_stream_t stream);

// Raw launch with host virtual addresses for the user pointer bindings.
// Used by the arena-backed driver so global pointer callback arguments
// can be dereferenced by host callbacks.
acuda_status_t acuda_launch_raw_ex(acuda_context_t ctx,
                                   acuda_kernel_t *kernel,
                                   VkBuffer *bufs,
                                   const size_t *offsets,
                                   const size_t *ranges,
                                   uint32_t num_bufs,
                                   void **host_ptrs,
                                   acuda_buffer_t *tex_handles,
                                   acuda_buffer_t *surf_table,
                                   uint32_t nsurf_table,
                                   uint32_t surf_slots,
                                   acuda_buffer_t surf_meta,
                                   const void *push_data,
                                   size_t push_bytes,
                                   uint32_t grid_dim_x,
                                   uint32_t grid_dim_y,
                                   uint32_t grid_dim_z,
                                   uint32_t block_dim_x,
                                   uint32_t block_dim_y,
                                   uint32_t block_dim_z,
                                   uint32_t dyn_shared_bytes,
                                   acuda_stream_t stream);

#endif
