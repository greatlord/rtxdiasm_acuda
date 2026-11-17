#ifndef ACUDA_LAUNCH_INTERNAL_H
#define ACUDA_LAUNCH_INTERNAL_H

#include "acuda/runtime/internal.h"

#define ACUDA_BIND_CACHE_N 8
#define ACUDA_BIND_CACHE_PARAMS 8

typedef struct {
    uint64_t stream_id;
    uint64_t gen;
    VkDescriptorSet dset;
    uint32_t nbufs;
    int has_handles;
    VkBuffer bufs[ACUDA_BIND_CACHE_PARAMS];
    size_t offsets[ACUDA_BIND_CACHE_PARAMS];
    size_t ranges[ACUDA_BIND_CACHE_PARAMS];
    acuda_buffer_t handles[ACUDA_BIND_CACHE_PARAMS];
    VkDeviceAddress addrs[ACUDA_BIND_CACHE_PARAMS];
} acuda_bind_cache_t;

typedef struct {
    const acuda_kernel_t *kernel;
    uint64_t serial;
    acuda_context_t ctx;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t dyn_words;
    VkShaderModule mod;
    VkDescriptorSetLayout dsl;
    VkDescriptorSetLayout *surf_dsl;
    uint32_t surf_dsl_count;
    VkPipelineLayout pl;
    VkPipeline pipe;
    acuda_bind_cache_t bind_cache[ACUDA_BIND_CACHE_N];
    uint32_t bind_next;
    acuda_bind_cache_t steady;
    VkDescriptorPool steady_pool;
    int steady_valid;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    uint64_t cmd_stream_id;
    VkDescriptorSet cmd_dset;
    uint32_t cmd_grid;
    uint32_t cmd_grid_y;
    uint32_t cmd_grid_z;
    uint32_t cmd_push_bytes;
    uint8_t cmd_push[512];
    int cmd_pending;
    int cmd_valid;
} launch_cache_t;

typedef struct {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSet sets[1 + ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t set_count;
    const void *push_data;
    size_t push_bytes;
    uint32_t grid;
    uint32_t grid_y;
    uint32_t grid_z;
    VkBuffer *bufs;
    const size_t *offsets;
    const size_t *ranges;
    uint32_t num_bufs;
    int has_storage;
    int mem_barrier;
    acuda_stream_t stream;
    acuda_buffer_t *handles;
    const uint32_t *written;
} launch_rec_t;

extern int lprof_on;
extern uint64_t lprof_n, lprof_pre;
double lprof_now(void);
void lprof_dump(void);

void bind_cache_addresses(acuda_context_t ctx, acuda_bind_cache_t *c,
                          const VkBuffer *bufs, uint32_t nbufs);
uint32_t surf_max_slots(acuda_context_t ctx, uint32_t count);
VkDescriptorType binding_vktype(uint32_t kind);
acuda_status_t cache_lookup(acuda_context_t ctx, const acuda_kernel_t *k,
                            const acuda_kernel_info_t *info,
                            uint32_t block_dim_x, uint32_t block_dim_y,
                            uint32_t block_dim_z, uint32_t dyn_words,
                            launch_cache_t **out);
acuda_status_t record_launch(VkCommandBuffer cmd, void *user);
acuda_status_t cache_record_cmd(acuda_context_t ctx, launch_cache_t *e,
                                const launch_rec_t *rec);
uint32_t tex_addr_code(uint32_t a);
void tex_meta_words(acuda_buffer_t h, uint32_t kind, uint32_t out[7]);

acuda_status_t launch_raw_once(acuda_context_t ctx,
                               acuda_kernel_t *kernel,
                               VkBuffer *bufs, const size_t *offsets,
                               const size_t *ranges, uint32_t num_bufs,
                               acuda_buffer_t *handles,
                               acuda_buffer_t *surf_table,
                               uint32_t nsurf_table, uint32_t surf_slots,
                               acuda_buffer_t surf_meta,
                               const void *push_data, size_t push_bytes,
                               uint32_t grid_dim_x, uint32_t grid_dim_y,
                               uint32_t grid_dim_z, uint32_t block_dim_x,
                               uint32_t block_dim_y, uint32_t block_dim_z,
                               uint32_t dyn_shared_bytes,
                               acuda_stream_t stream);

acuda_status_t launch_host_chain_raw(acuda_context_t ctx,
                                     acuda_kernel_t *kernel,
                                     VkBuffer *bufs,
                                     const size_t *offsets,
                                     const size_t *ranges,
                                     uint32_t num_bufs,
                                     void **host_ptrs,
                                     const void *push_data,
                                     size_t push_bytes,
                                     uint32_t grid_dim_x,
                                     uint32_t grid_dim_y,
                                     uint32_t grid_dim_z,
                                     uint32_t block_dim_x,
                                     uint32_t block_dim_y,
                                     uint32_t block_dim_z,
                                     uint32_t dyn_shared_bytes,
                                     acuda_stream_t stream,
                                     const acuda_kernel_info_t *info);

acuda_status_t launch_host_chain(acuda_context_t ctx,
                                 acuda_kernel_t *kernel,
                                 acuda_buffer_t *bufs,
                                 uint32_t num_bufs,
                                 const void *push_data,
                                 size_t push_bytes,
                                 uint32_t grid_dim_x,
                                 uint32_t grid_dim_y,
                                 uint32_t grid_dim_z,
                                 uint32_t block_dim_x,
                                 uint32_t block_dim_y,
                                 uint32_t block_dim_z,
                                 uint32_t dyn_shared_bytes,
                                 acuda_stream_t stream,
                                 const acuda_kernel_info_t *info);

#endif

